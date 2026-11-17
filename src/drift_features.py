"""Add drift-weighted supplier features to the existing feature set.

Input : data/processed/<scenario>/features.parquet   (from build_features.py)
Output: data/processed/<scenario>/features.parquet   (same rows + new columns)
        data/processed/<scenario>/feature_columns.json (new columns added to delay_features)
        data/processed/<scenario>/features_v1.parquet, feature_columns_v1.json
            (backup of the original feature set, written once, on the first run)
        results/<scenario>/diagnostics/feature_set_comparison.csv

Why
---
The diagnostics showed the models beat the naive baseline mainly on suppliers
whose behaviour is changing. A fixed 30-day window reacts slowly and treats a
shipment from 29 days ago the same as one from yesterday. These features
weight recent shipments more heavily, so they pick up a change sooner.

New columns (all as of 72 hours before planned arrival, like every other
history feature: only shipments whose truck had actually arrived count)
------------------------------------------------------------------------
sup_ewm_n            effective number of recent shipments: sum of weights
                     w = 0.5 ** (age_in_days / HALF_LIFE_DAYS)
sup_late_rate_ewm    exponentially weighted late rate, shrunk toward the
                     supplier's all-history late rate:
                     (sum w * late + K_RATE * rate_all) / (sum w + K_RATE)
sup_late_delay_ewm   exponentially weighted mean delay of late shipments,
                     shrunk toward the all-history mean late delay:
                     (sum w * late * delay + K_DELAY * delay_all) / (sum w * late + K_DELAY)

Shrinkage keeps a handful of recent trucks from swinging the feature wildly:
with few recent shipments the feature stays near the long-run level; with many,
it follows the recent data.

The weighted sums use the same cumulative-sum and as-of lookup as
build_features.py. A weight 0.5 ** ((c - t) / h) factors into
exp(t / tau) * exp(-c / tau), so running totals of v * exp(t / tau) looked up at
the cutoff c give every weighted sum in one pass.

The original columns are never changed. Models pick up the new columns through
models.FEATURE_SET = "v2"; set it to "v1" to use the original predictors.

Run order: generate_all -> build_features -> drift_features -> build_folds -> evaluate_models
(this script rebuilds the folds itself, so build_folds only needs rerunning if
you change it).

Usage:
    python drift_features.py                     # all scenarios
    python drift_features.py --scenario drift
    python drift_features.py --skip-compare      # features only, no fast-model comparison
"""
from __future__ import annotations

import argparse
import json
import shutil
import warnings

import numpy as np
import pandas as pd

import build_folds
from build_features import _asof, _cumulative_history, shipment_outcomes
from build_folds import PROCESSED, SCENARIOS
from models import DRIFT_COLUMNS

HALF_LIFE_DAYS = 14.0
K_RATE = 5.0       # pseudo-shipments pulling the recent late rate toward the long-run rate
K_DELAY = 2.0      # pseudo-late-shipments pulling the recent late delay toward the long-run delay
TAU = HALF_LIFE_DAYS / np.log(2)


# --------------------------------------------------------------------------- #
# Step 1: exponentially weighted history, as of each cutoff
# --------------------------------------------------------------------------- #
def ewm_sums(ships: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Weighted sums sum(w), sum(w * late), sum(w * late * delay) per shipment,
    over that supplier's shipments known by the shipment's cutoff."""
    t_known = (ships["known_time"] - origin).dt.total_seconds() / 86400
    growth = np.exp(t_known / TAU)                     # exp(t / tau), finite for a year of data
    events = ships.assign(
        w=growth,
        w_late=growth * ships["late"].fillna(0.0),
        w_late_delay=growth * np.where(ships["late"] == 1, ships["delay_hours"], 0.0),
    )
    hist = _cumulative_history(events, ["supplier_code"], ["w", "w_late", "w_late_delay"])
    at_cutoff = _asof(ships, hist, ["supplier_code"], "cutoff")
    t_cut = (ships["cutoff"] - origin).dt.total_seconds() / 86400
    decay = np.exp(-t_cut / TAU).to_numpy()[:, None]   # exp(-c / tau)
    sums = at_cutoff[["cum_w", "cum_w_late", "cum_w_late_delay"]].to_numpy() * decay
    return pd.DataFrame(sums, columns=["sum_w", "sum_w_late", "sum_w_late_delay"], index=ships.index)


def drift_features(features: pd.DataFrame) -> pd.DataFrame:
    """The three new columns, one row per delivery row of `features`."""
    ships = shipment_outcomes(features)
    origin = features["production_date"].min().normalize()
    s = ewm_sums(ships, origin)

    first = features.drop_duplicates("shipment_id").set_index("shipment_id")
    rate_all = first.loc[ships["shipment_id"], "sup_late_rate_all"].to_numpy()
    delay_all = first.loc[ships["shipment_id"], "sup_mean_late_delay_all"].to_numpy()

    out = pd.DataFrame({
        "shipment_id": ships["shipment_id"].to_numpy(),
        "sup_ewm_n": s["sum_w"].to_numpy(),
        "sup_late_rate_ewm": (s["sum_w_late"] + K_RATE * rate_all) / (s["sum_w"] + K_RATE),
        "sup_late_delay_ewm": (s["sum_w_late_delay"] + K_DELAY * delay_all) / (s["sum_w_late"] + K_DELAY),
    })
    # With no long-run history the shrinkage target is missing -> leave NaN for the fold imputer
    out.loc[np.isnan(rate_all), "sup_late_rate_ewm"] = np.nan
    out.loc[np.isnan(delay_all), "sup_late_delay_ewm"] = np.nan
    return features[["shipment_id"]].merge(out, on="shipment_id", how="left")[list(DRIFT_COLUMNS)] \
        .set_index(features.index)


# --------------------------------------------------------------------------- #
# Step 2: leakage check (brute force on a sample)
# --------------------------------------------------------------------------- #
def check_no_leakage(features: pd.DataFrame, n_samples: int = 200, seed: int = 0) -> None:
    """Recompute the EWM features by brute force, using only shipments known by
    the cutoff, and confirm they match."""
    ships = shipment_outcomes(features)
    sample = features.drop_duplicates("shipment_id").sample(n_samples, random_state=seed)
    for _, row in sample.iterrows():
        h = ships[(ships["supplier_code"] == row["supplier_code"]) & (ships["known_time"] <= row["cutoff"])]
        age = (row["cutoff"] - h["known_time"]).dt.total_seconds() / 86400
        w = 0.5 ** (age / HALF_LIFE_DAYS)
        late = h["late"].fillna(0.0)
        assert np.isclose(w.sum(), row["sup_ewm_n"], rtol=1e-6, atol=1e-9), "sup_ewm_n mismatch"
        if pd.notna(row["sup_late_rate_all"]):
            expected = ((w * late).sum() + K_RATE * row["sup_late_rate_all"]) / (w.sum() + K_RATE)
            assert np.isclose(expected, row["sup_late_rate_ewm"], rtol=1e-6), "sup_late_rate_ewm mismatch"
    print(f"  leakage check passed on {n_samples} sampled shipments")


# --------------------------------------------------------------------------- #
# Step 3: save alongside the original feature set
# --------------------------------------------------------------------------- #
def save(scenario: str, features: pd.DataFrame, new: pd.DataFrame) -> None:
    folder = PROCESSED / scenario
    truth = [c for c in features.columns if c.startswith("true_")]
    out = pd.concat([features.drop(columns=truth), new, features[truth]], axis=1)
    out.to_parquet(folder / "features.parquet", index=False)

    roles = json.loads((folder / "feature_columns_v1.json").read_text())
    roles["delay_features"] = roles["delay_features"] + list(DRIFT_COLUMNS)
    (folder / "feature_columns.json").write_text(json.dumps(roles, indent=2))


def load_base(scenario: str) -> pd.DataFrame:
    """The original feature set. Backed up on the first run so reruns never stack."""
    folder = PROCESSED / scenario
    if not (folder / "features_v1.parquet").exists():
        shutil.copy(folder / "features.parquet", folder / "features_v1.parquet")
        shutil.copy(folder / "feature_columns.json", folder / "feature_columns_v1.json")
    base = pd.read_parquet(folder / "features_v1.parquet")
    return base.drop(columns=[c for c in DRIFT_COLUMNS if c in base])


# --------------------------------------------------------------------------- #
# Step 4: does the new feature set help? (fast model 1, all CV folds)
# --------------------------------------------------------------------------- #
def compare_feature_sets(scenario: str) -> pd.DataFrame:
    from diagnostics import run as run_diagnostics
    v1 = run_diagnostics(scenario, "v1", verbose=False)["fold_comparison"]
    v2 = run_diagnostics(scenario, "v2", verbose=False)["fold_comparison"]
    table = pd.DataFrame({"fold_id": v1["fold_id"], "naive": v1["naive"],
                          "fast_1_v1": v1["fast_1"], "fast_1_v2": v2["fast_1"]})
    table["v2_minus_v1"] = table["fast_1_v2"] - table["fast_1_v1"]
    table["v2_vs_v1_pct"] = 100 * table["v2_minus_v1"] / table["fast_1_v1"]
    out = build_folds.ROOT / "results" / scenario / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "feature_set_comparison.csv", index=False)
    return table


def run(scenario: str, compare: bool = True) -> None:
    print(f"[drift_features] scenario={scenario}")
    features = load_base(scenario)
    new = drift_features(features)
    save(scenario, features, new)
    check_no_leakage(pd.concat([features, new], axis=1))
    print(f"  wrote {PROCESSED / scenario / 'features.parquet'} (+{len(DRIFT_COLUMNS)} columns)")

    print("  rebuilding folds so imputation covers the new columns")
    build_folds.run(scenario)
    if compare:
        table = compare_feature_sets(scenario)
        print(table.round(4).to_string(index=False))
        print(f"  mean change v2 vs v1: {table['v2_minus_v1'].mean():+.4f} CRPS "
              f"({table['v2_vs_v1_pct'].mean():+.2f}%), better in {(table['v2_minus_v1'] < 0).sum()}/{len(table)} folds")


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()
    for scenario in (SCENARIOS if args.scenario == "all" else (args.scenario,)):
        run(scenario, compare=not args.skip_compare)


if __name__ == "__main__":
    main()
