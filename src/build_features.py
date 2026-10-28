"""Build leakage-safe modelling features from the master panel.

Input : data/raw/<scenario>/master_panel.parquet   (from data_generation/)
Output: data/processed/<scenario>/features.parquet
        data/processed/<scenario>/feature_columns.json   (column roles)

The leakage rule
----------------
The dashboard forecasts deliveries up to 72 hours ahead. So every history
feature for a delivery is computed "as of" a cutoff:

    cutoff = planned_delivery_datetime - 72 hours

and only uses shipments whose outcome was KNOWN by then, i.e. whose truck had
actually arrived (actual_delivery_datetime <= cutoff). Unscanned receipts never
become known, so they never enter any history.

Feature groups
--------------
delay model (does the truck arrive late, and by how much?)
    supplier history: shipment counts, late rate and mean delay over 30 / 90
    days and all history, delay size and volatility when late, trend,
    days since last late shipment, history length, cold-start flag
    region history: late rate over the last 7 days (picks up disruptions)
    known in advance: hour, weekday, month, winter flag, trucks per day,
    lead time, distance, region, commodity

shortage step (how much would a late truck hurt?)
    safety stock coverage in hours, usage per hour, planned hours,
    availability in other plants, part shortages in the previous 30 days

Shortage label
--------------
`shortage_flag` = the delay was longer than the hours of production that
safety stock covers (delay_hours > safety_coverage_hours). This is a
simplification: it assumes the line is running throughout the delay, so it
slightly overstates shortages for trucks that are late overnight.

Missing values are NOT imputed here, because imputation values must be fit
on training data only. Missing-value indicator columns are added instead.

Usage:
    python build_features.py                    # all scenarios
    python build_features.py --scenario drift
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
SCENARIOS = ("clusters", "no_clusters", "drift")

HORIZON = pd.Timedelta(hours=72)
SUPPLIER_WINDOWS = {"30d": pd.Timedelta(days=30), "90d": pd.Timedelta(days=90)}
REGION_WINDOW = pd.Timedelta(days=7)
PART_SHORTAGE_WINDOW = pd.Timedelta(days=30)
COLD_START_MIN_SHIPMENTS = 20
WINTER_MONTHS = (12, 1, 2)


# --------------------------------------------------------------------------- #
# Step 1: shipment-level outcomes
# --------------------------------------------------------------------------- #
def shipment_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per truck. Parts on the same truck share one delay, so supplier
    history is counted in shipments, not part rows (otherwise suppliers with
    many parts would be over-weighted)."""
    cols = ["shipment_id", "supplier_code", "plant", "region", "production_date",
            "planned_delivery_datetime", "actual_delivery_datetime", "delay_hours", "is_late"]
    ships = panel[cols].drop_duplicates("shipment_id").copy()
    ships["cutoff"] = ships["planned_delivery_datetime"] - HORIZON
    ships["known_time"] = ships["actual_delivery_datetime"]          # outcome known on arrival
    ships["late"] = ships["is_late"].astype("float")                 # NaN if unscanned
    ships["log_delay"] = np.where(ships["late"] == 1, np.log(ships["delay_hours"].clip(lower=1e-3)), np.nan)
    return ships


# --------------------------------------------------------------------------- #
# Step 2: as-of history via cumulative sums
# --------------------------------------------------------------------------- #
def _cumulative_history(events: pd.DataFrame, by: list[str], value_cols: list[str]) -> pd.DataFrame:
    """Running totals per group, ordered by the time each outcome became known."""
    known = events.dropna(subset=["known_time"]).sort_values("known_time").copy()
    known["n"] = 1.0
    sums = ["n", *value_cols]
    known[[f"cum_{c}" for c in sums]] = known.groupby(by)[sums].cumsum()
    return known[[*by, "known_time", *[f"cum_{c}" for c in sums]]]


def _asof(queries: pd.DataFrame, history: pd.DataFrame, by: list[str], time_col: str) -> pd.DataFrame:
    """Cumulative totals at `time_col` for each query row (0 if no history yet)."""
    q = queries[[*by, time_col]].reset_index().sort_values(time_col)
    out = pd.merge_asof(q, history, left_on=time_col, right_on="known_time", by=by,
                        direction="backward", allow_exact_matches=True)
    cum_cols = [c for c in history.columns if c.startswith("cum_")]
    out[cum_cols] = out[cum_cols].fillna(0.0)
    return out.set_index("index").loc[queries.index, cum_cols]


def _window(queries, history, by, time_col, window=None):
    """Totals over (time - window, time]; whole history if window is None."""
    at_t = _asof(queries, history, by, time_col)
    if window is None:
        return at_t
    q = queries[[*by]].copy()
    q["_start"] = queries[time_col] - window
    at_start = _asof(q, history, by, "_start")
    return at_t - at_start.values


def _safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def supplier_history_features(ships: pd.DataFrame) -> pd.DataFrame:
    ev = ships.assign(
        delay=ships["delay_hours"],
        late_delay=np.where(ships["late"] == 1, ships["delay_hours"], 0.0),
        log_d=ships["log_delay"].fillna(0.0),
        log_d2=ships["log_delay"].fillna(0.0) ** 2,
    )
    hist = _cumulative_history(ev, ["supplier_code"], ["late", "delay", "late_delay", "log_d", "log_d2"])
    feats = pd.DataFrame(index=ships.index)

    for label, window in [("all", None), *SUPPLIER_WINDOWS.items()]:
        w = _window(ships, hist, ["supplier_code"], "cutoff", window)
        n, k = w["cum_n"].to_numpy(), w["cum_late"].to_numpy()
        feats[f"sup_n_ship_{label}"] = n
        feats[f"sup_late_rate_{label}"] = _safe_div(k, n)
        feats[f"sup_mean_delay_{label}"] = _safe_div(w["cum_delay"].to_numpy(), n)
        if label in ("all", "90d"):
            mean_log = _safe_div(w["cum_log_d"].to_numpy(), k)
            var_log = _safe_div(w["cum_log_d2"].to_numpy(), k) - mean_log ** 2
            feats[f"sup_mean_late_delay_{label}"] = _safe_div(w["cum_late_delay"].to_numpy(), k)
            feats[f"sup_log_delay_sd_{label}"] = np.where(k >= 2, np.sqrt(np.clip(var_log, 0, None)), np.nan)

    feats["sup_late_rate_trend"] = feats["sup_late_rate_30d"] - feats["sup_late_rate_90d"]

    # Days since last late shipment / since first known shipment
    known = ships.dropna(subset=["known_time"]).sort_values("known_time")
    last_late = known.assign(t=known["known_time"].where(known["late"] == 1)) \
                     .assign(t=lambda d: d.groupby("supplier_code")["t"].ffill()) \
                     .dropna(subset=["t"])[["supplier_code", "known_time", "t"]]
    first = known.groupby("supplier_code")["known_time"].min().rename("first_known")

    q = ships[["supplier_code", "cutoff"]].reset_index().sort_values("cutoff")
    m = pd.merge_asof(q, last_late, left_on="cutoff", right_on="known_time", by="supplier_code",
                      direction="backward").set_index("index").loc[ships.index]
    feats["sup_days_since_late"] = (ships["cutoff"] - m["t"]).dt.total_seconds() / 86400
    hist_days = (ships["cutoff"] - ships["supplier_code"].map(first)).dt.total_seconds() / 86400
    feats["sup_history_days"] = hist_days.where(hist_days >= 0)
    feats["sup_is_cold_start"] = feats["sup_n_ship_all"] < COLD_START_MIN_SHIPMENTS
    return feats


def region_history_features(ships: pd.DataFrame) -> pd.DataFrame:
    hist = _cumulative_history(ships, ["region"], ["late"])
    w = _window(ships, hist, ["region"], "cutoff", REGION_WINDOW)
    return pd.DataFrame({"region_late_rate_7d": _safe_div(w["cum_late"].to_numpy(), w["cum_n"].to_numpy())},
                        index=ships.index)


# --------------------------------------------------------------------------- #
# Step 3: features known in advance
# --------------------------------------------------------------------------- #
def calendar_features(ships: pd.DataFrame) -> pd.DataFrame:
    t = ships["planned_delivery_datetime"]
    trips = ships.groupby(["supplier_code", "plant", "production_date"])["shipment_id"].transform("size")
    return pd.DataFrame({
        "planned_hour": t.dt.hour + t.dt.minute / 60,
        "planned_dow": t.dt.dayofweek,
        "planned_month": t.dt.month,
        "is_winter": t.dt.month.isin(WINTER_MONTHS),
        "trips_per_day": trips,
    }, index=ships.index)


# --------------------------------------------------------------------------- #
# Step 4: shortage label and shortage-step features (part x plant level)
# --------------------------------------------------------------------------- #
def shortage_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel[["part_code", "plant", "planned_delivery_datetime", "actual_delivery_datetime",
                "delay_hours", "safety_inv_boxes", "box_qty", "usage_per_hour"]].copy()
    coverage = df["safety_inv_boxes"].astype("float") * df["box_qty"] / df["usage_per_hour"]
    shortage = (df["delay_hours"] > coverage).astype("float")
    shortage[df["delay_hours"].isna() | coverage.isna()] = np.nan

    events = pd.DataFrame({"part_code": df["part_code"], "plant": df["plant"],
                           "known_time": df["actual_delivery_datetime"],
                           "shortage": shortage.fillna(0.0)})
    hist = _cumulative_history(events, ["part_code", "plant"], ["shortage"])
    q = df[["part_code", "plant"]].assign(cutoff=df["planned_delivery_datetime"] - HORIZON)
    w = _window(q, hist, ["part_code", "plant"], "cutoff", PART_SHORTAGE_WINDOW)

    return pd.DataFrame({
        "safety_coverage_hours": coverage.round(2),
        "part_shortages_30d": w["cum_shortage"].to_numpy(),
        "shortage_flag": shortage.astype("boolean"),
    }, index=panel.index)


# --------------------------------------------------------------------------- #
# Step 5: assemble, label column roles, check for leakage
# --------------------------------------------------------------------------- #
COLUMN_ROLES = {
    "id": ["delivery_id", "shipment_id", "supplier_code", "part_code", "plant",
           "production_date", "planned_delivery_datetime", "cutoff"],
    "target": ["delay_hours", "is_late", "shortage_flag"],
    "outcome_do_not_use": ["actual_delivery_datetime"],
    "delay_features": [
        "region", "commodity", "distance_km", "distance_km_missing", "lead_time_days",
        "planned_hour", "planned_dow", "planned_month", "is_winter", "trips_per_day",
        "sup_n_ship_all", "sup_late_rate_all", "sup_mean_delay_all",
        "sup_mean_late_delay_all", "sup_log_delay_sd_all",
        "sup_n_ship_30d", "sup_late_rate_30d", "sup_mean_delay_30d",
        "sup_n_ship_90d", "sup_late_rate_90d", "sup_mean_delay_90d",
        "sup_mean_late_delay_90d", "sup_log_delay_sd_90d", "sup_late_rate_trend",
        "sup_days_since_late", "sup_history_days", "sup_is_cold_start",
        "region_late_rate_7d",
    ],
    "shortage_features": [
        "safety_coverage_hours", "safety_inv_missing", "usage_per_hour", "box_qty",
        "planned_hours", "ordered_qty_boxes", "avail_in_other_plants", "n_plants_using",
        "part_shortages_30d",
    ],
}


def build(panel: pd.DataFrame) -> pd.DataFrame:
    ships = shipment_outcomes(panel)
    ship_feats = pd.concat([ships[["shipment_id", "cutoff"]],
                            supplier_history_features(ships),
                            region_history_features(ships),
                            calendar_features(ships)], axis=1)

    df = panel.merge(ship_feats, on="shipment_id", how="left", validate="many_to_one")
    df = pd.concat([df, shortage_features(df)], axis=1)
    df["distance_km_missing"] = df["distance_km"].isna()
    df["safety_inv_missing"] = df["safety_inv_boxes"].isna()

    truth = [c for c in df.columns if c.startswith("true_")]
    ordered = [c for role in COLUMN_ROLES.values() for c in role]
    return df[ordered + truth].sort_values("planned_delivery_datetime").reset_index(drop=True)


def check_no_leakage(features: pd.DataFrame, panel: pd.DataFrame, n_samples: int = 200, seed: int = 0) -> None:
    """Recompute a sample of supplier features by brute force, keeping only
    shipments known by the cutoff, and confirm they match."""
    ships = shipment_outcomes(panel)
    sample = features.drop_duplicates("shipment_id").sample(n_samples, random_state=seed)
    for _, row in sample.iterrows():
        h = ships[(ships["supplier_code"] == row["supplier_code"]) & (ships["known_time"] <= row["cutoff"])]
        h30 = h[h["known_time"] > row["cutoff"] - SUPPLIER_WINDOWS["30d"]]
        assert len(h) == row["sup_n_ship_all"], "all-history count mismatch"
        assert len(h30) == row["sup_n_ship_30d"], "30-day count mismatch"
        if len(h30):
            assert np.isclose(h30["late"].mean(), row["sup_late_rate_30d"]), "30-day late rate mismatch"
    print(f"  leakage check passed on {n_samples} sampled shipments")


# --------------------------------------------------------------------------- #
# Helper for the clustering step (used later, inside each training window)
# --------------------------------------------------------------------------- #
def supplier_snapshot(panel: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per-supplier summary using only shipments known by `as_of`.

    Fit clustering on this snapshot at the end of each training window, never
    on the full dataset.
    """
    ships = shipment_outcomes(panel)
    h = ships[ships["known_time"] <= as_of]
    g = h.groupby("supplier_code")
    snap = pd.DataFrame({
        "n_shipments": g.size(),
        "late_rate": g["late"].mean(),
        "mean_delay": g["delay_hours"].mean(),
        "p90_delay": g["delay_hours"].quantile(0.9),
        "mean_late_delay": h[h["late"] == 1].groupby("supplier_code")["delay_hours"].mean(),
        "log_delay_sd": h.groupby("supplier_code")["log_delay"].std(),
    })
    static = panel.drop_duplicates("supplier_code").set_index("supplier_code")[
        ["region", "distance_km", "lead_time_days", "commodity"]]
    return snap.join(static, how="left").reset_index()


# --------------------------------------------------------------------------- #
def run(scenario: str, check: bool = True) -> pd.DataFrame:
    print(f"[features] scenario={scenario}")
    panel = pd.read_parquet(RAW_DIR / scenario / "master_panel.parquet")
    features = build(panel)
    if check:
        check_no_leakage(features, panel)

    out = OUT_DIR / scenario
    out.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out / "features.parquet", index=False)
    roles = {**COLUMN_ROLES, "evaluation_only": [c for c in features.columns if c.startswith("true_")]}
    (out / "feature_columns.json").write_text(json.dumps(roles, indent=2))
    print(f"  wrote {out / 'features.parquet'} ({len(features):,} rows, {features.shape[1]} columns)")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--skip-check", action="store_true", help="Skip the brute-force leakage check.")
    args = parser.parse_args()
    for scenario in (SCENARIOS if args.scenario == "all" else (args.scenario,)):
        run(scenario, check=not args.skip_check)


if __name__ == "__main__":
    main()
