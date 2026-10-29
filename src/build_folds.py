"""Walk-forward folds, per-fold imputation and per-fold supplier clustering.

Input : data/processed/<scenario>/features.parquet   (from features/build_features.py)
Output: data/processed/<scenario>/folds/
            fold_definitions.json      dates, imputation values, chosen k per algorithm
            cluster_labels.parquet     one row per fold x supplier
            clustering_diagnostics.csv one row per fold x algorithm x k

Rather than saving a copy of the training data for every fold, this script saves
the small fold "artifacts" above. Call `load_fold(scenario, fold_id)` to rebuild
any fold's train and test sets, already imputed and labelled.

Fold design (months counted from the first production day)
-----------------------------------------------------------
    month 0          burn-in: used only as history for features, never as rows
    folds 1-4        expanding training window from month 1, tested on months 6, 7, 8, 9
    holdout          trained on months 1-9, tested once on months 10-12 (final report)

A 96-hour embargo before each training end drops training rows whose outcome
might not be known yet: a truck due just before the cutoff could still be up to
96 h late, and its delay would not be known at training time.

Inside each fold (training data only)
-------------------------------------
1. Imputation values (medians, plus a few rule-based fills) are learned from
   training rows.
2. A supplier snapshot is taken at the training end (shipments known by then)
   and three clustering algorithms are fit: k-means, Gaussian mixture and
   hierarchical (Ward). k from 2-7 is chosen per algorithm:
       keep k with bootstrap stability ARI >= 0.75 (if none, keep all), then
       k-means / hierarchical: highest silhouette; GMM: lowest BIC
3. Clusters are renamed by late rate (C1 = most reliable) so they read the
   same across folds. Suppliers with too little history get "NEW".

The algorithm used by models 2 and 4 is chosen per scenario by a pre-set rule
on training-only diagnostics: highest mean stability at its chosen k across
folds, ties broken by silhouette. Agreement with the simulator's true clusters
(true_ari) is reported as a diagnostic only, never used to choose.

Usage:
    python build_folds.py                    # all scenarios
    python build_folds.py --scenario drift
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))
from build_features import supplier_snapshot  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
SCENARIOS = ("clusters", "no_clusters", "drift")
SEED = 20261028

BURN_IN_MONTHS = 1
FOLD_TEST_MONTHS = (6, 7, 8, 9)          # each fold tests one month
HOLDOUT_MONTHS = (10, 12)                # final report: months 10 up to 12
EMBARGO = pd.Timedelta(hours=96)         # = max simulated delay

K_RANGE = range(2, 8)
N_BOOTSTRAP = 20
MIN_STABILITY = 0.75
MIN_SNAPSHOT_SHIPMENTS = 20              # suppliers below this are "NEW"
ALGORITHMS = ("kmeans", "gmm", "hierarchical")
CLUSTER_FEATURES = ["late_rate", "mean_late_delay", "log_delay_sd"]


# --------------------------------------------------------------------------- #
# Step 1: fold definitions
# --------------------------------------------------------------------------- #
def make_folds(start: pd.Timestamp) -> list[dict]:
    month = lambda m: start + pd.DateOffset(months=m)
    folds = []
    for i, m in enumerate(FOLD_TEST_MONTHS, start=1):
        folds.append(dict(fold_id=f"fold_{i}", train_start=month(BURN_IN_MONTHS),
                          train_end=month(m), test_start=month(m), test_end=month(m + 1)))
    folds.append(dict(fold_id="holdout", train_start=month(BURN_IN_MONTHS),
                      train_end=month(HOLDOUT_MONTHS[0]),
                      test_start=month(HOLDOUT_MONTHS[0]), test_end=month(HOLDOUT_MONTHS[1])))
    return folds


def split(features: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = features["planned_delivery_datetime"]
    train = features[(t >= fold["train_start"]) & (t < fold["train_end"] - EMBARGO)]
    test = features[(t >= fold["test_start"]) & (t < fold["test_end"])]
    return train, test


# --------------------------------------------------------------------------- #
# Step 2: imputation learned on training rows only
# --------------------------------------------------------------------------- #
def fit_imputer(train: pd.DataFrame, numeric_cols: list[str]) -> dict:
    medians = train[numeric_cols].median(numeric_only=True)
    return {c: float(v) for c, v in medians.items() if pd.notna(v)}


def apply_imputer(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    df = df.copy()
    # Never late so far -> days since late = length of history (not the median)
    if "sup_days_since_late" in df:
        df["sup_days_since_late"] = df["sup_days_since_late"].fillna(df["sup_history_days"])
    # No shipments in a window -> no delay evidence; fall back to longer windows
    for short, long in [("sup_late_rate_30d", "sup_late_rate_90d"),
                        ("sup_late_rate_90d", "sup_late_rate_all"),
                        ("sup_mean_delay_30d", "sup_mean_delay_90d"),
                        ("sup_mean_delay_90d", "sup_mean_delay_all")]:
        if short in df and long in df:
            df[short] = df[short].fillna(df[long])
    df["sup_late_rate_trend"] = df["sup_late_rate_30d"] - df["sup_late_rate_90d"]
    for col, value in medians.items():
        if col in df:
            df[col] = df[col].astype("float").fillna(value)
    return df


# --------------------------------------------------------------------------- #
# Step 3: supplier clustering on a training-end snapshot
# --------------------------------------------------------------------------- #
def clustering_matrix(snapshot: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Eligible suppliers and their standardized clustering features."""
    s = snapshot[snapshot["n_shipments"] >= MIN_SNAPSHOT_SHIPMENTS].copy()
    x = pd.DataFrame({
        "late_rate": np.log(s["late_rate"].clip(0.005, 0.995) / (1 - s["late_rate"].clip(0.005, 0.995))),
        "mean_late_delay": np.log(s["mean_late_delay"]),
        "log_delay_sd": s["log_delay_sd"],
    }, index=s.index)
    x = x.fillna(x.median())
    x = (x - x.mean()) / x.std(ddof=0)
    return s, x.to_numpy()


def _fit_predict(algorithm: str, k: int, x_fit: np.ndarray, x_all: np.ndarray, seed: int):
    """Fit on x_fit, return (labels for x_all, fitted model)."""
    if algorithm == "kmeans":
        model = KMeans(k, n_init=10, random_state=seed).fit(x_fit)
        return model.predict(x_all), model
    if algorithm == "gmm":
        model = GaussianMixture(k, n_init=3, covariance_type="full", random_state=seed).fit(x_fit)
        return model.predict(x_all), model
    model = AgglomerativeClustering(k, linkage="ward").fit(x_fit)
    centroids = np.vstack([x_fit[model.labels_ == j].mean(axis=0) for j in range(k)])
    labels = ((x_all[:, None, :] - centroids[None]) ** 2).sum(-1).argmin(1)
    return labels, model


def stability(algorithm: str, k: int, x: np.ndarray, reference: np.ndarray, rng) -> float:
    """Mean ARI between the full-data labels and labels from bootstrap refits."""
    scores = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(x), len(x))
        labels, _ = _fit_predict(algorithm, k, x[idx], x, int(rng.integers(1e9)))
        scores.append(adjusted_rand_score(reference, labels))
    return float(np.mean(scores))


def true_clusters_at(snapshot_suppliers: pd.DataFrame, features: pd.DataFrame,
                     as_of: pd.Timestamp, start: pd.Timestamp) -> pd.Series:
    """Simulator truth at `as_of`, for diagnostics only."""
    truth = features.drop_duplicates("supplier_code").set_index("supplier_code")[
        ["true_cluster", "true_cluster_after_drift", "true_drift_month"]]
    switch = truth["true_drift_month"].map(lambda m: start + pd.DateOffset(months=int(m)) if pd.notna(m) else pd.NaT)
    current = truth["true_cluster"].where(~(switch <= as_of).fillna(False), truth["true_cluster_after_drift"])
    return current.reindex(snapshot_suppliers["supplier_code"]).reset_index(drop=True)


def cluster_fold(features: pd.DataFrame, fold: dict, start: pd.Timestamp, rng):
    """Fit all algorithms for all k on one fold's training snapshot."""
    as_of = fold["train_end"]
    snapshot = supplier_snapshot(features, as_of)
    eligible, x = clustering_matrix(snapshot)
    truth = true_clusters_at(eligible, features, as_of, start)
    has_truth = truth.notna().all()

    rows, labels = [], {}
    for algorithm in ALGORITHMS:
        for k in K_RANGE:
            lab, model = _fit_predict(algorithm, k, x, x, SEED)
            labels[(algorithm, k)] = lab
            rows.append(dict(
                fold_id=fold["fold_id"], algorithm=algorithm, k=k, n_suppliers=len(x),
                silhouette=silhouette_score(x, lab),
                bic=model.bic(x) if algorithm == "gmm" else np.nan,
                stability_ari=stability(algorithm, k, x, lab, rng),
                true_ari=adjusted_rand_score(truth.astype(str), lab) if has_truth else np.nan,
            ))
    diag = pd.DataFrame(rows)
    return snapshot, eligible, diag, labels


def choose_k(diag: pd.DataFrame) -> pd.DataFrame:
    """Pre-set rule: among stable k, best silhouette (BIC for GMM)."""
    picks = []
    for (fold_id, algorithm), g in diag.groupby(["fold_id", "algorithm"]):
        stable = g[g["stability_ari"] >= MIN_STABILITY]
        g = stable if len(stable) else g
        best = g.loc[g["bic"].idxmin()] if algorithm == "gmm" else g.loc[g["silhouette"].idxmax()]
        picks.append(best)
    chosen = pd.DataFrame(picks)[["fold_id", "algorithm", "k"]].assign(chosen=True)
    return diag.merge(chosen, on=["fold_id", "algorithm", "k"], how="left").fillna({"chosen": False})


def ordered_labels(raw: np.ndarray, late_rate: pd.Series) -> np.ndarray:
    """Rename clusters C1..Ck by mean late rate, so C1 is always the most reliable."""
    order = pd.Series(late_rate.to_numpy()).groupby(raw).mean().sort_values().index
    names = {old: f"C{new + 1}" for new, old in enumerate(order)}
    return np.array([names[v] for v in raw])


def select_algorithm(diag: pd.DataFrame) -> str:
    """Pre-set rule on training-only diagnostics across the CV folds (not the holdout)."""
    chosen = diag[diag["chosen"] & (diag["fold_id"] != "holdout")]
    summary = chosen.groupby("algorithm")[["stability_ari", "silhouette"]].mean()
    return summary.sort_values(["stability_ari", "silhouette"], ascending=False).index[0]


# --------------------------------------------------------------------------- #
# Step 4: run everything for a scenario
# --------------------------------------------------------------------------- #
def run(scenario: str) -> None:
    print(f"[folds] scenario={scenario}")
    features = pd.read_parquet(PROCESSED / scenario / "features.parquet")
    roles = json.loads((PROCESSED / scenario / "feature_columns.json").read_text())
    numeric = [c for c in roles["delay_features"] + roles["shortage_features"]
               if pd.api.types.is_numeric_dtype(features[c]) and not pd.api.types.is_bool_dtype(features[c])]

    start = features["production_date"].min().normalize()
    folds = make_folds(start)
    rng = np.random.default_rng(SEED)

    all_diag, all_labels = [], []
    for fold in folds:
        train, test = split(features, fold)
        assert train["planned_delivery_datetime"].max() < fold["train_end"] - EMBARGO
        assert test["planned_delivery_datetime"].min() >= fold["train_end"]
        fold["imputation_medians"] = fit_imputer(train, numeric)
        fold["n_train_rows"], fold["n_test_rows"] = len(train), len(test)

        snapshot, eligible, diag, labels = cluster_fold(features, fold, start, rng)
        diag = choose_k(diag)
        all_diag.append(diag)

        lab = snapshot[["supplier_code", "n_shipments"]].copy()
        lab.insert(0, "fold_id", fold["fold_id"])
        fold["chosen_k"] = {}
        for algorithm in ALGORITHMS:
            k = int(diag[(diag["algorithm"] == algorithm) & diag["chosen"]]["k"].iloc[0])
            fold["chosen_k"][algorithm] = k
            named = pd.Series(ordered_labels(labels[(algorithm, k)], eligible["late_rate"]),
                              index=eligible.index)
            lab[f"cluster_{algorithm}"] = named.reindex(lab.index).fillna("NEW")
        all_labels.append(lab)
        print(f"  {fold['fold_id']}: train {len(train):,} rows, test {len(test):,} rows, "
              f"{len(eligible)} suppliers clustered, k={fold['chosen_k']}")

    diag = pd.concat(all_diag, ignore_index=True)
    selected = select_algorithm(diag)
    labels = pd.concat(all_labels, ignore_index=True)
    labels["cluster"] = labels[f"cluster_{selected}"]
    print(f"  selected algorithm for models 2 and 4: {selected}")

    out = PROCESSED / scenario / "folds"
    out.mkdir(parents=True, exist_ok=True)
    diag.to_csv(out / "clustering_diagnostics.csv", index=False)
    labels.to_parquet(out / "cluster_labels.parquet", index=False)
    serial = [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in f.items()} for f in folds]
    (out / "fold_definitions.json").write_text(json.dumps(
        {"selected_algorithm": selected, "embargo_hours": EMBARGO.total_seconds() / 3600,
         "folds": serial}, indent=2))
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Used by the modelling scripts
# --------------------------------------------------------------------------- #
def load_fold(scenario: str, fold_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild one fold's train and test sets: split, impute, attach cluster labels."""
    folder = PROCESSED / scenario / "folds"
    meta = json.loads((folder / "fold_definitions.json").read_text())
    fold = next(f for f in meta["folds"] if f["fold_id"] == fold_id)
    fold = {k: (pd.Timestamp(v) if k.endswith(("_start", "_end")) else v) for k, v in fold.items()}

    features = pd.read_parquet(PROCESSED / scenario / "features.parquet")
    labels = pd.read_parquet(folder / "cluster_labels.parquet")
    labels = labels[labels["fold_id"] == fold_id].drop(columns=["fold_id", "n_shipments"])

    out = []
    for part in split(features, fold):
        part = apply_imputer(part, fold["imputation_medians"])
        part = part.merge(labels, on="supplier_code", how="left")
        cluster_cols = [c for c in part.columns if c.startswith("cluster")]
        part[cluster_cols] = part[cluster_cols].fillna("NEW")   # not yet in the snapshot
        out.append(part)
    return out[0], out[1]


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    args = parser.parse_args()
    for scenario in (SCENARIOS if args.scenario == "all" else (args.scenario,)):
        run(scenario)


if __name__ == "__main__":
    main()
