"""Fit and score the naive baseline and models 1-4 on every walk-forward fold.

Input : data/processed/<scenario>/features.parquet and folds/  (from build_folds.py)
Output: results/<scenario>/
            metrics.csv        one row per fold x model (all metrics, fit time, diagnostics)
            summary.csv        mean across CV folds + paired CRPS difference vs naive
            predictions/<fold>_<model>_shipments.parquet   per-shipment forecast summary
            predictions/<fold>_<model>_rows.parquet        per-delivery-row P(shortage)

For each fold:
    1. load_fold() gives imputed train/test rows with that fold's cluster labels
    2. collapse to shipments (one row per truck)
    3. each model produces an (n_test_shipments x n_draws) matrix of predictive delays.
       The Bayesian models fit "is it late?" on a subsample of n_train shipments
       (the same subsample for every model in a fold) and "how late?" on every
       late training shipment
    4. scoring.score() computes every metric from those draws

The final holdout is NOT run by default. Choose your model on the CV folds first,
then run once with --holdout for the final report.

Runs resume: a (fold, model) pair already in metrics.csv is skipped unless
--overwrite is passed, so a long run can be stopped and restarted.

Usage:
    python evaluate_models.py --quick                       # smoke test: 1 fold, small fits
    python evaluate_models.py --scenario clusters           # all CV folds, all models
    python evaluate_models.py --models naive model_1        # a subset of models
    python evaluate_models.py --scenario clusters --holdout # final report (run once)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from build_folds import PROCESSED, SCENARIOS, load_fold
from models import BAYES, MODELS, HurdleModel, naive_samples, to_shipments
from scoring import score, summarize

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 20261102

DEFAULTS = dict(n_train=30_000, draws=1000, tune=1000, chains=2, n_pred_draws=200)
# Smaller data and fewer draws, but still a full warm-up: hierarchical models
# (3 and 4) do not converge with short tuning, and an unconverged fit gives
# meaningless scores.
QUICK = dict(n_train=10_000, draws=500, tune=1000, chains=2, n_pred_draws=100)


def fold_seed(*parts: str) -> int:
    return int(np.random.SeedSequence([SEED, *[sum(map(ord, p)) for p in parts]]).generate_state(1)[0])


# --------------------------------------------------------------------------- #
# Step 1: prepare one fold
# --------------------------------------------------------------------------- #
def prepare_fold(scenario: str, fold_id: str):
    train_rows, test_rows = load_fold(scenario, fold_id)
    train_ships = to_shipments(train_rows)

    # Score only shipments whose delay is known (unscanned receipts have no truth)
    test_ships = to_shipments(test_rows).dropna(subset=["delay_hours"]).reset_index(drop=True)
    ship_idx = pd.Series(np.arange(len(test_ships)), index=test_ships["shipment_id"])
    test_rows = test_rows[test_rows["shipment_id"].isin(ship_idx.index)].copy()
    test_rows["ship_idx"] = test_rows["shipment_id"].map(ship_idx).to_numpy()
    return train_ships, test_ships, test_rows


# --------------------------------------------------------------------------- #
# Step 2: get predictive samples from one model
# --------------------------------------------------------------------------- #
def predict(model_name: str, train_ships, test_ships, cfg: dict, seed: int, sampler: str,
            subsample_seed: int = 0):
    rng = np.random.default_rng(seed)
    start = time.time()
    if model_name == "naive":
        samples, diag = naive_samples(train_ships, test_ships, cfg["n_pred_draws"], rng), {}
    else:
        model = HurdleModel(model_name).fit(train_ships, cfg["n_train"], cfg["draws"], cfg["tune"],
                                            cfg["chains"], seed % 2**32, subsample_seed, sampler)
        samples, diag = model.predict_samples(test_ships, cfg["n_pred_draws"], rng), model.diagnostics()
    return samples, diag, time.time() - start


def save_predictions(samples, test_ships, test_rows, out: Path, fold_id: str, model_name: str):
    out.mkdir(parents=True, exist_ok=True)
    q05, q50, q95 = np.quantile(samples, [0.05, 0.5, 0.95], axis=1)
    pd.DataFrame({
        "shipment_id": test_ships["shipment_id"], "supplier_code": test_ships["supplier_code"],
        "planned_delivery_datetime": test_ships["planned_delivery_datetime"],
        "delay_hours": test_ships["delay_hours"], "p_late": (samples > 0).mean(axis=1),
        "pred_mean": samples.mean(axis=1), "pred_q05": q05, "pred_median": q50, "pred_q95": q95,
    }).to_parquet(out / f"{fold_id}_{model_name}_shipments.parquet", index=False)

    rows = test_rows.dropna(subset=["safety_coverage_hours"])
    row_samples = samples[rows["ship_idx"].to_numpy()]
    pd.DataFrame({
        "delivery_id": rows["delivery_id"], "part_code": rows["part_code"], "plant": rows["plant"],
        "production_date": rows["production_date"],
        "p_shortage": (row_samples > rows["safety_coverage_hours"].to_numpy()[:, None]).mean(axis=1),
        "shortage_flag": rows["shortage_flag"],
    }).to_parquet(out / f"{fold_id}_{model_name}_rows.parquet", index=False)


# --------------------------------------------------------------------------- #
# Step 3: run a scenario
# --------------------------------------------------------------------------- #
def run(scenario: str, fold_ids: list[str], models: list[str], cfg: dict,
        sampler: str = "nutpie", overwrite: bool = False, tag: str = "") -> pd.DataFrame:
    out = RESULTS / f"{scenario}{tag}"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() and not overwrite else pd.DataFrame()
    done = set(zip(metrics.get("fold_id", []), metrics.get("model", [])))

    for fold_id in fold_ids:
        todo = [m for m in models if (fold_id, m) not in done]
        if not todo:
            continue
        train_ships, test_ships, test_rows = prepare_fold(scenario, fold_id)
        print(f"[{scenario} | {fold_id}] train {len(train_ships):,} shipments, "
              f"test {len(test_ships):,} shipments")
        subsample_seed = fold_seed(scenario, fold_id) % 2**32     # same subsample for every model
        for model_name in todo:
            samples, diag, seconds = predict(model_name, train_ships, test_ships, cfg,
                                             fold_seed(scenario, fold_id, model_name), sampler,
                                             subsample_seed)
            row = {"scenario": scenario, "fold_id": fold_id, "model": model_name,
                   **score(samples, test_ships, test_rows), "seconds": round(seconds, 1), **diag}
            metrics = pd.concat([metrics[~((metrics.get("fold_id") == fold_id) & (metrics.get("model") == model_name))]
                                 if len(metrics) else metrics, pd.DataFrame([row])], ignore_index=True)
            metrics.to_csv(metrics_path, index=False)          # save after every model (resumable)
            save_predictions(samples, test_ships, test_rows, out / "predictions", fold_id, model_name)
            flag = "" if model_name == "naive" or (diag.get("max_rhat", 1) < 1.01 and diag.get("divergences", 0) == 0) \
                else "  <-- check convergence"
            print(f"    {model_name:8s} CRPS {row['crps']:.4f}  brier_late {row['brier_late']:.4f}  "
                  f"cov90 {row['coverage_90']:.3f}  {seconds:5.0f}s{flag}")

    cv = metrics[metrics["fold_id"] != "holdout"]
    if len(cv) and "naive" in set(cv["model"]):
        summary = summarize(cv)
        summary.to_csv(out / "summary.csv")
        cols = ["crps", "crps_vs_ref", "crps_vs_ref_sd", "folds_better", "crps_cold_start",
                "brier_late", "coverage_90", "brier_shortage", "precision_at_10"]
        print(f"\n[{scenario}] mean over CV folds (crps_vs_ref < 0 = better than naive):")
        print(summary[[c for c in cols if c in summary]].round(4).to_string())
    return metrics


def main() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("pymc").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--folds", nargs="+", default=None, help="Default: all CV folds (not the holdout).")
    parser.add_argument("--holdout", action="store_true", help="Score the final holdout only. Run once, at the end.")
    parser.add_argument("--quick", action="store_true", help="Smoke test: fold_1 only, small fits.")
    parser.add_argument("--sampler", choices=("nutpie", "pymc"), default="nutpie")
    parser.add_argument("--overwrite", action="store_true")
    for k, v in DEFAULTS.items():
        parser.add_argument(f"--{k.replace('_', '-')}", type=int, default=None, help=f"default {v}")
    args = parser.parse_args()

    cfg = {**(QUICK if args.quick else DEFAULTS),
           **{k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None}}
    for scenario in (SCENARIOS if args.scenario == "all" else (args.scenario,)):
        meta = json.loads((PROCESSED / scenario / "folds" / "fold_definitions.json").read_text())
        cv_folds = [f["fold_id"] for f in meta["folds"] if f["fold_id"] != "holdout"]
        folds = ["holdout"] if args.holdout else (["fold_1"] if args.quick else (args.folds or cv_folds))
        run(scenario, folds, args.models, cfg, args.sampler, args.overwrite, "_quick" if args.quick else "")


if __name__ == "__main__":
    main()
