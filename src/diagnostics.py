"""Model diagnostics, and model 1 fit as a fast (non-Bayesian) regression.

Fast model 1
------------
Model 1 has no supplier or cluster effects, so it is an ordinary hurdle
regression and can be fit by maximum likelihood in seconds instead of minutes
of MCMC:
    P(late)            logistic regression on the late-head predictors
    log(delay) | late  normal regression, with log(sigma) also linear in the
                       sigma-head predictors (fit jointly by maximum likelihood)
It uses exactly the same predictors and transforms as the Bayesian models
(models.Design). With 150k+ training shipments, parameter uncertainty is tiny,
so plugging in the point estimates should forecast almost exactly like the
Bayesian version. Step 1 below checks that on every fold. If it holds, the fast
model can be used to try new features quickly.

Diagnostics (per scenario, over the CV folds)
---------------------------------------------
1. fold_comparison.csv  CRPS per fold: naive, fast model 1, Bayesian model 1
2. crps_by_group.csv    where each model loses CRPS: by outcome, region, season,
                        supplier history length, recent change, cold start
3. calibration.csv      P(late) reliability: predicted vs observed late rate in bins
4. pit.csv              PIT histogram: where each actual delay falls within its
                        predicted distribution (flat = well calibrated)

Outputs go to results/<scenario>/diagnostics/<feature_set>/.

Usage:
    python diagnostics.py                         # all scenarios, current feature set
    python diagnostics.py --scenario drift --feature-set v1
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from build_folds import PROCESSED, SCENARIOS
from evaluate_models import RESULTS, fold_seed, prepare_fold
from models import FEATURE_SET, MAX_PRED_DELAY_HOURS, Design, _logit, naive_samples
from scoring import crps_from_samples

N_DRAWS = 200
N_CALIBRATION_BINS = 10
N_PIT_BINS = 10


# --------------------------------------------------------------------------- #
# Fast model 1: maximum-likelihood hurdle regression
# --------------------------------------------------------------------------- #
class FastModel1:
    """Model 1 without MCMC: same predictors, maximum-likelihood fit."""

    def __init__(self, feature_set: str = FEATURE_SET):
        self.feature_set = feature_set

    def fit(self, train_ships: pd.DataFrame) -> "FastModel1":
        data = train_ships.dropna(subset=["delay_hours"])
        late = (data["delay_hours"] > 0).to_numpy()
        self.designs = {h: Design().fit(data if h == "late" else data[late], h, None, self.feature_set)
                        for h in ("late", "mu", "sigma")}

        # Occurrence: unpenalized logistic regression on every training shipment
        X = self.designs["late"].transform(data)
        self.logit = LogisticRegression(penalty=None, max_iter=2000).fit(X, late)

        # Severity: log(delay) ~ Normal(Xm g, exp(Xs h)), fit jointly by maximum likelihood
        sev = data[late]
        Xm = np.column_stack([np.ones(len(sev)), self.designs["mu"].transform(sev)])
        Xs = np.column_stack([np.ones(len(sev)), self.designs["sigma"].transform(sev)])
        y = np.log(sev["delay_hours"].to_numpy(float))

        def nll(theta):
            g, h = theta[:Xm.shape[1]], theta[Xm.shape[1]:]
            r, log_s = y - Xm @ g, Xs @ h
            s2 = np.exp(2 * log_s)
            value = np.sum(log_s + r**2 / (2 * s2))
            grad = np.concatenate([-Xm.T @ (r / s2), Xs.T @ (1 - r**2 / s2)])
            return value, grad

        start = np.zeros(Xm.shape[1] + Xs.shape[1])
        start[0], start[Xm.shape[1]] = y.mean(), np.log(y.std())
        result = minimize(nll, start, jac=True, method="L-BFGS-B")
        self.g, self.h = result.x[:Xm.shape[1]], result.x[Xm.shape[1]:]
        self.converged = bool(result.success)
        return self

    def predict_params(self, ships: pd.DataFrame):
        """P(late), and mu and sigma of log(delay) given late, per shipment."""
        p = self.logit.predict_proba(self.designs["late"].transform(ships))[:, 1]
        mu = self.g[0] + self.designs["mu"].transform(ships) @ self.g[1:]
        sigma = np.exp(self.h[0] + self.designs["sigma"].transform(ships) @ self.h[1:])
        return p, mu, sigma

    def predict_samples(self, ships: pd.DataFrame, n_draws: int, rng) -> np.ndarray:
        p, mu, sigma = self.predict_params(ships)
        n = len(ships)
        late = rng.random((n, n_draws)) < p[:, None]
        log_d = mu[:, None] + sigma[:, None] * rng.standard_normal((n, n_draws))
        return np.where(late, np.exp(np.minimum(log_d, np.log(MAX_PRED_DELAY_HOURS))), 0.0)

    def pit(self, ships: pd.DataFrame, rng) -> np.ndarray:
        """Probability integral transform of each actual delay.

        For a late truck, PIT = F(y) = (1 - p) + p * Phi((log y - mu) / sigma).
        For an on-time truck (y = 0) the forecast puts a lump of probability
        1 - p on zero, so PIT is drawn uniformly from [0, 1 - p] (the standard
        randomized PIT for a point mass). A calibrated model gives flat PITs.
        """
        p, mu, sigma = self.predict_params(ships)
        y = ships["delay_hours"].to_numpy(float)
        late_cdf = (1 - p) + p * norm.cdf((np.log(np.where(y > 0, y, 1)) - mu) / sigma)
        return np.where(y > 0, late_cdf, rng.uniform(0, 1, len(y)) * (1 - p))


# --------------------------------------------------------------------------- #
# Diagnostic tables
# --------------------------------------------------------------------------- #
def diagnostic_groups(ships: pd.DataFrame) -> pd.DataFrame:
    """Observable groupings of test shipments (no simulator truth)."""
    change = (_logit(ships["sup_late_rate_30d"]) - _logit(ships["sup_late_rate_90d"])).abs()
    return pd.DataFrame({
        "outcome": np.where(ships["delay_hours"] > 0, "late", "on time"),
        "region": ships["region"],
        "season": np.where(ships["is_winter"], "winter", "other"),
        "history": pd.cut(ships["sup_n_ship_all"], [-1, 19, 99, 299, np.inf],
                          labels=["<20 shipments", "20-99", "100-299", "300+"]).astype(str),
        "recent_change": np.where(change >= 0.5, "changing (30d vs 90d gap >= 0.5 logit)", "stable"),
        "cold_start": np.where(ships["is_cold"], "cold start", "established"),
    }, index=ships.index)


def crps_by_group(per_ship: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for group in ["outcome", "region", "season", "history", "recent_change", "cold_start"]:
        g = per_ship.groupby(group)
        table = g[[f"crps_{m}" for m in models]].mean()
        table.columns = models
        table["n_shipments"] = g.size()
        table["share_of_total_crps_naive"] = g["crps_naive"].sum() / per_ship["crps_naive"].sum()
        table["fast_vs_naive_pct"] = 100 * (table["fast_1"] / table["naive"] - 1)
        rows.append(table.reset_index().rename(columns={group: "level"}).assign(group=group))
    cols = ["group", "level", "n_shipments", *models, "fast_vs_naive_pct", "share_of_total_crps_naive"]
    return pd.concat(rows, ignore_index=True)[cols]


def calibration_table(p_late: np.ndarray, late: np.ndarray, model: str) -> pd.DataFrame:
    bins = pd.qcut(p_late, N_CALIBRATION_BINS, duplicates="drop")
    t = pd.DataFrame({"p": p_late, "y": late, "bin": bins}).groupby("bin", observed=True)
    return pd.DataFrame({"model": model, "mean_predicted": t["p"].mean(),
                         "observed_rate": t["y"].mean(), "n": t.size()}).reset_index(drop=True)


def pit_histogram(pit: np.ndarray) -> pd.DataFrame:
    counts, edges = np.histogram(pit, bins=N_PIT_BINS, range=(0, 1))
    return pd.DataFrame({"bin_low": edges[:-1], "bin_high": edges[1:],
                         "share": counts / counts.sum(), "uniform_share": 1 / N_PIT_BINS})


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def cv_folds(scenario: str) -> list[str]:
    meta = json.loads((PROCESSED / scenario / "folds" / "fold_definitions.json").read_text())
    return [f["fold_id"] for f in meta["folds"] if f["fold_id"] != "holdout"]


def run(scenario: str, feature_set: str = FEATURE_SET, verbose: bool = True) -> dict:
    per_ship, fold_rows, pits = [], [], []
    pred = {"naive": [], "fast_1": [], "late": []}
    for fold_id in cv_folds(scenario):
        train_ships, test_ships, _ = prepare_fold(scenario, fold_id)
        rng = np.random.default_rng(fold_seed(scenario, fold_id, "diagnostics"))
        y = test_ships["delay_hours"].to_numpy(float)
        late = (y > 0).astype(float)

        naive = naive_samples(train_ships, test_ships, N_DRAWS, rng)
        fast = FastModel1(feature_set).fit(train_ships)
        fast_draws = fast.predict_samples(test_ships, N_DRAWS, rng)
        p_fast, _, _ = fast.predict_params(test_ships)

        crps = {"naive": crps_from_samples(naive, y), "fast_1": crps_from_samples(fast_draws, y)}
        per_ship.append(diagnostic_groups(test_ships).assign(
            fold_id=fold_id, crps_naive=crps["naive"], crps_fast_1=crps["fast_1"]))
        fold_rows.append({"fold_id": fold_id, "naive": crps["naive"].mean(),
                          "fast_1": crps["fast_1"].mean(), "fast_converged": fast.converged})
        pred["naive"].append((naive > 0).mean(axis=1))
        pred["fast_1"].append(p_fast)
        pred["late"].append(late)
        pits.append(fast.pit(test_ships, rng))
        if verbose:
            print(f"  {fold_id}: CRPS naive {crps['naive'].mean():.4f}  fast_1 {crps['fast_1'].mean():.4f}")

    folds = pd.DataFrame(fold_rows)
    metrics = RESULTS / scenario / "metrics.csv"
    if metrics.exists():                     # Bayesian model 1, if it has been run
        m = pd.read_csv(metrics)
        bayes = m[m["model"] == "model_1"].set_index("fold_id")["crps"]
        folds["bayes_model_1"] = folds["fold_id"].map(bayes)
    per_ship = pd.concat(per_ship, ignore_index=True)
    late_all = np.concatenate(pred["late"])            # pool all folds, then bin once
    calibration = pd.concat([calibration_table(np.concatenate(pred[m]), late_all, m)
                             for m in ("naive", "fast_1")], ignore_index=True)
    tables = {"fold_comparison": folds, "crps_by_group": crps_by_group(per_ship, ["naive", "fast_1"]),
              "calibration": calibration, "pit": pit_histogram(np.concatenate(pits))}

    out = RESULTS / scenario / "diagnostics" / feature_set
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    if verbose:
        print(f"  wrote {out}")
    return tables


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--feature-set", choices=("v1", "v2"), default=FEATURE_SET)
    args = parser.parse_args()
    for scenario in (SCENARIOS if args.scenario == "all" else (args.scenario,)):
        print(f"[diagnostics] scenario={scenario} feature_set={args.feature_set}")
        tables = run(scenario, args.feature_set)
        print(tables["fold_comparison"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
