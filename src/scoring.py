"""Scoring for probabilistic delay forecasts.

Every model, including the naive baseline, hands back the same thing: a matrix
of predictive samples, one row per test shipment and one column per draw
(delay in hours, 0 = on time). All metrics are computed from those samples, so
every model is scored by exactly the same code.

Delay metrics are scored per SHIPMENT (each truck counted once). Shortage
metrics are scored per DELIVERY ROW (part x plant), because a shortage is
about one part running out.

Primary metric: CRPS (lower is better). It scores the whole predictive
distribution, so it rewards a model for getting both the on-time spike at zero
and the long delay tail right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PRECISION_AT_K = 10     # dashboard shows the top 10 shortage risks per day


def crps_from_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS per observation from an (n, S) matrix of predictive draws.

    CRPS = E|X - y| - 0.5 E|X - X'|. The second term uses the sorted-sample
    identity E|X - X'| = (2 / S^2) * sum_i (2i - S - 1) x_(i), which is O(S log S).
    """
    s = np.sort(samples, axis=1)
    n_draws = s.shape[1]
    term1 = np.abs(s - y[:, None]).mean(axis=1)
    weights = 2 * np.arange(1, n_draws + 1) - n_draws - 1
    term2 = (s * weights).sum(axis=1) / n_draws**2
    return term1 - term2


def pinball(samples: np.ndarray, y: np.ndarray, q: float) -> np.ndarray:
    pred = np.quantile(samples, q, axis=1)
    diff = y - pred
    return np.maximum(q * diff, (q - 1) * diff)


def delay_metrics(samples: np.ndarray, y: np.ndarray, late: np.ndarray) -> dict:
    """Shipment-level metrics for one model on one test window."""
    p_late = (samples > 0).mean(axis=1)
    p_clip = np.clip(p_late, 1e-4, 1 - 1e-4)
    lo, hi = np.quantile(samples, [0.05, 0.95], axis=1)
    mean = samples.mean(axis=1)
    return {
        "crps": crps_from_samples(samples, y).mean(),
        "mae_median": np.abs(np.median(samples, axis=1) - y).mean(),
        "rmse_mean": np.sqrt(((mean - y) ** 2).mean()),
        "bias_mean": (mean - y).mean(),                      # negative = under-predicts lateness
        "brier_late": ((p_late - late) ** 2).mean(),
        "log_loss_late": -(late * np.log(p_clip) + (1 - late) * np.log(1 - p_clip)).mean(),
        "coverage_90": ((y >= lo) & (y <= hi)).mean(),
        "width_90": (hi - lo).mean(),
        "pinball_q90": pinball(samples, y, 0.90).mean(),
    }


def shortage_metrics(row_samples: np.ndarray, coverage_hours: np.ndarray,
                     shortage: np.ndarray, day: np.ndarray) -> dict:
    """Row-level shortage metrics: P(delay > safety stock coverage)."""
    p_short = (row_samples > coverage_hours[:, None]).mean(axis=1)
    top = (pd.DataFrame({"day": day, "p": p_short, "y": shortage})
           .sort_values("p", ascending=False).groupby("day").head(PRECISION_AT_K))
    return {
        "brier_shortage": ((p_short - shortage) ** 2).mean(),
        "pr_auc_shortage": average_precision_score(shortage, p_short) if shortage.any() else np.nan,
        f"precision_at_{PRECISION_AT_K}": top["y"].mean(),
    }


def score(samples: np.ndarray, test_ships: pd.DataFrame, test_rows: pd.DataFrame) -> dict:
    """All metrics for one model on one fold's test window.

    test_ships: one row per shipment, aligned with `samples`
    test_rows:  one row per delivery row, with a `ship_idx` column pointing
                into test_ships, plus safety_coverage_hours and shortage_flag
    """
    y = test_ships["delay_hours"].to_numpy(float)
    late = (y > 0).astype(float)
    out = delay_metrics(samples, y, late)

    cold = test_ships["is_cold"].to_numpy(bool)
    out["crps_cold_start"] = crps_from_samples(samples[cold], y[cold]).mean() if cold.any() else np.nan
    out["n_cold_start"] = int(cold.sum())

    rows = test_rows.dropna(subset=["shortage_flag", "safety_coverage_hours"])
    out.update(shortage_metrics(samples[rows["ship_idx"].to_numpy()],
                                rows["safety_coverage_hours"].to_numpy(float),
                                rows["shortage_flag"].to_numpy(bool).astype(float),
                                rows["production_date"].to_numpy()))
    out["n_test_shipments"] = len(y)
    return out


def summarize(metrics: pd.DataFrame, reference: str = "naive") -> pd.DataFrame:
    """Mean of each metric across folds, plus the paired CRPS difference vs a
    reference model (negative = better than the reference) and how many folds
    the model beat it in. A gain that flips sign across folds is not a win."""
    mean = metrics.groupby("model").mean(numeric_only=True)
    ref = metrics[metrics["model"] == reference].set_index("fold_id")["crps"]
    diffs = metrics.assign(crps_diff=metrics["crps"] - metrics["fold_id"].map(ref))
    paired = diffs.groupby("model")["crps_diff"].agg(
        crps_vs_ref="mean", crps_vs_ref_sd="std", folds_better=lambda d: int((d < 0).sum()))
    return mean.join(paired)
