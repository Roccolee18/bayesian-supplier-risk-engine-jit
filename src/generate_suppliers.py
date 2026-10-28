"""Generate the `suppliers` table: one row per supplier.

Primary key: supplier_code

Observable columns (what a real supplier master would hold):
    supplier_code, region, distance_km, lead_time_days, commodity, onboard_date

Evaluation-only columns (prefixed `true_`): the simulator's latent truth. A real
plant never sees these, so they must NOT be used as model features.
    true_cluster, true_cluster_after_drift, true_drift_month,
    true_p_late, true_delay_mu, true_delay_sigma,
    true_p_late_after, true_delay_mu_after, true_delay_sigma_after

Scenarios:
    clusters     suppliers belong to 4 latent reliability clusters
    no_clusters  reliability drawn from one continuous distribution (no groups)
    drift        as `clusters`, but 20% switch cluster at month 3, 10% at month 7

Usage:
    python generate_suppliers.py --scenario clusters --seed 20261027
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _sample_categorical(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise categorical draw from an (n, k) matrix of weights."""
    probs = weights / weights.sum(axis=1, keepdims=True)
    u = rng.random(len(probs))[:, None]
    return (u > probs.cumsum(axis=1)).sum(axis=1)


def _static_attributes(rng: np.random.Generator) -> pd.DataFrame:
    n = C.N_SUPPLIERS
    region_names = list(C.REGIONS)
    region_w = np.array([C.REGIONS[r]["weight"] for r in region_names])
    region = rng.choice(region_names, size=n, p=region_w / region_w.sum())

    lo = np.array([C.REGIONS[r]["km"][0] for r in region])
    hi = np.array([C.REGIONS[r]["km"][1] for r in region])
    distance_km = np.round(rng.uniform(lo, hi), 0)

    lead_time_days = np.clip(np.round(1 + distance_km / 600 + rng.normal(0, 0.5, n)), 1, None).astype(int)

    # Most suppliers are established; ~12% onboard during the simulation (cold start)
    is_new = rng.random(n) < 0.12
    established = C.START_DATE - pd.to_timedelta(rng.integers(180, 3650, n), unit="D")
    new = C.START_DATE + pd.to_timedelta(rng.integers(14, 240, n), unit="D")
    onboard_date = pd.Series(np.where(is_new, new, established)).dt.normalize()

    return pd.DataFrame({
        "supplier_code": [f"S{i:04d}" for i in range(1, n + 1)],
        "region": region,
        "distance_km": distance_km,
        "lead_time_days": lead_time_days,
        "commodity": rng.choice(C.COMMODITIES, size=n),
        "onboard_date": onboard_date,
    })


def _cluster_params(df: pd.DataFrame, rng: np.random.Generator):
    """Assign latent clusters and per-supplier params on the (logit p, mu, log sigma) scale."""
    n = len(df)
    weights = np.tile(C.CLUSTER_WEIGHTS, (n, 1)).astype(float)
    far = df["region"].isin(C.FAR_REGIONS).to_numpy()
    weights[far, C.CLUSTER_NAMES.index("high_volatility")] *= 2.0   # long lanes are more volatile
    weights[far, C.CLUSTER_NAMES.index("chronic_problem")] *= 1.3
    cluster_idx = _sample_categorical(weights, rng)

    centres = np.array([[C.logit(c["p_late"]), c["mu"], np.log(c["sigma"])] for c in C.CLUSTERS.values()])
    offsets = rng.normal(0, C.SUPPLIER_NOISE_SD, size=(n, 3))   # supplier keeps its offset after drifting
    return cluster_idx, centres, offsets


def _continuous_params(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """No-cluster scenario: one correlated multivariate normal, no group structure.

    The mean and spread roughly match the cluster mixture so that marginal
    lateness looks similar across scenarios; only the group structure differs.
    """
    centres = np.array([[C.logit(c["p_late"]), c["mu"], np.log(c["sigma"])] for c in C.CLUSTERS.values()])
    mean = C.CLUSTER_WEIGHTS @ centres
    sd = np.array([0.9, 0.6, 0.35])
    corr = np.array([[1.0, 0.5, 0.2],
                     [0.5, 1.0, 0.3],
                     [0.2, 0.3, 1.0]])
    cov = corr * np.outer(sd, sd)
    params = rng.multivariate_normal(mean, cov, size=len(df))
    far = df["region"].isin(C.FAR_REGIONS).to_numpy()
    params[far, 0] += 0.3    # far lanes are a bit more often late
    params[far, 2] += 0.15   # ... and more volatile
    return params


def _to_natural(params: np.ndarray) -> dict:
    return {
        "p": np.round(C.expit(params[:, 0]), 4),
        "mu": np.round(params[:, 1], 4),
        "sigma": np.round(np.exp(params[:, 2]), 4),
    }


def generate(scenario: str, seed: int = C.BASE_SEED) -> pd.DataFrame:
    shared_rng = C.get_rng("suppliers", "shared", seed)
    df = _static_attributes(shared_rng)           # identical across scenarios
    n = len(df)

    if scenario == "no_clusters":
        before = after = _continuous_params(df, C.get_rng("suppliers", "no_clusters", seed))
        cluster_before = cluster_after = pd.Series([pd.NA] * n, dtype="string")
        drift_month = np.full(n, np.nan)
    else:
        # Same clusters in `clusters` and `drift`, so drift is the only difference
        cluster_idx, centres, offsets = _cluster_params(df, C.get_rng("suppliers", "clustered", seed))
        after_idx = cluster_idx.copy()
        drift_month = np.full(n, np.nan)

        if scenario == "drift":
            rng = C.get_rng("suppliers", "drift", seed)
            remaining = np.arange(n)
            for month, share in C.DRIFT_SCHEDULE:
                k = int(round(share * n))
                chosen = rng.choice(remaining, size=k, replace=False)
                remaining = np.setdiff1d(remaining, chosen)
                drift_month[chosen] = month
                # move to a different cluster, chosen uniformly
                shift = rng.integers(1, len(C.CLUSTER_NAMES), size=k)
                after_idx[chosen] = (cluster_idx[chosen] + shift) % len(C.CLUSTER_NAMES)

        before = centres[cluster_idx] + offsets
        after = centres[after_idx] + offsets
        names = np.array(C.CLUSTER_NAMES)
        cluster_before = pd.Series(names[cluster_idx], dtype="string")
        cluster_after = pd.Series(names[after_idx], dtype="string")

    b, a = _to_natural(before), _to_natural(after)
    df["true_cluster"] = cluster_before
    df["true_cluster_after_drift"] = cluster_after
    df["true_drift_month"] = pd.Series(drift_month).astype("Int64")
    df["true_p_late"], df["true_delay_mu"], df["true_delay_sigma"] = b["p"], b["mu"], b["sigma"]
    df["true_p_late_after"], df["true_delay_mu_after"], df["true_delay_sigma_after"] = a["p"], a["mu"], a["sigma"]

    df = C.mask_missing(df, "suppliers", "distance_km", shared_rng)
    return df


def main() -> None:
    args = C.parse_args(__doc__.splitlines()[0])
    print(f"[suppliers] scenario={args.scenario} seed={args.seed}")
    C.write_table(generate(args.scenario, args.seed), "suppliers", args.scenario, args.out_dir, args.format)


if __name__ == "__main__":
    main()
