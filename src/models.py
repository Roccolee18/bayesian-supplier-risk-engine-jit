"""Naive baseline and the four Bayesian hurdle models.

All four Bayesian models share one structure, a hurdle (zero-inflated
log-normal) model with three linear predictors ("heads"):

    late_i       ~ Bernoulli(logit^-1(eta_i))           will the truck be late?
    log(delay_i) ~ Normal(mu_i, sigma_i)    if late      how late, on the log scale
    eta_i       = a0 + X_late_i  . b + effects
    mu_i        = c0 + X_mu_i    . g + effects
    log sigma_i = d0 + X_sigma_i . h + effects           volatility can differ too

Each head gets only the predictors relevant to it, and overlapping predictors
are entered as differences (e.g. 30-day minus 90-day late rate) so they are not
strongly correlated. Correlated predictors make the posterior hard to sample.

The two parts share no parameters, so their posteriors are independent and they
are fit as two separate PyMC models:
    occurrence (late head):         a random subsample of training shipments
    severity (mu and sigma heads):  every late training shipment

The four models differ only in the "effects" term (the 2x2 design):

    model_1  pooled:        none; suppliers differ only through their history features
    model_2  + clusters:    a fixed effect per cluster (C2..Ck, NEW; C1 is the reference)
    model_3  hierarchical:  a random effect per supplier, effect_s ~ Normal(0, tau)
    model_4  nested:        cluster fixed effects + supplier effects with a
                            cluster-specific spread, effect_s ~ Normal(0, tau_cluster)

Supplier effects use the non-centred form (effect_s = tau * z_s). Hierarchical
models need a proper warm-up: with 200 tuning steps model 4's chains did not
agree; with 1000 they converge (r-hat ~1.02, no divergences).

Supplier effects are added to the late and mu heads. The sigma head is left
without them: with ~10 late shipments per supplier, a per-supplier volatility
is barely identified and made the sampler fail.

A supplier unseen in training gets a fresh effect drawn from its population
(model 3) or its cluster's population (model 4) at prediction time, which is
how partial pooling handles cold-start suppliers.

The naive baseline needs no fitting: each test shipment's forecast is the
empirical distribution of its supplier's delays in the training window
(pooled across all suppliers if it has fewer than 20 training shipments).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

MODELS = ("naive", "model_1", "model_2", "model_3", "model_4")
BAYES = {
    "model_1": dict(clusters=False, suppliers=None),
    "model_2": dict(clusters=True, suppliers=None),
    "model_3": dict(clusters=False, suppliers="shared"),
    "model_4": dict(clusters=True, suppliers="by_cluster"),
}
HEADS = ("late", "mu", "sigma")
MAX_PRED_DELAY_HOURS = 336.0     # 2 weeks: a truck later than this is a cancellation, not a delay
SUPPLIER_EFFECT_HEADS = ("late", "mu")
NAIVE_MIN_SHIPMENTS = 20
PRIORS = {   # (intercept mean, intercept sd, coefficient sd) on the standardized scale
    "late": (-2.0, 1.5, 0.5),           # ~12% late at the intercept
    "mu": (1.3, 1.0, 0.5),              # median late delay ~3.7 h
    "sigma": (np.log(0.7), 0.5, 0.3),
}


# --------------------------------------------------------------------------- #
# Shipment-level data and design matrices
# --------------------------------------------------------------------------- #
def to_shipments(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per truck (parts on a truck share one delay)."""
    ships = rows.drop_duplicates("shipment_id").reset_index(drop=True)
    ships["is_cold"] = ships["sup_is_cold_start"].astype(bool) | (ships["cluster"] == "NEW")
    return ships


def _logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def head_predictors(df: pd.DataFrame, head: str) -> pd.DataFrame:
    """Transformed predictors for one head. Rates on the logit scale, delays on
    the log scale; overlapping windows entered as differences."""
    late_all, late_90, late_30 = (_logit(df[f"sup_late_rate_{w}"]) for w in ("all", "90d", "30d"))
    delay_all = np.log(df["sup_mean_late_delay_all"].clip(lower=0.05))
    delay_90 = np.log(df["sup_mean_late_delay_90d"].clip(lower=0.05))
    common = {
        "cold_start": df["sup_is_cold_start"].astype(float),
        "region_late_7d": df["region_late_rate_7d"],
        "is_winter": df["is_winter"].astype(float),
    }
    if head == "late":
        cols = {
            "logit_late_all": late_all,
            "late_90d_vs_all": late_90 - late_all,
            "late_30d_vs_90d": late_30 - late_90,
            "log_n_ship": np.log1p(df["sup_n_ship_all"]),
            "log_days_since_late": np.log1p(df["sup_days_since_late"].clip(lower=0)),
            "planned_hour": df["planned_hour"],
            "trips_per_day": df["trips_per_day"],
            **common,
        }
    elif head == "mu":
        cols = {"log_late_delay_all": delay_all, "late_delay_90d_vs_all": delay_90 - delay_all, **common}
    else:
        cols = {
            "log_delay_sd_all": df["sup_log_delay_sd_all"],
            "delay_sd_90d_vs_all": df["sup_log_delay_sd_90d"] - df["sup_log_delay_sd_all"],
            "cold_start": common["cold_start"],
        }
    x = pd.DataFrame(cols, index=df.index)
    if head in ("late", "mu"):
        regions = pd.get_dummies(df["region"], prefix="region", dtype=float)
        x = pd.concat([x, regions], axis=1)
    return x


class Design:
    """Per-head predictors, standardized with training means and SDs only.

    For the cluster models (2 and 4), `groups` holds each row's cluster and every
    predictor is also centred on its training cluster mean (a within/between
    split). Clusters are built from these same history features, so without
    centring the cluster effects and the feature coefficients explain the same
    thing and the sampler cannot separate them. After centring, cluster effects
    capture differences between clusters and coefficients capture differences
    within a cluster.
    """

    def fit(self, train: pd.DataFrame, head: str, groups: pd.Series | None = None) -> "Design":
        self.head = head
        x = head_predictors(train, head)
        region_cols = sorted(c for c in x.columns if c.startswith("region_"))
        x = x.drop(columns=region_cols[:1])            # reference region, avoids collinearity
        self.columns = list(x.columns)
        self.mean, self.sd = x.mean(), x.std().replace(0, 1)
        self.group_means = None
        if groups is not None:
            z = ((x - self.mean) / self.sd).fillna(0.0)
            self.group_means = z.groupby(groups.to_numpy()).mean()
        return self

    def transform(self, df: pd.DataFrame, groups: pd.Series | None = None) -> np.ndarray:
        x = head_predictors(df, self.head).reindex(columns=self.columns, fill_value=0.0)
        z = ((x - self.mean) / self.sd).fillna(0.0)
        if self.group_means is not None:
            z = z - self.group_means.reindex(groups.to_numpy()).fillna(0.0).to_numpy()
        return z.to_numpy(float)


# --------------------------------------------------------------------------- #
# Naive baseline
# --------------------------------------------------------------------------- #
def naive_samples(train_ships: pd.DataFrame, test_ships: pd.DataFrame,
                  n_draws: int, rng: np.random.Generator) -> np.ndarray:
    known = train_ships.dropna(subset=["delay_hours"])
    pooled = known["delay_hours"].to_numpy()
    by_supplier = {s: g.to_numpy() for s, g in known.groupby("supplier_code")["delay_hours"]
                   if len(g) >= NAIVE_MIN_SHIPMENTS}
    out = np.empty((len(test_ships), n_draws))
    for supplier, idx in test_ships.groupby("supplier_code").indices.items():
        source = by_supplier.get(supplier, pooled)
        out[idx] = rng.choice(source, size=(len(idx), n_draws))
    return out


# --------------------------------------------------------------------------- #
# Bayesian hurdle models
# --------------------------------------------------------------------------- #
class HurdleModel:
    """One of model_1..model_4, fit with PyMC and predicted in NumPy."""

    def __init__(self, name: str):
        self.name = name
        self.use_clusters = BAYES[name]["clusters"]
        self.supplier_effects = BAYES[name]["suppliers"]

    def _cluster_dummies(self, clusters: pd.Series) -> np.ndarray:
        return np.column_stack([(clusters == c).to_numpy(float) for c in self.cluster_levels[1:]]) \
            if len(self.cluster_levels) > 1 else np.zeros((len(clusters), 0))

    def _add_head(self, head: str, data: pd.DataFrame) -> "pm.TensorVariable":
        """Linear predictor for one head, inside the active PyMC model."""
        groups = data["cluster"] if self.use_clusters else None
        design = Design().fit(data, head, groups)
        self.designs[head] = design
        X = design.transform(data, groups)
        loc, scale, coef_sd = PRIORS[head]
        intercept = pm.Normal(f"{head}_intercept", loc, scale)
        beta = pm.Normal(f"{head}_beta", 0, coef_sd, shape=X.shape[1])
        lp = intercept + pm.math.dot(X, beta)

        if self.use_clusters and len(self.cluster_levels) > 1:
            # cluster shifts get the same prior scale as that head's coefficients, x2;
            # a wide prior on the sigma head let small clusters (e.g. NEW) blow up the tail
            alpha = pm.Normal(f"{head}_cluster", 0, 2 * coef_sd, shape=len(self.cluster_levels) - 1)
            lp = lp + pm.math.dot(self._cluster_dummies(data["cluster"]), alpha)

        if self.supplier_effects and head in SUPPLIER_EFFECT_HEADS:
            sup_idx = np.searchsorted(self.suppliers, data["supplier_code"].to_numpy())
            if self.supplier_effects == "shared":
                tau = pm.HalfNormal(f"{head}_tau", 0.5)
                spread = tau
            else:
                tau = pm.HalfNormal(f"{head}_tau", 0.5, shape=len(self.cluster_levels))
                spread = tau[self.sup_cluster]
            z = pm.Normal(f"{head}_z", 0, 1, shape=len(self.suppliers))         # non-centred
            effect = pm.Deterministic(f"{head}_effect", spread * z)
            lp = lp + effect[sup_idx]
        return lp

    def fit(self, train_ships: pd.DataFrame, n_occurrence: int, draws: int, tune: int,
            chains: int, seed: int, subsample_seed: int = 0, sampler: str = "nutpie"):
        data = train_ships.dropna(subset=["delay_hours"]).reset_index(drop=True)
        self.designs, self.idata = {}, {}

        # Shared indexing: suppliers, clusters (C1 = reference), each supplier's cluster
        self.cluster_levels = sorted(data["cluster"].unique(), key=lambda c: (c == "NEW", c))
        self.suppliers = np.array(sorted(data["supplier_code"].unique()))
        level = {c: i for i, c in enumerate(self.cluster_levels)}
        self.sup_cluster = (data.drop_duplicates("supplier_code").set_index("supplier_code")
                            .loc[self.suppliers, "cluster"].map(level).to_numpy())

        occurrence = data.sample(min(n_occurrence, len(data)), random_state=subsample_seed)
        severity = data[data["delay_hours"] > 0]
        kwargs = dict(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False)
        if sampler == "nutpie":
            kwargs["nuts_sampler"] = "nutpie"

        with pm.Model():
            lp = self._add_head("late", occurrence)
            pm.Bernoulli("late_obs", logit_p=lp, observed=(occurrence["delay_hours"] > 0).to_numpy(int))
            self.idata["occurrence"] = pm.sample(**kwargs)
        with pm.Model():
            mu = self._add_head("mu", severity)
            log_sigma = self._add_head("sigma", severity)
            pm.Normal("log_delay_obs", mu=mu, sigma=pm.math.exp(log_sigma),
                      observed=np.log(severity["delay_hours"].to_numpy(float)))
            self.idata["severity"] = pm.sample(**kwargs)
        return self

    # -- prediction -----------------------------------------------------------
    def predict_samples(self, test_ships: pd.DataFrame, n_draws: int, rng) -> np.ndarray:
        """(n_test, n_draws) predictive delays. Each column uses one posterior draw
        per part (occurrence, severity), shared by every parameter in that part."""
        n = len(test_ships)
        known = np.isin(test_ships["supplier_code"], self.suppliers)
        sup_idx = np.searchsorted(self.suppliers, test_ships["supplier_code"].to_numpy())
        sup_idx = np.where(known, np.minimum(sup_idx, len(self.suppliers) - 1), 0)
        level = {c: i for i, c in enumerate(self.cluster_levels)}
        cl_idx = test_ships["cluster"].map(level).fillna(level.get("NEW", 0)).to_numpy(int)

        draw_ids = {}
        for part, idata in self.idata.items():
            total = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
            draw_ids[part] = rng.choice(total, size=n_draws, replace=total < n_draws)

        def post(var: str) -> np.ndarray:
            part = "occurrence" if var.startswith("late") else "severity"
            p = self.idata[part].posterior[var].stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()
            return p[draw_ids[part]]

        lp = {}
        for head in HEADS:
            X = self.designs[head].transform(test_ships, test_ships["cluster"] if self.use_clusters else None)
            val = post(f"{head}_intercept")[None, :] + X @ post(f"{head}_beta").T            # (n, S)
            if self.use_clusters and len(self.cluster_levels) > 1:
                val = val + self._cluster_dummies(test_ships["cluster"]) @ post(f"{head}_cluster").T
            if self.supplier_effects and head in SUPPLIER_EFFECT_HEADS:
                effect, tau = post(f"{head}_effect"), post(f"{head}_tau")    # (S, n_sup), (S,) or (S, k)
                eff_known = effect[:, sup_idx].T
                if self.supplier_effects == "shared":
                    eff_new = tau[None, :] * rng.standard_normal((n, n_draws))
                else:
                    eff_new = tau[:, cl_idx].T * rng.standard_normal((n, n_draws))
                val = val + np.where(known[:, None], eff_known, eff_new)
            lp[head] = val

        late = rng.random((n, n_draws)) < 1 / (1 + np.exp(-lp["late"]))
        log_delay = lp["mu"] + np.exp(lp["sigma"]) * rng.standard_normal((n, n_draws))
        return np.where(late, np.exp(np.minimum(log_delay, np.log(MAX_PRED_DELAY_HOURS))), 0.0)

    def diagnostics(self) -> dict:
        import arviz as az
        out = {"max_rhat": 0.0, "min_ess_bulk": np.inf, "divergences": 0}
        for idata in self.idata.values():
            summ = az.summary(idata, kind="diagnostics")
            out["max_rhat"] = max(out["max_rhat"], float(summ["r_hat"].max()))
            out["min_ess_bulk"] = min(out["min_ess_bulk"], float(summ["ess_bulk"].min()))
            out["divergences"] += int(idata.sample_stats["diverging"].sum())
        return out
