"""Generate the `deliveries` table: one row per supplier x part x plant x scheduled arrival.

Primary key: delivery_id
Foreign keys:
    supplier_code                     -> suppliers
    (part_code, plant)                -> parts
    (part_code, plant, production_date) -> production_schedule

All parts from one supplier to one plant ride the same truck, so rows sharing a
`shipment_id` share the same delay. Trucks run only on production days, at
1-4 fixed times per day depending on how far away the supplier is.

Delay process (zero-inflated log-normal, per shipment):
    late       ~ Bernoulli(p_late)
    delay_hrs  ~ LogNormal(mu, sigma) if late, else 0     (capped at 96 h)
p_late and mu come from the supplier's latent parameters, adjusted for:
    - winter months (Dec-Feb), stronger on long, weather-exposed lanes
    - a few regional disruption events per region (e.g. storms, border delays)
In the drift scenario, a supplier's parameters switch at its drift month.

On-time trucks arrive up to 45 minutes early; their delay is recorded as 0.
About 0.5% of receipts are never scanned, so actual time and delay are missing.

Columns:
    delivery_id, shipment_id, supplier_code, part_code, plant, production_date,
    planned_delivery_datetime, actual_delivery_datetime, delay_hours, is_late,
    ordered_qty_boxes,
    true_cluster_current, true_disruption   (evaluation only)

Usage:
    python generate_deliveries.py --scenario clusters
    (requires suppliers, parts and production_schedule for the same scenario)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _build_shipments(suppliers, parts, schedule, rng) -> pd.DataFrame:
    """One row per truck: route x production day x trip."""
    plant_days = (schedule.groupby(["plant", "production_date"], as_index=False)["planned_hours"].first())
    plant_days = plant_days[plant_days["planned_hours"] > 0]

    routes = parts[["supplier_code", "plant"]].drop_duplicates().merge(suppliers, on="supplier_code")
    routes["trips_per_day"] = [rng.choice(C.REGIONS[r]["trips"]) for r in routes["region"]]
    # fixed per-route offset so not every truck arrives on the hour
    routes["route_offset_min"] = rng.choice(np.arange(-60, 61, 15), size=len(routes))

    ships = routes.merge(plant_days, on="plant")
    ships = ships[ships["production_date"] >= ships["onboard_date"]].reset_index(drop=True)
    ships = ships.loc[ships.index.repeat(ships["trips_per_day"])].reset_index(drop=True)
    ships["trip_no"] = ships.groupby(["supplier_code", "plant", "production_date"]).cumcount()

    hours_into_window = (ships["trip_no"] + 0.5) / ships["trips_per_day"] * C.DELIVERY_WINDOW_LENGTH_HOURS
    ships["planned_delivery_datetime"] = (
        ships["production_date"]
        + pd.to_timedelta(C.DELIVERY_WINDOW_START_HOUR + hours_into_window, unit="h")
        + pd.to_timedelta(ships["route_offset_min"], unit="m")
    ).dt.round("15min")
    return ships


def _current_params(ships: pd.DataFrame):
    """Pick before/after-drift parameters for each shipment's date."""
    drift_starts = {m: C.START_DATE + pd.DateOffset(months=m) for m, _ in C.DRIFT_SCHEDULE}
    drift_start = ships["true_drift_month"].map(drift_starts)
    after = (ships["planned_delivery_datetime"] >= drift_start).fillna(False).to_numpy(dtype=bool)

    p = np.where(after, ships["true_p_late_after"], ships["true_p_late"]).astype(float)
    mu = np.where(after, ships["true_delay_mu_after"], ships["true_delay_mu"]).astype(float)
    sigma = np.where(after, ships["true_delay_sigma_after"], ships["true_delay_sigma"]).astype(float)
    cluster = np.where(after, ships["true_cluster_after_drift"].astype(object),
                       ships["true_cluster"].astype(object))
    return p, mu, sigma, cluster


def _apply_conditions(ships, p, mu, rng):
    """Winter and regional disruption effects."""
    month = ships["planned_delivery_datetime"].dt.month
    winter = month.isin(C.WINTER_MONTHS).to_numpy()
    sensitive = ships["region"].map(lambda r: C.REGIONS[r]["winter_sensitive"]).to_numpy(dtype=bool)
    winter_mult = np.where(sensitive, C.WINTER_P_MULT[True], C.WINTER_P_MULT[False])
    p = np.where(winter, p * winter_mult, p)

    disrupted = np.zeros(len(ships), dtype=bool)
    period_days = (C.END_DATE - C.START_DATE).days
    for region in C.REGIONS:
        for _ in range(rng.poisson(C.DISRUPTIONS_PER_REGION)):
            start = C.START_DATE + pd.Timedelta(days=int(rng.integers(0, period_days)))
            end = start + pd.Timedelta(days=int(rng.integers(1, 5)))
            hit = ((ships["region"] == region)
                   & (ships["planned_delivery_datetime"] >= start)
                   & (ships["planned_delivery_datetime"] < end)).to_numpy()
            disrupted |= hit
    p = np.where(disrupted, p * C.DISRUPTION_P_MULT, p)
    mu = np.where(disrupted, mu + C.DISRUPTION_MU_SHIFT, mu)
    return np.clip(p, 0, 0.95), mu, disrupted


def generate(scenario: str, seed: int = C.BASE_SEED, root=C.DEFAULT_ROOT) -> pd.DataFrame:
    rng = C.get_rng("deliveries", "shared", seed)  # common random numbers across scenarios
    suppliers = C.read_table("suppliers", scenario, root)
    parts = C.read_table("parts", scenario, root)
    schedule = C.read_table("production_schedule", scenario, root)

    ships = _build_shipments(suppliers, parts, schedule, rng)
    p, mu, sigma, cluster = _current_params(ships)
    p, mu, disrupted = _apply_conditions(ships, p, mu, rng)

    # Zero-inflated log-normal delay, drawn once per truck
    n = len(ships)
    late = rng.random(n) < p
    delay = np.where(late, np.minimum(rng.lognormal(mu, sigma), C.MAX_DELAY_HOURS), 0.0)
    early_min = rng.uniform(0, 45, n)
    arrival_offset = pd.to_timedelta(np.where(late, delay * 60, -early_min), unit="m")

    ships["actual_delivery_datetime"] = (ships["planned_delivery_datetime"] + arrival_offset).dt.round("1min")
    ships["delay_hours"] = np.round(delay, 2)
    ships["true_cluster_current"] = pd.Series(cluster, dtype="string")
    ships["true_disruption"] = disrupted
    ships["shipment_id"] = [f"SH{i:07d}" for i in range(1, n + 1)]

    # Expand trucks to one row per part on board
    df = ships.merge(parts[["supplier_code", "plant", "part_code", "usage_per_hour", "box_qty"]],
                     on=["supplier_code", "plant"])
    hours_covered = df["planned_hours"] / df["trips_per_day"]
    df["ordered_qty_boxes"] = np.maximum(1, np.ceil(df["usage_per_hour"] * hours_covered / df["box_qty"])).astype(int)

    # Unscanned receipts: actual time unknown, so the delay is unknown too
    df = C.mask_missing(df, "deliveries", "actual_delivery_datetime", rng)
    missing = df["actual_delivery_datetime"].isna()
    df.loc[missing, "delay_hours"] = np.nan
    df["is_late"] = (df["delay_hours"] > 0).astype("boolean")
    df.loc[missing, "is_late"] = pd.NA

    df = df.sort_values(["planned_delivery_datetime", "supplier_code", "part_code"]).reset_index(drop=True)
    df["delivery_id"] = [f"D{i:08d}" for i in range(1, len(df) + 1)]
    return df[["delivery_id", "shipment_id", "supplier_code", "part_code", "plant", "production_date",
               "planned_delivery_datetime", "actual_delivery_datetime", "delay_hours", "is_late",
               "ordered_qty_boxes", "true_cluster_current", "true_disruption"]]


def main() -> None:
    args = C.parse_args(__doc__.splitlines()[0])
    print(f"[deliveries] scenario={args.scenario} seed={args.seed}")
    df = generate(args.scenario, args.seed, args.out_dir)
    C.write_table(df, "deliveries", args.scenario, args.out_dir, args.format)


if __name__ == "__main__":
    main()
