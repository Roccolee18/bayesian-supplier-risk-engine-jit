"""Shared settings and helpers for the supplier-risk data simulator.

Every generator script imports from here so that all four tables use the same
plants, calendar, scenarios and random-number seeding.

All data produced by these scripts is fully simulated. Parameters are chosen to
be plausible for a generic just-in-time assembly plant and are not based on any
real company's operations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
BASE_SEED = 20261027

SCENARIOS = ("clusters", "no_clusters", "drift")
_TABLE_IDS = {"suppliers": 11, "parts": 12, "production_schedule": 13, "deliveries": 14}
_STREAM_IDS = {"shared": 0, "clustered": 1, "no_clusters": 2, "drift": 3}


def get_rng(table: str, stream: str = "shared", seed: int = BASE_SEED) -> np.random.Generator:
    """Independent, reproducible RNG stream per (seed, table, stream).

    Seeding each table separately means regenerating one table never changes
    the draws used by another.

    The "shared" stream is identical across scenarios, so every scenario has the
    same suppliers, parts, calendar, truck schedule and weather/disruption
    events (common random numbers). Only the latent supplier reliability
    differs, so differences between scenarios come from the reliability
    structure alone:
        "clustered"    cluster assignment + offsets, shared by clusters and drift
        "no_clusters"  continuous reliability draws
        "drift"        which suppliers switch, and to which cluster
    """
    return np.random.default_rng([seed, _TABLE_IDS[table], _STREAM_IDS[stream]])


# --------------------------------------------------------------------------- #
# Simulation scope
# --------------------------------------------------------------------------- #
START_DATE = pd.Timestamp("2025-01-06")          # a Monday
N_MONTHS = 12
END_DATE = START_DATE + pd.DateOffset(months=N_MONTHS)   # exclusive

PLANTS = ["PLANT_A", "PLANT_B", "PLANT_C"]
PLANT_BASE_HOURS = {"PLANT_A": 16.0, "PLANT_B": 16.0, "PLANT_C": 20.0}  # weekday production hours
DELIVERY_WINDOW_START_HOUR = 6     # receiving dock opens 06:00
DELIVERY_WINDOW_LENGTH_HOURS = 16  # ... and closes 22:00

N_SUPPLIERS = 366

# Supplier regions: distance range (km), share of suppliers, possible trucks per
# day to each plant, and whether winter weather affects the lane.
REGIONS = {
    "ON_LOCAL":    dict(km=(20, 150),    weight=0.30, trips=(2, 3, 4), winter_sensitive=False),
    "ON_REGIONAL": dict(km=(150, 450),   weight=0.25, trips=(1, 2),    winter_sensitive=False),
    "QC":          dict(km=(450, 900),   weight=0.10, trips=(1, 2),    winter_sensitive=True),
    "US_MIDWEST":  dict(km=(400, 1100),  weight=0.20, trips=(1,),      winter_sensitive=True),
    "US_SOUTH":    dict(km=(1100, 2200), weight=0.10, trips=(1,),      winter_sensitive=True),
    "MEXICO":      dict(km=(2800, 3500), weight=0.05, trips=(1,),      winter_sensitive=True),
}
FAR_REGIONS = ("US_MIDWEST", "US_SOUTH", "MEXICO")

COMMODITIES = ("stamping", "electrical", "interior_trim", "powertrain", "fasteners", "plastics")

# --------------------------------------------------------------------------- #
# Delay process: zero-inflated log-normal
#   late ~ Bernoulli(p_late)
#   delay_hours | late ~ LogNormal(mu, sigma)   (mu, sigma on the log-hours scale)
#   delay_hours = 0 when on time
# --------------------------------------------------------------------------- #
CLUSTERS = {
    #                    P(late)   median delay when late = exp(mu) hours
    "stable":            dict(p_late=0.03, mu=0.7, sigma=0.5),   # ~2.0 h
    "consistently_late": dict(p_late=0.22, mu=1.2, sigma=0.4),   # ~3.3 h, tight spread
    "high_volatility":   dict(p_late=0.10, mu=1.6, sigma=1.1),   # ~5.0 h, long tail
    "chronic_problem":   dict(p_late=0.35, mu=2.3, sigma=0.8),   # ~10 h
}
CLUSTER_NAMES = tuple(CLUSTERS)
CLUSTER_WEIGHTS = np.array([0.55, 0.20, 0.15, 0.10])

# Supplier-level deviation from its cluster centre (logit p, mu, log sigma)
SUPPLIER_NOISE_SD = np.array([0.30, 0.15, 0.10])

# Drift scenario: (months after START_DATE, share of all suppliers switching)
DRIFT_SCHEDULE = ((3, 0.20), (7, 0.10))

MAX_DELAY_HOURS = 96.0
WINTER_MONTHS = (12, 1, 2)
WINTER_P_MULT = {True: 1.4, False: 1.15}   # keyed by region winter_sensitive
DISRUPTIONS_PER_REGION = 2.0               # expected events per region per year
DISRUPTION_P_MULT = 3.0
DISRUPTION_MU_SHIFT = 0.5

# --------------------------------------------------------------------------- #
# Missing data (share of rows set to missing, to mimic real system gaps)
# --------------------------------------------------------------------------- #
MISSING_RATES = {
    ("suppliers", "distance_km"): 0.03,                        # address not geocoded
    ("parts", "safety_inv_boxes"): 0.02,                       # safety stock not maintained in system
    ("production_schedule", "planned_usage_boxes"): 0.003,     # schedule load failures
    ("deliveries", "actual_delivery_datetime"): 0.005,         # receipt not scanned
}


def mask_missing(df: pd.DataFrame, table: str, column: str, rng: np.random.Generator) -> pd.DataFrame:
    """Set a random share of `column` to missing, using MISSING_RATES."""
    rate = MISSING_RATES[(table, column)]
    mask = rng.random(len(df)) < rate
    df.loc[mask, column] = pd.NaT if pd.api.types.is_datetime64_any_dtype(df[column]) else np.nan
    return df


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


def scenario_dir(scenario: str, root: Path | str = DEFAULT_ROOT) -> Path:
    path = Path(root) / scenario
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_table(df: pd.DataFrame, name: str, scenario: str,
                root: Path | str = DEFAULT_ROOT, fmt: str = "parquet") -> Path:
    path = scenario_dir(scenario, root) / f"{name}.{fmt}"
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"  wrote {path} ({len(df):,} rows)")
    return path


def read_table(name: str, scenario: str, root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    folder = Path(root) / scenario
    parquet, csv = folder / f"{name}.parquet", folder / f"{name}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        date_cols = [c for c in ("onboard_date", "production_date",
                                 "planned_delivery_datetime", "actual_delivery_datetime")]
        df = pd.read_csv(csv)
        for col in date_cols:
            if col in df:
                df[col] = pd.to_datetime(df[col])
        return df
    raise FileNotFoundError(f"{name} not found in {folder}. Run its generator script first.")


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--scenario", choices=SCENARIOS, default="clusters",
                        help="Which data-generating scenario to simulate.")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT,
                        help="Root folder; tables go in <out-dir>/<scenario>/.")
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    return parser.parse_args()


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))
