"""Generate the `production_schedule` table: one row per part, plant and day.

Primary key: (part_code, plant, production_date)
Foreign key: (part_code, plant) -> parts

Usage per hour is constant (a README assumption), so day-to-day demand varies
only through planned production hours:
    - weekdays run each plant's base hours (16 h, or 20 h at PLANT_C)
    - ~15% of Saturdays have 8 h of overtime; Sundays are off
    - all plants shut down over the winter holidays
    - each plant takes a 2-week summer shutdown in July
    - ~2% of weekdays have reduced hours (unplanned downtime)

Columns:
    part_code, plant, production_date, planned_hours,
    planned_usage_pieces, planned_usage_boxes (some missing)

Usage:
    python generate_production_schedule.py --scenario clusters
    (requires the parts table for the same scenario)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def plant_calendar(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(C.START_DATE, C.END_DATE - pd.Timedelta(days=1), freq="D")
    frames = []
    for plant in C.PLANTS:
        dow = dates.dayofweek.to_numpy()
        hours = np.where(dow < 5, C.PLANT_BASE_HOURS[plant], 0.0)
        saturday_ot = (dow == 5) & (rng.random(len(dates)) < 0.15)
        hours = np.where(saturday_ot, 8.0, hours)

        # Winter holiday shutdown: Dec 22 - Jan 2
        winter = ((dates.month == 12) & (dates.day >= 22)) | ((dates.month == 1) & (dates.day <= 2))
        hours = np.where(winter, 0.0, hours)

        # Summer shutdown: two weeks starting on one of the first two Mondays of July
        july_mondays = dates[(dates.month == 7) & (dates.dayofweek == 0)]
        start = july_mondays[rng.integers(0, 2)]
        summer = (dates >= start) & (dates < start + pd.Timedelta(days=14))
        hours = np.where(summer, 0.0, hours)

        # Unplanned downtime: reduced hours on ~2% of working days
        downtime = (hours > 0) & (rng.random(len(dates)) < 0.02)
        hours = np.where(downtime, np.round(hours * rng.uniform(0.25, 0.75, len(dates)) * 2) / 2, hours)

        frames.append(pd.DataFrame({"plant": plant, "production_date": dates, "planned_hours": hours}))
    return pd.concat(frames, ignore_index=True)


def generate(scenario: str, seed: int = C.BASE_SEED, root=C.DEFAULT_ROOT) -> pd.DataFrame:
    rng = C.get_rng("production_schedule", "shared", seed)
    parts = C.read_table("parts", scenario, root)
    calendar = plant_calendar(rng)

    df = parts[["part_code", "plant", "usage_per_hour", "box_qty"]].merge(calendar, on="plant", how="inner")
    df["planned_usage_pieces"] = (df["usage_per_hour"] * df["planned_hours"]).round().astype(int)
    df["planned_usage_boxes"] = (df["planned_usage_pieces"] / df["box_qty"]).round(2)
    df = C.mask_missing(df, "production_schedule", "planned_usage_boxes", rng)

    df = df.sort_values(["plant", "production_date", "part_code"]).reset_index(drop=True)
    return df[["part_code", "plant", "production_date", "planned_hours",
               "planned_usage_pieces", "planned_usage_boxes"]]


def main() -> None:
    args = C.parse_args(__doc__.splitlines()[0])
    print(f"[production_schedule] scenario={args.scenario} seed={args.seed}")
    df = generate(args.scenario, args.seed, args.out_dir)
    C.write_table(df, "production_schedule", args.scenario, args.out_dir, args.format)


if __name__ == "__main__":
    main()
