"""Join the four tables into one master panel for modelling (one row per delivery).

Join keys:
    deliveries --supplier_code--------------------------> suppliers
    deliveries --(part_code, plant)---------------------> parts
    deliveries --(part_code, plant, production_date)----> production_schedule

Every join is many-to-one and checked with pandas' `validate`, so the panel has
exactly one row per delivery. Columns prefixed `true_` are the simulator's
latent truth: keep them for evaluation, never use them as features.

Note: this panel is the raw join only. Leakage-safe features (e.g. a supplier's
trailing 30-day on-time rate) should be computed later, inside each training
window, from information available before each delivery.

Usage:
    python build_master_panel.py --scenario clusters
"""
from __future__ import annotations

import pandas as pd

import config as C


def build(scenario: str, root=C.DEFAULT_ROOT) -> pd.DataFrame:
    deliveries = C.read_table("deliveries", scenario, root)
    suppliers = C.read_table("suppliers", scenario, root)
    parts = C.read_table("parts", scenario, root)
    schedule = C.read_table("production_schedule", scenario, root)

    # Primary keys must be unique before joining
    for name, df, key in [("suppliers", suppliers, ["supplier_code"]),
                          ("parts", parts, ["part_code", "plant"]),
                          ("production_schedule", schedule, ["part_code", "plant", "production_date"]),
                          ("deliveries", deliveries, ["delivery_id"])]:
        assert not df.duplicated(key).any(), f"duplicate keys in {name}: {key}"

    panel = (deliveries
             .merge(suppliers, on="supplier_code", how="left", validate="many_to_one")
             .merge(parts.drop(columns="supplier_code"), on=["part_code", "plant"],
                    how="left", validate="many_to_one")
             .merge(schedule, on=["part_code", "plant", "production_date"],
                    how="left", validate="many_to_one"))
    assert len(panel) == len(deliveries), "join changed the row count"

    truth = [c for c in panel.columns if c.startswith("true_")]
    return panel[[c for c in panel.columns if c not in truth] + truth]


def main() -> None:
    args = C.parse_args(__doc__.splitlines()[0])
    print(f"[master_panel] scenario={args.scenario}")
    C.write_table(build(args.scenario, args.out_dir), "master_panel", args.scenario, args.out_dir, args.format)


if __name__ == "__main__":
    main()
