"""Generate the `parts` table: one row per part used at a plant.

Primary key: (part_code, plant)
Foreign key: supplier_code -> suppliers.supplier_code

Each part has a single supplier and is used at 1-3 plants. Usage and safety
stock are set per plant, because the same part can be consumed at different
rates in different plants.

Columns:
    part_code, plant, supplier_code, part_category, box_qty,
    usage_per_hour       pieces consumed per production hour (held constant)
    safety_inv_boxes     in-house safety stock target, in boxes (some missing)
    n_plants_using       number of plants that use this part
    avail_in_other_plants  True if another plant also stocks this part

Usage:
    python generate_parts.py --scenario clusters
    (requires suppliers table for the same scenario)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def generate(scenario: str, seed: int = C.BASE_SEED, root=C.DEFAULT_ROOT) -> pd.DataFrame:
    rng = C.get_rng("parts", "shared", seed)
    suppliers = C.read_table("suppliers", scenario, root)

    # 1-6 parts per supplier (mean ~2.6)
    n_parts = 1 + np.minimum(rng.poisson(1.6, len(suppliers)), 5)
    part_supplier = np.repeat(suppliers["supplier_code"].to_numpy(), n_parts)
    part_category = np.repeat(suppliers["commodity"].to_numpy(), n_parts)
    n_total = len(part_supplier)
    part_codes = np.array([f"P{10000 + i}" for i in range(1, n_total + 1)])
    box_qty = rng.choice([10, 20, 50, 100, 200], size=n_total, p=[0.15, 0.25, 0.3, 0.2, 0.1])

    # Plants using each part
    n_plants = rng.choice([1, 2, 3], size=n_total, p=[0.55, 0.30, 0.15])
    rows = []
    for i in range(n_total):
        for plant in rng.choice(C.PLANTS, size=n_plants[i], replace=False):
            rows.append((part_codes[i], plant, part_supplier[i], part_category[i], box_qty[i], n_plants[i]))
    df = pd.DataFrame(rows, columns=["part_code", "plant", "supplier_code", "part_category",
                                     "box_qty", "n_plants_using"])

    # Plant-specific consumption and safety stock
    n = len(df)
    df["usage_per_hour"] = np.maximum(1, np.round(rng.lognormal(np.log(40), 0.7, n))).astype(int)
    safety_hours = rng.uniform(2, 8, n)          # hours of production covered by safety stock
    df["safety_inv_boxes"] = np.ceil(df["usage_per_hour"] * safety_hours / df["box_qty"]).astype(float)
    df["avail_in_other_plants"] = df["n_plants_using"] > 1

    df = C.mask_missing(df, "parts", "safety_inv_boxes", rng)
    df["safety_inv_boxes"] = df["safety_inv_boxes"].astype("Int64")
    df = df.sort_values(["part_code", "plant"]).reset_index(drop=True)
    return df[["part_code", "plant", "supplier_code", "part_category", "box_qty",
               "usage_per_hour", "safety_inv_boxes", "n_plants_using", "avail_in_other_plants"]]


def main() -> None:
    args = C.parse_args(__doc__.splitlines()[0])
    print(f"[parts] scenario={args.scenario} seed={args.seed}")
    df = generate(args.scenario, args.seed, args.out_dir)
    C.write_table(df, "parts", args.scenario, args.out_dir, args.format)


if __name__ == "__main__":
    main()
