"""Generate every table, in dependency order, for one or all scenarios.

Order: suppliers -> parts -> production_schedule -> deliveries -> master_panel

Usage:
    python generate_all.py                      # all three scenarios
    python generate_all.py --scenario drift     # one scenario
    python generate_all.py --seed 7 --format csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import build_master_panel
import config as C
import generate_deliveries
import generate_parts
import generate_production_schedule
import generate_suppliers


def run(scenario: str, seed: int, root: Path, fmt: str) -> None:
    print(f"=== scenario: {scenario} (seed {seed}) ===")
    C.write_table(generate_suppliers.generate(scenario, seed), "suppliers", scenario, root, fmt)
    for name, module in [("parts", generate_parts),
                         ("production_schedule", generate_production_schedule),
                         ("deliveries", generate_deliveries)]:
        C.write_table(module.generate(scenario, seed, root), name, scenario, root, fmt)
    C.write_table(build_master_panel.build(scenario, root), "master_panel", scenario, root, fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=(*C.SCENARIOS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=C.BASE_SEED)
    parser.add_argument("--out-dir", type=Path, default=C.DEFAULT_ROOT)
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    args = parser.parse_args()

    scenarios = C.SCENARIOS if args.scenario == "all" else (args.scenario,)
    for scenario in scenarios:
        run(scenario, args.seed, args.out_dir, args.format)


if __name__ == "__main__":
    main()
