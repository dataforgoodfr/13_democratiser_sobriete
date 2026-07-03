"""Build the eval_v1 intrusion set + calibration split.

Outputs under runs/eval_v1/:
  intrusion_items_500.jsonl                — 500 items, gold included
  calibration/items_gold.jsonl             — 5 per sector, gold included
  calibration/items_blind.jsonl            — same items, gold stripped
"""
from __future__ import annotations

import os
from pathlib import Path

from policy_analysis.clustering.eval.calibration import (
    split_calibration,
    write_blind,
    write_gold,
)
from policy_analysis.clustering.eval.loader import load_clusters
from policy_analysis.clustering.eval.sample import build_intrusion_items, save_items

DATA_ROOT = "~/data/wsl_sufficiency_eval/clusters_2026-03-18"
OUT_DIR = Path("runs/eval_v1")
N_ITEMS = 500
PER_SECTOR_CAL = 5
SEED = 1


def main() -> None:
    data_root = os.path.expanduser(DATA_ROOT)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_clusters(data_root)
    items = build_intrusion_items(df, n=N_ITEMS, seed=SEED)
    save_items(items, out_dir / "intrusion_items_500.jsonl")
    print(f"wrote {len(items)} intrusion items")

    cal, _ = split_calibration(items, per_sector=PER_SECTOR_CAL, seed=SEED)
    cal_dir = out_dir / "calibration"
    write_gold(cal, cal_dir / "items_gold.jsonl")
    write_blind(cal, cal_dir / "items_blind.jsonl")

    by_sector: dict[str, int] = {}
    for it in cal:
        by_sector[it["sector"]] = by_sector.get(it["sector"], 0) + 1
    print(f"\ncalibration set: {len(cal)} items across {len(by_sector)} sectors")
    for s, c in sorted(by_sector.items()):
        print(f"  {s:14s} {c}")


if __name__ == "__main__":
    main()
