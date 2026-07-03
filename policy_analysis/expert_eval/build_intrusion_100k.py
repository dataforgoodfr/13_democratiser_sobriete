"""Build intrusion items for the 100k pipeline run.

Selects up to N clusters (default 500) from clusters_2026-03-18 that have at
least ITEM_SIZE members, stratified by size bucket so the judge sees small,
medium, and large clusters. Reuses the intrusion builder from
``policy_analysis.clustering.eval.sample``.

Output:
    runs/api_100k/intrusion_items.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from policy_analysis.clustering.eval.loader import load_clusters
from policy_analysis.clustering.eval.sample import build_intrusion_items

DATA_ROOT = os.path.expanduser("~/data/wsl_sufficiency_eval/clusters_2026-03-18")
OUT = Path("runs/api_100k")
DEFAULT_N = 500
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_clusters(DATA_ROOT)

    # Oversample and stratify: request as many as we can, then downsample per bucket
    max_pool = min(args.n * 3, 900)
    items = build_intrusion_items(df, n=max_pool, seed=args.seed)
    rng = random.Random(args.seed)
    by_bucket: dict[str, list[dict]] = {"small": [], "medium": [], "large": []}
    for it in items:
        by_bucket[it["size_bucket"]].append(it)

    per_bucket = args.n // 3
    remainder = args.n - per_bucket * 3
    picked = []
    for bucket, pool in by_bucket.items():
        rng.shuffle(pool)
        target = per_bucket + (remainder if bucket == "large" else 0)
        picked.extend(pool[:target])

    rng.shuffle(picked)
    picked = picked[: args.n]

    with (OUT / "intrusion_items.jsonl").open("w") as f:
        for it in picked:
            f.write(json.dumps(it) + "\n")

    print(f"wrote {len(picked)} intrusion items")
    bucket_counts = {b: sum(1 for it in picked if it["size_bucket"] == b) for b in by_bucket}
    print(f"bucket coverage: {bucket_counts}")
    sector_counts: dict[str, int] = {}
    for it in picked:
        sector_counts[it["sector"]] = sector_counts.get(it["sector"], 0) + 1
    print("sector coverage:")
    for s, c in sorted(sector_counts.items()):
        print(f"  {s:15s}  {c}")


if __name__ == "__main__":
    main()
