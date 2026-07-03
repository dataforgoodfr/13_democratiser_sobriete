"""Build the full 1.47M `to_classify.jsonl` from the sector cluster parquets.

Each row identifies a distinct policy text via (openalex_id, chunk_idx,
policy_text). We construct `policy_uid = openalex_id::chunk_idx` and also
keep the cluster_uid for the aggregation join later.

Usage:
    uv run python build_classify_input.py \
        --clusters data/hf/clusters_2026-03-18 \
        --out data/to_classify_full.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _carbon import track

SECTORS = [
    "BUILDING", "ENERGY", "FOOD", "INDUSTRY", "LOGISTICS",
    "MACROECONOMIC", "MATERIALS", "MOBILITY", "NATURE", "SOCIAL", "URBAN",
]
DATE = "2026-03-18"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True, help="dir with per-sector parquets")
    ap.add_argument("--out", required=True, help="output JSONL")
    args = ap.parse_args()

    clusters = Path(args.clusters)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with track("build_classify_input"), out.open("w") as fh:
        for sector in SECTORS:
            path = clusters / f"{sector}_clustered_policies_with_representatives_{DATE}.parquet"
            df = pd.read_parquet(path)
            df["sector"] = sector
            df["policy_uid"] = df["openalex_id"].astype(str) + "::" + df["chunk_idx"].astype(int).astype(str)
            df["cluster_uid"] = sector + "__" + df["cluster_id"].astype(int).astype(str)
            for r in df[["policy_uid", "sector", "cluster_id", "cluster_uid", "policy_text"]].itertuples(index=False):
                fh.write(json.dumps({
                    "policy_uid": r.policy_uid,
                    "sector": r.sector,
                    "cluster_id": int(r.cluster_id),
                    "cluster_uid": r.cluster_uid,
                    "policy_text": r.policy_text,
                }) + "\n")
            total += len(df)
            print(f"  {sector:15s}  {len(df):>7,} rows (total {total:,})")

    print(f"\nwrote {total:,} rows to {out}")


if __name__ == "__main__":
    main()
