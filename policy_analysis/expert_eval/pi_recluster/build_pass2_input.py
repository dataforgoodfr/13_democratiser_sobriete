"""Build the pass-2 classifier input from the clustered corpus.

Pass 2 re-plays classification AFTER clustering, over exactly the rows that
made it into the shipped clustering (645k rows, vs 1.47M in pass 1), with the
tightened prompt (PROMPTS_VERSION=pass2). Downstream, a policy whose pass-2
category is no longer `sufficiency` is dropped from its cluster; a changed
sub_code flags the cluster for cell reassignment. This is the systemic
replacement for hand-patching the expert-flagged clusters.

Emits the same item schema as build_classify_input.py so classify_vllm.py is
reused unchanged:
    {"policy_uid", "sector", "policy_text", "cluster_uid"}
where cluster_uid is the NEW (stratified) cluster uid, so verdicts aggregate
per shipped cluster.

Usage:
    uv run python build_pass2_input.py \
        --clustering data/reclustered \
        --out data/to_classify_pass2.jsonl
    PROMPTS_VERSION=pass2 uv run --extra gpu python classify_vllm.py \
        --items data/to_classify_pass2.jsonl \
        --out data/classifications_pass2.jsonl \
        --few-shot gold/few_shot_v1.jsonl \
        --single-turn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clustering", required=True,
                    help="directory of per-sector reclustered parquets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cluster-size", type=int, default=2,
                    help="skip clusters smaller than this (singletons have "
                         "their own expert-review path)")
    args = ap.parse_args()

    frames = [pd.read_parquet(p) for p in sorted(Path(args.clustering).glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    sizes = df.groupby("new_cluster_uid")["policy_text"].transform("size")
    kept = df[sizes >= args.min_cluster_size]
    print(f"{len(df):,} clustered rows -> {len(kept):,} in clusters "
          f">= {args.min_cluster_size}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in kept.itertuples(index=False):
            fh.write(json.dumps({
                "policy_uid": f"{r.openalex_id}::{r.chunk_idx}",
                "sector": r.sector,
                "policy_text": r.policy_text,
                "cluster_uid": r.new_cluster_uid,
            }) + "\n")
    print(f"wrote {len(kept):,} items to {out}")


if __name__ == "__main__":
    main()
