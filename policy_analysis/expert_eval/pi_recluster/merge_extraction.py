"""Merge extract_vllm outputs into the policies parquet for embedding/clustering.

Explodes each chunk's `policies` list into one row per policy text (columns:
openalex_id, chunk_idx, policy_uid, policy_text). Error records are counted
and skipped — rebuild the queue with --exclude-done-from to retry them.
Queue outputs are disjoint by construction, so no cross-unit dedup is needed;
exact duplicate rows (same chunk, same policy string) are dropped defensively.

Next stage: python compute_embeddings.py --input policies.parquet \\
    --text-column policy_text ...

Usage:
    python merge_extraction.py --outputs "$ROOT/extract/outputs" \\
        --out "$ROOT/policies.parquet"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _carbon import track
from _events import emit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, help="extract_vllm output directory")
    ap.add_argument("--out", required=True, help="policies parquet")
    args = ap.parse_args()

    rows = []
    errors = chunks = with_policies = 0
    with track("merge_extraction"):
        for f in sorted(Path(args.outputs).glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                chunks += 1
                if "error" in rec:
                    errors += 1
                    continue
                if rec.get("contains_policies") and rec.get("policies"):
                    with_policies += 1
                    for p in rec["policies"]:
                        p = p.strip()
                        if p:
                            rows.append((rec["openalex_id"], rec["chunk_idx"],
                                         rec["policy_uid"], p))

        df = pd.DataFrame(
            rows, columns=["openalex_id", "chunk_idx", "policy_uid", "policy_text"]
        ).drop_duplicates()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)

    print(f"{chunks:,} chunk records ({errors:,} errors, {with_policies:,} with "
          f"policies) → {len(df):,} policy rows in {out}")
    emit("extraction_merged", chunk_records=chunks, errors=errors,
         chunks_with_policies=with_policies, policy_rows=len(df))


if __name__ == "__main__":
    main()
