"""Build extraction items JSONL from a chunked-corpus parquet.

Bridges chunk_corpus.py (or a historical chunked_conclusions parquet, where
openalex_id may live in the index) to extract_vllm.py / build_queue.py.
Items carry policy_uid = openalex_id::chunk_idx so build_queue's
done-detection works unchanged.

Usage:
    python build_extract_input.py --chunks "$ROOT/chunked_corpus.parquet" \\
        --out "$ROOT/extract/items.jsonl"
    python build_queue.py --items "$ROOT/extract/items.jsonl" \\
        --queue-dir "$ROOT/extract/queue" \\
        --exclude-done-from "$ROOT/extract/outputs" --rebuild
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
    ap.add_argument("--chunks", required=True, help="chunked-corpus parquet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.chunks)
    if "openalex_id" not in df.columns:  # historical files index by openalex_id
        df = df.reset_index()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with track("build_extract_input"), out.open("w") as fh:
        for r in df[["openalex_id", "chunk_idx", "text"]].itertuples(index=False):
            fh.write(json.dumps({
                "policy_uid": f"{r.openalex_id}::{int(r.chunk_idx)}",
                "openalex_id": str(r.openalex_id),
                "chunk_idx": int(r.chunk_idx),
                "text": r.text,
            }) + "\n")

    print(f"wrote {len(df):,} items to {out}")
    emit("extract_input_built", items=len(df))


if __name__ == "__main__":
    main()
