"""Build the prescreening items JSONL from stage-1 OpenAlex metadata parquets.

Feeds the Gemma prescreener (classify_vllm.py with PROMPTS_VERSION=prescreen)
that replaces the SetFit stage-2 classifier. Reads the stage-1 chunk parquets
(columns: id, title, abstract, language), keeps rows with an abstract, and
writes classify_vllm items: policy_uid = openalex id, policy_text =
title + abstract. Unlike SetFit, the LLM is not English-only, so no language
filter is applied by default (--english-only restores the old behaviour for
comparison runs).

Usage (login node, then queue it):
    python build_prescreen_input.py \\
        --parquet "$ROOT/prescreen/stage1/chunk_*.parquet" \\
        --out "$ROOT/prescreen/items.jsonl"
    python build_queue.py --items "$ROOT/prescreen/items.jsonl" \\
        --queue-dir "$ROOT/prescreen/queue" \\
        --exclude-done-from "$ROOT/prescreen/outputs" --rebuild
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from _carbon import track
from _events import emit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, nargs="+",
                    help="stage-1 metadata parquet path(s) or glob(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--english-only", action="store_true",
                    help="keep only language == 'en' (the old SetFit constraint)")
    args = ap.parse_args()

    paths = sorted(p for pat in args.parquet for p in glob.glob(pat))
    if not paths:
        raise SystemExit(f"no parquet matches {args.parquet}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = kept = 0
    with track("build_prescreen_input"), out.open("w") as fh:
        for path in paths:
            df = pd.read_parquet(path, columns=["id", "title", "abstract", "language"])
            total += len(df)
            df = df[df["abstract"].notna()]
            if args.english_only:
                df = df[df["language"] == "en"]
            for r in df.itertuples(index=False):
                title = (r.title or "").strip()
                fh.write(json.dumps({
                    "policy_uid": str(r.id),
                    "sector": "ABSTRACT",
                    "policy_text": f"Title: {title}\n\nAbstract: {r.abstract.strip()}",
                }) + "\n")
            kept += len(df)
            print(f"  {Path(path).name}: {len(df):,} kept (running total {kept:,})")

    print(f"\nwrote {kept:,} items from {total:,} rows across {len(paths)} parquet(s)")
    emit("prescreen_input_built", items=kept, source_rows=total, files=len(paths))


if __name__ == "__main__":
    main()
