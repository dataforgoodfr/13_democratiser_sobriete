"""Chunk the processed corpus for policy extraction — CPU-only, offline-safe.

Ports the chunking half of `library/ingestion/chunk_embed.ipynb`; the
embedding half of that notebook serves the deferred Qdrant library index and
is NOT needed for the analysis pipeline. Merges the RESULTS and DISCUSSION
sections ("## RESULTS\\n..." / "## DISCUSSION\\n...", as the notebook did)
and splits into token windows (default 1024, overlap 100) with the embedding
model's tokenizer, emitting the historical chunked-conclusions schema:
columns openalex_id, chunk_idx, text.

Window boundaries come from a plain sliding window over token ids (not
langchain's SentenceTransformersTokenTextSplitter, which would load the full
model just to tokenize) — sizes match, exact boundaries may differ; a fresh
corpus gets fresh chunk_idx anyway. Only tokenizer files are needed, so pass
--tokenizer as a local path on Jean Zay
(e.g. $DSDIR/HuggingFace_Models/Qwen/Qwen3-Embedding-0.6B).

Usage (login/prepost node):
    python chunk_corpus.py \\
        --corpus "$ROOT/corpus/processed_texts*.parquet" \\
        --out "$ROOT/chunked_corpus.parquet" \\
        --tokenizer "$DSDIR/HuggingFace_Models/Qwen/Qwen3-Embedding-0.6B"
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from _carbon import track
from _events import emit

_tok = None
_cfg: dict = {}


def merge_sections(results: str | None, discussion: str | None) -> str:
    s = ""
    if results:
        s += "## RESULTS\n" + results
    if s and discussion:
        s += "\n\n"
    if discussion:
        s += "## DISCUSSION\n" + discussion
    return s


def _init_worker(tokenizer_path: str, tokens_per_chunk: int, overlap: int) -> None:
    global _tok, _cfg
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(tokenizer_path)
    _cfg = {"size": tokens_per_chunk, "stride": tokens_per_chunk - overlap}


def _chunk_row(job: tuple[str, str | None, str | None]) -> list[tuple[str, int, str]]:
    doc_id, results, discussion = job
    text = merge_sections(results, discussion)
    if not text:
        return []
    ids = _tok.encode(text, add_special_tokens=False)
    out = []
    for i, start in enumerate(range(0, max(len(ids), 1), _cfg["stride"])):
        window = ids[start : start + _cfg["size"]]
        if not window:
            break
        out.append((doc_id, i, _tok.decode(window)))
        if start + _cfg["size"] >= len(ids):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, nargs="+",
                    help="processed-texts parquet path(s) or glob(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-Embedding-0.6B",
                    help="tokenizer repo id or LOCAL PATH (offline)")
    ap.add_argument("--tokens-per-chunk", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--id-col", default=None,
                    help="column holding the openalex id (default: the index)")
    ap.add_argument("--results-col", default="results")
    ap.add_argument("--discussion-col", default="discussion")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()

    paths = sorted(p for pat in args.corpus for p in glob.glob(pat))
    if not paths:
        raise SystemExit(f"no parquet matches {args.corpus}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    emit("run_start", stage="chunk_corpus", files=len(paths),
         tokens_per_chunk=args.tokens_per_chunk, overlap=args.overlap)

    writer = None
    docs = chunks = 0
    ctx = mp.get_context("spawn")
    with track("chunk_corpus"), ctx.Pool(
        args.workers, initializer=_init_worker,
        initargs=(args.tokenizer, args.tokens_per_chunk, args.overlap),
    ) as pool:
        try:
            for path in paths:
                df = pd.read_parquet(
                    path,
                    columns=[c for c in (args.id_col, args.results_col,
                                         args.discussion_col) if c],
                )
                ids = df[args.id_col] if args.id_col else df.index
                jobs = list(zip(
                    ids.astype(str), df[args.results_col], df[args.discussion_col]
                ))
                rows: list[tuple[str, int, str]] = []
                for res in pool.imap_unordered(_chunk_row, jobs, chunksize=64):
                    rows.extend(res)
                table = pa.Table.from_pandas(
                    pd.DataFrame(rows, columns=["openalex_id", "chunk_idx", "text"]),
                    preserve_index=False,
                )
                if writer is None:
                    writer = pq.ParquetWriter(out, table.schema)
                writer.write_table(table)
                docs += len(jobs)
                chunks += len(rows)
                print(f"  {Path(path).name}: {len(jobs):,} docs → {len(rows):,} chunks "
                      f"(totals {docs:,} / {chunks:,})")
                emit("chunk_progress", docs=docs, chunks=chunks, file=Path(path).name)
        finally:
            if writer:
                writer.close()

    print(f"\nwrote {chunks:,} chunks from {docs:,} docs to {out}")
    emit("run_end", stage="chunk_corpus", docs=docs, chunks=chunks)


if __name__ == "__main__":
    main()
