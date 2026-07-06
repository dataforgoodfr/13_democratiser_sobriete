"""Filter classified policies to sufficiency + ambiguous, then join with embeddings.

Inputs:
    --classifications  runs/api_100k/classifications.jsonl  (or full 1.47M)
    --clusters         data/hf/clusters_2026-03-18/         (per-sector parquets)
    --embeddings       data/hf/embeddings_policies_Qwen3-4B_2026-03-05.parquet

Output:
    data/filtered.parquet  — one row per surviving policy with columns:
        policy_uid, sector, orig_cluster_uid, sub_code, category,
        policy_text, embedding (numpy array)

Design notes:
- The embeddings file is 7.15 GB. We load it via pyarrow.dataset to avoid
  materialising the full frame, and filter by (openalex_id, chunk_idx,
  policy_text) — the only tuple that uniquely identifies a policy row.
- `sub_code` is threaded through as an Int64 (nullable): only sufficiency
  rows have it; ambiguous rows keep it null.
- Ambiguous rows are kept because they are "clearly a policy, mechanism
  unclear" — we still want them re-clustered so a human can review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from _carbon import track


KEEP_CATEGORIES = {"sufficiency", "ambiguous"}

SECTORS = [
    "BUILDING", "ENERGY", "FOOD", "INDUSTRY", "LOGISTICS",
    "MACROECONOMIC", "MATERIALS", "MOBILITY", "NATURE", "SOCIAL", "URBAN",
]
CLUSTERS_DATE = "2026-03-18"


def _load_classifications(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        if "error" in df.columns:
            df = df[df["error"].isna()]
        return df
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error"):
            continue
        rows.append(rec)
    return pd.DataFrame(rows)


def _load_clusters(clusters_dir: Path) -> pd.DataFrame:
    frames = []
    for sector in SECTORS:
        path = clusters_dir / f"{sector}_clustered_policies_with_representatives_{CLUSTERS_DATE}.parquet"
        df = pd.read_parquet(path)
        df["sector"] = sector
        df["policy_uid"] = df["openalex_id"].astype(str) + "::" + df["chunk_idx"].astype(int).astype(str)
        df["orig_cluster_uid"] = sector + "__" + df["cluster_id"].astype(int).astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifications", required=True)
    ap.add_argument("--clusters", required=True, help="dir with per-sector cluster parquets")
    ap.add_argument("--embeddings", required=True, help="embeddings parquet path")
    ap.add_argument("--out", required=True, help="output filtered.parquet")
    args = ap.parse_args()

    tracker_cm = track("filter_and_stratify")
    tracker_cm.__enter__()
    classifications = _load_classifications(Path(args.classifications))
    print(f"loaded {len(classifications):,} classifications")

    kept = classifications[classifications["category"].isin(KEEP_CATEGORIES)].copy()
    print(f"  kept (sufficiency + ambiguous): {len(kept):,} "
          f"({len(kept)/len(classifications):.1%})")

    # Rename to be joined with cluster metadata by (policy_uid, cluster_uid, seq)
    kept["_seq"] = kept.groupby(["policy_uid", "cluster_uid"]).cumcount()
    kept = kept.rename(columns={"cluster_uid": "orig_cluster_uid"})

    clusters = _load_clusters(Path(args.clusters))
    print(f"loaded {len(clusters):,} cluster rows across {clusters['sector'].nunique()} sectors")
    clusters["_seq"] = clusters.groupby(["policy_uid", "orig_cluster_uid"]).cumcount()

    meta = kept.merge(
        clusters[["policy_uid", "orig_cluster_uid", "_seq", "policy_text",
                  "openalex_id", "chunk_idx"]],
        on=["policy_uid", "orig_cluster_uid", "_seq"],
        how="inner",
    )
    print(f"joined classifier + cluster metadata: {len(meta):,}")

    # Now pull embeddings only for the surviving (openalex_id, chunk_idx,
    # policy_text) tuples. Use a filter set for pyarrow.
    print(f"streaming embeddings from {args.embeddings}")
    dset = ds.dataset(args.embeddings, format="parquet")
    schema_names = dset.schema.names
    print(f"embeddings schema: {schema_names}")

    wanted_keys = set(zip(meta["openalex_id"], meta["chunk_idx"].astype(int)))
    print(f"looking up {len(wanted_keys):,} unique (openalex_id, chunk_idx) tuples")

    kept_chunks = []
    scanned = 0
    for batch in dset.to_batches(batch_size=65536):
        b = batch.to_pandas()
        scanned += len(b)
        if "chunk_idx" in b.columns:
            b["chunk_idx"] = b["chunk_idx"].astype(int)
        mask = list(zip(b["openalex_id"], b["chunk_idx"]))
        keep_mask = np.array([k in wanted_keys for k in mask])
        if keep_mask.any():
            kept_chunks.append(b.loc[keep_mask])
        if scanned % (65536 * 4) == 0:
            print(f"  scanned {scanned:,} embedding rows; kept-so-far {sum(len(c) for c in kept_chunks):,}")

    embed_df = pd.concat(kept_chunks, ignore_index=True)
    print(f"kept {len(embed_df):,} embedding rows from {scanned:,} scanned")

    # Join on (openalex_id, chunk_idx, policy_text). Embeddings and meta may both
    # have multiple rows per chunk with distinct policy_text; text disambiguates.
    if "policy_text" in embed_df.columns:
        merged = meta.merge(
            embed_df[["openalex_id", "chunk_idx", "policy_text", "embedding"]],
            on=["openalex_id", "chunk_idx", "policy_text"],
            how="inner",
        )
    else:
        # fallback: no text column in embeddings → join on chunk key only
        merged = meta.merge(
            embed_df[["openalex_id", "chunk_idx", "embedding"]],
            on=["openalex_id", "chunk_idx"],
            how="inner",
        )
    print(f"final joined + embedded: {len(merged):,}")

    keep_cols = [
        "policy_uid", "sector", "orig_cluster_uid",
        "category", "sub_code", "policy_text",
        "openalex_id", "chunk_idx",
        "embedding",
    ]
    merged["sub_code"] = merged["sub_code"].astype("Int64")
    merged[keep_cols].to_parquet(args.out, index=False)
    print(f"wrote {args.out}")

    # summary stats
    print("\nBy sector:")
    print(merged.groupby("sector").size().sort_values(ascending=False).to_string())
    print("\nSufficiency sub-code distribution (Ambiguous has null):")
    sub_dist = merged["sub_code"].dropna().astype(int).value_counts().sort_index()
    print(sub_dist.to_string())

    tracker_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
