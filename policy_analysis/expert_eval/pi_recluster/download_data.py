"""Download the two data artefacts needed for re-clustering, from HuggingFace.

1. The clustered-policies parquets (~92 MB total, 11 sector files, gives us
   openalex_id, chunk_idx, policy_text, cluster_id, representative per row).
2. The Qwen3-4B embeddings parquet (7.15 GB, one row per source policy with an
   `embedding` column plus openalex_id, chunk_idx, sector, policy_text).

The two are joined on (openalex_id, chunk_idx, policy_text) — text disambiguates
the multi-policy-per-chunk case.

Usage:
    uv run python download_data.py --dest data/hf
    uv run python download_data.py --dest data/hf --skip-embeddings   # only clusters
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from _carbon import track

REPO_ID = "sufficiencylab/sufficiency-library"
CLUSTERS_DATE = "2026-03-18"
EMBEDDINGS_FILE = "embeddings_policies_Qwen3-4B_2026-03-05.parquet"

SECTORS = [
    "BUILDING", "ENERGY", "FOOD", "INDUSTRY", "LOGISTICS",
    "MACROECONOMIC", "MATERIALS", "MOBILITY", "NATURE", "SOCIAL", "URBAN",
]


def download_clusters(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"clusters_{CLUSTERS_DATE}"
    out.mkdir(exist_ok=True)
    print(f"→ downloading {len(SECTORS)} cluster parquets to {out}")
    for sector in SECTORS:
        fname = f"{sector}_clustered_policies_with_representatives_{CLUSTERS_DATE}.parquet"
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=fname,
            local_dir=out,
        )
    return out


def download_embeddings(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"→ downloading embeddings ({EMBEDDINGS_FILE}, ~7.15 GB) to {dest}")
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=EMBEDDINGS_FILE,
        local_dir=dest,
    )
    return Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/hf", help="output root")
    ap.add_argument("--skip-embeddings", action="store_true",
                    help="skip the 7.15 GB embeddings download")
    ap.add_argument("--skip-clusters", action="store_true",
                    help="skip the cluster parquet download")
    args = ap.parse_args()

    # Enable hf_transfer for parallel/fast downloads
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    dest = Path(args.dest)

    with track("download"):
        if not args.skip_clusters:
            clusters_root = download_clusters(dest)
            print(f"clusters: {clusters_root}")
        if not args.skip_embeddings:
            embed_path = download_embeddings(dest)
            print(f"embeddings: {embed_path}")


if __name__ == "__main__":
    main()
