"""Intrinsic clustering metrics that do not require embeddings.

For v0 we only have cluster_id and policy_text (see DATA.md). Embedding-based
metrics (silhouette, medoid coherence) are deferred. We report cluster-size
statistics per sector and overall.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def _size_stats(sizes: pd.Series) -> dict:
    return {
        "n_clusters": int(len(sizes)),
        "n_items": int(sizes.sum()),
        "min": int(sizes.min()),
        "median": float(sizes.median()),
        "p90": float(sizes.quantile(0.90)),
        "p99": float(sizes.quantile(0.99)),
        "max": int(sizes.max()),
        "mean": float(sizes.mean()),
        "largest_cluster_fraction": float(sizes.max() / sizes.sum()),
        "small_cluster_count_le_5": int((sizes <= 5).sum()),
    }


def compute_intrinsic(df: pd.DataFrame) -> dict:
    overall_sizes = df.groupby("cluster_uid").size()
    per_sector = {}
    for sector, sub in df.groupby("sector"):
        sizes = sub.groupby("cluster_id").size()
        per_sector[sector] = _size_stats(sizes)
    return {
        "overall": _size_stats(overall_sizes),
        "per_sector": per_sector,
    }


def save_intrinsic(metrics: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics_intrinsic.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


if __name__ == "__main__":
    import os
    from .loader import load_clusters
    root = os.path.expanduser("~/data/wsl_sufficiency_eval/clusters_2026-03-18")
    df = load_clusters(root)
    metrics = compute_intrinsic(df)
    print(json.dumps(metrics["overall"], indent=2))
    print("---")
    for sector, m in metrics["per_sector"].items():
        print(f"{sector}: n={m['n_clusters']} median={m['median']:.0f} max={m['max']} "
              f"largest_frac={m['largest_cluster_fraction']:.3f}")
