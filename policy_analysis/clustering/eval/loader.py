"""Load per-sector clustering outputs from the downloaded HF dataset.

Schema is locked in DATA.md. Sector is implicit in the file name; we inject it
as a column. Cluster IDs are sector-local, so we also build a globally-unique
`cluster_uid = f"{sector}__{cluster_id}"`.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

SECTORS = [
    "BUILDING", "ENERGY", "FOOD", "INDUSTRY", "LOGISTICS",
    "MACROECONOMIC", "MATERIALS", "MOBILITY", "NATURE", "SOCIAL", "URBAN",
]
DATE = "2026-03-18"


def _parquet_path(root: Path, sector: str) -> Path:
    return root / f"{sector}_clustered_policies_with_representatives_{DATE}.parquet"


def load_clusters(root: str | Path, sectors: list[str] | None = None) -> pd.DataFrame:
    """Concatenate per-sector parquets into one DataFrame.

    Returns columns: openalex_id, chunk_idx, policy_text, cluster_id (sector-local),
    representative, sector, cluster_uid (globally unique).
    """
    root = Path(root)
    sectors = sectors or SECTORS
    frames = []
    for sector in sectors:
        path = _parquet_path(root, sector)
        if not path.exists():
            raise FileNotFoundError(f"missing parquet for sector {sector}: {path}")
        df = pd.read_parquet(path)
        df["sector"] = sector
        df["cluster_uid"] = sector + "__" + df["cluster_id"].astype(str)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


if __name__ == "__main__":
    import os
    root = os.path.expanduser("~/data/wsl_sufficiency_eval/clusters_2026-03-18")
    df = load_clusters(root)
    print(f"rows: {len(df):,}")
    print(f"sectors: {df['sector'].nunique()} -> {sorted(df['sector'].unique())}")
    print(f"clusters (global): {df['cluster_uid'].nunique():,}")
    print(f"representatives: {int(df['representative'].sum()):,}")
    print("per-sector cluster count:")
    print(df.groupby("sector")["cluster_id"].nunique().sort_values(ascending=False).to_string())
