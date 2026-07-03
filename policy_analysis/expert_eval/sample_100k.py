"""Sample 100k policies stratified across sectors from clusters_2026-03-18.

Strategy: proportional-with-floor. Each sector gets at least ``MIN_PER_SECTOR``
rows; the rest is proportional to full-sector size. This keeps small sectors
(LOGISTICS ~9k) visible while giving big sectors (SOCIAL ~438k) proper weight.

Output:
    runs/api_100k/to_classify.jsonl
        {"policy_uid", "sector", "cluster_id", "cluster_uid", "policy_text"}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = os.path.expanduser("~/data/wsl_sufficiency_eval/clusters_2026-03-18")
OUT = Path("runs/api_100k")
TARGET = 100_000
MIN_PER_SECTOR = 3_000
SEED = 11


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in os.listdir(DATA_ROOT) if f.endswith(".parquet"))

    sizes = {}
    dfs = {}
    for f in files:
        sector = f.split("_")[0]
        df = pd.read_parquet(os.path.join(DATA_ROOT, f))
        dfs[sector] = df
        sizes[sector] = len(df)

    total = sum(sizes.values())
    n_sectors = len(sizes)
    floor_budget = MIN_PER_SECTOR * n_sectors
    remaining = TARGET - floor_budget

    quotas: dict[str, int] = {}
    for sector, size in sizes.items():
        proportional_extra = int(round(remaining * size / total))
        quotas[sector] = MIN_PER_SECTOR + proportional_extra
    drift = TARGET - sum(quotas.values())
    if drift != 0:
        big = max(quotas, key=lambda k: sizes[k])
        quotas[big] += drift

    rng = np.random.default_rng(SEED)
    picked = []
    for sector, quota in quotas.items():
        df = dfs[sector]
        n = min(quota, len(df))
        idx = rng.choice(len(df), size=n, replace=False)
        sub = df.iloc[idx].copy()
        sub["sector"] = sector
        picked.append(sub)

    out = pd.concat(picked, ignore_index=True)
    out["policy_uid"] = (
        out["openalex_id"].astype(str) + "::" + out["chunk_idx"].astype(int).astype(str)
    )
    out["cluster_uid"] = out["sector"] + "__" + out["cluster_id"].astype(int).astype(str)

    with (OUT / "to_classify.jsonl").open("w") as fh:
        for r in out[["policy_uid", "sector", "cluster_id", "cluster_uid", "policy_text"]].to_dict(orient="records"):
            fh.write(json.dumps(r) + "\n")

    print(f"wrote {len(out):,} rows to {OUT/'to_classify.jsonl'}")
    print("\nquota per sector (drawn / requested / full-sector size):")
    for sector in sizes:
        drawn = int((out["sector"] == sector).sum())
        print(f"  {sector:15s}  {drawn:>7,} / {quotas[sector]:>7,} / {sizes[sector]:>8,}")


if __name__ == "__main__":
    main()
