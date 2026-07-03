"""Build intrusion items over the RECLUSTERED per-sector parquets.

Reuses the intrusion-task format from the 100k pipeline test, but sourced
from the new stratified clustering (`runs/api_100k_recluster/reclustered/`).

Sampling:
  - up to N clusters, stratified across size buckets (small/med/large)
  - each item = medoid + 3 non-medoid members + 1 intruder from another
    cluster in the same sector

Output:
    runs/api_100k_recluster/intrusion_items.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

IN_DIR = Path("runs/api_100k_recluster/reclustered")
OUT = Path("runs/api_100k_recluster")
DEFAULT_N = 400
SEED = 51
ITEM_SIZE = 5


def _bucket(n: int) -> str:
    if n <= 20:
        return "small"
    if n <= 100:
        return "medium"
    return "large"


def _load_all() -> pd.DataFrame:
    frames = []
    for f in sorted(IN_DIR.glob("*_reclustered_*.parquet")):
        frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = _load_all()
    print(f"loaded {len(df):,} rows, {df['new_cluster_uid'].nunique()} clusters")

    sizes = df.groupby("new_cluster_uid").size()
    eligible = sizes[sizes >= ITEM_SIZE].index.tolist()
    print(f"eligible clusters (≥{ITEM_SIZE} members): {len(eligible)}")

    rng = random.Random(args.seed)

    # medoids + members
    medoid_by_cluster: dict[str, dict] = {}
    for _, r in df[df["representative"]].iterrows():
        medoid_by_cluster[r["new_cluster_uid"]] = {
            "text": r["policy_text"],
            "sector": r["sector"],
        }
    members_by_cluster: dict[str, list[str]] = (
        df[~df["representative"]]
        .groupby("new_cluster_uid")["policy_text"]
        .apply(list)
        .to_dict()
    )
    sector_pool: dict[str, list[str]] = (
        df[["sector", "new_cluster_uid"]]
        .drop_duplicates()
        .groupby("sector")["new_cluster_uid"]
        .apply(list)
        .to_dict()
    )

    # stratify by size bucket
    by_bucket: dict[str, list[str]] = {"small": [], "medium": [], "large": []}
    for uid in eligible:
        by_bucket[_bucket(int(sizes.loc[uid]))].append(uid)
    per_bucket = args.n // 3
    remainder = args.n - per_bucket * 3
    picked = []
    for bucket, pool in by_bucket.items():
        rng.shuffle(pool)
        take = per_bucket + (remainder if bucket == "large" else 0)
        picked.extend(pool[:take])
    rng.shuffle(picked)
    picked = picked[: args.n]

    items = []
    for uid in picked:
        sector = medoid_by_cluster[uid]["sector"]
        cluster_size = int(sizes.loc[uid])

        member_pool = members_by_cluster.get(uid, [])
        needed = ITEM_SIZE - 1 - 1
        if len(member_pool) < needed:
            continue
        cluster_stmts = [medoid_by_cluster[uid]["text"]] + rng.sample(member_pool, needed)

        other_uids = [u for u in sector_pool[sector] if u != uid]
        if not other_uids:
            continue
        intruder_uid = rng.choice(other_uids)
        intruder_pool = members_by_cluster.get(intruder_uid) or [medoid_by_cluster[intruder_uid]["text"]]
        intruder_text = rng.choice(intruder_pool)

        candidates = (
            [{"text": t, "source_cluster_uid": uid} for t in cluster_stmts]
            + [{"text": intruder_text, "source_cluster_uid": intruder_uid}]
        )
        rng.shuffle(candidates)
        gold_idx = next(i for i, c in enumerate(candidates) if c["source_cluster_uid"] == intruder_uid)

        items.append({
            "cluster_uid": uid,
            "sector": sector,
            "cluster_id": int(uid.split("__L")[-1]),
            "cluster_size": cluster_size,
            "size_bucket": _bucket(cluster_size),
            "items": candidates,
            "gold_intruder_index": gold_idx,
        })

    with (OUT / "intrusion_items.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    print(f"wrote {len(items)} intrusion items")
    print("bucket coverage:", {b: sum(1 for it in items if it["size_bucket"] == b) for b in by_bucket})
    per_sector: dict[str, int] = {}
    for it in items:
        per_sector[it["sector"]] = per_sector.get(it["sector"], 0) + 1
    print("sector coverage:")
    for s, c in sorted(per_sector.items()):
        print(f"  {s:15s} {c}")


if __name__ == "__main__":
    main()
