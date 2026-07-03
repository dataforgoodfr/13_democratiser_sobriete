"""Build intrusion-task items from clustered policies.

Each item: 4 statements from one cluster + 1 statement from a different
cluster in the same sector. We sample uniformly across global clusters and
record the gold intruder index after shuffling.

v0: random intruder. A later iteration will pick the *nearest* different
cluster (needs embeddings).
"""
from __future__ import annotations
import json
import random
from pathlib import Path
import pandas as pd

ITEM_SIZE = 5
N_MEMBERS = 4  # 4 from the cluster + 1 intruder


def _bucket(size: int) -> str:
    if size <= 50:
        return "small"
    if size <= 500:
        return "medium"
    return "large"


def build_intrusion_items(
    df: pd.DataFrame,
    n: int = 50,
    seed: int = 0,
) -> list[dict]:
    """Sample ``n`` intrusion items.

    Requires that every cluster has at least ``N_MEMBERS`` non-medoid items
    so we can take medoid + 4 distinct members. The downloaded data has
    min cluster size 3 in places (e.g. LOGISTICS), so we drop those.
    """
    rng = random.Random(seed)

    cluster_sizes = df.groupby("cluster_uid").size()
    eligible = cluster_sizes[cluster_sizes >= ITEM_SIZE].index.tolist()
    if len(eligible) < n:
        raise ValueError(
            f"only {len(eligible)} clusters meet the >= {ITEM_SIZE}-members "
            f"requirement; can't sample {n} items"
        )

    # Map of cluster_uid -> medoid policy_text (one row per cluster).
    medoid_map = (
        df[df["representative"]]
        .set_index("cluster_uid")["policy_text"]
        .to_dict()
    )

    # Per-sector pool of cluster_uids for cheap intruder draws.
    sectors_to_uids: dict[str, list[str]] = (
        df[["sector", "cluster_uid"]]
        .drop_duplicates()
        .groupby("sector")["cluster_uid"]
        .apply(list)
        .to_dict()
    )

    # Precompute non-medoid texts per cluster for fast member sampling.
    non_medoid = df[~df["representative"]]
    members_by_cluster: dict[str, list[str]] = (
        non_medoid.groupby("cluster_uid")["policy_text"]
        .apply(list)
        .to_dict()
    )

    target_uids = rng.sample(eligible, n)
    items = []
    for uid in target_uids:
        sector = uid.split("__", 1)[0]
        cluster_id = int(uid.split("__", 1)[1])

        # 4 distinct members (we always include the medoid + 3 others to keep
        # the cluster signal anchored on the medoid).
        member_pool = members_by_cluster.get(uid, [])
        if len(member_pool) < N_MEMBERS - 1:
            # very small cluster; fall back to using whatever we have
            sampled_members = list(member_pool)
        else:
            sampled_members = rng.sample(member_pool, N_MEMBERS - 1)
        cluster_statements = [medoid_map[uid]] + sampled_members

        # Intruder: a non-medoid statement from a *different* cluster in
        # the same sector.
        other_uids = [u for u in sectors_to_uids[sector] if u != uid]
        intruder_uid = rng.choice(other_uids)
        intruder_pool = members_by_cluster.get(intruder_uid) or [medoid_map[intruder_uid]]
        intruder_text = rng.choice(intruder_pool)

        # Assemble + shuffle, tracking the intruder index.
        candidates = [
            {"text": t, "source_cluster_uid": uid} for t in cluster_statements
        ] + [{"text": intruder_text, "source_cluster_uid": intruder_uid}]
        rng.shuffle(candidates)
        gold_idx = next(i for i, c in enumerate(candidates) if c["source_cluster_uid"] == intruder_uid)

        cluster_size = int(cluster_sizes.loc[uid])
        items.append({
            "cluster_uid": uid,
            "sector": sector,
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
            "size_bucket": _bucket(cluster_size),
            "items": candidates,
            "gold_intruder_index": gold_idx,
        })
    return items


def save_items(items: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return out_path


if __name__ == "__main__":
    import os
    from .loader import load_clusters
    root = os.path.expanduser("~/data/wsl_sufficiency_eval/clusters_2026-03-18")
    df = load_clusters(root)
    items = build_intrusion_items(df, n=50, seed=0)
    print(f"built {len(items)} items")
    print("first item:")
    print(json.dumps(items[0], indent=2)[:1200])
    print("\nsector coverage:")
    sector_counts: dict[str, int] = {}
    for it in items:
        sector_counts[it["sector"]] = sector_counts.get(it["sector"], 0) + 1
    for s, c in sorted(sector_counts.items()):
        print(f"  {s}: {c}")
    print("\nbucket coverage:")
    bucket_counts: dict[str, int] = {}
    for it in items:
        bucket_counts[it["size_bucket"]] = bucket_counts.get(it["size_bucket"], 0) + 1
    for b, c in sorted(bucket_counts.items()):
        print(f"  {b}: {c}")
