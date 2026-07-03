"""Stratified sampler for the expert-annotation dataset.

Per sector, draw ~50 policies stratified by cluster size:
  - small  (size <= 50):      10 items
  - medium (51 <= size <= 500): 20 items
  - large  (size > 500):       20 items

If a sector lacks enough material in a bucket, the deficit redistributes
to the other buckets (keeping 50/sector).

Within each bucket, we sample distinct cluster IDs and take TWO rows per
cluster: the medoid (representative=True) + one random non-medoid member.
This keeps medoid and periphery represented while capping the per-cluster
footprint at 2, so stratifying by *current* clusters does not leak the
current cluster assignment into the future ground truth.

The annotator groups rows by topical similarity from the policy text alone.
`source_cluster_id` is included in the CSV for reproducibility but the
README instructs the annotator to ignore it.

Outputs:
  policy_analysis/clustering/eval/annotation/to_annotate/{SECTOR}.csv
  policy_analysis/clustering/eval/annotation/to_annotate/all_sectors.xlsx
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from policy_analysis.clustering.eval.loader import SECTORS, load_clusters

DEFAULT_DATA_ROOT = "~/data/wsl_sufficiency_eval/clusters_2026-03-18"
N_PER_SECTOR = 50
N_EXTRA_PER_SECTOR = 20
BUCKET_ALLOC = {"small": 10, "medium": 20, "large": 20}
SMALL_MAX = 50
MEDIUM_MAX = 500
SEED = 0
ITEMS_PER_CLUSTER = 2

COLUMNS = [
    "policy_id",
    "sector",
    "source_cluster_id",
    "within_cluster_role",
    "sample_kind",
    "in_sector",
    "policy_text",
    "group_id",
]


def _bucket(size: int) -> str:
    if size <= SMALL_MAX:
        return "small"
    if size <= MEDIUM_MAX:
        return "medium"
    return "large"


def _cluster_sizes(sector_df: pd.DataFrame) -> pd.DataFrame:
    """Per cluster_id: size, bucket, medoid_idx, member_idx (list of non-medoid row indices)."""
    rows = []
    for cid, g in sector_df.groupby("cluster_id", sort=False):
        size = len(g)
        medoids = g.index[g["representative"]].tolist()
        members = g.index[~g["representative"]].tolist()
        if not medoids:
            continue
        rows.append(
            {
                "cluster_id": cid,
                "size": size,
                "bucket": _bucket(size),
                "medoid_idx": medoids[0],
                "member_idx": members,
            }
        )
    return pd.DataFrame(rows)


def _allocate(buckets_available: dict[str, int]) -> dict[str, int]:
    """Allocate N_PER_SECTOR across buckets, redistributing deficits."""
    target = dict(BUCKET_ALLOC)
    taken = {b: min(target[b], buckets_available[b]) for b in target}
    deficit = N_PER_SECTOR - sum(taken.values())
    if deficit <= 0:
        return taken
    surplus_buckets = [
        b for b in target if buckets_available[b] - taken[b] > 0
    ]
    while deficit > 0 and surplus_buckets:
        for b in surplus_buckets:
            if deficit == 0:
                break
            if buckets_available[b] - taken[b] > 0:
                taken[b] += 1
                deficit -= 1
        surplus_buckets = [
            b for b in target if buckets_available[b] - taken[b] > 0
        ]
    return taken


def _sample_sector(
    sector_df: pd.DataFrame, sector: str, rng: np.random.Generator
) -> pd.DataFrame:
    clusters = _cluster_sizes(sector_df)
    if clusters.empty:
        return pd.DataFrame(columns=COLUMNS)

    by_bucket = {b: clusters[clusters["bucket"] == b] for b in BUCKET_ALLOC}
    # Items-available per bucket = 2 per cluster (medoid + 1 member), capped
    # at the cluster size.
    avail = {}
    for b, df in by_bucket.items():
        capacity = df["size"].clip(upper=ITEMS_PER_CLUSTER).sum()
        avail[b] = int(capacity)

    allocation = _allocate(avail)

    picked: list[dict] = []
    for bucket, n_items in allocation.items():
        if n_items == 0:
            continue
        candidates = by_bucket[bucket]
        if candidates.empty:
            continue
        # Number of clusters needed = ceil(n_items / 2) when picking 2 per cluster.
        n_clusters_needed = (n_items + ITEMS_PER_CLUSTER - 1) // ITEMS_PER_CLUSTER
        n_clusters_needed = min(n_clusters_needed, len(candidates))
        cluster_choices = candidates.sample(
            n=n_clusters_needed, random_state=int(rng.integers(0, 2**31 - 1))
        )

        remaining = n_items
        for _, c in cluster_choices.iterrows():
            if remaining == 0:
                break
            members = list(c["member_idx"])
            take_medoid = True
            take_member = len(members) > 0 and remaining >= 2

            if take_medoid:
                picked.append(
                    {
                        "_row_idx": int(c["medoid_idx"]),
                        "cluster_id": int(c["cluster_id"]),
                        "role": "medoid",
                    }
                )
                remaining -= 1
            if remaining == 0:
                break
            if take_member:
                mi = int(rng.choice(members))
                picked.append(
                    {
                        "_row_idx": mi,
                        "cluster_id": int(c["cluster_id"]),
                        "role": "periphery",
                    }
                )
                remaining -= 1

        # If we still owe rows (e.g. tiny clusters of size 1 in 'small'),
        # backfill from additional clusters in the same bucket.
        if remaining > 0:
            leftover = candidates.drop(cluster_choices.index)
            for _, c in leftover.sample(
                n=min(remaining, len(leftover)),
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).iterrows():
                if remaining == 0:
                    break
                picked.append(
                    {
                        "_row_idx": int(c["medoid_idx"]),
                        "cluster_id": int(c["cluster_id"]),
                        "role": "medoid",
                    }
                )
                remaining -= 1

    pick_df = pd.DataFrame(picked)
    pick_df["kind"] = "stratified"

    # Add N_EXTRA_PER_SECTOR random rows drawn uniformly from the sector,
    # excluding rows already picked. These are intended for catching
    # wrong-sector classifications, so we want broad random coverage.
    already = set(pick_df["_row_idx"].astype(int).tolist())
    remaining_idx = [i for i in sector_df.index.tolist() if int(i) not in already]
    n_extra = min(N_EXTRA_PER_SECTOR, len(remaining_idx))
    extra_idx = rng.choice(remaining_idx, size=n_extra, replace=False)
    extra_rows = []
    for ri in extra_idx:
        src = sector_df.loc[int(ri)]
        role = "medoid" if bool(src["representative"]) else "periphery"
        extra_rows.append(
            {
                "_row_idx": int(ri),
                "cluster_id": int(src["cluster_id"]),
                "role": role,
                "kind": "random",
            }
        )
    pick_df = pd.concat([pick_df, pd.DataFrame(extra_rows)], ignore_index=True)

    # Shuffle so cluster ID order does not leak the grouping, and stratified
    # vs. random rows are mixed together.
    pick_df = pick_df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(
        drop=True
    )

    out_rows = []
    for i, row in pick_df.iterrows():
        src = sector_df.loc[row["_row_idx"]]
        out_rows.append(
            {
                "policy_id": f"{sector}_{i + 1:03d}",
                "sector": sector,
                "source_cluster_id": int(row["cluster_id"]),
                "within_cluster_role": row["role"],
                "sample_kind": row["kind"],
                "in_sector": "",
                "policy_text": str(src["policy_text"]),
                "group_id": "",
            }
        )
    return pd.DataFrame(out_rows, columns=COLUMNS)


def build_dataset(
    data_root: str | Path,
    out_dir: str | Path,
    seed: int = SEED,
) -> tuple[list[Path], Path]:
    data_root = Path(os.path.expanduser(str(data_root)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_clusters(data_root)
    rng = np.random.default_rng(seed)

    csv_paths: list[Path] = []
    per_sector_frames: dict[str, pd.DataFrame] = {}
    for sector in SECTORS:
        sdf = df[df["sector"] == sector].reset_index(drop=False).rename(
            columns={"index": "_orig_index"}
        )
        # _sample_sector indexes by original DataFrame index → restore it.
        sdf.index = sdf["_orig_index"]
        sample = _sample_sector(sdf, sector, rng)
        out_path = out_dir / f"{sector}.csv"
        sample.to_csv(out_path, index=False)
        csv_paths.append(out_path)
        per_sector_frames[sector] = sample
        print(f"{sector:14s}  {len(sample):3d} rows -> {out_path}")

    xlsx_path = out_dir / "all_sectors.xlsx"
    instructions = pd.DataFrame(
        {
            "field": [
                "purpose",
                "task 1 — relevant",
                "task 2 — in_sector",
                "task 3 — group_id",
                "label scheme",
                "group size",
                "ignore",
                "unsure",
                "ordering",
                "sample_kind",
            ],
            "value": [
                "Build a ground-truth dataset to evaluate policy clustering.",
                (
                    "Zero pass: fill `relevant` with `yes` if the text is actually"
                    " a policy worth analysing, or `no` if it is junk / not a"
                    " policy / unintelligible. Skip `no` rows for tasks 2 and 3."
                ),
                (
                    "Among `relevant=yes` rows, fill `in_sector` with `y` if the"
                    " policy genuinely belongs to this sector, or `n` if it was"
                    " mis-routed here. Skip the row for task 3 if `n`."
                ),
                (
                    "Among `relevant=yes` and `in_sector=y` rows, fill `group_id`."
                    " Put policies describing the SAME topic in the SAME group."
                ),
                (
                    "Use any label scheme (numbers, short tags, whatever) — only"
                    " equality between rows matters."
                ),
                "Aim for groups of 2–8 items. Singletons are fine.",
                (
                    "Do NOT look at `source_cluster_id` — it is the current"
                    " (possibly wrong) clustering; we want your independent judgement."
                ),
                (
                    "If a policy does not fit any group, give it a unique label or"
                    " write `unsure`."
                ),
                (
                    "The rows are shuffled deliberately. Re-order them freely in"
                    " your spreadsheet."
                ),
                (
                    "`stratified` = drawn by cluster-size bucket (for grouping)."
                    " `random` = drawn uniformly (mainly to catch wrong-sector"
                    " items). Treat both the same when annotating."
                ),
            ],
        }
    )
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="_instructions", index=False)
        for sector in SECTORS:
            sheet_df = per_sector_frames[sector].copy()
            # Excel-only convenience column for marking junk / non-policy rows.
            sheet_df.insert(sheet_df.columns.get_loc("in_sector"), "relevant", "")
            sheet_df.to_excel(writer, sheet_name=sector, index=False)
    print(f"\nconsolidated workbook -> {xlsx_path}")
    return csv_paths, xlsx_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "to_annotate"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build_dataset(args.data, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
