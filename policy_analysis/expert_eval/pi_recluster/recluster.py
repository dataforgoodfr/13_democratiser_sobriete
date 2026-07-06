"""Stratified Leiden re-clustering per (sector, sub_code) cell.

Reads:
    data/filtered.parquet  (from filter_and_stratify.py)

Writes:
    data/reclustered/<SECTOR>_reclustered_2026-06-13.parquet  (one per sector)
        policy_uid, sector, orig_cluster_uid, category, sub_code, policy_text,
        new_cluster_id (sector-local), new_cluster_uid, representative

Design:
- Cells are (sector, sub_code). Ambiguous rows (sub_code null) form their own
  cell per sector, coded `sub_code = -1` for stratification.
- Cell size drives strategy:
    n < MIN_LEIDEN         → single cluster (label with sub_code)
    n < MEDIUM             → coarse Leiden (resolution 0.4)
    n ≥ MEDIUM             → default Leiden (resolution 1.0)
- k-NN graph: FAISS HNSW over L2-normalised vectors → cosine-similar top-k.
  Edges below COS_THRESHOLD are dropped so Leiden doesn't glue everything.
- Medoid ("representative") = highest cosine sum to other cluster members.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from pathlib import Path

import faiss
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd

from _carbon import track

MIN_LEIDEN = 30           # below this: single cluster
MEDIUM = 300              # below this: coarse Leiden
COARSE_RESOLUTION = 0.4
# Iteration 5: lowered from 1.0 → 0.5. At 1.0, RBConfiguration ejected fringe
# nodes from dense communities to gain modularity, producing 2,095 size-1
# clusters in the ≥300 cells (66% of all clusters were singletons). 0.5 keeps
# communities coarser; the singleton-absorption pass below mops up the rest.
DEFAULT_RESOLUTION = 0.5
K_NEIGHBOURS = 20
COS_THRESHOLD = 0.55      # edge threshold for building the Leiden graph
RANDOM_SEED = 42
# Absorption uses a LOWER floor than the graph threshold on purpose. A node that
# Leiden stranded as a singleton typically has its nearest neighbour just below
# 0.55 (median ~0.52 in LOGISTICS c09; 90% are ≥0.45). Rehoming such a stray to
# its nearest cluster is better than leaving it alone; only nodes with no
# neighbour ≥ this floor stay singletons (genuinely unique policies).
ABSORB_THRESHOLD = 0.45
# Below this cell size use an exact inner-product index instead of HNSW: HNSW's
# approximation drops a few true neighbours, and on the fringe that difference is
# exactly what strands a node as a singleton. Exact is cheap at these sizes.
EXACT_KNN_MAX = 5000
# After Leiden, reassign any size-1 cluster to the nearest cluster medoid in the
# same cell when cosine ≥ COS_THRESHOLD. Genuinely isolated policies (no peer
# above threshold) stay singletons, which is correct.
ABSORB_SINGLETONS = True


def _normalize(embs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(embs, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (embs / n).astype("float32")


def _knn_graph(embs: np.ndarray, k: int, exact: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, similarities) for the top-k+1 nearest neighbours."""
    n, d = embs.shape
    if exact:
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
    index.add(embs)
    sims, idx = index.search(embs, min(k + 1, n))   # +1 because the top hit is self
    return idx, sims


def _leiden(embs: np.ndarray, resolution: float) -> np.ndarray:
    n = embs.shape[0]
    idx, sims = _knn_graph(embs, K_NEIGHBOURS, exact=(n <= EXACT_KNN_MAX))
    edges = []
    weights = []
    for i in range(n):
        for j_i, s in zip(idx[i], sims[i]):
            j = int(j_i)
            if j == i or j < 0:
                continue
            if s < COS_THRESHOLD:
                continue
            edges.append((i, j))
            weights.append(float(s))
    if not edges:
        return np.zeros(n, dtype=np.int64)
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=RANDOM_SEED,
    )
    return np.array(partition.membership, dtype=np.int64)


def _absorb_singletons(embs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Merge each size-1 cluster into the cluster of its nearest neighbour.

    For every singleton point we take its single nearest neighbour among ALL
    points (any cluster); if cosine ≥ ABSORB_THRESHOLD we union the two clusters.
    This catches both singleton→big-cluster and singleton↔singleton cases (two
    near-duplicate policies that each became their own cluster). Genuinely
    isolated policies — no neighbour above threshold — keep their own cluster.
    `embs` must be L2-normalised so inner product == cosine.
    """
    labels = labels.copy()
    counts = Counter(labels.tolist())
    singletons = [c for c, n in counts.items() if n == 1]
    if not singletons:
        return labels

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    single_idx = [int(np.where(labels == c)[0][0]) for c in singletons]
    sims, nbr = index.search(embs[single_idx], 2)   # col 0 is self

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for k, c in enumerate(singletons):
        if float(sims[k, 1]) >= ABSORB_THRESHOLD:
            union(int(labels[nbr[k, 1]]), int(c))   # c joins the neighbour's cluster

    if not parent:
        return labels
    return np.array([find(int(l)) for l in labels], dtype=labels.dtype)


def _pick_medoids(embs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """For each cluster, medoid = arg-max Σ cosine to fellow members."""
    n = len(labels)
    picks = np.zeros(n, dtype=bool)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 1:
            picks[idx[0]] = True
            continue
        sub = embs[idx]
        sims = sub @ sub.T
        scores = sims.sum(axis=1)
        picks[idx[int(np.argmax(scores))]] = True
    return picks


def _handle_cell(cell: pd.DataFrame, cell_key: str) -> pd.DataFrame:
    n = len(cell)
    if n < MIN_LEIDEN:
        strategy = "single"
        labels = np.zeros(n, dtype=np.int64)
    else:
        embs = _normalize(np.stack(cell["embedding"].values))
        if n < MEDIUM:
            strategy = f"coarse@{COARSE_RESOLUTION}"
            labels = _leiden(embs, COARSE_RESOLUTION)
        else:
            strategy = f"default@{DEFAULT_RESOLUTION}"
            labels = _leiden(embs, DEFAULT_RESOLUTION)
        n_pre = len(set(labels))
        if ABSORB_SINGLETONS:
            labels = _absorb_singletons(embs, labels)

    cell = cell.copy()
    cell["_cell_key"] = cell_key
    cell["_local_label"] = labels
    if n >= MIN_LEIDEN:
        cell["representative"] = _pick_medoids(embs, labels)
    else:
        # single-cluster fallback: mark first row as representative
        rep = np.zeros(n, dtype=bool)
        rep[0] = True
        cell["representative"] = rep

    if n >= MIN_LEIDEN:
        absorbed = n_pre - len(set(labels))
        print(f"  [{cell_key}] n={n:>6,}  strategy={strategy}  "
              f"clusters={len(set(labels))} (absorbed {absorbed} singletons)")
    else:
        print(f"  [{cell_key}] n={n:>6,}  strategy={strategy}  clusters=1")
    return cell


def _cell_key(sector: str, sub_code) -> str:
    code = -1 if pd.isna(sub_code) else int(sub_code)
    return f"{sector}__c{code:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker_cm = track("recluster")
    tracker_cm.__enter__()
    df = pd.read_parquet(args.filtered)
    print(f"loaded {len(df):,} filtered rows across {df['sector'].nunique()} sectors")
    df["_sub_code_int"] = df["sub_code"].fillna(-1).astype(int)

    # group by (sector, sub_code)
    all_results = []
    t0 = time.time()
    grouped = list(df.groupby(["sector", "_sub_code_int"], sort=True))
    print(f"grouping into {len(grouped)} cells")
    for (sector, code), cell in grouped:
        key = _cell_key(sector, code if code != -1 else np.nan)
        result = _handle_cell(cell, key)
        all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True)

    # Build sector-local new_cluster_id: within each sector, number cells+labels
    # sequentially. new_cluster_uid = f"{sector}__c{code}__L{label}".
    combined["new_cluster_uid"] = (
        combined["_cell_key"] + "__L" + combined["_local_label"].astype(str)
    )

    # sector-local new_cluster_id (0..N-1 per sector)
    per_sector = combined.groupby("sector", sort=True)["new_cluster_uid"].apply(
        lambda s: pd.Series(pd.Categorical(s).codes.astype(int), index=s.index)
    )
    combined["new_cluster_id"] = per_sector.values

    n_new = combined.groupby("sector")["new_cluster_uid"].nunique()
    print("\nNew cluster count per sector:")
    for s, k in n_new.items():
        print(f"  {s:15s}  {k}")

    # emit per-sector parquets, matching original schema
    keep = ["openalex_id", "chunk_idx", "policy_text", "sector",
            "orig_cluster_uid", "category", "sub_code",
            "new_cluster_id", "new_cluster_uid", "representative"]
    for sector, group in combined.groupby("sector"):
        out_path = out_dir / f"{sector}_reclustered_2026-06-13.parquet"
        group[keep].to_parquet(out_path, index=False)
        print(f"  wrote {out_path} with {len(group):,} rows, {group['new_cluster_uid'].nunique()} clusters")

    dt = time.time() - t0
    print(f"\ndone in {dt/60:.1f} min")

    tracker_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
