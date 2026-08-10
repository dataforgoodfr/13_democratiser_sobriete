"""Rehome the expert-validated sufficiency singletons into real clusters.

Input: the expert singleton review workbook (2,223 rows). Rows with
`reviewer_verdict == "sufficiency"` carry a `merge_into_cluster` instruction of
the form "code N" — the expert-assigned sub-code, which may differ from the
classifier's. Each such singleton is re-assigned to a cluster inside its
(sector, expert sub-code) cell of the shipped clustering.

Assignment rule — similarity-weighted k-NN vote, consistent with how the
clusters were built (Leiden on a k-NN cosine graph, k=20):
  1. candidates = members of all clusters in the target cell, excluding
     singleton clusters (size 1) so strays can't chain onto other strays.
  2. take the singleton's k=20 nearest candidates by cosine similarity,
     keep those >= FLOOR (0.45 — iteration 5: a Leiden-stranded singleton's
     nearest neighbour sits ~0.52, below the 0.55 graph threshold, and 90%
     have one >= 0.45).
  3. winner = cluster with the largest sum of similarities among the kept
     neighbours. No neighbour >= FLOOR -> unassigned (left as singleton).
  4. cross-check: nearest cluster centroid in the cell; agreement reported.

Usage:
    uv run python rehome_singletons.py \
        --review "2026-06-27_Singletons_Final Assessemnt_YS and team.xlsx" \
        --clustering ../../runs/full_recluster_noambig/reclustered \
        --embeddings ../../runs/full_recluster/filtered.parquet \
        --out ../../runs/singleton_rehoming
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

K = 20
FLOOR = 0.45


def load_keepers(review_path: Path) -> pd.DataFrame:
    df = pd.read_excel(review_path)
    keep = df[df["reviewer_verdict"] == "sufficiency"].copy()
    codes = keep["merge_into_cluster"].astype(str).str.extract(r"code\s*(\d)")[0]
    if codes.isna().any():
        bad = keep.loc[codes.isna(), "merge_into_cluster"].unique()
        raise SystemExit(f"unparseable merge_into_cluster values: {bad}")
    keep["expert_code"] = codes.astype(int)
    keep["policy_uid"] = keep["policy_uid"].astype(str)
    return keep


def load_membership(clustering_dir: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(clustering_dir.glob("*.parquet"))]
    m = pd.concat(frames, ignore_index=True)
    m["policy_uid"] = m["openalex_id"].astype(str) + "::" + m["chunk_idx"].astype(str)
    return m


def collect_embeddings(emb_path: Path, wanted: set[tuple[str, str, str]]) -> dict:
    """Stream filtered.parquet and keep embeddings for the wanted join keys.

    Key = (policy_uid, orig_cluster_uid, policy_text) — policy_uid alone is not
    unique (see RESEARCH_NOTES, iteration 3 data quirk).
    """
    out: dict[tuple, np.ndarray] = {}
    pf = pq.ParquetFile(emb_path)
    cols = ["policy_uid", "orig_cluster_uid", "policy_text", "embedding"]
    for batch in pf.iter_batches(batch_size=32_768, columns=cols):
        uids = batch.column("policy_uid").to_pylist()
        cuids = batch.column("orig_cluster_uid").to_pylist()
        texts = batch.column("policy_text").to_pylist()
        embs = batch.column("embedding")
        for i, key in enumerate(zip(uids, cuids, texts)):
            if key in wanted and key not in out:
                v = np.asarray(embs[i].as_py(), dtype=np.float32)
                out[key] = v / np.linalg.norm(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--clustering", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--floor", type=float, default=FLOOR)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = load_keepers(Path(args.review))
    print(f"{len(keep)} expert-validated sufficiency singletons to rehome")

    m = load_membership(Path(args.clustering))
    sizes = m.groupby("new_cluster_uid")["policy_uid"].size()
    singleton_uids = set(sizes[sizes == 1].index)
    print(f"shipped clustering: {len(m):,} rows, {sizes.size} clusters "
          f"({len(singleton_uids)} singletons excluded as targets)")

    # candidate pool: members of non-singleton clusters in the target cells
    cells = set(zip(keep["sector"], keep["expert_code"]))
    cand = m[
        m[["sector", "sub_code"]].apply(tuple, axis=1).isin(cells)
        & ~m["new_cluster_uid"].isin(singleton_uids)
    ].copy()
    print(f"candidate members across {len(cells)} (sector, code) cells: {len(cand):,}")

    cand_keys = set(zip(cand["policy_uid"], cand["orig_cluster_uid"], cand["policy_text"]))
    single_keys = set(zip(keep["policy_uid"], keep["orig_cluster_uid"], keep["policy_text"]))
    print("streaming embeddings ...")
    emb = collect_embeddings(Path(args.embeddings), cand_keys | single_keys)
    print(f"collected {len(emb):,} embeddings")

    # per-cell candidate matrices
    cell_members: dict[tuple, list[tuple[str, np.ndarray]]] = defaultdict(list)
    missing_cand = 0
    for uid, cuid, text, sector, code, new_uid in zip(
        cand["policy_uid"], cand["orig_cluster_uid"], cand["policy_text"],
        cand["sector"], cand["sub_code"], cand["new_cluster_uid"],
    ):
        v = emb.get((uid, cuid, text))
        if v is None:
            missing_cand += 1
            continue
        cell_members[(sector, int(code))].append((new_uid, v))
    if missing_cand:
        print(f"[warn] {missing_cand} candidate members had no embedding row")

    cell_mat: dict[tuple, tuple[np.ndarray, list[str]]] = {}
    cell_centroids: dict[tuple, tuple[np.ndarray, list[str]]] = {}
    for cell, pairs in cell_members.items():
        labels = [p[0] for p in pairs]
        mat = np.stack([p[1] for p in pairs])
        cell_mat[cell] = (mat, labels)
        cent_lab = sorted(set(labels))
        cents = np.stack([
            mat[[i for i, l in enumerate(labels) if l == cl]].mean(axis=0)
            for cl in cent_lab
        ])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True)
        cell_centroids[cell] = (cents, cent_lab)

    records = []
    for _, r in keep.iterrows():
        key = (r["policy_uid"], r["orig_cluster_uid"], r["policy_text"])
        v = emb.get(key)
        cell = (r["sector"], int(r["expert_code"]))
        rec = {
            "policy_uid": r["policy_uid"],
            "orig_cluster_uid": r["orig_cluster_uid"],
            "singleton_cluster_uid": r["new_cluster_uid"],
            "sector": r["sector"],
            "classifier_code": r["sub_code"],
            "expert_code": int(r["expert_code"]),
            "policy_text": r["policy_text"],
            "assigned_cluster_uid": None,
            "top_sim": np.nan,
            "vote_share": np.nan,
            "assigned_cluster_size": np.nan,
            "centroid_cluster_uid": None,
            "centroid_agrees": None,
            "status": None,
        }
        if v is None:
            rec["status"] = "no_embedding"
        elif cell not in cell_mat:
            rec["status"] = "empty_cell"
        else:
            mat, labels = cell_mat[cell]
            sims = mat @ v
            top = np.argsort(sims)[::-1][: args.k]
            top = [i for i in top if sims[i] >= args.floor]
            if not top:
                rec["status"] = "below_floor"
                rec["top_sim"] = float(sims.max())
            else:
                votes: dict[str, float] = defaultdict(float)
                for i in top:
                    votes[labels[i]] += float(sims[i])
                winner = max(votes, key=votes.get)
                cents, cent_lab = cell_centroids[cell]
                cent_winner = cent_lab[int(np.argmax(cents @ v))]
                rec.update(
                    status="assigned",
                    assigned_cluster_uid=winner,
                    top_sim=float(sims.max()),
                    vote_share=votes[winner] / sum(votes.values()),
                    assigned_cluster_size=int(sizes.loc[winner]),
                    centroid_cluster_uid=cent_winner,
                    centroid_agrees=cent_winner == winner,
                )
        records.append(rec)

    res = pd.DataFrame(records)
    res.to_parquet(out_dir / "rehomed_singletons.parquet", index=False)
    res.drop(columns=["policy_text"]).to_csv(out_dir / "rehomed_singletons.csv", index=False)

    n = len(res)
    assigned = res[res["status"] == "assigned"]
    print(f"\n=== rehoming summary (k={args.k}, floor={args.floor}) ===")
    print(res["status"].value_counts().to_string())
    if len(assigned):
        print(f"\nassigned: {len(assigned)}/{n} ({len(assigned)/n:.0%})")
        print(f"median top similarity: {assigned['top_sim'].median():.3f}")
        print(f"median vote share:     {assigned['vote_share'].median():.2f}")
        print(f"centroid cross-check agreement: {assigned['centroid_agrees'].mean():.0%}")
        print(f"recoded rows (expert != classifier code): "
              f"{(assigned['classifier_code'] != assigned['expert_code']).sum()}")

    # apply: rewrite the shipped per-sector parquets with the 209 moved
    apply_dir = out_dir / "reclustered"
    apply_dir.mkdir(exist_ok=True)
    moves = {
        (r["policy_uid"], r["orig_cluster_uid"], r["policy_text"]):
            (r["assigned_cluster_uid"], r["expert_code"])
        for _, r in assigned.iterrows()
    }
    moved = 0
    for p in sorted(Path(args.clustering).glob("*.parquet")):
        d = pd.read_parquet(p)
        d["policy_uid"] = d["openalex_id"].astype(str) + "::" + d["chunk_idx"].astype(str)
        keys = list(zip(d["policy_uid"], d["orig_cluster_uid"], d["policy_text"]))
        hit = [i for i, k in enumerate(keys) if k in moves]
        for i in hit:
            new_uid, code = moves[keys[i]]
            d.loc[d.index[i], ["new_cluster_uid", "sub_code", "representative"]] = \
                [new_uid, code, False]
            d.loc[d.index[i], "new_cluster_id"] = int(new_uid.split("__L")[-1])
        moved += len(hit)
        d.drop(columns=["policy_uid"]).to_parquet(apply_dir / p.name, index=False)
    print(f"\napplied: {moved} rows moved; amended clustering in {apply_dir}")


if __name__ == "__main__":
    main()
