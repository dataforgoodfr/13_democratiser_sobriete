"""Sample policies for the API smoke test of the sufficiency classifier.

Outputs under runs/sufficiency_api_test/:
  production.jsonl     — 200 policies stratified by sector (for the run)
  calibration.jsonl    — 30 policies stratified by sector (for Claude to label
                         blind; 9 will be promoted to few-shot examples, 21
                         held out as gold)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from policy_analysis.clustering.eval.loader import SECTORS, load_clusters

DEFAULT_DATA = "~/data/wsl_sufficiency_eval/clusters_2026-03-18"
DEFAULT_OUT = "runs/sufficiency_api_test"
N_PROD = 200
N_CAL = 30
SEED = 1


def _sample_stratified(df, n_total, rng, exclude_uids=set()):
    per_sector = max(1, n_total // len(SECTORS))
    out = []
    leftover = n_total - per_sector * len(SECTORS)
    for sector in SECTORS:
        pool = df[(df["sector"] == sector) & (~df["policy_uid"].isin(exclude_uids))]
        if pool.empty:
            continue
        take = per_sector + (1 if leftover > 0 else 0)
        if leftover > 0:
            leftover -= 1
        take = min(take, len(pool))
        out.append(pool.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    import pandas as pd
    return pd.concat(out).sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-prod", type=int, default=N_PROD)
    ap.add_argument("--n-cal", type=int, default=N_CAL)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    df = load_clusters(os.path.expanduser(args.data))
    df["policy_uid"] = df["openalex_id"].astype(str) + "::" + df["chunk_idx"].astype(str)
    rng = np.random.default_rng(args.seed)

    cal = _sample_stratified(df, args.n_cal, rng)
    prod = _sample_stratified(df, args.n_prod, rng, exclude_uids=set(cal["policy_uid"]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("calibration", cal), ("production", prod)]:
        path = out / f"{name}.jsonl"
        with path.open("w") as f:
            for _, r in frame.iterrows():
                f.write(json.dumps({
                    "policy_uid": r["policy_uid"],
                    "sector": r["sector"],
                    "cluster_uid": r["cluster_uid"],
                    "policy_text": str(r["policy_text"]),
                }) + "\n")
        print(f"wrote {len(frame):3d} -> {path}")


if __name__ == "__main__":
    main()
