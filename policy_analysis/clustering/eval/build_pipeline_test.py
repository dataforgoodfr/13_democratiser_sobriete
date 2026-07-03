"""Whole-loop test on 200 already-produced clusters.

For each of 200 randomly-picked clusters:
  - build an intrusion item (medoid + 3 members + 1 intruder from another cluster)
  - sample up to 5 members for the sufficiency classifier

Also builds the 4-example judge few-shot file by picking cases where Claude
and DeepSeek both agreed with the manufactured gold in the prior 55-item
calibration run.

Outputs under runs/pipeline_test/:
  intrusion_items.jsonl   — 200 items for the judge
  cluster_members.jsonl   — ~1000 policies for the sufficiency classifier
  judge_few_shot.jsonl    — 4 examples to inject into the judge prompt
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

from policy_analysis.clustering.eval.loader import load_clusters
from policy_analysis.clustering.eval.sample import build_intrusion_items

DATA_ROOT = "~/data/wsl_sufficiency_eval/clusters_2026-03-18"
OUT = Path("runs/pipeline_test")
N_ITEMS = 200
SEED = 7  # different from eval_v1 so we hit clusters we haven't judged before
N_MEMBERS_FOR_CLASSIFIER = 5


def build_few_shot():
    """Pick 4 judge-calibration examples where Claude + DeepSeek + gold agreed.

    Uses the prior 55-item calibration in runs/eval_v1/calibration/.
    """
    base = Path("runs/eval_v1/calibration")
    gold = {x["cluster_uid"]: x for x in (json.loads(l) for l in (base / "items_gold.jsonl").read_text().splitlines() if l.strip())}
    claude = {x["cluster_uid"]: x for x in (json.loads(l) for l in (base / "claude_labels.jsonl").read_text().splitlines() if l.strip())}
    deep = {x["cluster_uid"]: x for x in (json.loads(l) for l in (base / "judgments.jsonl").read_text().splitlines() if l.strip())}

    candidates = []
    for uid, item in gold.items():
        c = claude.get(uid)
        d = deep.get(uid)
        if not c or not d:
            continue
        if c["intruder_index"] == item["gold_intruder_index"] == d["intruder_index"]:
            candidates.append({
                "cluster_uid": uid,
                "sector": item["sector"],
                "statements": [c["text"] for c in item["items"]],
                "intruder_index": item["gold_intruder_index"],
                "topic_label": c.get("topic_label", "(unspecified)"),
                "confidence": c.get("confidence", 5),
            })

    rng = random.Random(0)
    rng.shuffle(candidates)
    picked = []
    seen_sectors = set()
    for c in candidates:
        if c["sector"] in seen_sectors:
            continue
        if c["confidence"] < 4:
            continue
        picked.append(c)
        seen_sectors.add(c["sector"])
        if len(picked) == 4:
            break
    return picked


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_clusters(os.path.expanduser(DATA_ROOT))

    # 1) intrusion items
    eval_v1_uids = set()
    eval_v1_path = Path("runs/eval_v1/intrusion_items_500.jsonl")
    if eval_v1_path.exists():
        eval_v1_uids = {json.loads(l)["cluster_uid"] for l in eval_v1_path.read_text().splitlines() if l.strip()}

    # Build many items and keep the first 200 with cluster_uids NOT in eval_v1.
    items_pool = build_intrusion_items(df, n=min(N_ITEMS * 4, 900), seed=SEED)
    fresh = [it for it in items_pool if it["cluster_uid"] not in eval_v1_uids][:N_ITEMS]
    if len(fresh) < N_ITEMS:
        # Fall back to whatever we got
        fresh = items_pool[:N_ITEMS]
    with (OUT / "intrusion_items.jsonl").open("w") as f:
        for it in fresh:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(fresh)} intrusion items (avoiding eval_v1 overlap: {len(items_pool) - len(fresh)} dropped)")

    # 2) cluster members for classifier
    rng = np.random.default_rng(SEED)
    items_uids = [it["cluster_uid"] for it in fresh]
    members = []
    for uid in items_uids:
        rows = df[df["cluster_uid"] == uid]
        n = min(N_MEMBERS_FOR_CLASSIFIER, len(rows))
        picks = rows.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))
        for _, r in picks.iterrows():
            members.append({
                "policy_uid": f"{r['openalex_id']}::{int(r['chunk_idx'])}",
                "sector": r["sector"],
                "cluster_uid": uid,
                "policy_text": str(r["policy_text"]),
            })
    with (OUT / "cluster_members.jsonl").open("w") as f:
        for m in members:
            f.write(json.dumps(m) + "\n")
    print(f"wrote {len(members)} cluster-member rows ({N_MEMBERS_FOR_CLASSIFIER} per cluster, 200 clusters)")

    # 3) judge few-shot
    fs = build_few_shot()
    with (OUT / "judge_few_shot.jsonl").open("w") as f:
        for ex in fs:
            f.write(json.dumps(ex) + "\n")
    print(f"wrote {len(fs)} judge few-shot examples (sectors: {[e['sector'] for e in fs]})")


if __name__ == "__main__":
    main()
