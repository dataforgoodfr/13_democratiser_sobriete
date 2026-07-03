"""Pick a balanced few-shot set from the expert gold for prompt v1.

Selection recipe:
- 4 sufficiency examples across sub-codes {1, 4, 6, 9} (the most common codes)
- 2 efficiency
- 2 consistency
- 1 ambiguous
- 1 not_a_policy (with a non-trivial text length)

Reasoning strings are templated from the label so they stay grounded in the
taxonomy definitions rather than hand-written interpretations.

Outputs:
  runs/expert_gold/few_shot_v1.jsonl
  runs/expert_gold/few_shot_v1_ids.txt   (for excluding from evaluation)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

OUT = Path("runs/expert_gold")
SEED = 3


REASONING_TEMPLATES = {
    "sufficiency": {
        0: "matches sub-code 0 (involuntary demand cut)",
        1: "matches sub-code 1 (cap/limit/ban): absolute ceiling on activity",
        2: "matches sub-code 2 (demand-suppressing price/tax)",
        3: "matches sub-code 3 (reduce/right-size the quantity of the service)",
        4: "matches sub-code 4 (passive/climate-responsive design avoids active system demand)",
        5: "matches sub-code 5 (proximity/compactness avoids underlying travel/space demand)",
        6: "matches sub-code 6 (modal or provisioning shift to a lighter option)",
        7: "matches sub-code 7 (share/reuse/repair/prolong — longevity, not throughput)",
        8: "matches sub-code 8 (dietary/food-system sufficiency)",
        9: "matches sub-code 9 (public/collective provisioning at low throughput)",
    },
    "efficiency": "same service delivered with less input per unit — not absolute reduction",
    "consistency": "substitution to a cleaner or renewable input — not absolute reduction",
    "ambiguous": "text IS a policy but mechanism is mixed/unclear across categories",
    "not_a_policy": "descriptive statement / data / implementation detail, not a policy instrument",
}


def _pick(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=seed)


def _reasoning(row: pd.Series) -> str:
    cat = row["category"]
    if cat == "sufficiency" and pd.notna(row["sufficiency_code"]):
        return REASONING_TEMPLATES["sufficiency"][int(row["sufficiency_code"])]
    return REASONING_TEMPLATES[cat]


def main() -> None:
    gold = pd.read_parquet(OUT / "gold.parquet")
    picks: list[pd.Series] = []

    rng = random.Random(SEED)

    # 4 sufficiency across the priority codes
    for code in (1, 4, 6, 9):
        pool = gold[(gold["category"] == "sufficiency") & (gold["sufficiency_code"] == code)]
        if len(pool):
            picks.append(pool.sample(n=1, random_state=SEED + code).iloc[0])

    # 2 efficiency, 2 consistency
    for cat, n, base_seed in [("efficiency", 2, SEED + 10), ("consistency", 2, SEED + 20)]:
        pool = gold[gold["category"] == cat]
        chosen = pool.sample(n=min(n, len(pool)), random_state=base_seed)
        for _, r in chosen.iterrows():
            picks.append(r)

    # 1 ambiguous
    pool = gold[gold["category"] == "ambiguous"]
    picks.append(pool.sample(n=1, random_state=SEED + 30).iloc[0])

    # 1 not_a_policy with reasonable text length
    pool = gold[
        (gold["category"] == "not_a_policy") & (gold["policy_text"].str.len().between(60, 200))
    ]
    picks.append(pool.sample(n=1, random_state=SEED + 40).iloc[0])

    # De-duplicate on policy_id in case of overlap (shouldn't happen but be safe)
    seen: set[str] = set()
    unique = []
    for r in picks:
        if r["policy_id"] in seen:
            continue
        seen.add(r["policy_id"])
        unique.append(r)

    fs_rows = []
    for r in unique:
        row = {
            "policy_id": r["policy_id"],
            "sector": r["sector"],
            "policy_text": r["policy_text"],
            "category": r["category"],
            "reasoning": _reasoning(r),
        }
        if r["category"] == "sufficiency" and pd.notna(r["sufficiency_code"]):
            row["sub_code"] = int(r["sufficiency_code"])
        fs_rows.append(row)

    with (OUT / "few_shot_v1.jsonl").open("w") as f:
        for row in fs_rows:
            f.write(json.dumps(row) + "\n")

    with (OUT / "few_shot_v1_ids.txt").open("w") as f:
        for row in fs_rows:
            f.write(row["policy_id"] + "\n")

    print(f"wrote {len(fs_rows)} few-shot examples\n")
    for r in fs_rows:
        head = r["policy_text"][:100].replace("\n", " ")
        lbl = r["category"] + (f" (code {r['sub_code']})" if "sub_code" in r else "")
        print(f"  [{lbl:35s}] {r['policy_id']:15s} {head}")


if __name__ == "__main__":
    main()
