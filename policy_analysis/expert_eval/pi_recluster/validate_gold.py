"""Score a `classifications.jsonl` against expert gold and print go/no-go.

Companion to `classify_vllm.py`: run the classifier on the 770 expert-gold
rows first, then this script tells you whether the model is good enough to
commit to the full 1.47M run.

Decision rule (mirrors PI-recluster planning):
    binary_acc >= 72% AND strict_recall >= 70%  → SHIP on 1.47M
    binary_acc >= 65%                            → borderline, consider larger model
    otherwise                                     → do not scale

Usage:
    uv run python validate_gold.py \\
        --classifications data/gold_predictions.jsonl \\
        --gold ../../../runs/expert_gold/gold.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BINARY_MAP_MODEL = {
    "sufficiency": "sufficiency_related",
    "ambiguous": "sufficiency_related",
    "efficiency": "not_related",
    "consistency": "not_related",
    "not_a_policy": "not_related",
}
BINARY_MAP_EXPERT = BINARY_MAP_MODEL   # same mapping applied to expert labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifications", required=True)
    ap.add_argument("--gold", required=True, help="expert gold parquet (policy_id, category, ...)")
    args = ap.parse_args()

    preds = pd.DataFrame(
        [json.loads(l) for l in Path(args.classifications).read_text().splitlines() if l.strip()]
    )
    if "error" in preds.columns:
        fails = int(preds["error"].notna().sum())
        preds = preds[preds["error"].isna()]
    else:
        fails = 0
    preds = preds.rename(columns={"policy_uid": "policy_id"})

    gold = pd.read_parquet(args.gold)
    df = gold.merge(
        preds[["policy_id", "category", "sub_code", "confidence"]],
        on="policy_id", how="inner",
        suffixes=("_expert", "_model"),
    ).rename(columns={"category_expert": "expert_category", "category_model": "model_category"})

    total = len(df)
    if total == 0:
        print("no overlap between classifications and gold — check --gold path")
        return 1

    df["model_bin"] = df["model_category"].map(BINARY_MAP_MODEL)
    df["expert_bin"] = df["expert_category"].map(BINARY_MAP_EXPERT)

    binary_acc = (df["model_bin"] == df["expert_bin"]).mean()
    tp = ((df["model_bin"] == "sufficiency_related") & (df["expert_bin"] == "sufficiency_related")).sum()
    fp = ((df["model_bin"] == "sufficiency_related") & (df["expert_bin"] == "not_related")).sum()
    fn = ((df["model_bin"] == "not_related") & (df["expert_bin"] == "sufficiency_related")).sum()
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    strict = df[df["expert_category"] == "sufficiency"]
    strict_rec = (strict["model_category"] == "sufficiency").mean() if len(strict) else 0

    five_class = (df["model_category"] == df["expert_category"]).mean()

    sub = df[(df["expert_category"] == "sufficiency") & (df["model_category"] == "sufficiency")]
    sub = sub.dropna(subset=["sub_code"])
    sub_acc = (sub["sub_code"].astype(int) == sub["sufficiency_code"].astype(int)).mean() if len(sub) else 0

    print(f"scored {total:,} rows from gold  (failed classifications: {fails})")
    print()
    print(f"binary accuracy      {binary_acc:.1%}")
    print(f"binary precision     {prec:.1%}")
    print(f"binary recall        {rec:.1%}")
    print(f"binary F1            {f1:.2f}")
    print(f"strict suff recall   {strict_rec:.1%}   (n={len(strict)})")
    print(f"5-class exact match  {five_class:.1%}")
    print(f"sub-code accuracy    {sub_acc:.1%}   (n={len(sub)})")
    print()

    if binary_acc >= 0.72 and strict_rec >= 0.70:
        verdict, code = "SHIP — proceed with full 1.47M run", 0
    elif binary_acc >= 0.65:
        verdict, code = "BORDERLINE — try a larger model or improve few-shot before scaling", 2
    else:
        verdict, code = "DO NOT SCALE — model is not fit for this task", 3
    print(f"VERDICT: {verdict}")
    return code


if __name__ == "__main__":
    sys.exit(main())
