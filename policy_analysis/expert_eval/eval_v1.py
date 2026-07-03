"""Head-to-head evaluation: baseline v0 vs prompt v1, both against expert gold.

Both sets excluded the 10 few-shot examples to give v1 a fair number.

Writes:
  runs/expert_gold/REPORT_v1.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("runs/expert_gold")

BINARY_MAP_MODEL_V0 = {
    "sufficiency": "sufficiency_related",
    "potential_sufficiency": "sufficiency_related",
    "not_sufficiency": "not_related",
}
BINARY_MAP_MODEL_V1 = {
    "sufficiency": "sufficiency_related",
    "ambiguous": "sufficiency_related",
    "efficiency": "not_related",
    "consistency": "not_related",
    "not_a_policy": "not_related",
}
BINARY_MAP_EXPERT = {
    "sufficiency": "sufficiency_related",
    "ambiguous": "sufficiency_related",
    "efficiency": "not_related",
    "consistency": "not_related",
    "not_a_policy": "not_related",
}


def _load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _binary_stats(df, model_col, binary_map_model):
    d = df.copy()
    d["model_binary"] = d[model_col].map(binary_map_model)
    d["expert_binary"] = d["category"].map(BINARY_MAP_EXPERT)
    tp = ((d["model_binary"] == "sufficiency_related") & (d["expert_binary"] == "sufficiency_related")).sum()
    fp = ((d["model_binary"] == "sufficiency_related") & (d["expert_binary"] == "not_related")).sum()
    fn = ((d["model_binary"] == "not_related") & (d["expert_binary"] == "sufficiency_related")).sum()
    tn = ((d["model_binary"] == "not_related") & (d["expert_binary"] == "not_related")).sum()
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (d["model_binary"] == d["expert_binary"]).mean()
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _df_md(df: pd.DataFrame, index_name: str) -> str:
    cols = list(df.columns)
    lines = [f"| {index_name} | " + " | ".join(str(c) for c in cols) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for idx, row in df.iterrows():
        cells = [str(row[c]) for c in cols]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _crosstab_md(df, index, cols):
    ct = pd.crosstab(df[index], df[cols], margins=True, margins_name="TOTAL")
    return _df_md(ct, index)


def main() -> None:
    gold = pd.read_parquet(OUT / "gold.parquet")

    fs_ids = {l.strip() for l in (OUT / "few_shot_v1_ids.txt").read_text().splitlines() if l.strip()}
    gold_eval = gold[~gold["policy_id"].isin(fs_ids)].copy()

    v0 = pd.DataFrame(_load_jsonl(OUT / "classifications_v0.jsonl"))
    v0 = v0[~v0["policy_uid"].isin(fs_ids)]
    v0 = v0.rename(columns={"policy_uid": "policy_id", "category": "v0_category", "confidence": "v0_confidence", "reasoning": "v0_reasoning"})[
        ["policy_id", "v0_category", "v0_confidence", "v0_reasoning"]
    ]

    v1 = pd.DataFrame(_load_jsonl(OUT / "classifications_v1.jsonl"))
    v1 = v1.rename(columns={"policy_uid": "policy_id", "category": "v1_category", "confidence": "v1_confidence", "reasoning": "v1_reasoning"})[
        ["policy_id", "v1_category", "v1_confidence", "v1_reasoning", "sub_code"]
    ].rename(columns={"sub_code": "v1_sub_code"})

    df = gold_eval.merge(v0, on="policy_id", how="left").merge(v1, on="policy_id", how="left")
    df.to_parquet(OUT / "joined_v1.parquet", index=False)

    s_v0 = _binary_stats(df, "v0_category", BINARY_MAP_MODEL_V0)
    s_v1 = _binary_stats(df, "v1_category", BINARY_MAP_MODEL_V1)

    # 5-class exact-match accuracy for v1 (v0 has different enum so N/A)
    v1_exact = (df["v1_category"] == df["category"]).mean()

    # strict sufficiency recall
    strict = df[df["category"] == "sufficiency"]
    v0_strict = (strict["v0_category"] == "sufficiency").mean()
    v1_strict = (strict["v1_category"] == "sufficiency").mean()

    # sub_code accuracy on sufficiency where both model and expert predict sufficiency
    both_suff = df[(df["category"] == "sufficiency") & (df["v1_category"] == "sufficiency")]
    if len(both_suff):
        sub_acc = (both_suff["v1_sub_code"] == both_suff["sufficiency_code"]).mean()
    else:
        sub_acc = float("nan")

    per_sector = (
        df.groupby("sector")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "v0_bin_acc": (g["v0_category"].map(BINARY_MAP_MODEL_V0) == g["category"].map(BINARY_MAP_EXPERT)).mean(),
            "v1_bin_acc": (g["v1_category"].map(BINARY_MAP_MODEL_V1) == g["category"].map(BINARY_MAP_EXPERT)).mean(),
            "v1_5class_acc": (g["v1_category"] == g["category"]).mean(),
            "n_gold_suff": (g["category"] == "sufficiency").sum(),
            "v1_recall_gold_suff": (
                ((g["category"] == "sufficiency") & (g["v1_category"] == "sufficiency")).sum()
                / max(1, (g["category"] == "sufficiency").sum())
            ),
        }), include_groups=False)
        .sort_values("v1_bin_acc")
    )

    # regressions: cases v0 got right and v1 got wrong (binary)
    df["v0_bin"] = df["v0_category"].map(BINARY_MAP_MODEL_V0)
    df["v1_bin"] = df["v1_category"].map(BINARY_MAP_MODEL_V1)
    df["expert_bin"] = df["category"].map(BINARY_MAP_EXPERT)
    regressions = df[(df["v0_bin"] == df["expert_bin"]) & (df["v1_bin"] != df["expert_bin"])]
    v1_new_wins = df[(df["v0_bin"] != df["expert_bin"]) & (df["v1_bin"] == df["expert_bin"])]

    lines = [
        "# Prompt v1 vs baseline — expert gold",
        "",
        f"Model: `deepseek-v4-pro`. 10 few-shot examples excluded → **{len(df)} rows** in evaluation set.",
        "- Baseline prompt: `v0-sufficiency-3class-placeholder-def` (3-class, 6 few-shot).",
        "- v1 prompt: `v1-expert-5class-10sub` (5-class + 10-code sub-taxonomy, 10 few-shot from expert gold).",
        "",
        "## Headline (binary: sufficiency-related vs not)",
        "",
        "| metric | v0 baseline | v1 | Δ |",
        "|---|---:|---:|---:|",
        f"| accuracy | {s_v0['acc']:.1%} | **{s_v1['acc']:.1%}** | {(s_v1['acc']-s_v0['acc'])*100:+.1f} pt |",
        f"| precision | {s_v0['prec']:.1%} | **{s_v1['prec']:.1%}** | {(s_v1['prec']-s_v0['prec'])*100:+.1f} pt |",
        f"| recall | {s_v0['rec']:.1%} | **{s_v1['rec']:.1%}** | {(s_v1['rec']-s_v0['rec'])*100:+.1f} pt |",
        f"| F1 | {s_v0['f1']:.2f} | **{s_v1['f1']:.2f}** | {s_v1['f1']-s_v0['f1']:+.2f} |",
        f"| strict recall (expert=suff → model=suff) | {v0_strict:.1%} | **{v1_strict:.1%}** | {(v1_strict-v0_strict)*100:+.1f} pt |",
        "",
        f"5-class exact-match accuracy (v1 only, v0 enum incompatible): **{v1_exact:.1%}**",
        "",
        f"Sub-code exact-match accuracy where both label sufficiency (n={len(both_suff)}): **{sub_acc:.1%}**",
        "",
        "## v1 5-class confusion (rows = expert, cols = v1 model)",
        "",
        _crosstab_md(df, "category", "v1_category"),
        "",
        "## Per-sector",
        "",
        _df_md(per_sector.round(3), "sector"),
        "",
        f"## Regressions from v0 → v1 (binary): {len(regressions)}",
        "",
        f"## New wins v1 vs v0 (binary): {len(v1_new_wins)}",
        "",
    ]

    if len(regressions):
        lines += [
            "### Sample regressions",
            "",
        ]
        for _, r in regressions.head(10).iterrows():
            lines.append(f"- **{r['policy_id']}** expert={r['category']} v0={r['v0_category']} v1={r['v1_category']}: {r['policy_text'][:180]}")
            lines.append(f"  - v1 reasoning: {r['v1_reasoning']}")

    if len(v1_new_wins):
        lines += [
            "",
            "### Sample new wins",
            "",
        ]
        for _, r in v1_new_wins.head(10).iterrows():
            lines.append(f"- **{r['policy_id']}** expert={r['category']} v0={r['v0_category']} v1={r['v1_category']}: {r['policy_text'][:180]}")

    # Sub-code breakdown
    lines += ["", "## Sub-code performance on sufficiency"]
    if len(both_suff):
        sub_ct = pd.crosstab(both_suff["sufficiency_code"], both_suff["v1_sub_code"], margins=True, margins_name="TOTAL")
        lines += ["", "Rows = expert sub-code, cols = v1 sub-code (only where both label `sufficiency`):", "", _df_md(sub_ct, "expert_code")]

    (OUT / "REPORT_v1.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'REPORT_v1.md'}")
    print("\n--- HEADLINE ---")
    print(f"binary  v0: acc={s_v0['acc']:.1%}  prec={s_v0['prec']:.1%}  rec={s_v0['rec']:.1%}  F1={s_v0['f1']:.2f}")
    print(f"binary  v1: acc={s_v1['acc']:.1%}  prec={s_v1['prec']:.1%}  rec={s_v1['rec']:.1%}  F1={s_v1['f1']:.2f}")
    print(f"strict recall  v0: {v0_strict:.1%}   v1: {v1_strict:.1%}")
    print(f"v1 5-class exact-match: {v1_exact:.1%}")
    print(f"sub-code accuracy on sufficiency (both agree suff): {sub_acc:.1%}")


if __name__ == "__main__":
    main()
