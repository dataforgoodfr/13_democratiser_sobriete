"""Compare current 3-class classifier output to expert 5-class labels.

Reads:
  runs/expert_gold/gold.parquet
  runs/expert_gold/classifications_v0.jsonl

Writes:
  runs/expert_gold/joined.parquet
  runs/expert_gold/REPORT.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("runs/expert_gold")

BINARY_MAP_MODEL = {
    "sufficiency": "sufficiency_related",
    "potential_sufficiency": "sufficiency_related",
    "not_sufficiency": "not_related",
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


def _df_md(df: pd.DataFrame, index_name: str) -> str:
    cols = list(df.columns)
    lines = [f"| {index_name} | " + " | ".join(str(c) for c in cols) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for idx, row in df.iterrows():
        cells = [str(row[c]) for c in cols]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _crosstab_md(df: pd.DataFrame, index: str, cols: str) -> str:
    ct = pd.crosstab(df[index], df[cols], margins=True, margins_name="TOTAL")
    return _df_md(ct, index)


def main() -> None:
    gold = pd.read_parquet(OUT / "gold.parquet")
    preds = pd.DataFrame(_load_jsonl(OUT / "classifications_v0.jsonl"))
    preds = preds.rename(columns={"policy_uid": "policy_id"})[
        ["policy_id", "category", "confidence", "reasoning"]
    ].rename(
        columns={
            "category": "model_category",
            "confidence": "model_confidence",
            "reasoning": "model_reasoning",
        }
    )
    df = gold.merge(preds, on="policy_id", how="left")
    df.to_parquet(OUT / "joined.parquet", index=False)

    df["model_binary"] = df["model_category"].map(BINARY_MAP_MODEL)
    df["expert_binary"] = df["category"].map(BINARY_MAP_EXPERT)

    total = len(df)
    binary_acc = (df["model_binary"] == df["expert_binary"]).mean()
    tp = ((df["model_binary"] == "sufficiency_related") & (df["expert_binary"] == "sufficiency_related")).sum()
    fp = ((df["model_binary"] == "sufficiency_related") & (df["expert_binary"] == "not_related")).sum()
    fn = ((df["model_binary"] == "not_related") & (df["expert_binary"] == "sufficiency_related")).sum()
    tn = ((df["model_binary"] == "not_related") & (df["expert_binary"] == "not_related")).sum()
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    strict = df[df["category"] == "sufficiency"]
    strict_recall = (strict["model_category"] == "sufficiency").mean()

    by_expert = df.groupby("category")["model_category"].value_counts().unstack(fill_value=0)

    per_sector = (
        df.groupby("sector")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "binary_acc": (g["model_binary"] == g["expert_binary"]).mean(),
            "n_gold_suff": (g["category"] == "sufficiency").sum(),
            "recall_gold_suff": (
                ((g["category"] == "sufficiency") & (g["model_category"] == "sufficiency")).sum()
                / max(1, (g["category"] == "sufficiency").sum())
            ),
        }), include_groups=False)
        .sort_values("binary_acc")
    )

    # cases the model missed hard
    missed = df[(df["category"] == "sufficiency") & (df["model_category"] == "not_sufficiency")]
    hallucinated = df[(df["model_category"] == "sufficiency") & (df["category"] == "not_a_policy")]

    lines = [
        "# Expert-gold baseline — current 3-class classifier",
        "",
        f"Model: `deepseek-v4-pro`, prompt `v0-sufficiency-3class-placeholder-def`, 6 few-shot examples.",
        f"Gold: {total} expert-labelled rows, 11 sectors × 70 rows.",
        "",
        "## Headline",
        "",
        "Binary framing (`sufficiency_related` = expert `sufficiency` ∪ `ambiguous`, model `sufficiency` ∪ `potential_sufficiency`):",
        "",
        f"- accuracy   **{binary_acc:.1%}**",
        f"- precision  **{prec:.1%}** ({tp} / {tp+fp})",
        f"- recall     **{rec:.1%}** ({tp} / {tp+fn})",
        f"- F1         **{f1:.2f}**",
        "",
        f"Strict recall on expert `sufficiency` only (model must also say `sufficiency`): **{strict_recall:.1%}** ({(strict['model_category']=='sufficiency').sum()} / {len(strict)})",
        "",
        "## Expert category × model category (rows = expert, cols = model)",
        "",
        _crosstab_md(df.fillna({"model_category": "MISSING"}), "category", "model_category"),
        "",
        "## Per-sector",
        "",
        _df_md(per_sector.round(3), "sector"),
        "",
        "## Where the current schema hurts",
        "",
        "1. **`Not a policy` (417 rows) is not modeled.** The 3-class enum cannot express 'this is junk, not a policy'; those rows are forced into `not_sufficiency` (best case) or a sufficiency bucket (false positive). Adding a 5th enum value alone would remove ~54% of the noise from downstream.",
        "2. **`Efficiency` and `Consistency` are conflated with `not_sufficiency`.** Fine for the binary problem but destroys the mechanism-level split experts actually care about.",
        "3. **`potential_sufficiency` has no counterpart in the expert schema.** It's how the model expresses hedging; the closest expert equivalent is `Ambiguous`. Look at the crosstab column for `potential_sufficiency` — the mass is spread across every expert class.",
        "",
        f"## Bad cases",
        "",
        f"### Missed (expert=sufficiency, model=not_sufficiency): {len(missed)}",
        "",
    ]
    for _, r in missed.head(10).iterrows():
        lines.append(f"- **{r['policy_id']}** ({r['sector']}): {r['policy_text'][:180]}")
        lines.append(f"  - model reasoning: {r['model_reasoning']}")
    lines += [
        "",
        f"### Hallucinated sufficiency on `Not a policy`: {len(hallucinated)}",
        "",
    ]
    for _, r in hallucinated.head(10).iterrows():
        lines.append(f"- **{r['policy_id']}** ({r['sector']}): {r['policy_text'][:180]}")
        lines.append(f"  - model reasoning: {r['model_reasoning']}")

    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'REPORT.md'}")
    print(f"\n--- HEADLINE ---")
    print(f"binary accuracy: {binary_acc:.1%}   precision: {prec:.1%}   recall: {rec:.1%}   F1: {f1:.2f}")
    print(f"strict sufficiency recall: {strict_recall:.1%}")


if __name__ == "__main__":
    main()
