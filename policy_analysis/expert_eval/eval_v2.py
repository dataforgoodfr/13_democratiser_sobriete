"""Head-to-head evaluation: prompt v1 vs prompt v2, both against expert gold.

Both use the same DeepSeek model and the same 10 few-shot examples (excluded
from scoring). v2 = six-pillar definition; v1 = single-discriminator definition.

Writes:
  runs/expert_gold/REPORT_v2.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

OUT = Path("runs/expert_gold")

# 5-class → binary (sufficiency-related vs not). Identical mapping for both
# prompt versions and for the expert labels.
BINARY_MAP = {
    "sufficiency": "sufficiency_related",
    "ambiguous": "sufficiency_related",
    "efficiency": "not_related",
    "consistency": "not_related",
    "not_a_policy": "not_related",
}


def _load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _binary_stats(df, model_col):
    d = df.copy()
    d["mb"] = d[model_col].map(BINARY_MAP)
    d["eb"] = d["category"].map(BINARY_MAP)
    d = d.dropna(subset=["mb", "eb"])
    tp = ((d["mb"] == "sufficiency_related") & (d["eb"] == "sufficiency_related")).sum()
    fp = ((d["mb"] == "sufficiency_related") & (d["eb"] == "not_related")).sum()
    fn = ((d["mb"] == "not_related") & (d["eb"] == "sufficiency_related")).sum()
    tn = ((d["mb"] == "not_related") & (d["eb"] == "not_related")).sum()
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    acc = (d["mb"] == d["eb"]).mean()
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "n": len(d)}


def _df_md(df: pd.DataFrame, index_name: str) -> str:
    cols = list(df.columns)
    lines = [f"| {index_name} | " + " | ".join(str(c) for c in cols) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _crosstab_md(df, index, cols):
    ct = pd.crosstab(df[index], df[cols], margins=True, margins_name="TOTAL")
    return _df_md(ct, index)


def _prep(name):
    v = pd.DataFrame(_load_jsonl(OUT / f"classifications_{name}.jsonl"))
    v = v[~v.get("error", pd.Series([None] * len(v))).notna()] if "error" in v.columns else v
    v = v.rename(columns={
        "policy_uid": "policy_id",
        "category": f"{name}_category",
        "confidence": f"{name}_confidence",
        "reasoning": f"{name}_reasoning",
        "sub_code": f"{name}_sub_code",
    })
    keep = ["policy_id", f"{name}_category", f"{name}_confidence",
            f"{name}_reasoning", f"{name}_sub_code"]
    return v[[c for c in keep if c in v.columns]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v1", help="baseline classifications_<base>.jsonl")
    ap.add_argument("--challenger", default="v2", help="challenger classifications_<chal>.jsonl")
    ap.add_argument("--out", default=None, help="report path (default REPORT_<chal>.md)")
    args = ap.parse_args()
    B, C = args.base, args.challenger
    report = Path(args.out) if args.out else OUT / f"REPORT_{C}.md"

    gold = pd.read_parquet(OUT / "gold.parquet")
    fs_ids = {l.strip() for l in (OUT / "few_shot_v1_ids.txt").read_text().splitlines() if l.strip()}
    gold_eval = gold[~gold["policy_id"].isin(fs_ids)].copy()

    b = _prep(B)
    c = _prep(C)
    df = gold_eval.merge(b, on="policy_id", how="left").merge(c, on="policy_id", how="left")

    # scoring only where both versions returned a label (fair comparison)
    both = df.dropna(subset=[f"{B}_category", f"{C}_category"]).copy()

    s_b = _binary_stats(both, f"{B}_category")
    s_c = _binary_stats(both, f"{C}_category")

    b_exact = (both[f"{B}_category"] == both["category"]).mean()
    c_exact = (both[f"{C}_category"] == both["category"]).mean()

    strict = both[both["category"] == "sufficiency"]
    b_strict = (strict[f"{B}_category"] == "sufficiency").mean()
    c_strict = (strict[f"{C}_category"] == "sufficiency").mean()

    # sub-code accuracy where model & expert both say sufficiency
    def sub_acc(name):
        bs = both[(both["category"] == "sufficiency") & (both[f"{name}_category"] == "sufficiency")]
        if not len(bs):
            return float("nan"), 0
        return (bs[f"{name}_sub_code"] == bs["sufficiency_code"]).mean(), len(bs)
    b_sub, b_sub_n = sub_acc(B)
    c_sub, c_sub_n = sub_acc(C)

    # per-sector binary accuracy, both versions
    def _bin_acc(g, col):
        return (g[col].map(BINARY_MAP) == g["category"].map(BINARY_MAP)).mean()
    per_sector = (
        both.groupby("sector")
        .apply(lambda g: pd.Series({
            "n": len(g),
            f"{B}_bin": round(_bin_acc(g, f"{B}_category"), 3),
            f"{C}_bin": round(_bin_acc(g, f"{C}_category"), 3),
            "Δ": round(_bin_acc(g, f"{C}_category") - _bin_acc(g, f"{B}_category"), 3),
            f"{B}_5cls": round((g[f"{B}_category"] == g["category"]).mean(), 3),
            f"{C}_5cls": round((g[f"{C}_category"] == g["category"]).mean(), 3),
        }), include_groups=False)
        .sort_values("Δ")
    )

    # movement analysis (binary)
    both["bb"] = both[f"{B}_category"].map(BINARY_MAP)
    both["cb"] = both[f"{C}_category"].map(BINARY_MAP)
    both["eb"] = both["category"].map(BINARY_MAP)
    regressions = both[(both["bb"] == both["eb"]) & (both["cb"] != both["eb"])]
    new_wins = both[(both["bb"] != both["eb"]) & (both["cb"] == both["eb"])]

    def dpt(a, x):  # delta in percentage points
        return f"{(x - a) * 100:+.1f} pt"

    lines = [
        f"# Prompt {C} vs {B} — expert gold, DeepSeek V4",
        "",
        f"Same model, same 10 few-shot (excluded). Scored on **{len(both)} rows** "
        "where both versions returned a valid label.",
        "",
        "## Headline (binary: sufficiency-related vs not)",
        "",
        f"| metric | {B} | {C} | Δ |",
        "|---|---:|---:|---:|",
        f"| accuracy | {s_b['acc']:.1%} | **{s_c['acc']:.1%}** | {dpt(s_b['acc'], s_c['acc'])} |",
        f"| precision | {s_b['prec']:.1%} | **{s_c['prec']:.1%}** | {dpt(s_b['prec'], s_c['prec'])} |",
        f"| recall | {s_b['rec']:.1%} | **{s_c['rec']:.1%}** | {dpt(s_b['rec'], s_c['rec'])} |",
        f"| F1 | {s_b['f1']:.2f} | **{s_c['f1']:.2f}** | {s_c['f1'] - s_b['f1']:+.2f} |",
        f"| strict recall (expert=suff→model=suff) | {b_strict:.1%} | **{c_strict:.1%}** | {dpt(b_strict, c_strict)} |",
        "",
        f"5-class exact-match: {B} **{b_exact:.1%}** → {C} **{c_exact:.1%}** ({dpt(b_exact, c_exact)})",
        "",
        f"Sub-code exact-match on sufficiency: {B} **{b_sub:.1%}** (n={b_sub_n}) → "
        f"{C} **{c_sub:.1%}** (n={c_sub_n})",
        "",
        f"Confusion matrices (binary): {B} TP={s_b['tp']} FP={s_b['fp']} FN={s_b['fn']} TN={s_b['tn']} | "
        f"{C} TP={s_c['tp']} FP={s_c['fp']} FN={s_c['fn']} TN={s_c['tn']}",
        "",
        f"## {C} 5-class confusion (rows = expert, cols = {C} model)",
        "",
        _crosstab_md(both, "category", f"{C}_category"),
        "",
        f"## Per-sector (binary accuracy, {B} vs {C})",
        "",
        _df_md(per_sector, "sector"),
        "",
        f"## Movement {B} → {C} (binary): {len(new_wins)} new wins, {len(regressions)} regressions",
        "",
    ]

    if len(new_wins):
        lines += [f"### Sample new wins ({C} fixed what {B} got wrong)", ""]
        for _, r in new_wins.head(12).iterrows():
            lines.append(
                f"- **{r['policy_id']}** expert={r['category']} "
                f"{B}={r[f'{B}_category']} → {C}={r[f'{C}_category']}: {r['policy_text'][:160]}")
            lines.append(f"  - {C}: {r[f'{C}_reasoning']}")
    if len(regressions):
        lines += ["", f"### Sample regressions ({C} broke what {B} got right)", ""]
        for _, r in regressions.head(12).iterrows():
            lines.append(
                f"- **{r['policy_id']}** expert={r['category']} "
                f"{B}={r[f'{B}_category']} → {C}={r[f'{C}_category']}: {r['policy_text'][:160]}")
            lines.append(f"  - {C}: {r[f'{C}_reasoning']}")

    report.write_text("\n".join(lines) + "\n")
    print(f"wrote {report}\n")
    print(f"--- HEADLINE (binary): {B} vs {C} ---")
    print(f"{B}: acc={s_b['acc']:.1%}  prec={s_b['prec']:.1%}  rec={s_b['rec']:.1%}  F1={s_b['f1']:.2f}")
    print(f"{C}: acc={s_c['acc']:.1%}  prec={s_c['prec']:.1%}  rec={s_c['rec']:.1%}  F1={s_c['f1']:.2f}")
    print(f"strict recall  {B}={b_strict:.1%}  {C}={c_strict:.1%}")
    print(f"5-class exact  {B}={b_exact:.1%}  {C}={c_exact:.1%}")
    print(f"sub-code acc   {B}={b_sub:.1%}(n={b_sub_n})  {C}={c_sub:.1%}(n={c_sub_n})")
    print(f"movement: {len(new_wins)} new wins, {len(regressions)} regressions")


if __name__ == "__main__":
    main()
