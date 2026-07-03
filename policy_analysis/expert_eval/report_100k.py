"""Final 2×2 pipeline report for the 100k API run.

Joins per-cluster sufficiency density (from 100k classifications) with
per-cluster judge verdicts (intrusion task on existing clusters).

Reads:
    runs/api_100k/per_cluster_stats.parquet
    runs/api_100k/judgments.jsonl

Writes:
    runs/api_100k/per_cluster.csv
    runs/api_100k/REPORT.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("runs/api_100k")


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


SUB_LABELS = {
    0: "Involuntary demand cut",
    1: "Caps, limits & bans",
    2: "Demand-suppressing prices/taxes",
    3: "Reduce & right size",
    4: "Passive/climate design",
    5: "Proximity & compactness",
    6: "Modal & provisioning shift",
    7: "Share, reuse, repair",
    8: "Dietary/food-system",
    9: "Public provisioning",
}


def main() -> None:
    stats = pd.read_parquet(OUT / "per_cluster_stats.parquet")
    judgments = pd.DataFrame(_load_jsonl(OUT / "judgments.jsonl"))
    judgments = judgments[judgments.get("error").isna()] if "error" in judgments.columns else judgments
    judgments["judge_correct"] = (judgments["intruder_index"] == judgments["gold_intruder_index"]).astype(int)

    j = judgments[
        ["cluster_uid", "sector", "cluster_size", "size_bucket", "judge_correct",
         "confidence", "topic_label", "specificity", "intruder_index", "gold_intruder_index"]
    ].rename(columns={"confidence": "judge_confidence"})
    df = j.merge(stats, on=["cluster_uid", "sector"], how="left")
    df = df.dropna(subset=["n_sampled"])
    df.to_csv(OUT / "per_cluster.csv", index=False)

    df["judge_ok"] = df["judge_correct"] == 1
    df["suff_high"] = df["density_loose"].fillna(0) >= 0.5

    counts = df.groupby(["judge_ok", "suff_high"]).size()

    def cell(jo, sh):
        try: return int(counts.loc[(jo, sh)])
        except KeyError: return 0
    A = cell(True, True)
    B = cell(True, False)
    C = cell(False, True)
    D = cell(False, False)

    overall_acc = df["judge_correct"].mean()

    by_sector = (
        df.groupby("sector")
        .agg(n_clusters=("cluster_uid", "size"),
             judge_acc=("judge_correct", "mean"),
             mean_density=("density_loose", "mean"),
             mean_strict=("density_strict", "mean"))
        .sort_values("judge_acc")
    )

    keep = df[df["judge_ok"] & df["suff_high"]].sort_values(["judge_confidence", "density_loose"], ascending=False).head(15)
    recluster = df[(~df["judge_ok"]) & df["suff_high"]].sort_values("density_loose", ascending=False).head(15)
    drop = df[df["judge_ok"] & (~df["suff_high"])].sort_values("judge_confidence", ascending=False).head(10)

    # sub-code themes
    stats_top = stats.dropna(subset=["top_subcode"]).copy()
    stats_top["top_subcode"] = stats_top["top_subcode"].astype(int)
    sub_hist = stats_top["top_subcode"].value_counts().sort_index()

    # global stats from the 100k classification
    global_dist = (
        pd.read_json(OUT / "classifications.jsonl", lines=True)
    )
    ok = "error" not in global_dist.columns or global_dist["error"].isna().all()
    if "error" in global_dist.columns:
        global_dist = global_dist[global_dist["error"].isna()]
    class_dist = global_dist["category"].value_counts()

    N = len(global_dist)
    n_clusters_with_data = int(stats["n_sampled"].notna().sum())

    lines = [
        "# 100k pipeline test — DeepSeek V4 API end-to-end",
        "",
        f"- Classified {N:,} policies via DeepSeek V4 API (`v1-expert-5class-10sub` prompt, 10 few-shot from expert gold).",
        f"- Judged {len(df):,} existing clusters from `clusters_2026-03-18` via intrusion task with calibrated judge.",
        f"- Note: this run **evaluates the existing clustering**; re-clustering was not run (embeddings not local yet).",
        "",
        "## Classifier outcome on 100k",
        "",
    ]
    for k, v in class_dist.items():
        lines.append(f"- **{k}**: {v:,} ({v/N:.1%})")
    lines.append("")
    lines.append(f"After filtering (dropping `not_a_policy` + `efficiency` + `consistency`), **{int((class_dist.get('sufficiency', 0) + class_dist.get('ambiguous', 0))):,}** policies remain ({(class_dist.get('sufficiency', 0) + class_dist.get('ambiguous', 0))/N:.1%}). That is the input to the re-clustering stage in the full pipeline.")
    lines.append("")

    lines += [
        "## Judge outcome",
        "",
        f"- {len(df):,} clusters judged.",
        f"- Judge accuracy: **{overall_acc:.1%}** (chance = 20% for 5-way intrusion).",
        "",
        "## 2×2 decision matrix",
        "",
        "Rows = judge verdict.  Cols = sufficiency density ≥ 50% (loose, includes ambiguous).",
        "",
        "|                       | sufficiency-relevant (≥50%) | off-topic (<50%) | row total |",
        "|---|---:|---:|---:|",
        f"| **coherent (judge ok)**  | **{A} — KEEP** | {B} — drop for downstream | {A+B} |",
        f"| **incoherent (judge wrong)** | {C} — re-cluster | {D} — drop entirely | {C+D} |",
        f"| col total | {A+C} | {B+D} | {A+B+C+D} |",
        "",
        "Each quadrant → decision:",
        "- **A (KEEP)** — coherent, sufficiency-dense clusters. Ship to downstream index.",
        "- **B (drop for sufficiency use)** — coherent but off-topic; not needed for a sufficiency index.",
        "- **C (RE-CLUSTER)** — sufficiency-dense but incoherent. Highest-value cell: run Leiden with finer resolution on this slice.",
        "- **D (drop)** — noisy and irrelevant.",
        "",
        "## Per-sector",
        "",
        _df_md(by_sector.round(3), "sector"),
        "",
        "## Sufficiency sub-code themes (dominant per cluster, n=" + str(int(len(stats_top))) + " clusters with any sufficiency)",
        "",
        "| code | label | n clusters where dominant |",
        "|---:|---|---:|",
    ]
    for code, n in sub_hist.items():
        lines.append(f"| {code} | {SUB_LABELS.get(code, '?')} | {n} |")

    lines += [
        "",
        "## Top 15 KEEP candidates",
        "",
        "| cluster_uid | sector | size | judge_conf | density_loose | density_strict | top_subcode | topic_label |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for _, r in keep.iterrows():
        sub = f"{int(r['top_subcode'])} {SUB_LABELS.get(int(r['top_subcode']), '')}" if pd.notna(r["top_subcode"]) else ""
        lines.append(f"| {r['cluster_uid']} | {r['sector']} | {int(r['cluster_size'])} | {int(r['judge_confidence'])} | {r['density_loose']:.2f} | {r['density_strict']:.2f} | {sub} | {r['topic_label']} |")

    lines += [
        "",
        "## Top 15 RE-CLUSTER candidates",
        "",
        "| cluster_uid | sector | size | density_loose | judge picked vs gold | topic_label |",
        "|---|---|---:|---:|---|---|",
    ]
    for _, r in recluster.iterrows():
        lines.append(f"| {r['cluster_uid']} | {r['sector']} | {int(r['cluster_size'])} | {r['density_loose']:.2f} | {int(r['intruder_index'])} vs {int(r['gold_intruder_index'])} | {r['topic_label']} |")

    lines += [
        "",
        "## Top 10 coherent-but-off-topic clusters (B)",
        "",
        "| cluster_uid | sector | size | density_loose | topic_label |",
        "|---|---|---:|---:|---|",
    ]
    for _, r in drop.iterrows():
        lines.append(f"| {r['cluster_uid']} | {r['sector']} | {int(r['cluster_size'])} | {r['density_loose']:.2f} | {r['topic_label']} |")

    lines += [
        "",
        "## Bottom line for the API vs Jean-Zay decision",
        "",
        f"- Classifier throughput: {N:,} policies via V4 API. Cost ≈ ${N*0.00019:.2f} at ~$0.00019/call, wall clock scales linearly.",
        f"- Judge throughput: {len(df)} clusters judged with calibrated few-shot. Cost ≈ ${len(df)*0.0005:.3f}.",
        f"- Combined API cost for the full 1.5M classification + full-cluster judge: ~${(1_470_000*0.00019 + 2000*0.0005):.0f}.",
        "- Re-clustering stage is CPU-only Leiden on filtered subset; it needs embeddings (7.15GB from HF or Scaleway API) but runs the same way regardless of where the classifier lives.",
        "- Conclusion: for the current iteration, the API path is viable end-to-end at a ~$300 budget. Jean-Zay makes sense if we need >5 iterations or start swapping in models the API doesn't offer.",
    ]

    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'REPORT.md'}")
    print("\n--- HEADLINE ---")
    print(f"judge accuracy: {overall_acc:.1%}")
    print(f"A (keep)      : {A}")
    print(f"B (off-topic) : {B}")
    print(f"C (re-cluster): {C}")
    print(f"D (drop)      : {D}")


if __name__ == "__main__":
    main()
