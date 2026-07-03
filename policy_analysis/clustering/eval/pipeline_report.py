"""Per-cluster pipeline report for the 200-cluster smoke test.

Joins:
  - judgments.jsonl       (cluster judged correct? confidence?)
  - classifications.jsonl (sufficiency density of cluster members)

Outputs:
  per_cluster.csv  — one row per cluster, with all signals
  REPORT.md        — narrative + 2x2 decision matrix
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

OUT = Path("runs/pipeline_test")


def _load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    judgments = _load(OUT / "judgments.jsonl")
    classifications = _load(OUT / "classifications.jsonl")

    # per-cluster sufficiency density
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for c in classifications:
        if "error" in c:
            continue
        by_cluster[c["cluster_uid"]].append(c["category"])

    rows = []
    for j in judgments:
        uid = j["cluster_uid"]
        if "error" in j:
            continue
        cats = by_cluster.get(uid, [])
        n = len(cats)
        suff_strict = sum(1 for c in cats if c == "sufficiency")
        suff_loose = suff_strict + sum(1 for c in cats if c == "potential_sufficiency")
        rows.append({
            "cluster_uid": uid,
            "sector": j["sector"],
            "size_bucket": j["size_bucket"],
            "cluster_size": j["cluster_size"],
            "judge_correct": int(j["intruder_index"] == j["gold_intruder_index"]),
            "judge_confidence": j["confidence"],
            "judge_specificity": j["specificity"],
            "topic_label": j["topic_label"],
            "n_members_classified": n,
            "sufficiency_density_strict": suff_strict / n if n else None,
            "sufficiency_density_loose": suff_loose / n if n else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_cluster.csv", index=False)
    print(f"wrote {OUT/'per_cluster.csv'} with {len(df)} clusters")

    overall_acc = df["judge_correct"].mean()
    by_sector = df.groupby("sector")["judge_correct"].agg(["mean", "count"]).rename(columns={"mean": "judge_acc", "count": "n"}).sort_values("judge_acc")

    # 2x2 quadrants:
    df["judge_ok"] = df["judge_correct"] == 1
    df["suff_high"] = df["sufficiency_density_loose"].fillna(0) >= 0.5

    q = df.groupby(["judge_ok", "suff_high"]).size().unstack(fill_value=0)
    # rows: judge_ok False/True. cols: suff_high False/True

    def cell(jok, sh):
        try: return int(q.loc[jok, sh])
        except KeyError: return 0
    A = cell(True, True)    # coherent + sufficiency-relevant: KEEP
    B = cell(True, False)   # coherent but off-topic for sufficiency: DROP
    C = cell(False, True)   # sufficiency-relevant but incoherent: RE-CLUSTER
    D = cell(False, False)  # incoherent and off-topic: DROP

    lines = [
        "# Pipeline test — 200 clusters",
        "",
        f"Inputs: {len(df)} clusters from existing `clusters_2026-03-18` (different seed than `eval_v1`, no overlap).",
        f"Judge: DeepSeek V4 with 4 few-shot calibration examples injected.",
        f"Classifier: sufficiency 3-class on 5 sampled members per cluster (placeholder definition).",
        "",
        "## Headline",
        f"- Judge accuracy: **{overall_acc:.1%}** (chance = 20%).",
        f"- Mean sufficiency density (loose: `sufficiency` ∪ `potential_sufficiency`): **{df['sufficiency_density_loose'].mean():.1%}**.",
        f"- Mean sufficiency density (strict: `sufficiency` only): **{df['sufficiency_density_strict'].mean():.1%}**.",
        "",
        "## 2×2 decision matrix",
        "",
        f"Rows = judge verdict.  Columns = sufficiency density ≥ 50% (loose).",
        "",
        "|                       | sufficiency-relevant (≥50%) | off-topic (<50%) | row total |",
        "|---|---:|---:|---:|",
        f"| **coherent (judge ok)**  | **{A} — KEEP** | {B} — drop for downstream | {A+B} |",
        f"| **incoherent (judge wrong)** | {C} — re-cluster | {D} — drop entirely | {C+D} |",
        f"| col total | {A+C} | {B+D} | {A+B+C+D} |",
        "",
        "Each quadrant tells you what to do with that group of clusters:",
        "- **A (keep)** — clusters that pass both tests; they go into the downstream policy index.",
        "- **B (drop for sufficiency use)** — coherent topical clusters that aren't about sufficiency. Useful if you want to map adjacent policy areas, otherwise drop.",
        "- **C (re-cluster)** — sufficiency-rich material grouped poorly. Most informative quadrant: re-run Leiden with finer resolution on this slice, or split by sub-topic.",
        "- **D (drop)** — noisy and irrelevant. Don't waste downstream attention.",
        "",
        "## Per-sector judge accuracy",
        "| sector | n | judge accuracy |",
        "|---|---:|---:|",
    ]
    for s, row in by_sector.iterrows():
        lines.append(f"| {s} | {int(row['n'])} | {row['judge_acc']:.2f} |")

    # which sectors are sufficiency-dense
    sec_suff = df.groupby("sector")["sufficiency_density_loose"].mean().sort_values(ascending=False)
    lines += [
        "",
        "## Per-sector sufficiency density (loose)",
        "| sector | mean sufficiency density |",
        "|---|---:|",
    ]
    for s, v in sec_suff.items():
        lines.append(f"| {s} | {v:.2f} |")

    # actionable shortlist
    keep = df[df["judge_ok"] & df["suff_high"]].sort_values("judge_confidence", ascending=False).head(10)
    recluster = df[(~df["judge_ok"]) & df["suff_high"]].sort_values("sufficiency_density_loose", ascending=False).head(10)
    lines += [
        "",
        "## Top 10 KEEP candidates (high judge confidence, sufficiency-dense)",
        "| cluster_uid | sector | size | judge_conf | suff_density | topic_label |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, r in keep.iterrows():
        lines.append(f"| {r['cluster_uid']} | {r['sector']} | {r['cluster_size']} | {r['judge_confidence']} | {r['sufficiency_density_loose']:.2f} | {r['topic_label']} |")
    lines += [
        "",
        "## Top 10 RE-CLUSTER candidates (sufficiency-dense but judge failed)",
        "| cluster_uid | sector | size | suff_density | judge picked vs gold | topic_label |",
        "|---|---|---:|---:|---:|---|",
    ]
    # Need to re-join for intruder picks; pull from judgments
    by_uid = {j["cluster_uid"]: j for j in judgments if "error" not in j}
    for _, r in recluster.iterrows():
        j = by_uid[r["cluster_uid"]]
        lines.append(f"| {r['cluster_uid']} | {r['sector']} | {r['cluster_size']} | {r['sufficiency_density_loose']:.2f} | {j['intruder_index']} vs gold {j['gold_intruder_index']} | {r['topic_label']} |")

    lines += [
        "",
        "## What decisions can be made from this",
        "",
        "1. **Cluster admission**: cells A and C go into the downstream sufficiency index; B and D do not.",
        "2. **Clustering hyperparameter tuning**: if quadrant C is big, the Leiden resolution is too coarse for sufficiency-relevant material. Re-run with finer resolution on the filtered subset.",
        "3. **Per-sector triage**: sectors with low judge accuracy AND high sufficiency density need attention first (most upside).",
        "4. **Sufficiency-classifier rubric**: spot-check the policies in quadrant B — if many are genuinely sufficiency-adjacent, tighten the classifier's `potential_sufficiency` definition; if not, the rubric is fine.",
        "5. **Judge trust**: judge accuracy here vs. the prior 500-item un-calibrated run (73%) tells you how much the few-shot calibration helped.",
        "",
        f"Decision threshold used: sufficiency_density_loose ≥ 0.50. Adjust by replaying `pipeline_report.py` with a different cut-off if you want stricter or looser admission.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'REPORT.md'}")
    print("\n--- headline ---")
    print(f"judge accuracy: {overall_acc:.1%}")
    print(f"A (keep)      : {A}")
    print(f"B (off-topic) : {B}")
    print(f"C (re-cluster): {C}")
    print(f"D (drop)      : {D}")


if __name__ == "__main__":
    main()
