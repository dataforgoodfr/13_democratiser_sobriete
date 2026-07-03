"""Aggregate judgments + intrinsic metrics into a markdown report."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path


def _accuracy(judgments: list[dict]) -> float:
    if not judgments:
        return float("nan")
    correct = sum(1 for j in judgments if j["intruder_index"] == j["gold_intruder_index"])
    return correct / len(judgments)


def _weighted_accuracy(judgments: list[dict]) -> float:
    """Accuracy weighted by judge's self-reported confidence (1..5)."""
    if not judgments:
        return float("nan")
    num = sum(
        j["confidence"] * (1 if j["intruder_index"] == j["gold_intruder_index"] else 0)
        for j in judgments
    )
    den = sum(j["confidence"] for j in judgments)
    return num / den if den else float("nan")


def aggregate(
    judgments: list[dict],
    intrinsic: dict | None = None,
) -> dict:
    # Dedup by cluster_uid keeping the latest record (so a successful retry
    # overrides an earlier error line in the same file).
    latest: dict[str, dict] = {}
    for j in judgments:
        uid = j.get("cluster_uid")
        if uid:
            latest[uid] = j
    judgments = list(latest.values())

    errors = [j for j in judgments if "error" in j]
    valid = [j for j in judgments if "error" not in j]

    overall = {
        "n_total": len(judgments),
        "n_valid": len(valid),
        "n_errors": len(errors),
        "error_rate": len(errors) / len(judgments) if judgments else float("nan"),
        "accuracy": _accuracy(valid),
        "accuracy_weighted_by_confidence": _weighted_accuracy(valid),
        "mean_specificity": (
            sum(j["specificity"] for j in valid) / len(valid)
            if valid else float("nan")
        ),
        "mean_confidence": (
            sum(j["confidence"] for j in valid) / len(valid)
            if valid else float("nan")
        ),
    }

    by_sector: dict[str, dict] = {}
    sectors = defaultdict(list)
    for j in valid:
        sectors[j["sector"]].append(j)
    for sector, js in sorted(sectors.items()):
        by_sector[sector] = {
            "n": len(js),
            "accuracy": _accuracy(js),
            "mean_specificity": sum(j["specificity"] for j in js) / len(js),
        }

    by_bucket: dict[str, dict] = {}
    buckets = defaultdict(list)
    for j in valid:
        buckets[j["size_bucket"]].append(j)
    for bucket, js in sorted(buckets.items()):
        by_bucket[bucket] = {
            "n": len(js),
            "accuracy": _accuracy(js),
        }

    topic_labels = Counter(j["topic_label"].strip().lower() for j in valid)

    prompt_versions = sorted({j.get("prompt_version", "?") for j in valid})

    error_examples = [
        {"cluster_uid": j["cluster_uid"], "error": j["error"]}
        for j in errors[:5]
    ]

    return {
        "prompt_versions": prompt_versions,
        "overall": overall,
        "by_sector": by_sector,
        "by_size_bucket": by_bucket,
        "topic_label_top20": topic_labels.most_common(20),
        "error_examples": error_examples,
        "intrinsic": intrinsic,
    }


def render_markdown(summary: dict) -> str:
    o = summary["overall"]
    lines = [
        "# Clustering Evaluation Report",
        "",
        f"Prompt version(s): {', '.join(summary['prompt_versions'])}",
        "",
        "## Intrusion task — overall",
        f"- n total = {o['n_total']}",
        f"- n valid = {o['n_valid']}  ({o['n_errors']} errors, {o['error_rate']:.1%})",
        f"- accuracy = {o['accuracy']:.3f}  (chance = 0.20)",
        f"- accuracy weighted by judge confidence = {o['accuracy_weighted_by_confidence']:.3f}",
        f"- mean specificity (1-5) = {o['mean_specificity']:.2f}",
        f"- mean confidence (1-5) = {o['mean_confidence']:.2f}",
        "",
        "## By sector",
        "| sector | n | accuracy | mean specificity |",
        "|---|---:|---:|---:|",
    ]
    for sector, s in summary["by_sector"].items():
        lines.append(f"| {sector} | {s['n']} | {s['accuracy']:.3f} | {s['mean_specificity']:.2f} |")
    lines += [
        "",
        "## By cluster-size bucket",
        "| bucket | n | accuracy |",
        "|---|---:|---:|",
    ]
    for bucket, s in summary["by_size_bucket"].items():
        lines.append(f"| {bucket} | {s['n']} | {s['accuracy']:.3f} |")
    lines += [
        "",
        "## Top topic labels (judge's free-form, lowercased)",
    ]
    for label, count in summary["topic_label_top20"]:
        lines.append(f"- {count:>3}  {label}")

    if summary.get("error_examples"):
        lines += ["", "## Error examples (first 5)"]
        for e in summary["error_examples"]:
            lines.append(f"- {e['cluster_uid']}: {e['error']}")

    intr = summary.get("intrinsic")
    if intr:
        ov = intr["overall"]
        lines += [
            "",
            "## Intrinsic metrics",
            f"- n_clusters: {ov['n_clusters']}",
            f"- n_items: {ov['n_items']:,}",
            f"- cluster size: min={ov['min']} median={ov['median']:.0f} p90={ov['p90']:.0f} max={ov['max']}",
            f"- largest_cluster_fraction: {ov['largest_cluster_fraction']:.4f}",
            f"- small clusters (size <= 5): {ov['small_cluster_count_le_5']}",
            "",
            "### Per-sector cluster counts and median size",
            "| sector | n_clusters | median size | max size |",
            "|---|---:|---:|---:|",
        ]
        for sector, m in intr["per_sector"].items():
            lines.append(f"| {sector} | {m['n_clusters']} | {m['median']:.0f} | {m['max']} |")
    return "\n".join(lines) + "\n"


def write_report(
    judgments_path: str | Path,
    intrinsic_path: str | Path | None,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    judgments_path = Path(judgments_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    judgments = [
        json.loads(line) for line in judgments_path.read_text().splitlines() if line.strip()
    ]
    intrinsic = None
    if intrinsic_path:
        intrinsic = json.loads(Path(intrinsic_path).read_text())

    summary = aggregate(judgments, intrinsic)

    summary_path = out_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    report_path = out_dir / "REPORT.md"
    report_path.write_text(render_markdown(summary))
    return summary_path, report_path
