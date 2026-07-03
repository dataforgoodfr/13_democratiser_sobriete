"""Score the calibration set: Claude vs gold vs DeepSeek.

Inputs (all JSONL under runs/eval_v1/calibration/):
  items_gold.jsonl        — has gold_intruder_index
  claude_labels.jsonl     — has cluster_uid, intruder_index, confidence
  judgments.jsonl         — DeepSeek output (cluster_uid, intruder_index, ...)

Writes calibration_report.md.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _accuracy(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def main() -> None:
    base = Path("runs/eval_v1/calibration")
    gold_items = {it["cluster_uid"]: it for it in _load(base / "items_gold.jsonl")}
    claude = {it["cluster_uid"]: it for it in _load(base / "claude_labels.jsonl")}
    deepseek = {it["cluster_uid"]: it for it in _load(base / "judgments.jsonl")}

    uids = sorted(set(gold_items) & set(claude) & set(deepseek))

    cd_vs_gold, ds_vs_gold, cd_vs_ds = [], [], []
    per_sector: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    rows = []
    for uid in uids:
        g = gold_items[uid]["gold_intruder_index"]
        c = claude[uid]["intruder_index"]
        d = deepseek[uid]["intruder_index"]
        sector = gold_items[uid]["sector"]
        cd_vs_gold.append((c, g))
        ds_vs_gold.append((d, g))
        cd_vs_ds.append((c, d))
        per_sector[sector]["c_g"].append((c, g))
        per_sector[sector]["d_g"].append((d, g))
        per_sector[sector]["c_d"].append((c, d))
        rows.append((uid, sector, g, c, d, claude[uid]["confidence"], deepseek[uid].get("confidence")))

    lines = [
        "# Calibration report — runs/eval_v1/calibration",
        "",
        f"n = {len(uids)} items across 11 sectors",
        "",
        "## Pairwise agreement",
        "| comparison | accuracy |",
        "|---|---:|",
        f"| Claude vs gold (task well-formed?) | {_accuracy(cd_vs_gold):.3f} |",
        f"| DeepSeek vs gold (judge accuracy) | {_accuracy(ds_vs_gold):.3f} |",
        f"| Claude vs DeepSeek (inter-rater) | {_accuracy(cd_vs_ds):.3f} |",
        f"| chance baseline | 0.200 |",
        "",
        "## Per-sector breakdown",
        "| sector | n | Claude/gold | DeepSeek/gold | Claude/DeepSeek |",
        "|---|---:|---:|---:|---:|",
    ]
    for sector in sorted(per_sector):
        m = per_sector[sector]
        lines.append(
            f"| {sector} | {len(m['c_g'])} | {_accuracy(m['c_g']):.2f} | {_accuracy(m['d_g']):.2f} | {_accuracy(m['c_d']):.2f} |"
        )

    # Where Claude AND DeepSeek both disagree with gold — these are the
    # cases where the *gold-by-construction* is probably wrong (the random
    # intruder happened to fit the cluster).
    both_wrong = [
        (uid, gold_items[uid]["sector"], g, c, d)
        for (uid, _s, g, c, d, _cc, _dc) in rows
        if c != g and d != g
    ]
    both_wrong_agree = [t for t in both_wrong if t[3] == t[4]]
    lines += [
        "",
        "## Where Claude AND DeepSeek both disagree with gold",
        f"- both disagree: {len(both_wrong)} items",
        f"- and pick the SAME alternative: {len(both_wrong_agree)} items (likely gold-by-construction failures)",
        "",
        "| cluster_uid | sector | gold | Claude | DeepSeek |",
        "|---|---|---:|---:|---:|",
    ]
    for uid, s, g, c, d in both_wrong:
        flag = " 🤝" if c == d else ""
        lines.append(f"| {uid} | {s} | {g} | {c} | {d}{flag} |")

    # Claude's own confidence vs accuracy.
    by_conf: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for uid, _s, g, c, _d, cc, _dc in rows:
        by_conf[int(cc)].append((c, g))
    lines += [
        "",
        "## Claude's accuracy by self-reported confidence",
        "| conf | n | Claude/gold |",
        "|---:|---:|---:|",
    ]
    for k in sorted(by_conf):
        lines.append(f"| {k} | {len(by_conf[k])} | {_accuracy(by_conf[k]):.2f} |")

    # DeepSeek confidence distribution.
    ds_conf = Counter(int(deepseek[u].get("confidence", 0)) for u in uids)
    lines += [
        "",
        "## DeepSeek confidence distribution",
        "| conf | n |",
        "|---:|---:|",
    ]
    for k in sorted(ds_conf):
        lines.append(f"| {k} | {ds_conf[k]} |")

    out = base.parent / "calibration_report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n" + "\n".join(lines[:14]))


if __name__ == "__main__":
    main()
