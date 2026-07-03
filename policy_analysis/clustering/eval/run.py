"""End-to-end driver for the v0 clustering evaluation.

Chains: load_clusters -> compute_intrinsic -> build_intrusion_items -> (LLM judge) -> report.

Usage:
  python -m policy_analysis.clustering.eval.run \\
      --data ~/data/wsl_sufficiency_eval/clusters_2026-03-18 \\
      --out runs/eval_v0 \\
      --n-items 50 \\
      --seed 0
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from .loader import load_clusters
from .intrinsic import compute_intrinsic, save_intrinsic
from .sample import build_intrusion_items, save_items
from .llm_judge import run_judge
from .report import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="folder of per-sector parquets")
    parser.add_argument("--out", required=True, help="output run folder (created)")
    parser.add_argument("--n-items", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-judge", action="store_true",
                        help="build items + intrinsic + items file, skip LLM calls")
    args = parser.parse_args()

    data = os.path.expanduser(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading clusters from {data}")
    df = load_clusters(data)
    print(f"      {len(df):,} rows, {df['cluster_uid'].nunique()} clusters, "
          f"{df['sector'].nunique()} sectors")

    print(f"[2/5] computing intrinsic metrics")
    intrinsic = compute_intrinsic(df)
    intrinsic_path = save_intrinsic(intrinsic, out)
    print(f"      wrote {intrinsic_path}")

    print(f"[3/5] sampling {args.n_items} intrusion items (seed={args.seed})")
    items = build_intrusion_items(df, n=args.n_items, seed=args.seed)
    items_path = save_items(items, out / "intrusion_items.jsonl")
    print(f"      wrote {items_path}")

    judgments_path = out / "judgments.jsonl"
    if args.skip_judge:
        print(f"[4/5] skipping judge (--skip-judge); no judgments to aggregate")
        print(f"[5/5] skipping report")
        return

    print(f"[4/5] running LLM judge -> {judgments_path}")
    run_judge(items_path, judgments_path)

    print(f"[5/5] writing report")
    summary_path, report_path = write_report(judgments_path, intrinsic_path, out)
    print(f"      wrote {summary_path}")
    print(f"      wrote {report_path}")


if __name__ == "__main__":
    main()
