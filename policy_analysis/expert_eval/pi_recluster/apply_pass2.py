"""Apply pass-2 verdicts to the shipped clustering.

Consumes classifications_pass2.jsonl (from classify_vllm.py with
PROMPTS_VERSION=pass2) and the clustered parquets, and emits:

  - reclustered_pass2/*.parquet — rows whose pass-2 category is still
    `sufficiency` (others dropped), with pass-2 sub_code recorded.
  - cluster_audit.csv — per cluster: retention rate, majority pass-2
    sub_code vs the cluster's cell code, and a flag when they disagree
    (the systemic replacement for hand-patching mislabeled clusters).

Usage:
    uv run python apply_pass2.py \
        --clustering data/reclustered \
        --pass2 data/classifications_pass2.jsonl \
        --out data/pass2_applied
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RETENTION_FLAG = 0.5  # flag clusters keeping less than half their members


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clustering", required=True)
    ap.add_argument("--pass2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "reclustered_pass2").mkdir(parents=True, exist_ok=True)

    p2 = pd.DataFrame([
        json.loads(l) for l in Path(args.pass2).read_text().splitlines() if l.strip()
    ])
    p2 = p2[~p2.get("error").notna()] if "error" in p2.columns else p2
    p2 = p2.rename(columns={"category": "pass2_category", "sub_code": "pass2_sub_code"})
    p2 = p2.drop_duplicates(subset=["policy_uid", "cluster_uid"], keep="last")
    print(f"pass-2 verdicts: {len(p2):,}")
    print(p2["pass2_category"].value_counts().to_string())

    audits = []
    total_in = total_kept = 0
    for p in sorted(Path(args.clustering).glob("*.parquet")):
        d = pd.read_parquet(p)
        d["policy_uid"] = d["openalex_id"].astype(str) + "::" + d["chunk_idx"].astype(str)
        d = d.merge(
            p2[["policy_uid", "cluster_uid", "pass2_category", "pass2_sub_code"]],
            left_on=["policy_uid", "new_cluster_uid"],
            right_on=["policy_uid", "cluster_uid"],
            how="left",
        ).drop(columns=["cluster_uid"])

        # rows without a pass-2 verdict (e.g. singletons excluded from the
        # replay) are kept as-is
        keep_mask = d["pass2_category"].isna() | (d["pass2_category"] == "sufficiency")
        kept = d[keep_mask].drop(columns=["policy_uid"])
        kept.to_parquet(out_dir / "reclustered_pass2" / p.name, index=False)
        total_in += len(d)
        total_kept += len(kept)

        judged = d[d["pass2_category"].notna()]
        for uid, g in judged.groupby("new_cluster_uid"):
            cell_code = g["sub_code"].mode().iat[0] if g["sub_code"].notna().any() else None
            suff = g[g["pass2_category"] == "sufficiency"]
            maj = suff["pass2_sub_code"].mode()
            maj_code = int(maj.iat[0]) if len(maj) and pd.notna(maj.iat[0]) else None
            retention = len(suff) / len(g)
            audits.append({
                "cluster_uid": uid,
                "sector": g["sector"].iat[0],
                "size": len(g),
                "retention": round(retention, 3),
                "cell_sub_code": cell_code,
                "pass2_majority_sub_code": maj_code,
                "code_mismatch": maj_code is not None and cell_code is not None
                                 and maj_code != cell_code,
                "low_retention": retention < RETENTION_FLAG,
            })

    audit = pd.DataFrame(audits).sort_values(["low_retention", "retention"],
                                             ascending=[False, True])
    audit.to_csv(out_dir / "cluster_audit.csv", index=False)

    print(f"\nrows: {total_in:,} -> {total_kept:,} kept "
          f"({total_kept / total_in:.1%} retention)")
    print(f"clusters audited: {len(audit):,}")
    print(f"  low retention (<{RETENTION_FLAG:.0%}):   {int(audit['low_retention'].sum()):,}")
    print(f"  sub-code mismatch vs cell: {int(audit['code_mismatch'].sum()):,}")
    print(f"audit: {out_dir / 'cluster_audit.csv'}")


if __name__ == "__main__":
    main()
