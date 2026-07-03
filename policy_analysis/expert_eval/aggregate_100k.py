"""Aggregate v1 classifications by existing cluster.

Reads:
    runs/api_100k/to_classify.jsonl         (to join back cluster_uid)
    runs/api_100k/classifications.jsonl     (v1 outputs)

Writes:
    runs/api_100k/per_cluster_stats.parquet
    runs/api_100k/headline.txt              (sector totals + histogram)

Column set on the per-cluster table:
    cluster_uid, sector, n_sampled,
    n_sufficiency, n_ambiguous, n_efficiency, n_consistency, n_not_a_policy,
    n_suff_related  (= sufficiency + ambiguous),
    sufficiency_density_strict, sufficiency_density_loose,
    top_subcode, top_subcode_share
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

OUT = Path("runs/api_100k")


def _jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main() -> None:
    items = pd.DataFrame(_jsonl(OUT / "to_classify.jsonl"))
    preds = pd.DataFrame(_jsonl(OUT / "classifications.jsonl"))
    if "error" in preds.columns:
        fails = preds["error"].notna().sum()
    else:
        fails = 0
    preds = preds[preds.get("error").isna()] if "error" in preds.columns else preds
    print(f"loaded {len(items):,} items, {len(preds):,} classifications, {fails} failures")

    # (policy_uid, cluster_uid) is still not unique — a chunk can yield several
    # distinct policy_texts that end up in the same cluster. Assign a
    # group-local sequence on both sides so the inner-join is 1:1 within each
    # duplicate group. Since the API only sees text and returns one category
    # per call, and we're aggregating per cluster, arbitrary pairing inside a
    # cluster does not affect per-cluster counts.
    items = items.copy()
    items["_seq"] = items.groupby(["policy_uid", "cluster_uid"]).cumcount()
    preds = preds.copy()
    preds["_seq"] = preds.groupby(["policy_uid", "cluster_uid"]).cumcount()
    df = items.merge(
        preds[["policy_uid", "cluster_uid", "_seq", "category", "sub_code", "confidence"]],
        on=["policy_uid", "cluster_uid", "_seq"],
        how="inner",
    )
    print(f"joined: {len(df):,} rows")

    dummies = pd.get_dummies(df["category"]).astype(int)
    for col in ["sufficiency", "ambiguous", "efficiency", "consistency", "not_a_policy"]:
        if col not in dummies:
            dummies[col] = 0
    df = pd.concat([df, dummies.add_prefix("is_")], axis=1)

    grouped = df.groupby(["sector", "cluster_uid"])
    agg = grouped.agg(
        n_sampled=("policy_uid", "size"),
        n_sufficiency=("is_sufficiency", "sum"),
        n_ambiguous=("is_ambiguous", "sum"),
        n_efficiency=("is_efficiency", "sum"),
        n_consistency=("is_consistency", "sum"),
        n_not_a_policy=("is_not_a_policy", "sum"),
    ).reset_index()
    agg["n_suff_related"] = agg["n_sufficiency"] + agg["n_ambiguous"]
    agg["density_strict"] = agg["n_sufficiency"] / agg["n_sampled"].clip(lower=1)
    agg["density_loose"] = agg["n_suff_related"] / agg["n_sampled"].clip(lower=1)

    # dominant sub_code per cluster (sufficiency only)
    suff = df[df["category"] == "sufficiency"].copy()
    sub_summary = (
        suff.dropna(subset=["sub_code"])
        .assign(sub_code=lambda x: x["sub_code"].astype(int))
        .groupby("cluster_uid")["sub_code"]
        .agg(lambda s: (Counter(s).most_common(1)[0]) if len(s) else (None, 0))
    )
    if len(sub_summary):
        top_sub = pd.DataFrame(sub_summary.tolist(), index=sub_summary.index, columns=["top_subcode", "top_subcode_count"]).reset_index()
        agg = agg.merge(top_sub, on="cluster_uid", how="left")
        agg["top_subcode_share"] = agg["top_subcode_count"] / agg["n_sufficiency"].replace(0, pd.NA)
    else:
        agg["top_subcode"] = pd.NA
        agg["top_subcode_count"] = 0
        agg["top_subcode_share"] = pd.NA

    agg.to_parquet(OUT / "per_cluster_stats.parquet", index=False)
    print(f"wrote {OUT/'per_cluster_stats.parquet'} with {len(agg):,} cluster rows")

    per_sector = (
        df.groupby("sector")
        .agg(
            n=("policy_uid", "size"),
            n_sufficiency=("is_sufficiency", "sum"),
            n_ambiguous=("is_ambiguous", "sum"),
            n_efficiency=("is_efficiency", "sum"),
            n_consistency=("is_consistency", "sum"),
            n_not_a_policy=("is_not_a_policy", "sum"),
        )
        .assign(
            pct_sufficiency=lambda x: x["n_sufficiency"] / x["n"],
            pct_suff_related=lambda x: (x["n_sufficiency"] + x["n_ambiguous"]) / x["n"],
            pct_not_a_policy=lambda x: x["n_not_a_policy"] / x["n"],
        )
    )

    lines = ["# 100k classification aggregate", ""]
    lines.append(f"Rows: {len(df):,}   clusters with data: {len(agg):,}")
    lines.append("")
    lines.append("## Global distribution")
    dist = df["category"].value_counts()
    for k, v in dist.items():
        lines.append(f"  {k:15s}  {v:>7,}  ({v/len(df):.1%})")
    lines.append("")
    lines.append("## Per-sector")
    lines.append(per_sector.round(3).to_string())
    lines.append("")
    lines.append("## Sufficiency sub-code distribution (where category=sufficiency)")
    if len(suff):
        subd = suff.dropna(subset=["sub_code"])["sub_code"].astype(int).value_counts().sort_index()
        for k, v in subd.items():
            lines.append(f"  code {k}  {v:>6,}")
    (OUT / "headline.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'headline.txt'}")
    print("\n--- HEADLINE ---")
    print("\n".join(lines[3:]))


if __name__ == "__main__":
    main()
