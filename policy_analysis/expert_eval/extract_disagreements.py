"""Extract the sufficiency-on-not_a_policy disagreement set for expert review.

Reads:
    runs/expert_gold/joined_v1.parquet

Writes:
    runs/expert_gold/disagreements_review.csv
        - one row per case where v1 said `sufficiency` but expert said `not_a_policy`
        - columns include everything a reviewer needs plus two empty columns
          (`expert_agrees_now`, `notes`) for the reviewer to fill in place.

Sorted by v1_confidence descending — highest-confidence model calls first,
because those are either (a) legitimate labeling disagreements the experts
may want to flip, or (b) the most instructive false-positives.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("runs/expert_gold")


def main() -> None:
    df = pd.read_parquet(OUT / "joined_v1.parquet")
    mask = (df["category"] == "not_a_policy") & (df["v1_category"] == "sufficiency")
    d = df.loc[mask].copy()
    d = d.sort_values(["v1_confidence", "sector", "policy_id"], ascending=[False, True, True])

    out_cols = [
        "policy_id",
        "sector",
        "source_cluster_id",
        "policy_text",
        "v1_category",
        "v1_sub_code",
        "v1_confidence",
        "v1_reasoning",
        "policy_categorisation_raw",
    ]
    out = d[out_cols].rename(
        columns={
            "policy_categorisation_raw": "expert_original_label",
            "v1_category": "model_category",
            "v1_sub_code": "model_sub_code",
            "v1_confidence": "model_confidence",
            "v1_reasoning": "model_reasoning",
        }
    )
    out["expert_agrees_now"] = ""   # reviewer to fill: y / n
    out["notes"] = ""

    path = OUT / "disagreements_review.csv"
    out.to_csv(path, index=False)

    n_by_sector = d.groupby("sector").size().sort_values(ascending=False)
    n_by_subcode = d["v1_sub_code"].value_counts().sort_index()

    print(f"wrote {path}  ({len(out)} rows)")
    print("\nby sector:")
    print(n_by_sector.to_string())
    print("\nby v1 sub_code:")
    print(n_by_subcode.to_string())


if __name__ == "__main__":
    main()
