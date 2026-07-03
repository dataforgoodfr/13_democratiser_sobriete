"""Read expert-labelled Excel and emit a clean gold dataset.

Input:
    policy_analysis/clustering/2026-06-10_Clustering ground truth.xlsx

Outputs:
    runs/expert_gold/gold.parquet
    runs/expert_gold/gold.jsonl

Normalisations applied:
- Two typo variants in `Sufficiency Categories` collapsed:
    "3-Reduce & right size"                    -> "3- Reduce & right size"
    "5 Proximity, compactness & land sufficiency"
                                               -> "5- Proximity, compactness & land sufficiency"
- Whitespace stripped on all string columns.
- `sufficiency_code` extracted as the leading integer 0..9 (nullable Int64).
- `category` reduced to a lowercase-slug column drawn from Policy categorisation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import openpyxl
import pandas as pd

XLSX = Path("policy_analysis/clustering/2026-06-10_Clustering ground truth.xlsx")
OUT = Path("runs/expert_gold")

SUB_TAXO_RENAME = {
    "3-Reduce & right size": "3- Reduce & right size",
    "5 Proximity, compactness & land sufficiency": (
        "5- Proximity, compactness & land sufficiency"
    ),
}

CATEGORY_SLUG = {
    "Not a policy": "not_a_policy",
    "Sufficiency": "sufficiency",
    "Efficiency": "efficiency",
    "Consistency": "consistency",
    "Ambiguous": "ambiguous",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.load_workbook(XLSX, data_only=True, read_only=True)
    frames = []
    for sheet in wb.sheetnames:
        if sheet == "_instructions":
            continue
        ws = wb[sheet]
        rows = list(ws.iter_rows(values_only=True))
        header = rows[0]
        data = [dict(zip(header, r)) for r in rows[1:] if any(c is not None for c in r)]
        df = pd.DataFrame(data)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    for col in ("Policy categorisation", "Sufficiency Categories", "policy_text"):
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

    df["Sufficiency Categories"] = df["Sufficiency Categories"].replace(SUB_TAXO_RENAME)

    df["category"] = df["Policy categorisation"].map(CATEGORY_SLUG)
    unknown = df.loc[df["category"].isna(), "Policy categorisation"].unique().tolist()
    if unknown:
        raise ValueError(f"Unmapped Policy categorisation values: {unknown}")

    def _extract_code(value: object) -> int | None:
        if not isinstance(value, str):
            return None
        m = re.match(r"^(\d+)", value)
        return int(m.group(1)) if m else None

    df["sufficiency_code"] = df["Sufficiency Categories"].map(_extract_code).astype("Int64")

    df = df.rename(
        columns={
            "Sufficiency Categories": "sufficiency_label",
            "Policy categorisation": "policy_categorisation_raw",
        }
    )

    keep = [
        "policy_id",
        "sector",
        "source_cluster_id",
        "within_cluster_role",
        "sample_kind",
        "policy_text",
        "policy_categorisation_raw",
        "category",
        "sufficiency_label",
        "sufficiency_code",
    ]
    df = df[keep].reset_index(drop=True)

    df.to_parquet(OUT / "gold.parquet", index=False)
    with (OUT / "gold.jsonl").open("w") as f:
        for row in df.to_dict(orient="records"):
            for k, v in list(row.items()):
                if pd.isna(v):
                    row[k] = None
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(df)} rows to {OUT/'gold.parquet'}")
    print("\ncategory distribution:")
    print(df["category"].value_counts())
    print("\nsufficiency_code distribution (non-null):")
    print(df["sufficiency_code"].dropna().value_counts().sort_index())
    print("\nper-sector counts:")
    print(df["sector"].value_counts())


if __name__ == "__main__":
    main()
