"""
Catalog Builder
===============
Builds two output CSV files from the new-pipeline indicator CSVs.

Output files (written to output/catalog/)
------------------------------------------
catalog_complete.csv
    All EU-27 countries × all indicators × all available years × all deciles.
    Same column layout as the legacy master_indicator_catalog.csv so it is a
    drop-in replacement.

catalog_ewbi_clean.csv
    "Minimal clean" dataset for index computation.
    For each (indicator, country) the LATEST year that passes ALL of:
      • All 10 income-decile values present (no NaN)
      • All 10 income-decile values > 0   (no zero values)
    Only decile rows (D1–D10) are included; the "All" aggregate is excluded.

Output columns (both files)
----------------------------
  code             – indicator code (e.g. "HQ-SILC-1")
  name             – indicator name in English
  source           – data source ("EU-SILC" | "HBS" | "LFS" | "EHIS")
  code_source      – raw Eurostat variable code(s) used
  country          – ISO 2-letter country code
  year             – reference year (int)
  decile           – income decile (1–10) or "All"
  value            – share of population (%) meeting the condition
  n_obs            – unweighted observation count
  n_weighted       – sum of weights
  already_included – "Y" if indicator is in the published EWBI set, else "N"
  level            – "household" or "personal"
  threshold        – human-readable flag condition
  indicator_group  – thematic dimension

Usage
-----
    python build_catalog.py

    # Only rebuild one output:
    python build_catalog.py --complete
    python build_catalog.py --clean
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Local imports ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import config
from indicators import BY_CODE

# ── Output directory ──────────────────────────────────────────────────────────
CATALOG_DIR = _HERE / "output" / "catalog"

# ── Column order matching master_indicator_catalog.csv ───────────────────────
OUTPUT_COLS = [
    "code", "name", "source", "code_source",
    "country", "year", "decile", "value", "n_obs", "n_weighted",
    "already_included", "level", "threshold", "indicator_group",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _meta_row(code: str) -> dict:
    """Return the metadata dict for one indicator code."""
    m = BY_CODE.get(code, {})
    return {
        "name":             m.get("name", code),
        "source":           m.get("source", ""),
        "code_source":      m.get("variable", ""),
        "already_included": "Y" if m.get("in_ewbi") else "N",
        "level":            m.get("level", ""),
        "threshold":        m.get("condition", ""),
        "indicator_group":  m.get("dimension", ""),
    }


def _load_indicator(csv_path: Path) -> pd.DataFrame:
    """Read one indicator CSV and return it (may be empty)."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()
    if df.empty or "country" not in df.columns:
        return pd.DataFrame()
    return df


def _load_all(eu27: list[str]) -> pd.DataFrame:
    """
    Load every indicator CSV from INDICATORS_DIR, attach metadata,
    filter to EU-27, and return a single concatenated DataFrame.
    """
    parts: list[pd.DataFrame] = []
    csv_files = sorted(config.INDICATORS_DIR.glob("*.csv"))

    for csv_path in csv_files:
        code = csv_path.stem
        df = _load_indicator(csv_path)
        if df.empty:
            continue

        # Filter to EU-27
        df = df[df["country"].isin(eu27)].copy()
        if df.empty:
            continue

        meta = _meta_row(code)
        df.insert(0, "code", code)
        for col, val in meta.items():
            df[col] = val

        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLS)

    combined = pd.concat(parts, ignore_index=True)

    # Ensure all expected columns exist
    for col in OUTPUT_COLS:
        if col not in combined.columns:
            combined[col] = ""

    return combined[OUTPUT_COLS].copy()


# ─── Output 1: Complete catalog ───────────────────────────────────────────────

def build_complete(eu27: list[str]) -> pd.DataFrame:
    """Return the full dataset — all countries, years, deciles."""
    print("Building complete catalog …")
    df = _load_all(eu27)
    # Sort for readability
    df = df.sort_values(["code", "country", "year", "decile"],
                        key=lambda s: s.astype(str)).reset_index(drop=True)
    print(f"  {len(df):,} rows — {df['code'].nunique()} indicators, "
          f"{df['country'].nunique()} countries")
    return df


# ─── Output 2: Clean dataset ─────────────────────────────────────────────────

def build_ewbi_clean(eu27: list[str]) -> pd.DataFrame:
    """
    Return ALL (indicator × country × year) observations that pass:
      • All 10 decile values present — no NaN, no partial NaN
      • Not all 10 decile values are zero

    The "All" aggregate row is kept for qualifying years.
    Only EU-27 countries. All clean years are retained (not just the latest).
    """
    print("Building EWBI clean dataset …")
    all_df = _load_all(eu27)

    if all_df.empty:
        return all_df

    all_df["value"] = pd.to_numeric(all_df["value"], errors="coerce")
    all_df["year"]  = pd.to_numeric(all_df["year"],  errors="coerce")

    # Identify clean (code, country, year) based on D1–D10 rows only
    decile_df = all_df[
        all_df["decile"].astype(str).isin([str(d) for d in range(1, 11)])
    ].copy()
    decile_df["decile"] = pd.to_numeric(decile_df["decile"], errors="coerce")

    grp = decile_df.groupby(["code", "country", "year"])["value"]

    n_present = grp.apply(lambda s: s.notna().sum())   # must equal 10
    sum_vals  = grp.apply(lambda s: s.fillna(0).sum()) # must be != 0

    cky = pd.DataFrame({"n_present": n_present, "sum_vals": sum_vals}).reset_index()
    clean_keys = cky[(cky["n_present"] == 10) & (cky["sum_vals"] != 0)][
        ["code", "country", "year"]
    ]

    if clean_keys.empty:
        print("  WARNING: no clean (code, country, year) triples found.")
        return pd.DataFrame(columns=OUTPUT_COLS)

    # Keep ALL rows (deciles + "All" aggregate) for qualifying years
    result = all_df.merge(clean_keys, on=["code", "country", "year"], how="inner")

    for col in OUTPUT_COLS:
        if col not in result.columns:
            result[col] = ""

    result = (result[OUTPUT_COLS]
              .sort_values(["code", "country", "year", "decile"],
                           key=lambda s: s.astype(str))
              .reset_index(drop=True))

    n_pairs = clean_keys.groupby(["code", "country"]).ngroups
    n_ky    = len(clean_keys)
    print(f"  {len(result):,} rows — {n_ky:,} clean (indicator × country × year) "
          f"combinations, {n_pairs:,} unique (indicator × country) pairs")
    return result


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build indicator catalog CSVs from new-pipeline output"
    )
    parser.add_argument("--complete", action="store_true",
                        help="Build catalog_complete.csv only")
    parser.add_argument("--clean", action="store_true",
                        help="Build catalog_ewbi_clean.csv only")
    args = parser.parse_args()

    nothing_selected = not args.complete and not args.clean
    do_complete = args.complete or nothing_selected
    do_clean    = args.clean    or nothing_selected

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    eu27 = config.EU27

    if do_complete:
        df_complete = build_complete(eu27)
        out = CATALOG_DIR / "catalog_complete.csv"
        df_complete.to_csv(out, index=False)
        print(f"  Saved → {out}")

    if do_clean:
        df_clean = build_ewbi_clean(eu27)
        out = CATALOG_DIR / "catalog_clean.csv"
        df_clean.to_csv(out, index=False)
        print(f"  Saved → {out}")
