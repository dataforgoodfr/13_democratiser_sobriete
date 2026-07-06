"""
Quality Checker
===============
Evaluates data quality for all indicator CSVs at the (code, country, year, decile) level.

Checks performed
----------------
Per (code, country, year):
  1. ABSENT       – all 10 decile values are NaN
                    → probes raw source files to explain why
  2. NAN_PARTIAL  – some deciles NaN but not all
                    (anomalous: if a country-year is covered all deciles must be present)
                    → reports n_obs per decile to identify small-sample cause
  3. ZERO_LOW     – value = 0 for D1, D2 or D3
                    (suspicious: lowest-income HH should have highest deprivation rates)
  4. YEAR_GAP     – country had data in earlier AND later years but not this year
                    → probes raw source to confirm file missing vs. computation failure
  5. LATE_START   – country starts ≥3 years later than the median country
  6. EARLY_END    – country ends ≥3 years earlier than the median country
  7. MONOTONICITY – no income gradient: |Spearman(decile, value)| < 0.20
                    (values should move consistently D1→D10)

Outputs
-------
  output/quality/quality_summary.csv   – one row per (code, country, year)
  output/quality/quality_issues.csv    – one row per detected issue
  output/quality/quality_report.xlsx   – multi-tab Excel workbook

Usage
-----
    python quality_check.py                   # all indicators, all outputs
    python quality_check.py --csv             # CSV only
    python quality_check.py --excel           # Excel only
    python quality_check.py --code HH-HBS-1  # single indicator
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config
from indicators import BY_CODE

config.ensure_dirs()

# ─── Thresholds ───────────────────────────────────────────────────────────────
LOW_DECILES          = {1, 2, 3}      # D1-D3 zero is suspicious
MONO_MIN_VALID       = 6              # min deciles with data to run Spearman
MONO_FLAT_THRESHOLD  = 0.20           # |r| below this → no gradient
YEAR_GAP_TOLERANCE   = 0             # consecutive missing years to flag (0 = any gap)
LATE_START_THRESHOLD = 3             # years later than median start → flag
EARLY_END_THRESHOLD  = 3             # years earlier than median end → flag

# ─── Known structural explanations ────────────────────────────────────────────
# These are appended to the explanation field when the code matches.
STRUCTURAL_NOTES: dict[str, str] = {
    "HH-HBS-1":  "EUR_HE041 (actual rent) is zero for owner-occupier HH → zero possible in high-ownership countries.",
    "HH-HBS-2":  "EUR_HE041 absent if country did not collect imputed rent sub-components.",
    "EC-HBS-1":  "EUR_HJ08 (communications) not collected in HBS 2010 for many countries.",
    "EC-HBS-2":  "EUR_HJ08 (communications) not collected in HBS 2010 for many countries.",
    "TS-HBS-1":  "EUR_HJ90 (travel & tourism) absent in HBS 2010 for most countries.",
    "TS-HBS-2":  "EUR_HJ90 (travel & tourism) absent in HBS 2010 for most countries.",
    "IE-HBS-1":  "EUR_HE10 (education) very sparse; near-zero median causes instability.",
    "IE-HBS-2":  "EUR_HE10 (education) very sparse; near-zero median causes instability.",
    "RT-LFS-4":  "VARITIME (time flexibility) column absent in LFS before 2014.",
    "RT-LFS-5":  "SHIFTWK only asked to employees; NaN for self-employed.",
    "RT-LFS-6":  "NIGHTWK only asked to employees; NaN for self-employed.",
    "RT-LFS-7":  "SATWK only asked to employees; NaN for self-employed.",
    "RT-LFS-8":  "SUNWK only asked to employees; NaN for self-employed.",
    "SP-SILC-1": "Ad-hoc module: PS101/PS110 only present in EU-SILC 2015 and 2022.",
    "SP-SILC-2": "Ad-hoc module: PS102 only present in EU-SILC 2015 and 2022.",
    "GE-SILC-1": "Computed at country level (not per decile); decile values by construction identical.",
    "GE-SILC-2": "Computed at country level (not per decile); decile values by construction identical.",
    "AH-EHIS-1": "EHIS wave 1 (≈2007), wave 2 (≈2014), wave 3 (≈2019) only — 3 years total.",
    "AH-EHIS-2": "EHIS coverage: 3 waves only; year gaps expected between waves.",
    "AE-EHIS-1": "EHIS coverage: 3 waves only; year gaps expected between waves.",
    "AE-EHIS-2": "EHIS coverage: 3 waves only; year gaps expected between waves.",
    "AN-EHIS-1": "EHIS coverage: 3 waves only; year gaps expected between waves.",
    "AN-EHIS-2": "EHIS coverage: 3 waves only; year gaps expected between waves.",
    "AO-EHIS-1": "EHIS BMI derived variable; absent when self-reported height/weight not collected.",
    "HE-SILC-1": "HC060 introduced post-2010 in some countries; systematic gap before 2016 for some.",
    "HQ-SILC-8": "HC003 (renovation) not collected in all waves.",
}

# Expected years per source (for year-gap detection)
_SOURCE_YEARS: dict[str, list[int]] = {
    "EU-SILC": list(range(2004, 2024)),
    "HBS":     [2010, 2015, 2020],
    "LFS":     list(range(2005, 2024)),
    "EHIS":    [2007, 2014, 2019],   # approximate wave centres
}

# ─── Raw data availability probing ────────────────────────────────────────────

def _probe_silc(country: str, year: int) -> tuple[bool, str]:
    """Check if EU-SILC P and R files exist for country/year."""
    yy = str(year)[-2:]
    p = config.SILC_RAW_DIR / country / str(year) / f"UDB_c{country}{yy}P.csv"
    r = config.SILC_RAW_DIR / country / str(year) / f"UDB_c{country}{yy}R.csv"
    p_ok, r_ok = p.exists(), r.exists()
    if p_ok and r_ok:
        return True, "P+R files present"
    missing = []
    if not p_ok:
        missing.append(f"P-file missing: {p.name}")
    if not r_ok:
        missing.append(f"R-file missing: {r.name}")
    return False, "; ".join(missing)


def _probe_hbs(country: str, year: int) -> tuple[bool, str]:
    """Check if HBS parquet cache or raw Excel exists for country/wave."""
    cache = config.CACHE_DIR / "hbs" / f"HBS_{year}.parquet"
    if cache.exists():
        try:
            pq = pd.read_parquet(cache, columns=["COUNTRY"])
            if country in pq["COUNTRY"].unique():
                return True, f"HBS_{year}.parquet contains {country}"
            return False, f"HBS_{year}.parquet exists but {country} not in it"
        except Exception:
            pass
    # Fall back to raw Excel probe
    layout = {
        2010: ("HBS2010/HBS2010", f"{country}*_HBS_hh.xlsx"),
        2015: ("HBS2015/HBS2015", f"{country}*_MFR_hh.xlsx"),
        2020: ("HBS2020/HBS2020", f"HBS_HH_{country}*.xlsx"),
    }
    if year not in layout:
        return False, f"HBS wave {year} not in scope"
    subfolder, pat = layout[year]
    folder = config.HBS_RAW_DIR / subfolder
    files = list(folder.glob(pat)) if folder.exists() else []
    if files:
        return True, f"Raw Excel found: {files[0].name}"
    return False, f"No raw file matching '{pat}' in {folder}"


def _probe_lfs(country: str, year: int) -> tuple[bool, str]:
    """Check if LFS parquet cache or raw CSV exists for country/year."""
    cache = config.CACHE_DIR / "lfs" / f"LFS_{country}_{year}.parquet"
    if cache.exists():
        return True, f"LFS parquet cache present: {cache.name}"
    # Raw CSV scan
    folder = config.LFS_RAW_DIR / country / str(year)
    if not folder.exists():
        folder = config.LFS_RAW_DIR / str(year)  # alternate flat layout
    if folder.exists():
        csvs = list(folder.glob("*.csv"))
        if csvs:
            return True, f"Raw LFS CSV found in {folder}"
    return False, f"No LFS data for {country}/{year}"


def _probe_ehis(country: str, year: int) -> tuple[bool, str]:
    """Check if EHIS parquet cache or raw file exists for country near year."""
    # EHIS is wave-based; find the closest wave year
    wave_years = [2007, 2014, 2019]
    wave = min(wave_years, key=lambda y: abs(y - year))
    cache = config.CACHE_DIR / "ehis" / f"EHIS_{country}_{wave}.parquet"
    if cache.exists():
        return True, f"EHIS parquet cache present: {cache.name}"
    folder = config.EHIS_RAW_DIR / country
    if folder.exists():
        files = list(folder.glob("*.csv")) + list(folder.glob("*.sas7bdat"))
        if files:
            return True, f"Raw EHIS file found: {files[0].name}"
    return False, f"No EHIS data for {country} (wave ≈ {wave})"


def probe_raw(source: str, country: str, year: int) -> tuple[bool, str]:
    """Dispatch raw-data probe by source."""
    try:
        if source == "EU-SILC":
            return _probe_silc(country, year)
        if source == "HBS":
            return _probe_hbs(country, year)
        if source == "LFS":
            return _probe_lfs(country, year)
        if source == "EHIS":
            return _probe_ehis(country, year)
    except Exception as exc:
        return False, f"Probe error: {exc}"
    return False, f"Unknown source: {source}"


# ─── Per-indicator checks ─────────────────────────────────────────────────────

def check_nan_partial(dec_grp: pd.DataFrame, n_obs_col: bool) -> dict | None:
    """
    Return issue dict if some deciles are NaN but not all within a country-year.
    n_obs_col: whether the CSV has an 'n_obs' column for sample-size diagnosis.
    """
    values = dec_grp["value"]
    n_nan   = int(values.isna().sum())
    n_valid = int(values.notna().sum())
    if n_nan == 0 or n_valid == 0:
        return None   # either fully present or fully absent — handled elsewhere

    # Identify which deciles are NaN
    nan_deciles = sorted(dec_grp.loc[values.isna(), "decile"].astype(str).tolist())

    detail = f"NaN in decile(s): {', '.join(nan_deciles)} ({n_nan}/10 missing)"
    if n_obs_col and "n_obs" in dec_grp.columns:
        nan_obs = dec_grp.loc[values.isna(), "n_obs"]
        detail += f"; n_obs for those deciles: {list(nan_obs.fillna(0).astype(int))}"

    explanation = (
        "Partial NaN within a country-year is anomalous. "
        "Likely cause: too few raw observations in that income decile "
        "(weighted-quantile threshold falls outside the available income range, "
        "or decile contains 0 households after survey filters)."
    )
    return {"issue_type": "NAN_PARTIAL", "severity": "HIGH", "detail": detail,
            "explanation": explanation}


def check_zero_low_decile(dec_grp: pd.DataFrame) -> list[dict]:
    """Return one issue dict per low decile (D1-D3) whose value is exactly 0."""
    issues = []
    for _, row in dec_grp.iterrows():
        try:
            d = int(row["decile"])
        except (ValueError, TypeError):
            continue
        if d in LOW_DECILES and row["value"] == 0.0:
            issues.append({
                "issue_type": "ZERO_LOW_DECILE",
                "severity":   "MEDIUM",
                "detail":     f"D{d} = 0.0 % (0 households flagged in lowest income group)",
                "explanation": (
                    f"Decile {d} (lowest income) has 0% rate. "
                    "For a deprivation indicator, the poorest households should show the "
                    "highest rate. Zero could indicate: (a) the source variable was not "
                    "collected for these households, (b) the threshold is too extreme for "
                    "this country-year, or (c) a data-entry / coding issue in the raw file."
                ),
            })
    return issues


def check_monotonicity(dec_grp: pd.DataFrame) -> dict | None:
    """
    Check for an income gradient using Spearman rank correlation.
    Expects values to move consistently from D1 to D10 (either up or down).
    Flags if |r| < MONO_FLAT_THRESHOLD (no gradient detectable).
    """
    sub = dec_grp.copy()
    try:
        sub["decile_n"] = pd.to_numeric(sub["decile"], errors="coerce")
    except Exception:
        return None
    sub = sub.dropna(subset=["decile_n", "value"])
    if len(sub) < MONO_MIN_VALID:
        return None
    # Skip if values are constant — Spearman is undefined and gradient is trivially flat
    if sub["value"].nunique() <= 1:
        return {
            "issue_type":  "MONOTONICITY",
            "severity":    "LOW",
            "detail":      f"All {len(sub)} valid decile values are identical (constant input)",
            "explanation": (
                "All decile values are the same constant. "
                "No income gradient exists by construction. "
                "Possible causes: (a) indicator is computed at country level and replicated "
                "across deciles (e.g. GE-SILC-*), (b) 100%/0% saturation in every decile."
            ),
        }
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = spearmanr(sub["decile_n"].values, sub["value"].values)
    if np.isnan(r):
        return None
    if abs(r) < MONO_FLAT_THRESHOLD:
        return {
            "issue_type":  "MONOTONICITY",
            "severity":    "LOW",
            "detail":      f"Spearman r = {r:.3f} (p={p:.3f}); n_valid = {len(sub)}",
            "explanation": (
                f"|r|={abs(r):.3f} < {MONO_FLAT_THRESHOLD} threshold. "
                "No clear income gradient from D1 to D10. "
                "Possible causes: (a) indicator is not income-correlated in this country, "
                "(b) small sample sizes cause noisy decile estimates, "
                "(c) source variable has country-specific coding that breaks the expected gradient."
            ),
        }
    return None


# ─── Year-coverage checks ─────────────────────────────────────────────────────

def check_year_coverage(code: str, df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    For each country, detect:
      - YEAR_GAP:    missing year between first and last year with data
      - LATE_START:  country starts ≥ threshold years after median country start
      - EARLY_END:   country ends ≥ threshold years before median country end

    Returns a DataFrame of year-coverage issues.
    """
    dec_df = df[df["decile"] != "All"].copy()
    dec_df["value"] = pd.to_numeric(dec_df["value"], errors="coerce")
    # Only consider country-years with at least 1 valid value
    valid_cy = (
        dec_df.groupby(["country", "year"])["value"]
        .apply(lambda s: s.notna().any())
        .reset_index(name="has_data")
    )
    valid_cy = valid_cy[valid_cy["has_data"]]

    if valid_cy.empty:
        return pd.DataFrame()

    rows = []
    country_spans = (
        valid_cy.groupby("country")["year"]
        .agg(["min", "max", list])
        .rename(columns={"min": "first_year", "max": "last_year", "list": "years"})
    )

    # Median start / end across countries
    med_start = int(country_spans["first_year"].median())
    med_end   = int(country_spans["last_year"].median())

    for country, span in country_spans.iterrows():
        present = set(span["years"])
        first_y, last_y = int(span["first_year"]), int(span["last_year"])

        # LATE_START
        if first_y - med_start >= LATE_START_THRESHOLD:
            raw_ok, raw_msg = probe_raw(source, country, med_start)
            rows.append({
                "code": code, "country": country,
                "year": first_y, "issue_type": "LATE_START",
                "severity": "MEDIUM",
                "detail": f"First year = {first_y}; median country starts at {med_start}",
                "raw_file_present": raw_ok,
                "raw_probe_detail": raw_msg,
                "explanation": (
                    f"{country} starts {first_y - med_start} years later than median. "
                    f"Raw probe for {country}/{med_start}: {raw_msg}."
                ),
            })

        # EARLY_END
        if med_end - last_y >= EARLY_END_THRESHOLD:
            raw_ok, raw_msg = probe_raw(source, country, med_end)
            rows.append({
                "code": code, "country": country,
                "year": last_y, "issue_type": "EARLY_END",
                "severity": "MEDIUM",
                "detail": f"Last year = {last_y}; median country ends at {med_end}",
                "raw_file_present": raw_ok,
                "raw_probe_detail": raw_msg,
                "explanation": (
                    f"{country} ends {med_end - last_y} years earlier than median. "
                    f"Raw probe for {country}/{med_end}: {raw_msg}."
                ),
            })

        # YEAR_GAP — interior gaps between first and last year
        expected_interior = set(range(first_y, last_y + 1))
        # For non-annual sources, restrict to expected years for that source
        source_exp = set(_SOURCE_YEARS.get(source, []))
        if source_exp:
            expected_interior &= source_exp
        missing_years = sorted(expected_interior - present)
        for y in missing_years:
            raw_ok, raw_msg = probe_raw(source, country, y)
            rows.append({
                "code": code, "country": country,
                "year": y, "issue_type": "YEAR_GAP",
                "severity": "MEDIUM",
                "detail": (
                    f"Year {y} missing; country has data {first_y}–{last_y} "
                    f"(present years: {sorted(present)})"
                ),
                "raw_file_present": raw_ok,
                "raw_probe_detail": raw_msg,
                "explanation": (
                    f"{country}/{y}: no data despite coverage in adjacent years. "
                    f"Raw probe: {raw_msg}. "
                    + (STRUCTURAL_NOTES.get(code, ""))
                ),
            })

    return pd.DataFrame(rows)


# ─── Main runner ──────────────────────────────────────────────────────────────

def load_indicator_csvs(code_filter: str | None = None) -> dict[str, pd.DataFrame]:
    """Load all (or one) indicator CSV(s) from INDICATORS_DIR."""
    data: dict[str, pd.DataFrame] = {}
    if code_filter:
        f = config.INDICATORS_DIR / f"{code_filter}.csv"
        if not f.exists():
            print(f"  [ERROR] {f} not found.")
            return data
        files = [f]
    else:
        files = sorted(config.INDICATORS_DIR.glob("*.csv"))
    if not files:
        print(f"  No CSV files found in {config.INDICATORS_DIR}")
    for f in files:
        code = f.stem
        try:
            data[code] = pd.read_csv(f)
        except Exception as exc:
            print(f"  [WARN] {f.name}: {exc}")
    return data


def run_quality_check(code_filter: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all quality checks. Returns (summary_df, issues_df).
      summary_df : one row per (code, country, year) — aggregated status
      issues_df  : one row per issue detected
    """
    all_data = load_indicator_csvs(code_filter)
    if not all_data:
        print("No indicator data found. Run the extractors first.")
        return pd.DataFrame(), pd.DataFrame()

    summary_rows: list[dict] = []
    issue_rows:   list[dict] = []

    for code, df in sorted(all_data.items()):
        meta      = BY_CODE.get(code, {})
        full_name = meta.get("name", code)
        source    = meta.get("source", "")
        in_ewbi   = meta.get("in_ewbi", False)
        struct    = STRUCTURAL_NOTES.get(code, "")
        has_n_obs = "n_obs" in df.columns

        # ── Numeric coercion ──────────────────────────────────────────────────
        df = df.copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["decile"] = df["decile"].astype(str)

        dec_df = df[df["decile"] != "All"].copy()

        # ── Per (country, year) checks ────────────────────────────────────────
        for (country, year), grp in dec_df.groupby(["country", "year"]):
            year = int(year)
            grp = grp.copy()

            n_dec   = len(grp)
            n_valid = int(grp["value"].notna().sum())
            n_zero  = int((grp["value"] == 0).sum())
            n_nan   = int(grp["value"].isna().sum())

            # Collect issues for this (code, country, year)
            cy_issues: list[dict] = []

            # 1. ABSENT — all deciles NaN
            if n_valid == 0:
                raw_ok, raw_msg = probe_raw(source, country, year)
                explanation = (
                    f"All decile values are NaN — no source data extracted. "
                    f"Raw probe: {raw_msg}. "
                    + struct
                )
                cy_issues.append({
                    "issue_type": "ABSENT",
                    "severity":   "HIGH",
                    "detail":     f"0/{n_dec} deciles have a value",
                    "raw_file_present": raw_ok,
                    "raw_probe_detail": raw_msg,
                    "explanation": explanation,
                })

            else:
                # 2. NAN_PARTIAL — some deciles NaN
                partial = check_nan_partial(grp, has_n_obs)
                if partial:
                    raw_ok, raw_msg = probe_raw(source, country, year)
                    partial["raw_file_present"] = raw_ok
                    partial["raw_probe_detail"] = raw_msg
                    partial["explanation"] += f" Raw probe: {raw_msg}."
                    cy_issues.append(partial)

                # 3. ZERO_LOW_DECILE — D1/D2/D3 = 0
                cy_issues.extend(check_zero_low_decile(grp))

                # 4. MONOTONICITY — no income gradient
                mono = check_monotonicity(grp)
                if mono:
                    mono["raw_file_present"] = None
                    mono["raw_probe_detail"] = ""
                    cy_issues.append(mono)

            # Determine aggregate status for summary
            if n_valid == 0:
                status = "ABSENT"
            elif n_nan > 0:
                status = "NAN_PARTIAL"
            elif n_zero == n_valid:
                status = "ZERO_ALL"
            elif n_zero > 0:
                status = "ZERO_PARTIAL"
            else:
                status = "OK"

            # Collect all issue types for summary
            issue_types = list({i["issue_type"] for i in cy_issues})

            summary_rows.append({
                "code":           code,
                "indicator_name": full_name,
                "source":         source,
                "in_ewbi":        in_ewbi,
                "country":        country,
                "year":           year,
                "status":         status,
                "n_deciles":      n_dec,
                "n_valid":        n_valid,
                "n_zero":         n_zero,
                "n_nan":          n_nan,
                "issues":         "; ".join(sorted(issue_types)) if issue_types else "",
                "structural_note": struct,
            })

            # Append to issue list
            for iss in cy_issues:
                issue_rows.append({
                    "code":             code,
                    "indicator_name":   full_name,
                    "source":           source,
                    "in_ewbi":          in_ewbi,
                    "country":          country,
                    "year":             year,
                    **iss,
                })

        # ── Year-coverage checks (at indicator level, across countries) ───────
        year_issues = check_year_coverage(code, df, source)
        if not year_issues.empty:
            for _, row in year_issues.iterrows():
                issue_rows.append({
                    "code":           code,
                    "indicator_name": full_name,
                    "source":         source,
                    "in_ewbi":        in_ewbi,
                    "country":        row["country"],
                    "year":           int(row["year"]),
                    "issue_type":     row["issue_type"],
                    "severity":       row["severity"],
                    "detail":         row["detail"],
                    "raw_file_present": row.get("raw_file_present"),
                    "raw_probe_detail": row.get("raw_probe_detail", ""),
                    "explanation":    row["explanation"],
                })

    summary_df = pd.DataFrame(summary_rows)
    issues_df  = pd.DataFrame(issue_rows)
    return summary_df, issues_df


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_console_report(summary: pd.DataFrame, issues: pd.DataFrame):
    """Print a compact overview to stdout."""
    if summary.empty:
        return

    print("\n══════════════════════════════════════════════════════")
    print(" Quality Check — Summary")
    print("══════════════════════════════════════════════════════")

    # Status distribution
    counts = summary["status"].value_counts()
    total  = len(summary)
    print(f"\n{'Status':<14} {'Count':>7}  {'%':>6}")
    print("-" * 32)
    for st in ["OK", "ZERO_PARTIAL", "ZERO_ALL", "NAN_PARTIAL", "ABSENT"]:
        n = counts.get(st, 0)
        print(f"  {st:<12} {n:>7}  ({n/total*100:.1f}%)")
    print(f"  {'TOTAL':<12} {total:>7}")

    # Issue type distribution
    if not issues.empty:
        ic = issues["issue_type"].value_counts()
        print(f"\n{'Issue Type':<20} {'Count':>7}")
        print("-" * 30)
        for it, cnt in ic.items():
            print(f"  {it:<18} {cnt:>7}")

    # Worst indicators by issue count
    if not issues.empty:
        worst = (
            issues[issues["issue_type"].isin(["ABSENT", "ZERO_LOW_DECILE", "NAN_PARTIAL"])]
            .groupby(["code", "indicator_name"])
            .size()
            .reset_index(name="n_issues")
            .sort_values("n_issues", ascending=False)
            .head(20)
        )
        if not worst.empty:
            print("\nTop indicators by HIGH/MEDIUM issues (ABSENT + NAN_PARTIAL + ZERO_LOW):")
            print(worst.to_string(index=False))

    # Countries with most issues
    if not issues.empty:
        by_country = (
            issues[issues["severity"] == "HIGH"]
            .groupby("country")
            .size()
            .reset_index(name="n_high")
            .sort_values("n_high", ascending=False)
            .head(10)
        )
        if not by_country.empty:
            print("\nCountries with most HIGH-severity issues:")
            print(by_country.to_string(index=False))

    print()


# ─── Gap taxonomy ────────────────────────────────────────────────────────────
# Consistent problem labels used across all country-level reports.
GAP_LABELS: dict[str, str] = {
    "ABSENT":          "All data missing",
    "NAN_PARTIAL":     "Partial data — some deciles have NaN values",
    "ZERO_ALL":        "All decile values = 0%",
    "ZERO_PARTIAL":    "Some decile values = 0%",
    "ZERO_LOW_DECILE": "Low-income decile(s) = 0% — unexpected for deprivation indicator",
    "YEAR_GAP":        "Year missing within coverage window",
    "LATE_START":      "Coverage starts late vs other countries",
    "EARLY_END":       "Coverage ends early vs other countries",
    "MONOTONICITY":    "No income gradient detected",
}


def _format_year_range(years) -> str:
    """
    Convert a collection of integers into a compact range string.
    E.g. [2004,2005,2006,2008,2009] → '2004–2006, 2008–2009'
    """
    if not years:
        return ""
    vals = sorted(set(int(y) for y in years))
    ranges: list[str] = []
    start = end = vals[0]
    for y in vals[1:]:
        if y == end + 1:
            end = y
        else:
            ranges.append(f"{start}–{end}" if start != end else str(start))
            start = end = y
    ranges.append(f"{start}–{end}" if start != end else str(start))
    return ", ".join(ranges)


def save_csv(summary: pd.DataFrame, issues: pd.DataFrame):
    config.QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    sp = config.QUALITY_DIR / "quality_summary.csv"
    ip = config.QUALITY_DIR / "quality_issues.csv"
    summary.to_csv(sp, index=False)
    issues.to_csv(ip, index=False)
    print(f"  Summary CSV → {sp}")
    print(f"  Issues  CSV → {ip}")


def save_excel(summary: pd.DataFrame, issues: pd.DataFrame):
    try:
        import xlsxwriter
    except ImportError:
        print("  xlsxwriter not installed. Skipping Excel output.")
        return

    out_path = config.QUALITY_DIR / "quality_report.xlsx"
    config.QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    STATUS_COLORS = {
        "ABSENT":       "FFCCCC",
        "NAN_PARTIAL":  "FFEEAA",
        "ZERO_ALL":     "FFDDAA",
        "ZERO_PARTIAL": "FFEEDD",
        "OK":           "DDFFDD",
    }
    ISSUE_COLORS = {
        "ABSENT":          "FFAAAA",
        "NAN_PARTIAL":     "FFEE88",
        "ZERO_LOW_DECILE": "FFD080",
        "YEAR_GAP":        "AADDFF",
        "LATE_START":      "CCDDFF",
        "EARLY_END":       "CCDDFF",
        "MONOTONICITY":    "DDDDFF",
    }
    SEV_COLORS = {"HIGH": "FF8888", "MEDIUM": "FFCC88", "LOW": "FFFFAA"}

    wb = xlsxwriter.Workbook(str(out_path))
    hdr_fmt = wb.add_format({"bold": True, "bg_color": "1F4E79",
                              "font_color": "FFFFFF", "border": 1,
                              "text_wrap": True})

    def _fmt(color: str):
        return wb.add_format({"bg_color": color, "border": 1})

    status_fmts = {s: _fmt(c) for s, c in STATUS_COLORS.items()}
    issue_fmts  = {t: _fmt(c) for t, c in ISSUE_COLORS.items()}
    sev_fmts    = {s: _fmt(c) for s, c in SEV_COLORS.items()}

    def write_df(ws, df: pd.DataFrame, color_col: str | None = None,
                 color_map: dict | None = None, col_widths: dict | None = None):
        cols = list(df.columns)
        for j, c in enumerate(cols):
            ws.write(0, j, c, hdr_fmt)
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            color_key = str(row.get(color_col, "")) if color_col else None
            row_fmt   = color_map.get(color_key) if (color_map and color_key) else None
            for j, c in enumerate(cols):
                v = row[c]
                if isinstance(v, (np.integer,)):  v = int(v)
                elif isinstance(v, (np.floating,)): v = None if np.isnan(v) else float(v)
                elif isinstance(v, float) and np.isnan(v): v = None
                ws.write(i, j, v, row_fmt)
        # Auto-widths
        if col_widths:
            for col_name, width in col_widths.items():
                if col_name in cols:
                    ws.set_column(cols.index(col_name), cols.index(col_name), width)

    # ── Sheet 1: Summary ──────────────────────────────────────────────────────
    ws_sum = wb.add_worksheet("Summary")
    write_df(ws_sum, summary.sort_values(["code", "country", "year"]),
             color_col="status", color_map=status_fmts,
             col_widths={"code": 14, "indicator_name": 44, "country": 8,
                         "year": 6, "structural_note": 60})

    # ── Sheet 2: All Issues ───────────────────────────────────────────────────
    ws_all = wb.add_worksheet("All Issues")
    if not issues.empty:
        write_df(ws_all,
                 issues.sort_values(["severity", "issue_type", "code", "country", "year"]),
                 color_col="issue_type", color_map=issue_fmts,
                 col_widths={"code": 14, "indicator_name": 40, "country": 8,
                             "detail": 55, "explanation": 80, "raw_probe_detail": 50})

    # ── Sheet 3: ABSENT (with raw probe) ──────────────────────────────────────
    ws_absent = wb.add_worksheet("ABSENT")
    absent = issues[issues["issue_type"] == "ABSENT"].copy() if not issues.empty else pd.DataFrame()
    if not absent.empty:
        write_df(ws_absent, absent.sort_values(["code", "country", "year"]),
                 color_col="raw_file_present",
                 color_map={True: _fmt("CCFFCC"), False: _fmt("FFAAAA")},
                 col_widths={"code": 14, "indicator_name": 40, "explanation": 80})

    # ── Sheet 4: NAN_PARTIAL ─────────────────────────────────────────────────
    ws_nan = wb.add_worksheet("NAN_PARTIAL")
    nan_p = issues[issues["issue_type"] == "NAN_PARTIAL"].copy() if not issues.empty else pd.DataFrame()
    if not nan_p.empty:
        write_df(ws_nan, nan_p.sort_values(["code", "country", "year"]),
                 col_widths={"detail": 60, "explanation": 80})

    # ── Sheet 5: ZERO in low deciles ─────────────────────────────────────────
    ws_zero = wb.add_worksheet("ZERO_LOW_DECILE")
    zeros = issues[issues["issue_type"] == "ZERO_LOW_DECILE"].copy() if not issues.empty else pd.DataFrame()
    if not zeros.empty:
        write_df(ws_zero, zeros.sort_values(["code", "country", "year"]),
                 col_widths={"detail": 55, "explanation": 80})

    # ── Sheet 6: Year gaps ────────────────────────────────────────────────────
    ws_gap = wb.add_worksheet("Year Gaps")
    gaps = issues[issues["issue_type"].isin(["YEAR_GAP", "LATE_START", "EARLY_END"])].copy() \
        if not issues.empty else pd.DataFrame()
    if not gaps.empty:
        write_df(ws_gap, gaps.sort_values(["issue_type", "code", "country", "year"]),
                 color_col="issue_type", color_map=issue_fmts,
                 col_widths={"code": 14, "detail": 60, "raw_probe_detail": 50,
                             "explanation": 80})

    # ── Sheet 7: Monotonicity ─────────────────────────────────────────────────
    ws_mono = wb.add_worksheet("Monotonicity")
    mono = issues[issues["issue_type"] == "MONOTONICITY"].copy() if not issues.empty else pd.DataFrame()
    if not mono.empty:
        write_df(ws_mono, mono.sort_values(["code", "country", "year"]),
                 col_widths={"detail": 55, "explanation": 80})

    # ── Sheet 8: Indicator overview ───────────────────────────────────────────
    ws_ov = wb.add_worksheet("Indicator Overview")
    if not summary.empty:
        ov = (
            summary.groupby(["code", "indicator_name", "source", "in_ewbi"])
            .agg(
                n_country_years  = ("country", "count"),
                n_countries      = ("country", "nunique"),
                n_years          = ("year",    "nunique"),
                pct_ok           = ("status",  lambda s: (s == "OK").mean() * 100),
                pct_absent       = ("status",  lambda s: (s == "ABSENT").mean() * 100),
                pct_nan_partial  = ("status",  lambda s: (s == "NAN_PARTIAL").mean() * 100),
                pct_zero_all     = ("status",  lambda s: (s == "ZERO_ALL").mean() * 100),
            )
            .reset_index()
            .sort_values("pct_ok", ascending=True)
            .round(1)
        )
        write_df(ws_ov, ov,
                 col_widths={"code": 14, "indicator_name": 44, "source": 10})

    wb.close()
    print(f"  Excel report → {out_path}")


def save_excel_country_report(summary: pd.DataFrame, issues: pd.DataFrame):
    """
    Export one Excel sheet per EU-27 country.
    Each sheet has one row per indicator with columns:
      Code | Indicator Name | Source | Full Coverage | Country Coverage | Gap Analysis
    Gap Analysis contains one line per problematic year using the GAP_LABELS taxonomy.
    """
    try:
        import xlsxwriter
    except ImportError:
        print("  xlsxwriter not installed. Skipping country report.")
        return

    if summary.empty:
        return

    out_path = config.QUALITY_DIR / "country_report.xlsx"
    config.QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pre-compute per-indicator full-coverage metadata ─────────────────────
    indicator_codes = sorted(summary["code"].unique())

    ind_meta: dict[str, dict] = {}
    for code in indicator_codes:
        meta    = BY_CODE.get(code, {})
        source  = meta.get("source", "")
        ind_df  = summary[summary["code"] == code]
        # Full coverage = union of all years where ANY country has valid data
        full_years = sorted(ind_df.loc[ind_df["n_valid"] > 0, "year"].unique())
        if full_years:
            full_min, full_max = int(min(full_years)), int(max(full_years))
        else:
            full_min, full_max = None, None
        ind_meta[code] = {
            "name":      meta.get("name", code),
            "source":    source,
            "full_years": full_years,
            "full_str":  _format_year_range(full_years) if full_years else "No data",
        }

    # ── Pre-index issues by (code, country) for fast lookup ──────────────────
    iss_by_cc: dict[tuple[str, str], pd.DataFrame] = {}
    if not issues.empty:
        for (code, country), grp in issues.groupby(["code", "country"]):
            iss_by_cc[(code, country)] = grp.sort_values("year")

    # ── Workbook formats ──────────────────────────────────────────────────────
    wb = xlsxwriter.Workbook(str(out_path))

    hdr_fmt  = wb.add_format({"bold": True, "bg_color": "1F4E79",
                               "font_color": "FFFFFF", "border": 1,
                               "text_wrap": True, "valign": "top"})
    ok_fmt   = wb.add_format({"bg_color": "DDFFDD", "border": 1,
                               "text_wrap": True, "valign": "top"})
    warn_fmt = wb.add_format({"bg_color": "FFF2CC", "border": 1,
                               "text_wrap": True, "valign": "top"})
    bad_fmt  = wb.add_format({"bg_color": "FFCCCC", "border": 1,
                               "text_wrap": True, "valign": "top"})
    base_fmt = wb.add_format({"border": 1, "text_wrap": True, "valign": "top"})

    HEADERS    = ["Code", "Indicator Name", "Source",
                  "Full Coverage (all countries)",
                  "Country Coverage",
                  "Gap Analysis"]
    COL_WIDTHS = [14, 48, 10, 28, 28, 80]

    eu27 = config.EU_COUNTRIES

    for country in eu27:
        ws = wb.add_worksheet(country[:31])
        for j, (h, w) in enumerate(zip(HEADERS, COL_WIDTHS)):
            ws.write(0, j, h, hdr_fmt)
            ws.set_column(j, j, w)

        row_i = 1
        for code in indicator_codes:
            m       = ind_meta[code]
            source  = m["source"]
            struct  = STRUCTURAL_NOTES.get(code, "")

            # Country-specific coverage
            ctry_rows = summary[(summary["code"] == code) & (summary["country"] == country)]
            ctry_years = sorted(
                ctry_rows.loc[ctry_rows["n_valid"] > 0, "year"].unique()
            )
            ctry_cov_str = _format_year_range(ctry_years) if ctry_years else "No data"

            # ── Gap analysis: one line per problematic year ───────────────────
            gap_lines: list[str] = []
            iss_df = iss_by_cc.get((code, country), pd.DataFrame())

            # Only flag ABSENT for years that appear in the indicator's global
            # coverage (i.e. at least one other country has valid data that year).
            # Years outside this set are simply not in scope — not meaningful gaps.
            full_years_set = set(m["full_years"])

            if not ctry_years and not iss_df.empty:
                # Country has no valid data at all — emit a single summary note
                # instead of per-year ABSENT spam.
                late_rows = iss_df[iss_df["issue_type"] == "LATE_START"]
                if not late_rows.empty:
                    detail = str(late_rows.iloc[0].get("detail", "") or "")
                    gap_lines.append(f"No data — {detail}")
                elif struct:
                    gap_lines.append(f"No data for this country. Note: {struct}")
                else:
                    gap_lines.append("No data for this country")

            elif not iss_df.empty:
                # Group issues by year so we don't repeat the same year twice
                for yr_val, yr_grp in iss_df.groupby("year"):
                    yr = int(yr_val)
                    for _, iss_row in yr_grp.iterrows():
                        it    = str(iss_row.get("issue_type", ""))

                        # Skip ABSENT issues for years that no country covers —
                        # those are pre-scope or universally-absent years, not gaps.
                        if it == "ABSENT" and yr not in full_years_set:
                            continue

                        label = GAP_LABELS.get(it, it)

                        if it == "ABSENT":
                            raw = str(iss_row.get("raw_probe_detail", "") or "")
                            expl = f"No data extracted — {raw}" if raw else "All values NaN"

                        elif it == "NAN_PARTIAL":
                            detail = str(iss_row.get("detail", "") or "")
                            expl = detail if detail else "Some decile groups have no valid responses"

                        elif it in ("ZERO_ALL", "ZERO_PARTIAL"):
                            expl = struct if struct else "All decile values are zero"

                        elif it == "ZERO_LOW_DECILE":
                            detail = str(iss_row.get("detail", "") or "")
                            expl = detail if detail else "Low-income decile value = 0%"

                        elif it == "YEAR_GAP":
                            raw  = str(iss_row.get("raw_probe_detail", "") or "")
                            note = struct if struct else ""
                            expl = (f"Year missing within coverage window"
                                    + (f" — {raw}" if raw else "")
                                    + (f". {note}" if note else ""))

                        elif it == "LATE_START":
                            detail = str(iss_row.get("detail", "") or "")
                            expl = detail if detail else "Country starts late vs median"

                        elif it == "EARLY_END":
                            detail = str(iss_row.get("detail", "") or "")
                            expl = detail if detail else "Country ends early vs median"

                        elif it == "MONOTONICITY":
                            detail = str(iss_row.get("detail", "") or "")
                            expl = f"No income gradient — {detail}" if detail else "No income gradient"

                        else:
                            expl = str(iss_row.get("detail", "") or "")

                        gap_lines.append(f"{yr}: {label} — {expl}")

            # Add structural note if no gaps were found but note exists
            if struct and not gap_lines and ctry_years:
                gap_lines.append(f"Note: {struct}")

            gap_str = "\n".join(gap_lines)

            # Row color: red if no data, yellow if issues, green if OK
            if not ctry_years:
                row_fmt = bad_fmt
            elif gap_lines:
                row_fmt = warn_fmt
            else:
                row_fmt = ok_fmt

            # Estimate row height (≈15 pts per line, minimum 15)
            n_lines = max(1, len(gap_lines))
            ws.set_row(row_i, min(15 * n_lines, 400))

            ws.write(row_i, 0, code,          row_fmt)
            ws.write(row_i, 1, m["name"],     row_fmt)
            ws.write(row_i, 2, source,        row_fmt)
            ws.write(row_i, 3, m["full_str"], row_fmt)
            ws.write(row_i, 4, ctry_cov_str,  row_fmt)
            ws.write(row_i, 5, gap_str,       row_fmt)
            row_i += 1

    wb.close()
    print(f"  Country report → {out_path}")


def save_excel_availability_matrix(summary: pd.DataFrame, issues: pd.DataFrame):
    """
    Export availability_matrix.xlsx with one sheet per problem type plus a summary.

    Sheets
    ------
    Summary    – Heat-map: number of distinct issue types per (indicator × country)
    Coverage   – Full / Partial / Missing availability per (indicator × country)
    <issue>    – One sheet per GAP_LABELS key: count of affected years per cell
    Stats      – Per-indicator and per-country counts (Full / Partial / Missing)
    """
    try:
        import xlsxwriter
    except ImportError:
        print("  xlsxwriter not installed. Skipping availability matrix.")
        return

    if summary.empty:
        return

    out_path = config.QUALITY_DIR / "availability_matrix.xlsx"
    config.QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    eu27 = config.EU27
    # Restrict to EU-27 for all computations
    summary = summary[summary["country"].isin(eu27)].copy()
    issues  = issues[issues["country"].isin(eu27)].copy() if not issues.empty else issues

    indicator_codes = sorted(summary["code"].unique())

    # ── Issue sheets definition: (issue_type, sheet_name, hex_color) ─────────
    ISSUE_SHEETS: list[tuple[str, str, str]] = [
        ("ABSENT",          "Absent",         "D9534F"),
        ("NAN_PARTIAL",     "NaN Partial",    "F0AD4E"),
        ("ZERO_ALL",        "Zero (All)",     "FF8080"),
        ("ZERO_PARTIAL",    "Zero (Partial)", "FFAA55"),
        ("ZERO_LOW_DECILE", "Zero Low Dec.",  "E06000"),
        ("YEAR_GAP",        "Year Gap",       "FFD700"),
        ("LATE_START",      "Late Start",     "AADDFF"),
        ("EARLY_END",       "Early End",      "88BBEE"),
        ("MONOTONICITY",    "Monotonicity",   "CC99FF"),
    ]

    # ── Per-indicator expected year count (EU-27, global coverage window) ─────
    ind_meta: dict[str, dict] = {}
    for code in indicator_codes:
        meta   = BY_CODE.get(code, {})
        ind_df = summary[summary["code"] == code]
        full_years = sorted(ind_df.loc[ind_df["n_valid"] > 0, "year"].unique())
        ind_meta[code] = {
            "name":   meta.get("name", code),
            "source": meta.get("source", ""),
            "n_exp":  len(full_years),
        }

    # ── Valid-year counts per (code, country) ─────────────────────────────────
    valid_counts: dict[tuple[str, str], int] = {
        (code, country): int((grp["n_valid"] > 0).sum())
        for (code, country), grp in summary.groupby(["code", "country"])
    }

    def availability(code: str, country: str) -> tuple[str, str]:
        """(label, status) — status in {Full, Partial, Missing}."""
        n_valid = valid_counts.get((code, country), 0)
        n_exp   = ind_meta[code]["n_exp"]
        if n_valid == 0:
            return "–", "Missing"
        if n_exp == 0 or n_valid >= n_exp:
            return "Full", "Full"
        return f"{n_valid}/{n_exp}", "Partial"

    # ── Issue-year counts per (issue_type, code, country) ────────────────────
    # ZERO_ALL and ZERO_PARTIAL live only in summary["status"], not in issues.
    # ABSENT must be filtered to years within the global coverage window only
    # (same fix as save_excel_country_report — avoids pre-scope spam).
    issue_year_counts: dict[str, dict[tuple[str, str], int]] = {}

    # 1. Types that exist in the issues DataFrame
    if not issues.empty and "issue_type" in issues.columns:
        # Pre-compute full_years_set per indicator (years where ≥1 EU-27 country has data)
        full_years_per_code: dict[str, set] = {
            code: set(
                summary.loc[(summary["code"] == code) & (summary["n_valid"] > 0), "year"].unique()
            )
            for code in indicator_codes
        }
        for it, grp in issues.groupby("issue_type"):
            cnt: dict[tuple[str, str], int] = {}
            for (code, country), sub in grp.groupby(["code", "country"]):
                if it == "ABSENT":
                    # Only count years within the global coverage window
                    fy = full_years_per_code.get(code, set())
                    sub = sub[sub["year"].isin(fy)]
                n = sub["year"].nunique()
                if n > 0:
                    cnt[(code, country)] = n
            if cnt:
                issue_year_counts[it] = cnt

    # 2. ZERO_ALL and ZERO_PARTIAL — derived from summary status
    for zero_type in ("ZERO_ALL", "ZERO_PARTIAL"):
        cnt = {}
        for (code, country), grp in summary.groupby(["code", "country"]):
            n = int((grp["status"] == zero_type).sum())
            if n > 0:
                cnt[(code, country)] = n
        if cnt:
            issue_year_counts[zero_type] = cnt

    # ── Total distinct issue types per (code, country) ────────────────────────
    total_issue_types: dict[tuple[str, str], int] = {}
    for cnt_map in issue_year_counts.values():
        for key, n in cnt_map.items():
            if n > 0:
                total_issue_types[key] = total_issue_types.get(key, 0) + 1

    # ── Shared format factory (formats must be created after wb is open) ──────
    wb = xlsxwriter.Workbook(str(out_path))

    def hdr(wb):
        return wb.add_format({"bold": True, "bg_color": "1F4E79",
                               "font_color": "FFFFFF", "border": 1,
                               "align": "center", "valign": "vcenter",
                               "text_wrap": True, "font_size": 9})

    def cell(wb, bg="FFFFFF", fg="000000", bold=False):
        return wb.add_format({"bg_color": bg, "font_color": fg, "border": 1,
                               "align": "center", "valign": "vcenter",
                               "font_size": 9, "bold": bold})

    def left_cell(wb, bold=False):
        return wb.add_format({"border": 1, "valign": "vcenter",
                               "font_size": 9, "bold": bold, "text_wrap": True})

    hdr_f     = hdr(wb)
    code_f    = left_cell(wb, bold=True)
    name_f    = left_cell(wb)
    src_f     = cell(wb)
    plain_f   = cell(wb)
    ok0_f     = cell(wb, bg="92D050")          # 0 issues — green
    ok1_f     = cell(wb, bg="FFFF99")          # 1 issue type
    ok2_f     = cell(wb, bg="FFCC00")          # 2 issue types
    ok3_f     = cell(wb, bg="FF9900")          # 3 issue types
    ok4_f     = cell(wb, bg="FF6B6B", fg="FFFFFF")  # 4+ issue types
    full_f    = cell(wb, bg="92D050")
    part_f    = cell(wb, bg="FFCC00")
    miss_f    = cell(wb, bg="FF6B6B", fg="FFFFFF")
    no_iss_f  = cell(wb, bg="F2F2F2", fg="AAAAAA")  # no issue — light gray

    AVAIL_FMT = {"Full": full_f, "Partial": part_f, "Missing": miss_f}
    HEAT_FMT  = {0: ok0_f, 1: ok1_f, 2: ok2_f, 3: ok3_f}  # 4+ → ok4_f

    # ── Helper: write left-column header block common to all matrix sheets ────
    def init_matrix_sheet(ws_):
        ws_.freeze_panes(1, 3)
        ws_.write(0, 0, "Code",           hdr_f)
        ws_.write(0, 1, "Indicator Name", hdr_f)
        ws_.write(0, 2, "Source",         hdr_f)
        ws_.set_column(0, 0, 13)
        ws_.set_column(1, 1, 40)
        ws_.set_column(2, 2,  7)
        for j, cc in enumerate(eu27):
            ws_.write(0, 3 + j, cc, hdr_f)
            ws_.set_column(3 + j, 3 + j, 7)

    def write_left_cols(ws_, row_i, code):
        m = ind_meta[code]
        ws_.set_row(row_i, 15)
        ws_.write(row_i, 0, code,       code_f)
        ws_.write(row_i, 1, m["name"],  name_f)
        ws_.write(row_i, 2, m["source"], src_f)

    # ════════════════════════════════════════════════════════════════════════
    # Sheet 1 – Summary heat-map (total distinct issue types per cell)
    # ════════════════════════════════════════════════════════════════════════
    ws_sum = wb.add_worksheet("Summary")
    init_matrix_sheet(ws_sum)

    for row_i, code in enumerate(indicator_codes, start=1):
        write_left_cols(ws_sum, row_i, code)
        for j, country in enumerate(eu27):
            n = total_issue_types.get((code, country), 0)
            fmt = HEAT_FMT.get(n, ok4_f)
            label = str(n) if n > 0 else "–"
            ws_sum.write(row_i, 3 + j, label, fmt)

    # ════════════════════════════════════════════════════════════════════════
    # Sheet 2 – Coverage (Full / Partial / Missing)
    # ════════════════════════════════════════════════════════════════════════
    ws_cov = wb.add_worksheet("Coverage")
    init_matrix_sheet(ws_cov)

    for row_i, code in enumerate(indicator_codes, start=1):
        write_left_cols(ws_cov, row_i, code)
        for j, country in enumerate(eu27):
            label, status = availability(code, country)
            ws_cov.write(row_i, 3 + j, label, AVAIL_FMT[status])

    # ════════════════════════════════════════════════════════════════════════
    # Sheets 3–11 – One per issue type
    # ════════════════════════════════════════════════════════════════════════
    for issue_key, sheet_name, color_hex in ISSUE_SHEETS:
        ws_i = wb.add_worksheet(sheet_name)
        # Title row: issue description as a merged header above the matrix
        full_label = GAP_LABELS.get(issue_key, issue_key)
        title_fmt = wb.add_format({"bold": True, "bg_color": color_hex,
                                   "border": 1, "align": "left",
                                   "valign": "vcenter", "font_size": 9,
                                   "text_wrap": True})
        ws_i.merge_range(0, 0, 0, 3 + len(eu27) - 1, full_label, title_fmt)
        ws_i.set_row(0, 20)

        # Sub-header row (row index 1)
        ws_i.write(1, 0, "Code",           hdr_f)
        ws_i.write(1, 1, "Indicator Name", hdr_f)
        ws_i.write(1, 2, "Source",         hdr_f)
        ws_i.set_column(0, 0, 13)
        ws_i.set_column(1, 1, 40)
        ws_i.set_column(2, 2,  7)
        for j, cc in enumerate(eu27):
            ws_i.write(1, 3 + j, cc, hdr_f)
            ws_i.set_column(3 + j, 3 + j, 7)
        ws_i.freeze_panes(2, 3)

        iss_fmt = wb.add_format({"bg_color": color_hex, "border": 1,
                                  "align": "center", "valign": "vcenter",
                                  "font_size": 9})
        cnt_map = issue_year_counts.get(issue_key, {})

        for row_i, code in enumerate(indicator_codes, start=2):
            m = ind_meta[code]
            ws_i.set_row(row_i, 15)
            ws_i.write(row_i, 0, code,       code_f)
            ws_i.write(row_i, 1, m["name"],  name_f)
            ws_i.write(row_i, 2, m["source"], src_f)
            for j, country in enumerate(eu27):
                n = cnt_map.get((code, country), 0)
                if n > 0:
                    ws_i.write(row_i, 3 + j, f"{n}y", iss_fmt)
                else:
                    ws_i.write(row_i, 3 + j, "–", no_iss_f)

    # ════════════════════════════════════════════════════════════════════════
    # Last sheet – Stats (per-indicator and per-country tables)
    # ════════════════════════════════════════════════════════════════════════
    ws_st = wb.add_worksheet("Stats")
    ws_st.freeze_panes(1, 0)

    ind_hdr = ["Code", "Indicator Name", "Source",
               "# Full", "# Partial", "# Missing", "Coverage %", "# Issue types"]
    for j, h in enumerate(ind_hdr):
        ws_st.write(0, j, h, hdr_f)
    ws_st.set_column(0, 0, 13)
    ws_st.set_column(1, 1, 40)
    ws_st.set_column(2, 2,  7)
    ws_st.set_column(3, 8, 12)

    for row_i, code in enumerate(indicator_codes, start=1):
        m = ind_meta[code]
        statuses = [availability(code, cc)[1] for cc in eu27]
        n_full    = statuses.count("Full")
        n_partial = statuses.count("Partial")
        n_missing = statuses.count("Missing")
        cov_pct   = round((n_full + 0.5 * n_partial) / len(eu27) * 100, 1)
        n_iss_types = sum(
            1 for cc in eu27 if total_issue_types.get((code, cc), 0) > 0
        )
        ws_st.set_row(row_i, 15)
        ws_st.write(row_i, 0, code,        code_f)
        ws_st.write(row_i, 1, m["name"],   name_f)
        ws_st.write(row_i, 2, m["source"], src_f)
        ws_st.write(row_i, 3, n_full,      full_f)
        ws_st.write(row_i, 4, n_partial,   part_f  if n_partial else plain_f)
        ws_st.write(row_i, 5, n_missing,   miss_f  if n_missing else plain_f)
        ws_st.write(row_i, 6, cov_pct,     plain_f)
        ws_st.write(row_i, 7, n_iss_types, plain_f)

    # Per-country summary table
    offset = len(indicator_codes) + 3
    ctry_hdr = ["Country", "# Full", "# Partial", "# Missing", "Coverage %", "# Indicators w/ issues"]
    for j, h in enumerate(ctry_hdr):
        ws_st.write(offset, j, h, hdr_f)

    for row_i, country in enumerate(eu27, start=offset + 1):
        statuses  = [availability(code, country)[1] for code in indicator_codes]
        n_full    = statuses.count("Full")
        n_partial = statuses.count("Partial")
        n_missing = statuses.count("Missing")
        cov_pct   = round((n_full + 0.5 * n_partial) / len(indicator_codes) * 100, 1)
        n_with_iss = sum(
            1 for code in indicator_codes if total_issue_types.get((code, country), 0) > 0
        )
        ws_st.set_row(row_i, 15)
        ws_st.write(row_i, 0, country,    src_f)
        ws_st.write(row_i, 1, n_full,     full_f)
        ws_st.write(row_i, 2, n_partial,  part_f  if n_partial else plain_f)
        ws_st.write(row_i, 3, n_missing,  miss_f  if n_missing else plain_f)
        ws_st.write(row_i, 4, cov_pct,    plain_f)
        ws_st.write(row_i, 5, n_with_iss, plain_f)

    wb.close()
    print(f"  Availability matrix → {out_path}")


# ─── HBS Coverage ─────────────────────────────────────────────────────────────

# Source column → human-readable label (used in distribution sheet headers)
_HBS_SRC_LABELS: dict[str, str] = {
    "EUR_HE01":  "Food",
    "EUR_HE04":  "Housing_total",
    "EUR_HE041": "Rent_services",
    "EUR_HE06":  "Health",
    "EUR_HE07":  "Transport",
    "EUR_HE09":  "ICT",
    "EUR_HE10":  "Education",
    "EUR_HJ08":  "Communications",
    "EUR_HJ90":  "Travel_tourism",
}

# Source column → indicator codes that use it
_HBS_SRC_INDICATORS: dict[str, list[str]] = {
    "EUR_HE01":  ["AE-HBS-1", "AE-HBS-2"],
    "EUR_HE04":  ["HH-HBS-3", "HH-HBS-4"],
    "EUR_HE041": ["HH-HBS-1", "HH-HBS-2"],
    "EUR_HE06":  ["AC-HBS-1", "AC-HBS-2"],
    "EUR_HE07":  ["TT-HBS-1", "TT-HBS-2"],
    "EUR_HE09":  ["IC-HBS-1", "IC-HBS-2"],
    "EUR_HE10":  ["IE-HBS-1", "IE-HBS-2"],
    "EUR_HJ08":  ["EC-HBS-1", "EC-HBS-2"],
    "EUR_HJ90":  ["TS-HBS-1", "TS-HBS-2"],
}


def save_excel_hbs_coverage() -> None:
    """
    HBS data coverage: n_obs and n_weighted per (indicator, country, year, decile).
    Source: individual indicator CSVs in INDICATORS_DIR.
    Output: output/quality/hbs_coverage.xlsx  (2 sheets: n_obs, n_weighted)
    """
    import xlsxwriter
    out_path = config.QUALITY_DIR / "hbs_coverage.xlsx"
    hbs_codes = sorted(
        code for code, m in BY_CODE.items() if m.get("source") == "HBS"
    )

    rows: list[pd.DataFrame] = []
    for code in hbs_codes:
        f = config.INDICATORS_DIR / f"{code}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "n_obs" not in df.columns or "n_weighted" not in df.columns:
            continue
        df = df[df["decile"].astype(str) != "All"].copy()
        df["decile"] = pd.to_numeric(df["decile"], errors="coerce")
        df = df[df["decile"].notna()].copy()
        df["decile"] = df["decile"].astype(int)
        df["code"] = code
        rows.append(df[["code", "country", "year", "decile", "n_obs", "n_weighted"]])

    if not rows:
        print("  [WARN] No HBS CSVs with n_obs/n_weighted found.")
        return

    all_df = pd.concat(rows, ignore_index=True)
    all_df["year"] = pd.to_numeric(all_df["year"], errors="coerce")
    all_df = all_df.sort_values(["code", "country", "year", "decile"])

    wb = xlsxwriter.Workbook(str(out_path))
    hdr_f  = wb.add_format({"bold": True, "bg_color": "#2F5496",
                             "font_color": "white", "border": 1, "align": "center"})
    base_f = wb.add_format({"border": 1})
    num_f  = wb.add_format({"border": 1, "num_format": "#,##0"})
    low_f  = wb.add_format({"border": 1, "bg_color": "#FFC7CE",
                             "font_color": "#9C0006", "num_format": "#,##0"})
    nan_f  = wb.add_format({"border": 1, "bg_color": "#D9D9D9"})

    D_COLS = [f"D{i}" for i in range(1, 11)]
    N_OBS_MIN = 30   # flag cells below this

    for sheet_name, value_col, min_thresh in [
        ("n_obs",      "n_obs",      N_OBS_MIN),
        ("n_weighted", "n_weighted", 0),
    ]:
        wide = all_df.pivot_table(
            index=["code", "country", "year"],
            columns="decile",
            values=value_col,
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        wide.columns = [f"D{c}" if isinstance(c, int) else c for c in wide.columns]

        ws = wb.add_worksheet(sheet_name)
        ws.freeze_panes(1, 3)
        ws.set_zoom(85)

        hdrs   = ["Code", "Country", "Year"] + D_COLS
        widths = [14, 9, 7] + [11] * 10
        for ci, (h, w) in enumerate(zip(hdrs, widths)):
            ws.write(0, ci, h, hdr_f)
            ws.set_column(ci, ci, w)

        for ri, row in enumerate(wide.itertuples(index=False), start=1):
            ws.write(ri, 0, row.code,    base_f)
            ws.write(ri, 1, row.country, base_f)
            ws.write(ri, 2, int(row.year) if not np.isnan(row.year) else "", base_f)
            for ci, dcol in enumerate(D_COLS, start=3):
                val = getattr(row, dcol, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    ws.write_blank(ri, ci, None, nan_f)
                else:
                    v   = int(round(val))
                    fmt = low_f if (min_thresh > 0 and v < min_thresh) else num_f
                    ws.write(ri, ci, v, fmt)

    wb.close()
    print(f"  HBS coverage  → {out_path}")
    print(f"    {len(all_df['code'].unique())} indicators, "
          f"{len(all_df[['code','country','year']].drop_duplicates())} (code×country×year) combos")


# ─── HBS Expense Distribution ─────────────────────────────────────────────────

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v, w = values[mask], weights[mask]
    if len(v) == 0:
        return np.nan
    order = np.argsort(v)
    v, w  = v[order], w[order]
    cumw  = np.cumsum(w)
    return float(np.interp(0.5, cumw / cumw[-1], v))


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    v, w = values[mask], weights[mask]
    if len(v) < 2:
        return np.nan
    wmean = float(np.average(v, weights=w))
    wvar  = float(np.average((v - wmean) ** 2, weights=w))
    return float(np.sqrt(wvar))


def save_excel_hbs_distribution() -> None:
    """
    For each (country, year, income_decile): weighted median and std of the
    expenditure share (category / EUR_HH099 × 100) for every HBS category.
    Source: HBS parquet cache (output/cache/hbs/).
    Output: output/quality/hbs_distribution.xlsx  (2 sheets: median, std_dev)
    """
    import xlsxwriter
    CACHE_HBS = config.CACHE_DIR / "hbs"
    out_path  = config.QUALITY_DIR / "hbs_distribution.xlsx"

    # ── Load parquet waves ────────────────────────────────────────────────────
    waves: list[pd.DataFrame] = []
    for year in [2010, 2015, 2020]:
        p = CACHE_HBS / f"HBS_{year}.parquet"
        if not p.exists():
            print(f"  [WARN] HBS_{year}.parquet not found — skipping.")
            continue
        df = pd.read_parquet(p)
        df["YEAR"] = year
        waves.append(df)

    if not waves:
        print("  [WARN] No HBS parquet cache found. Run extract_hbs.py first.")
        return

    raw = pd.concat(waves, ignore_index=True)
    for c in raw.columns:
        if c not in ("COUNTRY",):
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    exp_cols = [c for c in _HBS_SRC_LABELS if c in raw.columns]
    print(f"  HBS distribution: {len(raw):,} households, "
          f"{raw['COUNTRY'].nunique()} countries, "
          f"expense cols: {exp_cols}")

    # ── Net income denominator ────────────────────────────────────────────────
    net_inc = raw.get("EUR_HH099", pd.Series(np.nan, index=raw.index)).copy()
    net_inc = pd.to_numeric(net_inc, errors="coerce").replace(0, np.nan)
    if net_inc.notna().sum() == 0:
        net_inc = pd.to_numeric(
            raw.get("EUR_HH095", pd.Series(np.nan, index=raw.index)),
            errors="coerce"
        ).replace(0, np.nan)

    # ── Expenditure shares ────────────────────────────────────────────────────
    for col in exp_cols:
        raw[f"share_{col}"] = raw[col] / net_inc * 100

    # ── Equivalised income → income decile ───────────────────────────────────
    hb061   = raw.get("HB061",    pd.Series(np.nan, index=raw.index)).replace(0, np.nan)
    inc95   = raw.get("EUR_HH095", pd.Series(np.nan, index=raw.index)).fillna(0)
    hh012   = raw.get("EUR_HH012", pd.Series(0.0, index=raw.index)).fillna(0)
    hh023   = raw.get("EUR_HH023", pd.Series(0.0, index=raw.index)).fillna(0)
    has95   = raw.groupby(["COUNTRY", "YEAR"])["EUR_HH095"].transform(
        lambda s: (s.fillna(0) > 0).any()
    ).astype(bool)
    inc     = np.where(has95, inc95.to_numpy(), (hh012 + hh023).to_numpy())
    raw["equi_inc"] = inc / hb061.to_numpy()

    weight = raw.get("HA10", pd.Series(np.nan, index=raw.index)).to_numpy(dtype=float)

    decile_arr = np.full(len(raw), np.nan)
    for (country, year), grp_idx in raw.groupby(["COUNTRY", "YEAR"]).groups.items():
        gi  = list(grp_idx)
        inc_g = raw.loc[gi, "equi_inc"].to_numpy(dtype=float)
        wt_g  = weight[gi]
        valid = np.isfinite(inc_g) & np.isfinite(wt_g) & (wt_g > 0)
        if valid.sum() < 10:
            continue
        thresholds = np.array([
            np.interp(q, np.cumsum(wt_g[valid][np.argsort(inc_g[valid])]) /
                      np.sum(wt_g[valid]),
                      np.sort(inc_g[valid]))
            for q in np.arange(0.1, 1.0, 0.1)
        ])
        if np.any(np.isnan(thresholds)):
            continue
        res = np.full(len(gi), np.nan)
        res[valid] = (inc_g[valid, np.newaxis] > thresholds).sum(axis=1) + 1
        decile_arr[gi] = res

    raw["income_decile"] = decile_arr
    raw_dec = raw[raw["income_decile"].notna()].copy()
    raw_dec["income_decile"] = raw_dec["income_decile"].astype(int)

    # ── Compute stats per (country, year, decile, category) ──────────────────
    records: list[dict] = []
    for (country, year, decile), grp in raw_dec.groupby(
            ["COUNTRY", "YEAR", "income_decile"]):
        wts = grp["HA10"].to_numpy(dtype=float)
        rec: dict = {
            "country": country, "year": int(year), "decile": int(decile),
            "n_obs": len(grp), "n_weighted": float(np.nansum(wts)),
        }
        for col in exp_cols:
            vals = grp[f"share_{col}"].to_numpy(dtype=float)
            rec[f"{col}_median"] = _weighted_median(vals, wts)
            rec[f"{col}_std"]    = _weighted_std(vals, wts)
        records.append(rec)

    dist_df = pd.DataFrame(records).sort_values(["country", "year", "decile"])

    # ── Write Excel ───────────────────────────────────────────────────────────
    wb = xlsxwriter.Workbook(str(out_path))
    hdr_f  = wb.add_format({"bold": True, "bg_color": "#2F5496",
                             "font_color": "white", "border": 1,
                             "align": "center", "text_wrap": True})
    base_f = wb.add_format({"border": 1})
    num1_f = wb.add_format({"border": 1, "num_format": "0.0"})
    num2_f = wb.add_format({"border": 1, "num_format": "0.00"})
    hi_f   = wb.add_format({"border": 1, "bg_color": "#FFC7CE",
                             "font_color": "#9C0006", "num_format": "0.00"})

    # Sub-headers row: show indicator codes below each category label
    subhdr_f = wb.add_format({"italic": True, "bg_color": "#D9E2F3",
                               "border": 1, "font_size": 8, "text_wrap": True})

    for sheet_name, stat_suffix, stat_label in [
        ("Median_share_%", "_median", "Weighted median of expenditure share (%)"),
        ("StdDev_share_%", "_std",    "Weighted std-dev of expenditure share (%)"),
    ]:
        ws = wb.add_worksheet(sheet_name)
        ws.freeze_panes(2, 4)
        ws.set_zoom(80)

        # Row 1: column headers
        fixed_hdrs = ["Country", "Year", "Decile", "N obs", "N weighted"]
        ws.merge_range(0, 0, 0, len(fixed_hdrs) - 1 + len(exp_cols),
                       stat_label, hdr_f)
        for ci, h in enumerate(fixed_hdrs):
            ws.write(1, ci, h, hdr_f)
            ws.set_column(ci, ci, [9, 7, 8, 8, 12][ci])

        for ci, col in enumerate(exp_cols, start=len(fixed_hdrs)):
            label   = _HBS_SRC_LABELS.get(col, col)
            ind_str = ", ".join(_HBS_SRC_INDICATORS.get(col, []))
            ws.write(1, ci, f"{label}\n({ind_str})", hdr_f)
            ws.set_column(ci, ci, 14)

        ws.set_row(1, 28)

        for ri, row in enumerate(dist_df.itertuples(index=False), start=2):
            ws.write(ri, 0, row.country,         base_f)
            ws.write(ri, 1, int(row.year),        base_f)
            ws.write(ri, 2, int(row.decile),      base_f)
            ws.write(ri, 3, int(row.n_obs),       num1_f)
            ws.write(ri, 4, round(row.n_weighted, 0), num1_f)
            for ci, col in enumerate(exp_cols, start=len(fixed_hdrs)):
                val = getattr(row, f"{col}{stat_suffix}", None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    ws.write_blank(ri, ci, None, base_f)
                else:
                    # Flag very high std (>50%) or median >100% as suspicious
                    suspicious = (stat_suffix == "_std" and val > 50) or \
                                 (stat_suffix == "_median" and val > 100)
                    ws.write(ri, ci, round(float(val), 2),
                             hi_f if suspicious else num2_f)

    wb.close()
    print(f"  HBS distribution → {out_path}")
    print(f"    {len(dist_df):,} (country×year×decile) rows, "
          f"{len(exp_cols)} expense categories")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data quality checks on all indicators")
    parser.add_argument("--csv",              action="store_true", help="Save CSV outputs")
    parser.add_argument("--excel",            action="store_true", help="Save indicator-level Excel report")
    parser.add_argument("--country-report",   action="store_true", help="Save per-country Excel report")
    parser.add_argument("--matrix",           action="store_true", help="Save availability matrix Excel")
    parser.add_argument("--hbs-coverage",     action="store_true", help="HBS n_obs/n_weighted per decile")
    parser.add_argument("--hbs-distribution", action="store_true", help="HBS median/std of expense shares")
    parser.add_argument("--code",             default=None,        help="Limit to one indicator code")
    args = parser.parse_args()

    # Default: produce all outputs when no flag is given
    nothing_selected = not any([
        args.csv, args.excel, args.country_report, args.matrix,
        args.hbs_coverage, args.hbs_distribution,
    ])
    do_csv          = args.csv              or nothing_selected
    do_excel        = args.excel            or nothing_selected
    do_country      = args.country_report   or nothing_selected
    do_matrix       = args.matrix           or nothing_selected
    do_hbs_cov      = args.hbs_coverage     or nothing_selected
    do_hbs_dist     = args.hbs_distribution or nothing_selected

    if do_csv or do_excel or do_country or do_matrix:
        summary, issues = run_quality_check(code_filter=args.code)
        if summary.empty:
            sys.exit(1)
        print_console_report(summary, issues)
        if do_csv:
            save_csv(summary, issues)
        if do_excel:
            save_excel(summary, issues)
        if do_country:
            save_excel_country_report(summary, issues)
        if do_matrix:
            save_excel_availability_matrix(summary, issues)

    if do_hbs_cov:
        print("\nGenerating HBS coverage report …")
        save_excel_hbs_coverage()
    if do_hbs_dist:
        print("\nGenerating HBS distribution report …")
        save_excel_hbs_distribution()

