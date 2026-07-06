"""
EU-SILC Supplementary Indicator Pipeline
=========================================
Computes four new indicators from raw EU-SILC P/R files.
Decile is joined from the pre-built personal merged CSV.

Indicators
----------
SP-SILC-1 : Not participating in voluntary activity
            (PS101==6 in 2015, PS110==6 in 2022)
SP-SILC-2 : No active citizenship
            (PS102 in [2,3,4], 2015 and 2022 only)
GE-SILC-1 : Gender employment gap (pp)
            rate(employees, men) - rate(employees, women), age 20-64
            Employment: RB211==1 (2022+) or PL031 in [1,2] (before 2022)
GE-SILC-2 : Gender pay gap (%)
            (1 - weighted_median_PY010G_women / weighted_median_PY010G_men) * 100
            Among employees with PY010G > 0, age 20-64

Sex coding: RB090 in R-file (1 = male, 2 = female) -- available all years.
Age:        RB082 if available (2022+), else RB010 - RB080 (birth year).
Weight:     RB050 from R-file.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SILC_BASE = (
    r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr"
    r"/1_WSL/1_EWBI/0_data/EU-SILC"
    r"/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set"
)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", "output"))
SILC_OUT = os.path.join(OUTPUT_DIR, "0_raw_data_EUROSTAT", "0_EU-SILC", "3_final_merged_df")

PERSONAL_MERGED = os.path.join(SILC_OUT, "EU_SILC_personal_final_merged.csv")
OUT_PATH = os.path.join(SILC_OUT, "EU_SILC_supplementary_catalog.csv")

# ---------------------------------------------------------------------------
# Indicator metadata
# ---------------------------------------------------------------------------
SUPP_META = {
    "SP-SILC-1": {
        "name": "Not participating in voluntary activity",
        "source": "EU-SILC",
        "code_source": "PS101/PS110",
        "threshold": "PS101==6 (2015) or PS110==6 (2022)",
        "indicator_group": "Social Participation",
        "level": None,
        "already_included": "N",
    },
    "SP-SILC-2": {
        "name": "No active citizenship",
        "source": "EU-SILC",
        "code_source": "PS102",
        "threshold": "PS102 in [2,3,4]",
        "indicator_group": "Social Participation",
        "level": None,
        "already_included": "N",
    },
    "GE-SILC-1": {
        "name": "Gender employment gap",
        "source": "EU-SILC",
        "code_source": "RB211/PL031,RB090",
        "threshold": "rate(RB211==1|PL031 in [1,2], men) - rate(women), age 20-64 (pp)",
        "indicator_group": "Gender Equality",
        "level": None,
        "already_included": "N",
    },
    "GE-SILC-2": {
        "name": "Gender pay gap",
        "source": "EU-SILC",
        "code_source": "PY010G,RB090,RB211",
        "threshold": "(1 - med_PY010G_women/med_PY010G_men)*100, employed age 20-64 (%)",
        "indicator_group": "Gender Equality",
        "level": None,
        "already_included": "N",
    },
}

CATALOG_COLS = [
    "code", "name", "source", "code_source",
    "country", "year", "decile", "value",
    "already_included", "level", "threshold", "indicator_group",
]

# Years where participation module variables were asked
SP_YEARS = {2015, 2022}

# Minimum cell size for pay gap computation (to avoid noisy medians)
MIN_CELL = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median via cumulative-weight method."""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0) & (values > 0)
    if mask.sum() == 0:
        return np.nan
    v, w = values[mask], weights[mask]
    idx = np.argsort(v)
    v, w = v[idx], w[idx]
    cumw = np.cumsum(w)
    mid = cumw[-1] / 2.0
    pos = np.searchsorted(cumw, mid)
    return float(v[min(pos, len(v) - 1)])


def collect_files(base: str):
    """Yield (country, year, p_path, r_path) for all available country/years."""
    base_path = Path(base)
    for country_dir in sorted(base_path.iterdir()):
        if not country_dir.is_dir():
            continue
        country = country_dir.name
        for year_dir in sorted(country_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            yy = year_dir.name[-2:]
            p_file = year_dir / f"UDB_c{country}{yy}P.csv"
            r_file = year_dir / f"UDB_c{country}{yy}R.csv"
            if p_file.exists() and r_file.exists():
                yield country, year, str(p_file), str(r_file)


# Fixed sets of column names we may want from each file type
_R_WANT = {"RB010", "RB020", "RB030", "RB050", "RB080", "RB090", "RB082", "RB211"}
_P_WANT_BASE = {"PB010", "PB020", "PB030", "PY010G", "PL031"}
_P_WANT_2015 = _P_WANT_BASE | {"PS101", "PS102"}
_P_WANT_2022 = _P_WANT_BASE | {"PS102", "PS110"}


def load_r(r_path: str, year: int) -> pd.DataFrame:
    """Load relevant columns from one R-file (single read via usecols callable)."""
    df = pd.read_csv(r_path, usecols=lambda c: c in _R_WANT)

    df = df.rename(columns={
        "RB010": "year_r", "RB020": "country_r",
        "RB030": "PB030", "RB050": "weight",
        "RB080": "birth_year", "RB090": "sex",
    })
    df["PB030"] = df["PB030"].fillna(0).astype("int64").astype(str).str.strip()

    # Compute age
    if "RB082" in df.columns:
        df["age"] = pd.to_numeric(df["RB082"], errors="coerce")
        df.drop(columns=["RB082"], inplace=True)
    elif "birth_year" in df.columns:
        df["age"] = year - pd.to_numeric(df["birth_year"], errors="coerce")
    else:
        df["age"] = np.nan

    if "birth_year" in df.columns:
        df.drop(columns=["birth_year"], inplace=True)
    return df


def load_p(p_path: str, year: int) -> pd.DataFrame:
    """Load relevant columns from one P-file (single read via usecols callable)."""
    want = _P_WANT_2015 if year == 2015 else (_P_WANT_2022 if year == 2022 else _P_WANT_BASE)
    df = pd.read_csv(p_path, usecols=lambda c: c in want)
    df["PB030"] = df["PB030"].fillna(0).astype("int64").astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Indicator computation for one country/year slice
# ---------------------------------------------------------------------------

def compute_indicators(m: pd.DataFrame, country: str, year: int) -> list:
    """Compute all supplementary indicators for one country/year DataFrame."""
    results = []
    if m.empty:
        return results

    # Age filter 20-64
    age_ok = (m["age"] >= 20) & (m["age"] <= 64)
    m = m[age_ok]
    if m.empty:
        return results

    # Unified employment flag
    # Priority: RB211 (2022+ R-file) > PL031 (P-file, pre-2022)
    if "RB211" in m.columns:
        m = m.copy()
        m["employed"] = (pd.to_numeric(m["RB211"], errors="coerce") == 1).astype(float)
        # Mark NaN RB211 as NaN employment
        m.loc[pd.to_numeric(m["RB211"], errors="coerce").isna(), "employed"] = np.nan
    elif "PL031" in m.columns:
        m = m.copy()
        pl = pd.to_numeric(m["PL031"], errors="coerce")
        m["employed"] = pl.isin([1, 2]).astype(float)
        m.loc[pl.isna(), "employed"] = np.nan
    else:
        m = m.copy()
        m["employed"] = np.nan

    # Iterate over deciles + "All"
    deciles = sorted(m["decile"].dropna().unique().tolist()) + ["All"]

    for dec in deciles:
        sub = m if dec == "All" else m[m["decile"] == dec]
        if sub.empty:
            continue

        wt = pd.to_numeric(sub["weight"], errors="coerce").fillna(0).values
        total_w = wt.sum()
        if total_w == 0:
            continue

        base = {"country": country, "year": year, "decile": dec}

        # ── SP-SILC-1 ───────────────────────────────────────────────────────
        if year == 2015 and "PS101" in sub.columns:
            ps = pd.to_numeric(sub["PS101"], errors="coerce")
            valid = ps.notna().values
            flag = (ps == 6).astype(float).values
            tw_valid = (wt * valid).sum()
            if tw_valid > 0:
                val = np.nansum(flag * wt * valid) / tw_valid * 100
                results.append({**base, "code": "SP-SILC-1", "value": val})
        elif year == 2022 and "PS110" in sub.columns:
            ps = pd.to_numeric(sub["PS110"], errors="coerce")
            valid = ps.notna().values
            flag = (ps == 6).astype(float).values
            tw_valid = (wt * valid).sum()
            if tw_valid > 0:
                val = np.nansum(flag * wt * valid) / tw_valid * 100
                results.append({**base, "code": "SP-SILC-1", "value": val})

        # ── SP-SILC-2 ───────────────────────────────────────────────────────
        if year in SP_YEARS and "PS102" in sub.columns:
            ps = pd.to_numeric(sub["PS102"], errors="coerce")
            valid = ps.notna().values
            flag = ps.isin([2, 3, 4]).astype(float).values
            tw_valid = (wt * valid).sum()
            if tw_valid > 0:
                val = np.nansum(flag * wt * valid) / tw_valid * 100
                results.append({**base, "code": "SP-SILC-2", "value": val})

        # ── GE-SILC-1 ───────────────────────────────────────────────────────
        if "sex" in sub.columns and "employed" not in [np.nan]:
            sx = pd.to_numeric(sub["sex"], errors="coerce").values
            emp = pd.to_numeric(sub["employed"], errors="coerce").values
            emp_known = np.isfinite(emp)
            men = sx == 1
            women = sx == 2

            wt_mk = np.where(men & emp_known, wt, 0.0)
            wt_fk = np.where(women & emp_known, wt, 0.0)
            tw_mk = wt_mk.sum()
            tw_fk = wt_fk.sum()

            if tw_mk > 0 and tw_fk > 0:
                rate_m = np.nansum(np.where(emp_known, emp, 0.0) * wt_mk) / tw_mk
                rate_f = np.nansum(np.where(emp_known, emp, 0.0) * wt_fk) / tw_fk
                gap = (rate_m - rate_f) * 100  # in percentage points
                results.append({**base, "code": "GE-SILC-1", "value": gap})

        # ── GE-SILC-2 ───────────────────────────────────────────────────────
        if "sex" in sub.columns and "PY010G" in sub.columns:
            sx = pd.to_numeric(sub["sex"], errors="coerce").values
            inc = pd.to_numeric(sub["PY010G"], errors="coerce").values
            emp = pd.to_numeric(sub["employed"], errors="coerce").values
            emp_flag = emp == 1  # boolean array

            men_emp = emp_flag & (sx == 1) & np.isfinite(inc) & (inc > 0)
            women_emp = emp_flag & (sx == 2) & np.isfinite(inc) & (inc > 0)

            if men_emp.sum() >= MIN_CELL and women_emp.sum() >= MIN_CELL:
                med_m = weighted_median(inc[men_emp], wt[men_emp])
                med_f = weighted_median(inc[women_emp], wt[women_emp])
                if np.isfinite(med_m) and np.isfinite(med_f) and med_m > 0:
                    gap = (1.0 - med_f / med_m) * 100
                    results.append({**base, "code": "GE-SILC-2", "value": gap})

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load decile lookup (minimal columns)
    print("Loading decile lookup from personal merged CSV ...")
    decile_lookup = pd.read_csv(
        PERSONAL_MERGED,
        usecols=["PB010", "PB020", "PB030", "decile"],
        dtype={"PB030": str},
    )
    decile_lookup["PB030"] = decile_lookup["PB030"].str.strip()
    print(f"  {len(decile_lookup):,} rows | {decile_lookup['PB010'].nunique()} years | "
          f"{decile_lookup['PB020'].nunique()} countries")

    # 2. Pre-index decile lookup by (year, country) for fast access
    print("Indexing decile lookup by country/year ...")
    decile_by_cy = {
        (yr, ct): grp[["PB030", "decile"]].reset_index(drop=True)
        for (yr, ct), grp in decile_lookup.groupby(["PB010", "PB020"])
    }
    del decile_lookup  # free memory
    print(f"  {len(decile_by_cy)} country/year groups indexed")

    # 3. Process each country/year
    all_files = list(collect_files(SILC_BASE))
    print(f"\nProcessing {len(all_files)} country/year combinations ...")

    all_results = []

    for i, (country, year, p_path, r_path) in enumerate(all_files):
        sys.stdout.write(f"\r  [{i+1:4d}/{len(all_files)}] {country} {year}   ")
        sys.stdout.flush()

        try:
            r_df = load_r(r_path, year)
        except Exception as e:
            print(f"\n  WARNING: skipping R-file {r_path}: {e}")
            continue

        try:
            p_df = load_p(p_path, year)
        except Exception as e:
            print(f"\n  WARNING: skipping P-file {p_path}: {e}")
            continue

        # Join R + P on person ID
        merged = r_df.merge(p_df, on="PB030", how="inner")

        # Join with decile lookup for this country/year (fast O(1) lookup)
        lk = decile_by_cy.get((year, country), pd.DataFrame(columns=["PB030", "decile"]))
        merged = merged.merge(lk, on="PB030", how="inner")

        if merged.empty:
            continue

        results = compute_indicators(merged, country, year)
        all_results.extend(results)

    print("\n\nBuilding catalog ...")

    if not all_results:
        print("ERROR: No results generated. Check paths and variable availability.")
        sys.exit(1)

    df = pd.DataFrame(all_results)

    # Attach metadata
    for code, meta in SUPP_META.items():
        mask = df["code"] == code
        for col, val in meta.items():
            df.loc[mask, col] = val

    df = df[CATALOG_COLS]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(df):,} rows -> {OUT_PATH}")
    summary = df.groupby("code").agg(
        n_rows=("value", "count"),
        n_countries=("country", "nunique"),
        n_years=("year", "nunique"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
