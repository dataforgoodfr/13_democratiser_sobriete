"""
EU-SILC Extractor
=================
Reads raw EU-SILC cross-sectional microdata (H/D/P/R files) and computes one
CSV per indicator. No value filtering: zeros and NaNs are preserved as-is so
the quality checker can diagnose them.

Output per indicator: output/indicators/{CODE}.csv
  country, year, decile, value, n_obs, n_weighted
  decile: 1-10 or "All"
  value:  share of (weighted) population meeting the indicator condition (%)
          NaN when variable was entirely absent for that country-year

Usage
-----
    # Compute ALL EU-SILC indicators (skips existing files):
    python extract_eusilc.py

    # Force recompute specific codes:
    python extract_eusilc.py --force HQ-SILC-1 HE-SILC-2

    # Force recompute ALL:
    python extract_eusilc.py --force-all

Design notes
------------
- One merged-data cache per country-year (Parquet) is written to output/cache/eusilc/
  so that adding a new indicator doesn't require re-reading raw CSV files.
- The cache stores ONLY the columns needed across all indicators (keeps it lean).
- Income decile is assigned once during merge and stored in the cache.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── imports from this package ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config
from indicators import BY_SOURCE, BY_CODE

config.ensure_dirs()
CACHE_SILC = config.CACHE_DIR / "eusilc"
CACHE_SILC.mkdir(parents=True, exist_ok=True)

# ─── Column catalogue ─────────────────────────────────────────────────────────
# Columns needed from each file type.  Keys match file suffix (H/D/P/R).

H_COLS_NEEDED = [
    "HB010", "HB020", "HB030",   # year, country, household_id
    "DB090",                      # household weight
    "HY020",                      # equivalised disposable income (Eurostat computed)
    # Household indicators
    "HH030", "HH040",             # rooms, leaking roof
    "HD080",                      # replace furniture
    "HC070", "HC060", "HC003",    # cool, warm, renovation
    "HS160", "HS170", "HS180",   # dark, noise, pollution
    "HS011", "HS021",             # mortgage arrears, utility arrears
    "HS050", "HS060", "HS040",   # food, expenses, holiday
    "HS120",                      # make ends meet
]

D_COLS_NEEDED = [
    "DB010", "DB020", "DB030",   # year, country, household_id
    "DB040",                      # region (NUTS1)
    "DB090",                      # household weight (prefer over H version)
    "HY020",                      # equivalised income (prefer D version when available)
    # Household composition columns (for overcrowding calc)
    "HH030",
]

P_COLS_NEEDED = [
    "PB010", "PB020", "PB030",   # year, country, person_id
    "PB040",                      # person weight
    "PB140",                      # year of birth (for overcrowding age classification)
    "PB150",                      # sex (1=male, 2=female)
    "PB180", "PB190",            # marital status, other
    "PB200",                      # de facto partnership status (1/2=in couple; for overcrowding)
    # Personal indicators
    "PH010", "PH020", "PH030", "PH040", "PH050", "PH060",  # health
    "PL080", "PL086", "PL031", "PL141", "PL145",              # work (PL031=activity status pre-2022)
    "PE010", "PE041",                                        # education
    "PW010", "PW191",                                        # subjective
    "PD050", "PD060", "PD070",                              # social
    "PS101", "PS110", "PS102",                               # supplementary (ad-hoc)
    "PY010G",                                                # gross wages (gender gap)
]

R_COLS_NEEDED = [
    "RB010", "RB020", "RB030",   # year, country, person_id
    "RB050",                      # personal weight (cross-sectional)
    "RB080",                      # year of birth
    "RB090",                      # sex
    "RB211",                      # employment status (for gender gap)
    # Household link: personal ID → household ID
    # In EU-SILC the household ID in R/P files can be derived from the first 
    # (len-2) digits of the personal ID – confirmed in Eurostat UDB manuals.
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _two_digit_year(year: int) -> str:
    return str(year)[-2:]


def _read_silc_file(path: Path, cols_needed: list[str]) -> pd.DataFrame | None:
    """Read a EU-SILC file, keeping only available needed columns."""
    if not path.exists():
        return None
    try:
        sample = pd.read_csv(path, nrows=2)
        usecols = [c for c in cols_needed if c in sample.columns]
        if len(usecols) < 3:
            return None
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        return df
    except Exception as exc:
        tqdm.write(f"    ⚠ Could not read {path.name}: {exc}")
        return None


def _assign_deciles(income: pd.Series, weight: pd.Series) -> pd.Series:
    """
    Assign income decile (1-10) for each observation using weighted quantiles.
    Returns NaN for rows where income is NaN.
    """
    valid = income.notna() & weight.notna() & (weight > 0)
    if valid.sum() < 10:
        return pd.Series(np.nan, index=income.index)

    inc_v = income[valid].to_numpy(float)
    w_v   = weight[valid].to_numpy(float)

    # Weighted percentile boundaries (10th, 20th, ..., 90th)
    # Sort by income
    order = np.argsort(inc_v)
    inc_s, w_s = inc_v[order], w_v[order]
    cum_w = np.cumsum(w_s)
    total_w = cum_w[-1]
    breaks = total_w * np.arange(0.1, 1.0, 0.1)   # 9 thresholds

    thresholds = np.interp(breaks, cum_w, inc_s)

    # Assign decile: count how many thresholds the income exceeds, +1
    result = np.full(len(income), np.nan)
    inc_arr = income.to_numpy(float)
    valid_arr = valid.to_numpy()
    result[valid_arr] = (inc_arr[valid_arr, np.newaxis] > thresholds).sum(axis=1) + 1
    return pd.Series(result, index=income.index)


def _weighted_share(flag: np.ndarray, weight: np.ndarray) -> tuple[float, int, float]:
    """
    Compute weighted share (%) of flag==1 rows.
    Returns (value_pct, n_obs, n_weighted) where NaN means no valid data.
    """
    avail = ~np.isnan(flag) & ~np.isnan(weight) & (weight > 0)
    n_obs = int(avail.sum())
    if n_obs == 0:
        return np.nan, 0, 0.0
    f = flag[avail]
    w = weight[avail]
    n_weighted = float(w.sum())
    if n_weighted == 0:
        return np.nan, n_obs, 0.0
    value = float((f * w).sum() / n_weighted * 100)
    return value, n_obs, n_weighted


# ─── Household-level flag computation ─────────────────────────────────────────

def _build_hh_flags(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Returns {code: flag_series (float 0/1/NaN)} for all household indicators.
    NaN = variable absent for that row.
    """
    flags: dict[str, pd.Series] = {}

    def col(c):
        return df[c] if c in df.columns else pd.Series(np.nan, index=df.index)

    year = (df["HB010"] if "HB010" in df.columns
            else df["year"] if "year" in df.columns
            else df.get("DB010", pd.Series(2000, index=df.index)))

    # Housing quality
    if "HH030" in df.columns:
        # Overcrowding: compute required rooms from household composition
        # Simplified: available rooms < (n_adults + 1 + max(0, n_children-1)//2)
        # Since we don't always have composition in H file, use the overcrowd
        # column from the merged file when available; fall back to NaN
        if "overcrowded" in df.columns:
            flags["HQ-SILC-1"] = df["overcrowded"].astype(float)
        else:
            flags["HQ-SILC-1"] = pd.Series(np.nan, index=df.index)
    else:
        flags["HQ-SILC-1"] = pd.Series(np.nan, index=df.index)

    def binary(c, vals):
        s = col(c)
        valid = s.notna()
        f = pd.Series(np.nan, index=df.index)
        f[valid] = s[valid].isin(vals).astype(float)
        return f

    def cmp(c, op, v):
        s = col(c)
        valid = s.notna()
        f = pd.Series(np.nan, index=df.index)
        if op == "==":
            f[valid] = (s[valid] == v).astype(float)
        elif op == ">=":
            f[valid] = (s[valid] >= v).astype(float)
        elif op == ">":
            f[valid] = (s[valid] > v).astype(float)
        elif op == "!=":
            f[valid] = (s[valid] != v).astype(float)
        return f

    flags["HQ-SILC-2"] = binary("HD080",  [2, 3])
    flags["HQ-SILC-3"] = cmp("HC070",     "==", 2)
    flags["HQ-SILC-4"] = cmp("HS160",     "==", 1)
    flags["HQ-SILC-5"] = cmp("HS170",     "==", 1)
    flags["HQ-SILC-6"] = cmp("HH040",     "==", 1)
    flags["HQ-SILC-7"] = cmp("HS180",     "==", 1)
    flags["HQ-SILC-8"] = binary("HC003",  [4, 99])
    # HC060 coding changed in 2016 → set that year to NaN (known data quality issue)
    he1 = cmp("HC060", "==", 2)
    he1[year == 2016] = np.nan   # 2016 unreliable for this variable
    flags["HE-SILC-1"] = he1

    # Utility arrears: in [1,2] after 2008, ==1 before
    hs021 = col("HS021")
    valid_u = hs021.notna()
    f_u = pd.Series(np.nan, index=df.index)
    pre2008 = year < 2008
    f_u[valid_u &  pre2008] = (hs021[valid_u &  pre2008] == 1).astype(float)
    f_u[valid_u & ~pre2008] = hs021[valid_u & ~pre2008].isin([1, 2]).astype(float)
    flags["HE-SILC-2"] = f_u

    # Mortgage arrears
    hs011 = col("HS011")
    valid_m = hs011.notna()
    f_m = pd.Series(np.nan, index=df.index)
    f_m[valid_m &  pre2008] = (hs011[valid_m &  pre2008] == 1).astype(float)
    f_m[valid_m & ~pre2008] = hs011[valid_m & ~pre2008].isin([1, 2]).astype(float)
    flags["HH-SILC-1"] = f_m

    flags["AN-SILC-1"] = cmp("HS050", "==", 2)
    flags["ES-SILC-1"] = cmp("HS060", "==", 2)
    flags["ES-SILC-2"] = binary("HS120", [1, 2])
    flags["TS-SILC-1"] = cmp("HS040",  "==", 2)

    # Living alone: household_size == 1 → need household size variable
    # Use the 'household_size' column if present in merged data, else NaN
    if "household_size" in df.columns:
        hs = df["household_size"]
        f_la = pd.Series(np.nan, index=df.index)
        valid_la = hs.notna()
        f_la[valid_la] = (hs[valid_la] == 1).astype(float)
        flags["EC-SILC-4"] = f_la
    else:
        flags["EC-SILC-4"] = pd.Series(np.nan, index=df.index)

    return flags


# ─── Personal-level flag computation ──────────────────────────────────────────

def _build_pers_flags(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Returns {code: flag_series (float 0/1/NaN)} for all personal indicators.
    df must contain merged P+R columns + 'age' (computed from RB080) + 'decile'.
    """
    flags: dict[str, pd.Series] = {}

    def col(c):
        return df[c] if c in df.columns else pd.Series(np.nan, index=df.index)

    def binary(c, vals):
        s = col(c)
        valid = s.notna()
        f = pd.Series(np.nan, index=df.index)
        f[valid] = s[valid].isin(vals).astype(float)
        return f

    def cmp(c, op, v):
        s = col(c)
        valid = s.notna()
        f = pd.Series(np.nan, index=df.index)
        if op == "==":
            f[valid] = (s[valid] == v).astype(float)
        elif op == "<":
            f[valid] = (s[valid] < v).astype(float)
        elif op == ">":
            f[valid] = (s[valid] > v).astype(float)
        elif op == "<=":
            f[valid] = (s[valid] <= v).astype(float)
        return f

    # Health
    flags["EL-SILC-1"] = cmp("PW010",  "<",  3)
    flags["AH-SILC-1"] = binary("PH010", [4, 5])
    flags["AH-SILC-2"] = cmp("PH020",  "==", 1)
    flags["AH-SILC-3"] = binary("PH030", [1, 2])
    flags["AH-SILC-4"] = cmp("PL086",  ">",  0)
    flags["AC-SILC-1"] = cmp("PH050",  "==", 1)
    flags["AC-SILC-3"] = cmp("PH060",  "==", 1)
    flags["AC-SILC-4"] = cmp("PH040",  "==", 1)

    # Social
    flags["IC-SILC-1"] = binary("PD060", [2, 3])
    flags["IC-SILC-2"] = binary("PD070", [2, 3])
    flags["EC-SILC-1"] = binary("PD050", [2, 3])
    flags["EC-SILC-2"] = cmp("PW191",  "<",  3)
    flags["EC-SILC-3"] = binary("PD050", [2, 3])  # same variable as EC-SILC-1

    # Education
    age = col("age")
    pe041 = col("PE041")
    # IS-SILC-3: No formal education, age > 15
    f_is3 = pd.Series(np.nan, index=df.index)
    valid_age_edu = age.notna() & (age > 15)
    pe041_valid = pe041.notna() & valid_age_edu
    f_is3[valid_age_edu] = 0.0          # default: has education
    f_is3[pe041_valid]    = (pe041[pe041_valid] == 0).astype(float)
    f_is3[valid_age_edu & pe041.isna()] = 1.0  # NaN education = no education
    flags["IS-SILC-3"] = f_is3

    flags["IS-SILC-4"] = cmp("PE010", "==", 2)
    flags["IS-SILC-5"] = binary("PE041", [0, 100])  # ISCED 0 (no education) or 100 (primary only)

    # Work
    year_col = col("PB010")
    pl141 = col("PL141")
    # RT-SILC-1: fixed-term contracts (variable changed coding in 2021)
    f_rt1 = pd.Series(np.nan, index=df.index)
    adult = age.notna() & (age > 17)
    valid141 = pl141.notna() & adult
    pre2021 = year_col < 2021
    f_rt1[valid141 &  pre2021] = (pl141[valid141 &  pre2021] == 2).astype(float)
    f_rt1[valid141 & ~pre2021] = pl141[valid141 & ~pre2021].isin([11, 12]).astype(float)
    flags["RT-SILC-1"] = f_rt1

    pl145 = col("PL145")
    f_rt2 = pd.Series(np.nan, index=df.index)
    valid145 = pl145.notna() & adult
    f_rt2[valid145] = (pl145[valid145] == 2).astype(float)
    flags["RT-SILC-2"] = f_rt2

    flags["RU-SILC-1"] = cmp("PL080", ">", 5)

    # Supplementary (only present in specific ad-hoc waves)
    ps101 = col("PS101")
    ps110 = col("PS110")
    # SP-SILC-1: voluntary activity, year 2015 uses PS101, 2022 uses PS110
    f_sp1 = pd.Series(np.nan, index=df.index)
    v101 = ps101.notna()
    v110 = ps110.notna()
    f_sp1[v101] = (ps101[v101] == 6).astype(float)
    f_sp1[v110] = (ps110[v110] == 6).astype(float)
    flags["SP-SILC-1"] = f_sp1
    flags["SP-SILC-2"] = binary("PS102", [2, 3, 4])

    # Gender gap indicators are computed at country-year level (not per person),
    # so we mark them as NaN here – they are computed separately in the aggregation.
    flags["GE-SILC-1"] = pd.Series(np.nan, index=df.index)
    flags["GE-SILC-2"] = pd.Series(np.nan, index=df.index)

    return flags


# ─── Aggregation ──────────────────────────────────────────────────────────────

def _aggregate(df: pd.DataFrame, flags: dict[str, pd.Series],
                weight_col: str, codes: list[str]) -> pd.DataFrame:
    """
    Aggregate flags per (country, year, decile) + (country, year, 'All').
    Returns long-format DataFrame: country, year, decile, code, value, n_obs, n_weighted.
    """
    w = df[weight_col].to_numpy(dtype=float)
    country_col = df.get("country", df.get("HB020", df.get("PB020"))).to_numpy()
    year_col    = df.get("year",    df.get("HB010", df.get("PB010"))).to_numpy(dtype=float)
    decile_col  = df["decile"].to_numpy(dtype=float)

    rows = []
    for code in codes:
        if code not in flags:
            continue
        flag_arr = flags[code].to_numpy(dtype=float)

        # Per decile
        for d in range(1, 11):
            mask = (decile_col == d) & ~np.isnan(decile_col)
            if mask.sum() == 0:
                continue
            # Get unique country-years for this decile
            cy_in_d = set(zip(country_col[mask], year_col[mask]))
            for (ctry, yr) in cy_in_d:
                m = mask & (country_col == ctry) & (year_col == yr)
                val, n_obs, n_w = _weighted_share(flag_arr[m], w[m])
                rows.append((ctry, int(yr), d, code, val, n_obs, n_w))

        # "All" deciles aggregated
        cy_all = set(zip(country_col, year_col[~np.isnan(year_col)]))
        for (ctry, yr) in cy_all:
            m = (country_col == ctry) & (year_col == yr) & ~np.isnan(decile_col)
            val, n_obs, n_w = _weighted_share(flag_arr[m], w[m])
            rows.append((ctry, int(yr), "All", code, val, n_obs, n_w))

    return pd.DataFrame(rows, columns=["country", "year", "decile", "code", "value", "n_obs", "n_weighted"])


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _cache_path(country: str, year: int) -> Path:
    return CACHE_SILC / f"{country}_{year}.parquet"


def _build_and_cache(country: str, year: int) -> dict[str, pd.DataFrame | None]:
    """
    Read raw EU-SILC files for one country-year, merge, assign deciles,
    and save a Parquet cache.  Returns dict with 'household' and 'personal' dfs.
    """
    base = config.SILC_RAW_DIR / country / str(year)
    yy = _two_digit_year(year)

    # ── Household (H) ──────────────────────────────────────────────────────
    hpath = base / f"UDB_c{country}{yy}H.csv"
    dpath = base / f"UDB_c{country}{yy}D.csv"

    h_df = _read_silc_file(hpath, H_COLS_NEEDED)
    d_df = _read_silc_file(dpath, D_COLS_NEEDED)

    if h_df is None and d_df is None:
        return {"household": None, "personal": None}

    # ── Income: H file primary (HY020), D file fallback ────────────────────
    # ── Weight (DB090): always in D file; H file never has it  ─────────────
    # In most country-years, HY020 is in the H file and DB090 in the D file.
    # In some years, HY020 also appears in the D file (used as override).
    if h_df is not None:
        hh = h_df.copy()
    else:
        # Only D file available – rename to H conventions
        hh = d_df.rename(columns={"DB010": "HB010", "DB020": "HB020", "DB030": "HB030"})
        d_df = None   # already consumed

    if d_df is not None:
        # Build the D-side join frame (always bring DB090; optionally HY020)
        d_keep = ["DB010", "DB020", "DB030", "DB090"]
        if "HY020" in d_df.columns:
            d_keep.append("HY020")
        d_join = d_df[[c for c in d_keep if c in d_df.columns]].rename(
            columns={"DB010": "HB010", "DB020": "HB020", "DB030": "HB030",
                     "DB090": "_DB090_d", "HY020": "_HY020_d"}
        )
        hh = hh.merge(d_join, on=["HB010", "HB020", "HB030"], how="left")
        # Weight: D file is authoritative (DB090 never lives in H file)
        hh["DB090"] = hh.pop("_DB090_d")
        # Income: prefer H file; fall back to D file if H doesn't have it
        if "_HY020_d" in hh.columns:
            if "HY020" not in hh.columns:
                hh["HY020"] = hh.pop("_HY020_d")
            else:
                hh["HY020"] = hh["HY020"].combine_first(hh.pop("_HY020_d"))
    else:
        hh["DB090"] = hh.get("DB090", pd.Series(np.nan, index=hh.index))

    # Standardise columns
    hh = hh.rename(columns={"HB010": "year", "HB020": "country", "HB030": "hh_id"})
    hh["year"]    = year
    hh["country"] = country

    # Household size (count persons per hh from R file or use hh_members col)
    rpath = base / f"UDB_c{country}{yy}R.csv"
    r_df  = _read_silc_file(rpath, R_COLS_NEEDED)
    if r_df is not None:
        r_df = r_df.rename(columns={"RB010": "year", "RB020": "country", "RB030": "person_id"})
        r_df["year"] = year
        r_df["country"] = country
        # Derive household ID from person ID (same float-safe normalization)
        rid_clean = pd.to_numeric(r_df["person_id"], errors="coerce").astype("Int64").astype(str)
        r_df["hh_id"] = rid_clean.str[:-2].str.lstrip("0").replace("", "0")
        hh_sizes = r_df.groupby("hh_id").size().rename("household_size").reset_index()
        hh["hh_id"] = hh["hh_id"].astype(str)
        hh = hh.merge(hh_sizes, on="hh_id", how="left")

    # Assign income decile using equivalised disposable income (OECD modified scale)
    # First adult → 1.0, additional adults (≥14) → 0.5, children (<14) → 0.3
    inc_raw = pd.to_numeric(hh.get("HY020", pd.Series(np.nan)), errors="coerce")
    wgt     = pd.to_numeric(hh.get("DB090", pd.Series(np.nan)), errors="coerce")
    if r_df is not None and "RB080" in r_df.columns:
        r_age = year - pd.to_numeric(r_df["RB080"], errors="coerce")
        r_oecd = pd.DataFrame({"hh_id": r_df["hh_id"].values,
                               "age":   r_age.values})
        r_oecd["oecd_w"] = np.where(r_oecd["age"].isna(), 0.5,
                                    np.where(r_oecd["age"] < 14, 0.3, 0.5))
        # Oldest person per household becomes head → weight 1.0
        r_oecd = r_oecd.sort_values("age", ascending=False)
        head_idx = r_oecd.groupby("hh_id").head(1).index
        r_oecd.loc[head_idx, "oecd_w"] = 1.0
        equiv = (r_oecd.groupby("hh_id")["oecd_w"]
                 .sum().rename("equiv_size").reset_index())
        hh = hh.merge(equiv, on="hh_id", how="left")
        inc_equi = inc_raw / hh["equiv_size"].replace(0, np.nan)
        hh["decile"] = _assign_deciles(inc_equi, wgt)
    else:
        hh["decile"] = _assign_deciles(inc_raw, wgt)

    # Overcrowding computed below after personal data is loaded (requires PB140, PB200)

    # ── Personal (P+R) ──────────────────────────────────────────────────────
    ppath = base / f"UDB_c{country}{yy}P.csv"
    p_df = _read_silc_file(ppath, P_COLS_NEEDED)

    pers = None
    if p_df is not None:
        p_df = p_df.rename(columns={"PB010": "year", "PB020": "country", "PB030": "person_id"})
        p_df["year"]    = year
        p_df["country"] = country

        # ── Proper EU overcrowding definition ────────────────────────────────
        # Required rooms = 1 (living room)
        #   + 1 per couple (adults in de facto union, paired)
        #   + 1 per unpaired adult (single / widow / etc.)
        #   + 1 per pair of teens (12–17)
        #   + 1 per remaining solo teen
        #   + 1 per pair of children (<12)
        #   + 1 per remaining solo child
        # Mirrors 0_raw_indicator_EU-SILC.py → calculate_overcrowding()
        if "HH030" in hh.columns and "PB140" in p_df.columns:
            pid_oc = pd.to_numeric(p_df["person_id"], errors="coerce").astype("Int64").astype(str)
            oc_hh_id = pid_oc.str[:-2].str.lstrip("0").replace("", "0")
            oc_age   = year - pd.to_numeric(p_df["PB140"], errors="coerce")
            if "PB200" in p_df.columns:
                pb200 = pd.to_numeric(p_df["PB200"], errors="coerce")
                in_union = pb200.isin([1, 2])
            else:
                in_union = pd.Series(False, index=p_df.index)
            oc = pd.DataFrame({
                "hh_id":            oc_hh_id.values,
                "age":              oc_age.values,
                "in_union":         in_union.values,
            })
            oc["adult_union"]  = (oc["age"] >= 18) & oc["in_union"]
            oc["adult_single"] = (oc["age"] >= 18) & ~oc["in_union"]
            oc["teen"]         = (oc["age"] >= 12) & (oc["age"] < 18)
            oc["child"]        = oc["age"] < 12
            hh_comp = oc.groupby("hh_id").agg(
                n_au=("adult_union",  "sum"),
                n_as=("adult_single", "sum"),
                n_t =("teen",         "sum"),
                n_c =("child",        "sum"),
            ).reset_index()
            hh_comp["required_rooms"] = (
                1
                + hh_comp["n_au"] // 2
                + hh_comp["n_as"] + (hh_comp["n_au"] % 2)
                + hh_comp["n_t"] // 2 + hh_comp["n_t"] % 2
                + hh_comp["n_c"] // 2 + hh_comp["n_c"] % 2
            )
            hh = hh.merge(hh_comp[["hh_id", "required_rooms"]], on="hh_id", how="left")
            n_rooms = pd.to_numeric(hh["HH030"], errors="coerce")
            req     = pd.to_numeric(hh["required_rooms"], errors="coerce")
            hh["overcrowded"] = np.where(
                n_rooms.isna() | req.isna(), np.nan,
                (n_rooms < req).astype(float)
            )
            hh.drop(columns=["required_rooms"], errors="ignore", inplace=True)

        if r_df is not None:
            r_merge = r_df[["year", "country", "person_id", "RB050",
                             "RB080", "RB090"] + [c for c in ["RB211"] if c in r_df.columns]
                           ].copy()
            # Normalise IDs: some country-years store person IDs as floats (e.g. HR 2019)
            r_merge["person_id"] = pd.to_numeric(r_merge["person_id"], errors="coerce").astype("Int64").astype(str)
            p_df["person_id"]    = pd.to_numeric(p_df["person_id"],    errors="coerce").astype("Int64").astype(str)
            pers = p_df.merge(r_merge, on=["year", "country", "person_id"], how="left")
        else:
            pers = p_df
            for c in ["RB050", "RB080", "RB090"]:
                if c not in pers.columns:
                    pers[c] = np.nan

        # Derive household ID from person ID (last 2 chars = person number within HH)
        # Convert via numeric first to handle float-formatted IDs (e.g. HR 2019: 10001.0 → "10001")
        pid_clean = pd.to_numeric(pers["person_id"], errors="coerce").astype("Int64").astype(str)
        pers["hh_id"] = pid_clean.str[:-2].str.lstrip("0").replace("", "0")

        # Merge household income decile into personal data
        hh_dec = hh[["hh_id", "decile"]].drop_duplicates("hh_id")
        pers = pers.merge(hh_dec, on="hh_id", how="left")

        # Age from year of birth
        if "RB080" in pers.columns:
            pers["age"] = year - pd.to_numeric(pers["RB080"], errors="coerce")

    return {"household": hh, "personal": pers}


# ─── Compute gender-gap indicators separately ─────────────────────────────────

def _compute_gender_gap(pers: pd.DataFrame, country: str, year: int) -> list:
    """
    GE-SILC-1: gender employment gap (percentage points)
    GE-SILC-2: gender pay gap (%)
      Formula: (1 - median_wage_women_in_decile / median_wage_men_overall) * 100
      The men's reference is always the country-level (All) weighted median wage,
      so per-decile values show where women in that decile stand relative to the
      single national male benchmark.
    Computed at country-year level (one value per decile, not per person).
    """
    rows = []
    if pers is None or "RB090" not in pers.columns:
        return rows

    age = pers.get("age", pd.Series(np.nan, index=pers.index))
    working_age = age.notna() & (age >= 20) & (age <= 64)
    w = pd.to_numeric(pers.get("RB050", pd.Series(np.nan)), errors="coerce")
    sex = pd.to_numeric(pers.get("RB090", pd.Series(np.nan)), errors="coerce")

    # Employment: RB211 == 1 or PL031 in [1,2] if available
    if "RB211" in pers.columns:
        emp = pd.to_numeric(pers["RB211"], errors="coerce") == 1
    elif "PL031" in pers.columns:
        emp = pd.to_numeric(pers["PL031"], errors="coerce").isin([1, 2])
    else:
        return rows

    def _wmedian(vals, weights):
        """Weighted median."""
        fin = np.isfinite(vals) & np.isfinite(weights) & (weights > 0) & (vals > 0)
        if fin.sum() == 0:
            return np.nan
        v, w_ = vals[fin], weights[fin]
        order = np.argsort(v)
        sv, sw = v[order], w_[order]
        cumw = np.cumsum(sw)
        mid = cumw[-1] / 2.0
        idx = np.searchsorted(cumw, mid)
        return float(sv[min(idx, len(sv) - 1)])

    # Pre-compute overall men's median wage — single reference for GE-SILC-2
    py = pd.to_numeric(pers.get("PY010G", pd.Series(np.nan)), errors="coerce")
    men_all = working_age & (sex == 1) & emp & py.notna() & (py > 0)
    med_m_all = _wmedian(py[men_all].to_numpy(), w[men_all].to_numpy()) if men_all.sum() >= 5 else np.nan

    for d in list(range(1, 11)) + ["All"]:
        if d == "All":
            mask = working_age
        else:
            mask = working_age & (pers.get("decile", pd.Series(np.nan)) == d)

        if mask.sum() < 5:
            continue

        men   = mask & (sex == 1)
        women = mask & (sex == 2)

        def rate(sel):
            valid = sel & emp.notna()
            denom = (w[valid] * 1.0).sum()
            if denom == 0:
                return np.nan
            return (w[valid & emp].sum() / denom) * 100

        r_men   = rate(men)
        r_women = rate(women)
        gap1 = (r_men - r_women) if (not np.isnan(r_men) and not np.isnan(r_women)) else np.nan
        rows.append((country, year, d, "GE-SILC-1", gap1,
                     int(mask.sum()), float(w[mask].sum())))

        # Pay gap: women's median in this decile vs overall men's median
        emp_women = women & emp & py.notna() & (py > 0)
        if np.isfinite(med_m_all) and med_m_all > 0 and emp_women.sum() >= 5:
            med_w = _wmedian(py[emp_women].to_numpy(), w[emp_women].to_numpy())
            gap2 = (1 - med_w / med_m_all) * 100 if np.isfinite(med_w) else np.nan
        else:
            gap2 = np.nan
        rows.append((country, year, d, "GE-SILC-2", gap2,
                     int(mask.sum()), float(w[mask].sum())))

    return rows


# ─── Main extraction loop ──────────────────────────────────────────────────────

def extract_eusilc(force_codes: set[str] | None = None, force_all: bool = False):
    """
    Main extraction function.

    Parameters
    ----------
    force_codes : set of indicator codes to recompute even if CSV exists.
    force_all   : if True, recompute ALL EU-SILC indicators.
    """
    silc_indicators = BY_SOURCE.get("EU-SILC", [])
    all_codes = [ind["code"] for ind in silc_indicators]

    # Which codes need (re)computation?
    codes_to_run = set()
    if force_all:
        codes_to_run = set(all_codes)
    else:
        for code in all_codes:
            out_path = config.INDICATORS_DIR / f"{code}.csv"
            if not out_path.exists() or (force_codes and code in force_codes):
                codes_to_run.add(code)

    if not codes_to_run:
        print("EU-SILC: all indicator CSVs are up to date. Use --force to recompute.")
        return

    hh_codes   = [c for c in codes_to_run if BY_CODE.get(c, {}).get("level") == "household"
                  and c not in ("GE-SILC-1", "GE-SILC-2")]
    pers_codes = [c for c in codes_to_run if BY_CODE.get(c, {}).get("level") == "personal"]

    print(f"EU-SILC: computing {len(codes_to_run)} indicators "
          f"({len(hh_codes)} HH + {len(pers_codes)} personal)")

    # Accumulators: one list per code
    accum: dict[str, list] = {c: [] for c in codes_to_run}

    # Iterate over available country × year combinations
    raw_root = config.SILC_RAW_DIR
    if not raw_root.exists():
        print(f"  ERROR: EU-SILC raw data directory not found:\n  {raw_root}")
        return

    country_dirs = sorted(d.name for d in raw_root.iterdir() if d.is_dir())
    total_cy = sum(
        1 for c in country_dirs for y in config.SILC_YEARS
        if (raw_root / c / str(y)).exists()
    )

    pbar = tqdm(total=total_cy, desc="EU-SILC country-years", unit="cy")
    for country in country_dirs:
        if country not in config.EU_COUNTRIES:
            pbar.update(len([y for y in config.SILC_YEARS
                              if (raw_root / country / str(y)).exists()]))
            continue

        for year in config.SILC_YEARS:
            yr_path = raw_root / country / str(year)
            if not yr_path.exists():
                continue
            pbar.set_postfix_str(f"{country} {year}")

            try:
                data = _build_and_cache(country, year)
            except Exception as exc:
                tqdm.write(f"  ⚠ {country} {year}: {exc}")
                pbar.update(1)
                continue

            hh   = data["household"]
            pers = data["personal"]

            # ── Household indicators ──────────────────────────────────────
            if hh is not None and hh_codes:
                hh_flags = _build_hh_flags(hh)
                hh["decile"] = pd.to_numeric(hh["decile"], errors="coerce")
                w_arr = pd.to_numeric(hh.get("DB090", pd.Series(np.nan)), errors="coerce").to_numpy()
                dec_arr = hh["decile"].to_numpy(dtype=float)

                for code in hh_codes:
                    if code not in hh_flags:
                        continue
                    flag_arr = hh_flags[code].to_numpy(dtype=float)

                    for d in list(range(1, 11)) + ["All"]:
                        if d == "All":
                            mask = ~np.isnan(dec_arr)
                        else:
                            mask = dec_arr == d
                        val, n_obs, n_w = _weighted_share(flag_arr[mask], w_arr[mask])
                        accum[code].append((country, year, d, val, n_obs, n_w))

            # ── Personal indicators ───────────────────────────────────────
            if pers is not None and pers_codes:
                pers_flags = _build_pers_flags(pers)
                pers["decile"] = pd.to_numeric(pers.get("decile", pd.Series(np.nan)), errors="coerce")
                w_p = pd.to_numeric(pers.get("RB050", pd.Series(np.nan)), errors="coerce").to_numpy()
                dec_p = pers["decile"].to_numpy(dtype=float)

                pers_no_gap = [c for c in pers_codes if c not in ("GE-SILC-1", "GE-SILC-2")]
                for code in pers_no_gap:
                    if code not in pers_flags:
                        continue
                    flag_arr = pers_flags[code].to_numpy(dtype=float)
                    for d in list(range(1, 11)) + ["All"]:
                        if d == "All":
                            mask = ~np.isnan(dec_p)
                        else:
                            mask = dec_p == d
                        val, n_obs, n_w = _weighted_share(flag_arr[mask], w_p[mask])
                        accum[code].append((country, year, d, val, n_obs, n_w))

                # Gender gap  (rows: country, year, decile, code, value, n_obs, n_w)
                for row in _compute_gender_gap(pers, country, year):
                    code_g = row[3]
                    if code_g in accum:
                        # drop the embedded code field; accum stores 6-tuples
                        accum[code_g].append((row[0], row[1], row[2], row[4], row[5], row[6]))

            pbar.update(1)

    pbar.close()

    # ── Save one CSV per code ───────────────────────────────────────────────
    print("\nSaving indicator CSV files …")
    saved, skipped = 0, 0
    for code in sorted(codes_to_run):
        rows = accum.get(code, [])
        if not rows:
            tqdm.write(f"  [SKIP] {code}: no data computed")
            skipped += 1
            continue
        out_df = pd.DataFrame(rows, columns=["country", "year", "decile",
                                              "value", "n_obs", "n_weighted"])
        out_path = config.INDICATORS_DIR / f"{code}.csv"
        out_df.to_csv(out_path, index=False)
        saved += 1
        tqdm.write(f"  [OK] {code}: {len(out_df)} rows → {out_path.name}")

    print(f"\nEU-SILC done: {saved} saved, {skipped} skipped.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract EU-SILC indicators")
    parser.add_argument("codes", nargs="*", help="Specific codes to (re)compute")
    parser.add_argument("--force", dest="force", action="store_true",
                        help="Force recompute for given codes (or all if none specified)")
    parser.add_argument("--force-all", dest="force_all", action="store_true",
                        help="Force recompute ALL EU-SILC indicators")
    args = parser.parse_args()

    force_codes = set(args.codes) if args.codes else None
    extract_eusilc(force_codes=force_codes, force_all=args.force_all)
