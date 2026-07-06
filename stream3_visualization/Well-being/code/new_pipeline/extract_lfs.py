"""
LFS Extractor
=============
Reads raw Labour Force Survey yearly files ({COUNTRY}{YYYY}_y.csv) and
computes one CSV per indicator. No zero/NaN filtering.

Output per indicator: output/indicators/{CODE}.csv
  country, year, decile, value, n_obs, n_weighted

LFS income decile: from INCDECIL column (already in raw data).
LFS weight column: COEFFY

Usage
-----
    python extract_lfs.py                  # compute missing indicators
    python extract_lfs.py --force RT-LFS-1
    python extract_lfs.py --force-all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config
from indicators import BY_SOURCE, BY_CODE

config.ensure_dirs()

# ─── Parquet cache dir ────────────────────────────────────────────────────────
CACHE_LFS = config.CACHE_DIR / "lfs"
CACHE_LFS.mkdir(parents=True, exist_ok=True)

# ─── Columns needed from raw LFS files ───────────────────────────────────────
LFS_COLS = [
    "REFYEAR", "COUNTRY",
    "COEFFY",           # personal weight
    "INCDECIL",         # income decile (1-10 or special codes)
    # Indicator variables
    "NUMJOB",           # RT-LFS-1: multiple jobs
    "WISHMORE",         # RT-LFS-2: wish more hours
    "EXTRAHRS",         # RT-LFS-3: overtime/extra hours
    "VARITIME",         # RT-LFS-4: no flexibility on working time
    "SHIFTWK",          # RT-LFS-5: shift work
    "NIGHTWK",          # RT-LFS-6: night work
    "SATWK",            # RT-LFS-7: Saturday work
    "SUNWK",            # RT-LFS-8: Sunday work
    "ILOSTAT",          # RU-LFS-1: unemployed (ILO)
    "NEEDCARE",         # EL-LFS-2: no adequate childcare
    # Ancillary
    "AGE", "SEX", "EMPSTAT",
]

# Missing / not-applicable code in LFS
LFS_MISSING = {9, 99, 999}

# ─── Flag conditions per indicator ────────────────────────────────────────────
def _flag(df: pd.DataFrame, code: str) -> pd.Series:
    """Return float series (0/1/NaN) for a given LFS indicator code."""
    nan_s = pd.Series(np.nan, index=df.index)

    def col(c):
        return df[c] if c in df.columns else nan_s.copy()

    def binary(c, vals, exclude=LFS_MISSING):
        s = col(c)
        s = pd.to_numeric(s, errors="coerce")
        # Treat LFS missing codes as NaN
        s = s.where(~s.isin(exclude), other=np.nan)
        valid = s.notna()
        f = nan_s.copy()
        f[valid] = s[valid].isin(vals).astype(float)
        return f

    def cmp(c, op, v, exclude=LFS_MISSING):
        s = col(c)
        s = pd.to_numeric(s, errors="coerce")
        s = s.where(~s.isin(exclude), other=np.nan)
        valid = s.notna()
        f = nan_s.copy()
        if op == ">":
            f[valid] = (s[valid] > v).astype(float)
        elif op == "==":
            f[valid] = (s[valid] == v).astype(float)
        return f

    dispatch = {
        "RT-LFS-1": lambda: binary("NUMJOB",   [2, 3, 4]),
        "RT-LFS-2": lambda: binary("WISHMORE",  [2]),
        "RT-LFS-3": lambda: cmp("EXTRAHRS",    ">", 0),
        "RT-LFS-4": lambda: binary("VARITIME",  [3, 4]),
        "RT-LFS-5": lambda: binary("SHIFTWK",   [1]),
        "RT-LFS-6": lambda: binary("NIGHTWK",   [1, 2]),
        "RT-LFS-7": lambda: binary("SATWK",     [1, 2]),
        "RT-LFS-8": lambda: binary("SUNWK",     [1, 2]),
        "RU-LFS-1": lambda: binary("ILOSTAT",   [2]),
        "EL-LFS-2": lambda: binary("NEEDCARE",  [1, 2]),
    }

    fn = dispatch.get(code)
    return fn() if fn else nan_s


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _weighted_share(flag: np.ndarray, weight: np.ndarray):
    avail = ~np.isnan(flag) & ~np.isnan(weight) & (weight > 0)
    n = int(avail.sum())
    if n == 0:
        return np.nan, 0, 0.0
    f, w = flag[avail], weight[avail]
    n_w = float(w.sum())
    if n_w == 0:
        return np.nan, n, 0.0
    return float((f * w).sum() / n_w * 100), n, n_w


def _read_lfs_file(path: Path, cols: list[str]) -> pd.DataFrame | None:
    try:
        sample = pd.read_csv(path, nrows=2)
        usecols = [c for c in cols if c in sample.columns]
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df
    except Exception as exc:
        tqdm.write(f"    ⚠ {path.name}: {exc}")
        return None


# ─── Main extraction ───────────────────────────────────────────────────────────

def extract_lfs(force_codes: set[str] | None = None, force_all: bool = False):
    lfs_indicators = BY_SOURCE.get("LFS", [])
    all_codes = [ind["code"] for ind in lfs_indicators]

    codes_to_run = set()
    if force_all:
        codes_to_run = set(all_codes)
    else:
        for code in all_codes:
            out = config.INDICATORS_DIR / f"{code}.csv"
            if not out.exists() or (force_codes and code in force_codes):
                codes_to_run.add(code)

    if not codes_to_run:
        print("LFS: all indicator CSVs are up to date.")
        return

    print(f"LFS: computing {len(codes_to_run)} indicators …")

    raw_root = config.LFS_RAW_DIR
    if not raw_root.exists():
        print(f"  ERROR: LFS raw data not found:\n  {raw_root}")
        return

    accum: dict[str, list] = {c: [] for c in codes_to_run}

    # Discover all country-year files, grouped by country
    cy_by_country: dict[str, list[tuple[int, Path]]] = {}
    for folder in sorted(raw_root.iterdir()):
        if not folder.is_dir():
            continue
        m = re.match(r"([A-Z]{2})_YEAR", folder.name)
        if not m:
            continue
        ctry = m.group(1)
        if ctry not in config.EU_COUNTRIES:
            continue
        for f in folder.glob("*.csv"):
            fm = re.match(rf"{ctry}(\d{{4}})_y\.csv", f.name)
            if fm:
                cy_by_country.setdefault(ctry, []).append((int(fm.group(1)), f))

    for country in tqdm(sorted(cy_by_country), desc="LFS countries", unit="country"):
        cache_path = CACHE_LFS / f"{country}.parquet"

        # ── Load from Parquet cache if available ──
        if cache_path.exists():
            country_df = pd.read_parquet(cache_path)
        else:
            # Read all year files for this country
            year_dfs = []
            for year, fpath in sorted(cy_by_country[country]):
                df = _read_lfs_file(fpath, LFS_COLS)
                if df is not None:
                    df["country"] = country
                    df["year"] = year
                    year_dfs.append(df)
            if not year_dfs:
                continue
            country_df = pd.concat(year_dfs, ignore_index=True)
            try:
                country_df.to_parquet(cache_path, index=False)
            except Exception as exc:
                tqdm.write(f"  ⚠ Could not cache {country}: {exc}")

        # ── Decode decile and weight ──
        incdec = pd.to_numeric(country_df.get("INCDECIL", pd.Series(np.nan)),
                               errors="coerce")
        incdec = incdec.where(incdec.between(1, 10), other=np.nan)
        country_df["decile"] = incdec

        weight   = pd.to_numeric(country_df.get("COEFFY", pd.Series(np.nan)),
                                 errors="coerce").to_numpy()
        dec_arr  = country_df["decile"].to_numpy(dtype=float)
        year_arr = pd.to_numeric(country_df.get("year", pd.Series(np.nan)),
                                 errors="coerce").to_numpy(dtype=float)

        # Process by year so rows group correctly
        for year_val in np.unique(year_arr[~np.isnan(year_arr)]):
            mask_yr = year_arr == year_val
            df_yr   = country_df[mask_yr].reset_index(drop=True)
            w_yr    = weight[mask_yr]
            d_yr    = dec_arr[mask_yr]

            for code in codes_to_run:
                flag_arr = _flag(df_yr, code).to_numpy(dtype=float)

                for d in list(range(1, 11)) + ["All"]:
                    if d == "All":
                        m2 = ~np.isnan(d_yr)
                    else:
                        m2 = d_yr == d
                    val, n_obs, n_w = _weighted_share(flag_arr[m2], w_yr[m2])
                    accum[code].append((country, int(year_val), d, val, n_obs, n_w))

    print("\nSaving LFS indicator CSV files …")
    for code in sorted(codes_to_run):
        rows = accum.get(code, [])
        if not rows:
            print(f"  [SKIP] {code}: no data")
            continue
        out_df = pd.DataFrame(rows, columns=["country", "year", "decile",
                                              "value", "n_obs", "n_weighted"])
        out_path = config.INDICATORS_DIR / f"{code}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  [OK] {code}: {len(out_df)} rows → {out_path.name}")

    print("\nLFS done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LFS indicators")
    parser.add_argument("codes", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-all", dest="force_all", action="store_true")
    args = parser.parse_args()
    extract_lfs(force_codes=set(args.codes) if args.codes else None,
                force_all=args.force_all)
