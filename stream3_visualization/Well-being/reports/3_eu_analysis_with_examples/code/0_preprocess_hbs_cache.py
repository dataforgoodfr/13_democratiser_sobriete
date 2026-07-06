"""
HBS raw data → local parquet cache
===================================
Run this script ONCE before using 1_expense.py or any other HBS analysis.

It converts each country's HBS_HH_{CC}.xlsx (on OneDrive, ~100 MB each) into
a compact local parquet file.  Subsequent loads from parquet are instant
(< 0.5 s per country) vs. 3–10 min per country from xlsx.

Usage:
    py 0_preprocess_hbs_cache.py

Re-running skips countries that are already cached.  To force a rebuild:
    py 0_preprocess_hbs_cache.py --rebuild
"""

import os
import sys
import glob
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

_ONEDRIVE = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr"

# Year-specific source folder and household-file name pattern.
# Pattern tokens: {CC} = uppercase 2-letter country code.
YEAR_CONFIGS = {
    2010: {
        "folder":  f"{_ONEDRIVE}/1_WSL/1_EWBI/0_data/HBS/HBS2010/HBS2010",
        "pattern": "{CC}_HBS_hh.xlsx",
    },
    2015: {
        "folder":  f"{_ONEDRIVE}/1_WSL/1_EWBI/0_data/HBS/HBS2015/HBS2015",
        "pattern": "{CC}_MFR_hh.xlsx",
    },
    2020: {
        "folder":  f"{_ONEDRIVE}/1_WSL/1_EWBI/0_data/HBS/HBS2020/HBS2020",
        "pattern": "HBS_HH_{CC}.xlsx",
    },
}

RAW_CACHE_DIR = os.path.join(BASE_DIR, "outputs", "data", "hbs_raw_cache")
os.makedirs(RAW_CACHE_DIR, exist_ok=True)

HBS_COUNTRIES = [
    "AT","BE","BG","CY","CZ","DE","DK","EE","EL","ES",
    "FI","FR","HR","HU","IE","IT","LT","LU","LV","MT",
    "NL","NO","PL","PT","RO","SI","SK",
]

# ── workers ────────────────────────────────────────────────────────────────
# Keep low: OneDrive throttles parallel reads of large files.
MAX_WORKERS = 2


def cache_path(cc: str, year: int) -> str:
    return os.path.join(RAW_CACHE_DIR, f"{cc}_hbs{year}_raw.parquet")


def process_country(cc: str, year: int, rebuild: bool) -> tuple[str, str]:
    """
    Convert source xlsx for *cc* and *year* → parquet.
    Returns (cc, status) where status is 'cached', 'done', or an error message.
    """
    out = cache_path(cc, year)
    if not rebuild and os.path.exists(out):
        return cc, "cached"

    cfg     = YEAR_CONFIGS[year]
    folder  = cfg["folder"]
    fname   = cfg["pattern"].replace("{CC}", cc)
    pattern = os.path.join(folder, fname)
    matches = glob.glob(pattern)
    if not matches:
        # Case-insensitive fallback
        wildcard = cfg["pattern"].replace("{CC}", "*")
        all_files = glob.glob(os.path.join(folder, wildcard))
        matches = [f for f in all_files if cc.upper() in os.path.basename(f).upper()]
    if not matches:
        return cc, f"skip (file not found: {pattern})"

    src = matches[0]
    t0  = time.time()
    try:
        # calamine (Rust-based) is 3-8× faster than openpyxl for large files
        df = pd.read_excel(src, engine="calamine", dtype_backend="numpy_nullable")
    except Exception:
        # calamine may fail for some xlsx variants → fall back to openpyxl
        try:
            df = pd.read_excel(src, engine="openpyxl")
        except Exception as exc:
            return cc, f"ERROR: {exc}"

    df["year"] = str(year)
    try:
        df.to_parquet(out, index=False, compression="snappy")
    except Exception as exc:
        return cc, f"ERROR saving parquet: {exc}"

    elapsed = time.time() - t0
    size_mb = os.path.getsize(out) / 1e6
    return cc, f"done  ({df.shape[0]:,} rows, {size_mb:.1f} MB parquet, {elapsed:.0f}s)"


def _process_year(year: int, targets: list[str], rebuild: bool) -> dict:
    """Run all countries for one *year* and return {cc: status} dict."""
    cfg = YEAR_CONFIGS[year]
    already = sum(1 for cc in targets if os.path.exists(cache_path(cc, year)))
    todo    = len(targets) - already

    print(f"\n{'=' * 60}")
    print(f"HBS {year}  →  parquet cache")
    print(f"  Source    : {cfg['folder']}")
    print(f"  Pattern   : {cfg['pattern']}")
    print(f"  Countries : {len(targets)}  ({already} cached, {todo} to build)")
    print(f"  Workers   : {MAX_WORKERS}")
    print("=" * 60)

    if todo == 0 and not rebuild:
        print(f"  All {year} countries already cached.")
        return {cc: "cached" for cc in targets}

    results: dict = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_country, cc, year, rebuild): cc
            for cc in targets
        }
        for fut in as_completed(futures):
            cc, status = fut.result()
            results[cc] = status
            tag = "[cached]" if status == "cached" else f"[{cc}]"
            print(f"  {tag:8s}  {cc}  {status}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-build HBS parquet cache")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild even if parquet already exists")
    parser.add_argument("--country", metavar="CC",
                        help="Process a single country code only")
    parser.add_argument("--year", metavar="YEAR", type=int,
                        choices=list(YEAR_CONFIGS.keys()),
                        help="Process a single year only (default: all years)")
    args = parser.parse_args()

    targets  = [args.country.upper()] if args.country else HBS_COUNTRIES
    years    = [args.year] if args.year else sorted(YEAR_CONFIGS.keys())

    all_errors: list[tuple[int, str, str]] = []
    for yr in years:
        res = _process_year(yr, targets, args.rebuild)
        for cc, status in res.items():
            if status.startswith("ERROR"):
                all_errors.append((yr, cc, status))

    print(f"\n{'=' * 60}")
    print(f"All done.  {len(all_errors)} errors total.")
    if all_errors:
        print("Errors:")
        for yr, cc, s in all_errors:
            print(f"  {yr} {cc}: {s}")


if __name__ == "__main__":
    main()
