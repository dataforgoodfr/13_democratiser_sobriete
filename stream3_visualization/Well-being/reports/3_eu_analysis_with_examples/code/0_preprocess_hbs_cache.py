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

HBS_FOLDER = (
    r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr"
    r"/1_WSL/1_EWBI/0_data/HBS/HBS2020/HBS2020"
)

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


def cache_path(cc: str) -> str:
    return os.path.join(RAW_CACHE_DIR, f"{cc}_hbs2020_raw.parquet")


def process_country(cc: str, rebuild: bool) -> tuple[str, str]:
    """
    Convert HBS_HH_{cc}.xlsx → parquet.
    Returns (cc, status) where status is 'cached', 'done', 'skip', or an error msg.
    """
    out = cache_path(cc)
    if not rebuild and os.path.exists(out):
        return cc, "cached"

    pattern = os.path.join(HBS_FOLDER, f"HBS_HH_{cc}.xlsx")
    matches = glob.glob(pattern)
    if not matches:
        # Try case-insensitive search
        all_files = glob.glob(os.path.join(HBS_FOLDER, "HBS_HH_*.xlsx"))
        matches = [f for f in all_files if cc.upper() in os.path.basename(f).upper()]
    if not matches:
        return cc, f"ERROR: file not found ({pattern})"

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

    df["year"] = "2020"
    try:
        df.to_parquet(out, index=False, compression="snappy")
    except Exception as exc:
        return cc, f"ERROR saving parquet: {exc}"

    elapsed = time.time() - t0
    size_mb = os.path.getsize(out) / 1e6
    return cc, f"done  ({df.shape[0]:,} rows, {size_mb:.1f} MB parquet, {elapsed:.0f}s)"


def main():
    parser = argparse.ArgumentParser(description="Pre-build HBS parquet cache")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild even if parquet already exists")
    parser.add_argument("--country", metavar="CC",
                        help="Process a single country code only")
    args = parser.parse_args()

    targets = [args.country.upper()] if args.country else HBS_COUNTRIES
    already = sum(1 for cc in targets if os.path.exists(cache_path(cc)))
    todo    = len(targets) - already

    print("=" * 60)
    print("HBS raw data → parquet cache builder")
    print(f"  Cache dir : {RAW_CACHE_DIR}")
    print(f"  Countries : {len(targets)}  ({already} cached, {todo} to build)")
    print(f"  Engine    : calamine (fast Rust parser)")
    print(f"  Workers   : {MAX_WORKERS}")
    print("=" * 60)

    if todo == 0 and not args.rebuild:
        print("\nAll countries already cached.  Use --rebuild to force refresh.")
        return

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_country, cc, args.rebuild): cc
            for cc in targets
        }
        for fut in as_completed(futures):
            cc, status = fut.result()
            results[cc] = status
            tag = "[cached]" if status == "cached" else f"[{cc}]"
            print(f"  {tag:8s}  {cc}  {status}")

    errors = [(cc, s) for cc, s in results.items() if s.startswith("ERROR")]
    print(f"\nDone.  {len(targets) - len(errors)} OK, {len(errors)} errors.")
    if errors:
        print("Errors:")
        for cc, s in errors:
            print(f"  {cc}: {s}")


if __name__ == "__main__":
    main()
