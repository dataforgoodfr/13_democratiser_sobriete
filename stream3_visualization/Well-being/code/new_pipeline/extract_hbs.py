"""
HBS Extractor
=============
Reads raw Household Budget Survey (HBS) Excel files for waves 2010, 2015, 2020
and computes one CSV per indicator. No zero/NaN filtering.

Output per indicator: output/indicators/{CODE}.csv
  country, year, decile, value, n_obs, n_weighted

HBS income decile: derived from equivalised income (EUR_HH095 / HB061)
HBS weight column: HA10
HBS years: 2010, 2015, 2020

Usage
-----
    python extract_hbs.py                 # compute missing indicators
    python extract_hbs.py --force HH-HBS-1
    python extract_hbs.py --force-all
"""

from __future__ import annotations

import argparse
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
CACHE_HBS = config.CACHE_DIR / "hbs"
CACHE_HBS.mkdir(parents=True, exist_ok=True)

# ─── HBS raw-file layout per wave ─────────────────────────────────────────────
HBS_FILE_LAYOUT = {
    2010: {"subfolder": "HBS2010/HBS2010", "pattern": "*_HBS_hh.xlsx"},
    2015: {"subfolder": "HBS2015/HBS2015", "pattern": "*_MFR_hh.xlsx"},
    2020: {"subfolder": "HBS2020/HBS2020", "pattern": "HBS_HH_*.xlsx"},
}

# ─── Source column for each HBS indicator ─────────────────────────────────────
HBS_SRC_COL = {
    "HH-HBS-1": "EUR_HE041", "HH-HBS-2": "EUR_HE041",
    "HH-HBS-3": "EUR_HE04",  "HH-HBS-4": "EUR_HE04",
    "AC-HBS-1": "EUR_HE06",  "AC-HBS-2": "EUR_HE06",
    "AE-HBS-1": "EUR_HE01",  "AE-HBS-2": "EUR_HE01",
    "EC-HBS-1": "EUR_HJ08",  "EC-HBS-2": "EUR_HJ08",
    "IE-HBS-1": "EUR_HE10",  "IE-HBS-2": "EUR_HE10",
    "TT-HBS-1": "EUR_HE07",  "TT-HBS-2": "EUR_HE07",
    "TS-HBS-1": "EUR_HJ90",  "TS-HBS-2": "EUR_HJ90",
    "IC-HBS-1": "EUR_HE09",  "IC-HBS-2": "EUR_HE09",
}

# Columns to load from raw Excel files
HBS_COLS = [
    "COUNTRY", "YEAR",
    "HA10",                            # survey weight
    "HB061",                           # household size (number of members)
    "HB062",                           # modified OECD equivalence scale (preferred)
    "EUR_HH095",                        # disposable income (primary)
    "EUR_HH099",                        # net income (used for overburden threshold)
    "EUR_HH012", "EUR_HH023",           # income-in-kind fallbacks
    "INCDECIL",                         # pre-computed income decile (if present)
    # Standard expenditure columns
    "EUR_HE01", "EUR_HE04", "EUR_HE041", "EUR_HE06",
    "EUR_HE07", "EUR_HE09", "EUR_HE10",
    "EUR_HJ08", "EUR_HJ90",
    # Housing sub-components (EUR_HE04 = HE041+HE042+HE043+HE044+HE045)
    "EUR_HE042",                        # imputed rent for housing
    "EUR_HE043",                        # maintenance and repair of dwelling
    "EUR_HE044",                        # water supply and misc. dwelling services
    "EUR_HE045",                        # electricity, gas and other fuels
]

# ─── Overburden indicators (absolute income-share threshold) ──────────────────
# These are NOT relative-to-median: they flag if a household spends more than
# a fixed share of net income (EUR_HH099, fallback EUR_HH095) on essential needs.
HBS_OVERBURDEN = {
    # Total housing + utilities > 40 % of net income.
    # Prefer EUR_HE04 (total housing+utilities, directly measured).
    # Fallback: sum of components HE041+HE042+HE043+HE044+HE045,
    # derived in prepare_df() *before* EUR_HE041 is aggregated (avoids double-counting).
    "HH-HBS-5": {
        "cols": ["EUR_HE04"],
        "threshold": 0.40,
    },
}


# Indicators that use a 2-level weighted MEAN (instead of median) as national reference.
HBS_USE_MEAN: set[str] = set()  # no indicators currently use mean; kept for future use

# Indicators that compute the reference only on households with positive spending.
# Rationale: EC/TS/IE expenditure columns contain many zeros; the overall median
# would be 0 (degenerate), and the mean is not robust to outliers.  Using the
# median of positive-only spenders gives a stable, meaningful reference.
# Zero-spending households are still evaluated against the threshold and will
# naturally fall in the 'below 0.5×' group.
HBS_NONZERO_THRESHOLD = {"IE-HBS-1", "IE-HBS-2", "EC-HBS-1", "EC-HBS-2", "TS-HBS-1", "TS-HBS-2"}

# Custom lower-threshold multiplier for specific 'below' indicators.
# Default is 0.5× median; EC/TS use 0.2× because their spending distribution
# is heavily right-skewed — 0.5× still catches too many households.
HBS_LOWER_MULTIPLIER = {"EC-HBS-2": 0.2, "TS-HBS-2": 0.2}

# Custom upper-threshold multiplier for specific 'above' indicators.
# Default is 2.0× median; EC/TS use 1.5× to capture more high-spenders.
HBS_UPPER_MULTIPLIER = {"EC-HBS-1": 1.5, "TS-HBS-1": 1.5}

# Indicators where the share denominator is restricted to non-zero spenders.
# For IE-HBS, households with no informal-education spending are not part of
# the relevant population (no children / not engaged in education).  Keeping
# them in the denominator would dilute the share, and flagging them as
# 'below threshold' (direction='below') would be meaningless.
HBS_NONZERO_POPULATION = {"IE-HBS-1", "IE-HBS-2"}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def weighted_quantile(values: np.ndarray, weights: np.ndarray,
                      quantiles: np.ndarray) -> np.ndarray:
    """Weighted quantiles via sorted cumulative weight interpolation."""
    mask = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    v, w = values[mask], weights[mask]
    if len(v) == 0:
        return np.full(len(quantiles), np.nan)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumw = np.cumsum(w)
    return np.interp(quantiles, cumw / cumw[-1], v)


def assign_deciles_weighted(income: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Assign decile 1-10 using weighted percentile thresholds."""
    thresholds = weighted_quantile(income, weight, np.arange(0.1, 1.0, 0.1))
    if np.any(np.isnan(thresholds)):
        return np.full(len(income), np.nan)
    valid = ~np.isnan(income)
    result = np.full(len(income), np.nan)
    result[valid] = (income[valid, np.newaxis] > thresholds).sum(axis=1) + 1
    return result


def weighted_share(flag: np.ndarray, weight: np.ndarray):
    """Returns (value_pct, n_obs, n_weighted). NaN if no valid data."""
    avail = ~np.isnan(flag) & ~np.isnan(weight) & (weight > 0)
    n = int(avail.sum())
    if n == 0:
        return np.nan, 0, 0.0
    f, w = flag[avail], weight[avail]
    n_w = float(w.sum())
    if n_w == 0:
        return np.nan, n, 0.0
    return float((f * w).sum() / n_w * 100), n, n_w


def decile_balanced_weights(w_g: np.ndarray, dec_g: np.ndarray) -> np.ndarray:
    """
    Two-level weight for the national median computation.
    Level 1 : each income decile (1-10) contributes equally (1/10 of total).
    Level 2 : within each decile, relative HA10 survey weights are preserved.

    Implementation: normalise HA10 inside each decile so every decile sums
    to the same value (1). Households with missing decile get weight 0.
    """
    adj_w = np.zeros(len(w_g))
    for d in range(1, 11):
        mask = (dec_g == d) & np.isfinite(w_g) & (w_g > 0)
        dw = float(w_g[mask].sum())
        if dw > 0:
            adj_w[mask] = w_g[mask] / dw   # decile sums to 1
    return adj_w


# ─── Load one wave ─────────────────────────────────────────────────────────────

def load_wave(year: int) -> pd.DataFrame:
    """Read all Excel files for a HBS wave, with Parquet caching for speed."""
    cache_path = CACHE_HBS / f"HBS_{year}.parquet"
    if cache_path.exists():
        print(f"  Loading cached HBS {year} from Parquet …")
        return pd.read_parquet(cache_path)

    layout = HBS_FILE_LAYOUT[year]
    folder = config.HBS_RAW_DIR / layout["subfolder"]

    if not folder.exists():
        print(f"  [WARN] HBS {year} folder not found: {folder}")
        return pd.DataFrame()

    files = list(folder.glob(layout["pattern"]))
    if not files:
        print(f"  [WARN] No HBS {year} files matching '{layout['pattern']}' in {folder}")
        return pd.DataFrame()

    dfs = []
    for f in tqdm(files, desc=f"HBS {year} (Excel→Parquet)", leave=False):
        try:
            raw = pd.read_excel(f, engine="calamine")
            # Keep only available needed columns
            keep = [c for c in HBS_COLS if c in raw.columns]
            tmp = raw[keep].copy()
            # Infer country from filename if COUNTRY column absent
            if "COUNTRY" not in tmp.columns:
                tmp["COUNTRY"] = f.stem.split("_")[0].upper()[:2]
            if "YEAR" not in tmp.columns:
                tmp["YEAR"] = year
            dfs.append(tmp)
        except Exception as exc:
            tqdm.write(f"    [WARN] {f.name}: {exc}")

    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)

    # Coerce all numeric HBS columns from object (calamine may read spaces as str)
    for col in combined.columns:
        if combined[col].dtype == object and col not in ("COUNTRY",):
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    try:
        combined.to_parquet(cache_path, index=False)
        print(f"  Parquet cache written -> {cache_path.name}")
    except Exception as exc:
        print(f"  [WARN] Could not write Parquet cache: {exc}")
    return combined


# ─── Prepare: income, decile, expenditure shares ──────────────────────────────

def prepare_df(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Coerce types, compute equivalised income, assign income decile,
    compute expenditure shares (category / EUR_HH099 × 100).
    Returns enriched DataFrame.
    """
    df = df.copy()
    for c in df.columns:
        if c not in ("COUNTRY", "YEAR", "source_file"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["YEAR"] = year

    # Equivalised income = total_income / household_size
    hb061 = df.get("HB061", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    if "EUR_HH095" in df.columns:
        inc95   = df["EUR_HH095"].fillna(0)
        hh012   = df.get("EUR_HH012", pd.Series(0.0, index=df.index)).fillna(0)
        hh023   = df.get("EUR_HH023", pd.Series(0.0, index=df.index)).fillna(0)
        # Use EUR_HH095 per country-year group if it has positive values
        has95 = df.groupby(["COUNTRY", "YEAR"])["EUR_HH095"].transform(
            lambda s: (s.fillna(0) > 0).any()
        ).astype(bool)
        inc = np.where(has95, inc95, hh012 + hh023)
    else:
        hh012 = df.get("EUR_HH012", pd.Series(0.0, index=df.index)).fillna(0)
        hh023 = df.get("EUR_HH023", pd.Series(0.0, index=df.index)).fillna(0)
        inc   = (hh012 + hh023).to_numpy()

    df["equi_inc"] = inc / hb061.to_numpy()

    # Nullify equi_inc for country-years where no household has positive income.
    # Without this, fillna(0) above turns missing income into 0, which makes
    # assign_deciles_weighted put ALL households in decile 1 (0 > 0 is False
    # for every quantile threshold). Affected cases: IT all waves, LU 2010,
    # CZ 2020 — where EUR_HH095 / EUR_HH012 / EUR_HH023 are entirely absent.
    has_pos_inc = df.groupby(["COUNTRY", "YEAR"])["equi_inc"].transform(
        lambda s: (s > 0).any()
    ).astype(bool)
    no_income_pairs = (~has_pos_inc).sum()
    if no_income_pairs > 0:
        affected = (
            df.loc[~has_pos_inc, ["COUNTRY", "YEAR"]]
            .drop_duplicates()
            .apply(lambda r: f"{r['COUNTRY']}/{int(r['YEAR'])}", axis=1)
            .tolist()
        )
        tqdm.write(
            f"  [WARN] No income data — decile set to NaN for: {', '.join(affected)}"
        )
    df.loc[~has_pos_inc, "equi_inc"] = np.nan

    # Income decile
    weight = df.get("HA10", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    # Use INCDECIL if already available, else compute per country-year
    if "INCDECIL" in df.columns and df["INCDECIL"].notna().any():
        df["decile"] = pd.to_numeric(df["INCDECIL"], errors="coerce")
    else:
        deciles = np.full(len(df), np.nan)
        for (ctry, yr), grp in df.groupby(["COUNTRY", "YEAR"]):
            idx   = grp.index
            inc_g = df.loc[idx, "equi_inc"].to_numpy(dtype=float)
            w_g   = weight[df.index.get_indexer(idx)]
            deciles[df.index.get_indexer(idx)] = assign_deciles_weighted(inc_g, w_g)
        df["decile"] = deciles

    # ── Derive EUR_HE04 FIRST (from original sub-components, before any modification) ──
    # EUR_HE04 = HE041 + HE042 + HE043 + HE044 + HE045 (all housing+utilities sub-items).
    # Must be done before modifying EUR_HE041 to avoid double-counting HE042/HE043.
    if "EUR_HE04" not in df.columns or df["EUR_HE04"].isna().all():
        he04_parts = [c for c in ("EUR_HE041", "EUR_HE042", "EUR_HE043",
                                   "EUR_HE044", "EUR_HE045")
                      if c in df.columns]
        if he04_parts:
            df["EUR_HE04"] = df[he04_parts].fillna(0).sum(axis=1)

    # ── Aggregate rent sub-components for HH-HBS-1/2 (matches old pipeline behaviour) ──
    # Old pipeline: EUR_HE041 = actual rent + imputed rent + maintenance (HE041+HE042+HE043)
    rent_parts = [c for c in ("EUR_HE041", "EUR_HE042", "EUR_HE043") if c in df.columns]
    if len(rent_parts) > 1:
        df["EUR_HE041"] = df[rent_parts].fillna(0).sum(axis=1)

    # ── Equivalence scale for expense equivalization ──────────────────────────
    # Prefer HB062 (modified OECD scale), fall back to HB061 (household size).
    hb062 = pd.to_numeric(
        df.get("HB062", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).replace(0, np.nan)
    hb061 = pd.to_numeric(
        df.get("HB061", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).replace(0, np.nan)
    equiv_scale = hb062.where(hb062.notna(), hb061)

    # Equivalized expense = raw_expenditure / equiv_scale
    # Do NOT fillna(0): missing expenditure → NaN, excluded from median reference.
    exp_cols = [c for c in HBS_SRC_COL.values() if c in df.columns]
    for col in set(exp_cols):
        df[f"equiv_{col}"] = df[col] / equiv_scale

    # ── Net household income: EUR_HH099 (fallback EUR_HH095) ──────────────────
    # Used ONLY for the overburden indicator (HH-HBS-5: housing cost > 40% income).
    net_inc = df.get("EUR_HH099", pd.Series(np.nan, index=df.index))
    net_inc = pd.to_numeric(net_inc, errors="coerce").replace(0, np.nan)
    has_net = net_inc.notna().any()
    if not has_net:
        net_inc = pd.to_numeric(
            df.get("EUR_HH095", pd.Series(np.nan, index=df.index)), errors="coerce"
        ).replace(0, np.nan)

    for code, spec in HBS_OVERBURDEN.items():
        cols = [c for c in spec["cols"] if c in df.columns]
        if not cols:
            continue
        total = df[cols].fillna(0).sum(axis=1)
        df[f"_ob_{code}"] = total / net_inc   # ratio (0–1 scale)

    return df


# ─── Compute flags & aggregate ────────────────────────────────────────────────

def compute_hbs_indicators(df: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """
    For each HBS indicator code, compute the national-median threshold and
    flag each household (share > 2×median or < 0.5×median).
    Returns long-format DataFrame: country, year, decile, code, value, n_obs, n_weighted.
    """
    rows = []
    country_arr = df["COUNTRY"].to_numpy()
    year_arr    = df["YEAR"].to_numpy(dtype=float)
    decile_arr  = df["decile"].to_numpy(dtype=float)
    weight_arr  = df.get("HA10", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)

    for code in codes:
        src_col  = HBS_SRC_COL.get(code)
        equiv_col = f"equiv_{src_col}" if src_col else None

        # Overburden indicators bypass the standard median-expense logic entirely
        if code not in HBS_OVERBURDEN:
            if equiv_col is None or equiv_col not in df.columns:
                tqdm.write(f"  [SKIP] {code}: source column {src_col} not found")
                continue

        # ── Overburden indicators: absolute income-share threshold ──
        if code in HBS_OVERBURDEN:
            ob_col = f"_ob_{code}"
            if ob_col not in df.columns:
                tqdm.write(f"  [SKIP] {code}: overburden column not computed")
                continue
            ratio_arr = df[ob_col].to_numpy(dtype=float)
            threshold = HBS_OVERBURDEN[code]["threshold"]

            for (ctry, yr), grp_idx in df.groupby(["COUNTRY", "YEAR"]).groups.items():
                idx_g  = df.index.get_indexer(grp_idx)
                r_g    = ratio_arr[idx_g]
                w_g    = weight_arr[idx_g]
                dec_g  = decile_arr[idx_g]
                valid  = ~np.isnan(r_g)
                flag   = np.full(len(idx_g), np.nan)
                flag[valid] = (r_g[valid] > threshold).astype(float)

                for d in list(range(1, 11)) + ["All"]:
                    if d == "All":
                        mask = ~np.isnan(dec_g)
                    else:
                        mask = dec_g == d
                    val, n_obs, n_w = weighted_share(flag[mask], w_g[mask])
                    rows.append((ctry, int(yr), d, code, val, n_obs, n_w))
            continue

        # ── Standard indicators: 2× / 0.5× national median or mean of equivalized expense ──
        # Convention: odd-suffix codes (e.g. -1, -3) → above 2×; even (-2, -4) → below 0.5×
        last_digit = int(code.rsplit("-", 1)[-1])
        direction  = "above" if last_digit % 2 == 1 else "below"
        expv       = df[equiv_col].to_numpy(dtype=float)

        for (ctry, yr), grp_idx in df.groupby(["COUNTRY", "YEAR"]).groups.items():
            idx_g = df.index.get_indexer(grp_idx)
            e_g   = expv[idx_g]
            w_g   = weight_arr[idx_g]
            dec_g = decile_arr[idx_g]

            if code in HBS_NONZERO_THRESHOLD:
                # Education: raw HA10 weights on positive non-zero spenders only.
                # No decile balancing — spending is structurally higher in wealthier
                # deciles (more children → more education spend), so the unbalanced
                # distribution is meaningful.
                pos_m = (~np.isnan(e_g)) & (e_g > 0) & np.isfinite(w_g) & (w_g > 0)
                m = weighted_quantile(e_g[pos_m], w_g[pos_m], np.array([0.5]))[0] \
                    if pos_m.sum() > 0 else np.nan
            elif code in HBS_USE_MEAN:
                # Communications / Travel: 2-level weighted MEAN.
                # Many households have zero spend → median would be 0 or degenerate.
                # Level 1 — each income decile counts as exactly 1/10
                # Level 2 — HA10 as sub-weight within each decile
                bal_w   = decile_balanced_weights(w_g, dec_g)
                valid_m = ~np.isnan(e_g) & (bal_w > 0)
                if valid_m.sum() > 0:
                    m = float(np.average(e_g[valid_m], weights=bal_w[valid_m]))
                else:
                    m = np.nan
            else:
                # 2-level weighted median:
                #   Level 1 — each income decile counts as exactly 1/10
                #   Level 2 — HA10 as sub-weight within each decile
                bal_w   = decile_balanced_weights(w_g, dec_g)
                valid_m = ~np.isnan(e_g) & (bal_w > 0)
                m = weighted_quantile(e_g[valid_m], bal_w[valid_m], np.array([0.5]))[0] \
                    if valid_m.sum() > 0 else np.nan

            if np.isnan(m) or m == 0:
                threshold = np.nan
            else:
                if direction == "above":
                    upper_mult = HBS_UPPER_MULTIPLIER.get(code, 2.0)
                    threshold = upper_mult * m
                else:
                    lower_mult = HBS_LOWER_MULTIPLIER.get(code, 0.5)
                    threshold = lower_mult * m

            flag  = np.full(len(idx_g), np.nan)
            if not np.isnan(threshold):
                if code in HBS_NONZERO_POPULATION:
                    # Restrict to non-zero spenders: zero-spending households
                    # stay NaN and are excluded from both numerator and
                    # denominator of the weighted share.
                    valid = (~np.isnan(e_g)) & (e_g > 0)
                else:
                    valid = ~np.isnan(e_g)
                if direction == "above":
                    flag[valid] = (e_g[valid] > threshold).astype(float)
                else:
                    flag[valid] = (e_g[valid] < threshold).astype(float)

            w_g_a = w_g

            for d in list(range(1, 11)) + ["All"]:
                if d == "All":
                    mask = ~np.isnan(dec_g)
                else:
                    mask = dec_g == d
                val, n_obs, n_w = weighted_share(flag[mask], w_g_a[mask])
                rows.append((ctry, int(yr), d, code, val, n_obs, n_w))

    return pd.DataFrame(rows, columns=["country", "year", "decile",
                                        "code", "value", "n_obs", "n_weighted"])


# ─── Main extraction ───────────────────────────────────────────────────────────

def extract_hbs(force_codes: set[str] | None = None, force_all: bool = False):
    hbs_indicators = BY_SOURCE.get("HBS", [])
    all_codes = [ind["code"] for ind in hbs_indicators]

    codes_to_run = set()
    if force_all:
        codes_to_run = set(all_codes)
    else:
        for code in all_codes:
            out = config.INDICATORS_DIR / f"{code}.csv"
            if not out.exists() or (force_codes and code in force_codes):
                codes_to_run.add(code)

    if not codes_to_run:
        print("HBS: all indicator CSVs are up to date.")
        return

    print(f"HBS: computing {len(codes_to_run)} indicators …")

    accum: dict[str, list] = {c: [] for c in codes_to_run}

    for year in config.HBS_YEARS:
        print(f"\n  Loading wave {year} …")
        wave_df = load_wave(year)
        if wave_df.empty:
            print(f"  Skipping wave {year} – no data loaded.")
            continue

        wave_df = prepare_df(wave_df, year)
        result  = compute_hbs_indicators(wave_df, list(codes_to_run))

        for _, row in result.iterrows():
            code = row["code"]
            if code in accum:
                accum[code].append(
                    (row["country"], row["year"], row["decile"],
                     row["value"], row["n_obs"], row["n_weighted"])
                )

    print("\nSaving HBS indicator CSV files …")
    for code in sorted(codes_to_run):
        rows = accum.get(code, [])
        if not rows:
            print(f"  [SKIP] {code}: no data")
            continue
        out_df = pd.DataFrame(rows, columns=["country", "year", "decile",
                                              "value", "n_obs", "n_weighted"])
        out_path = config.INDICATORS_DIR / f"{code}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  [OK] {code}: {len(out_df)} rows -> {out_path.name}")

    print("\nHBS done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract HBS indicators")
    parser.add_argument("codes", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-all", dest="force_all", action="store_true")
    args = parser.parse_args()
    extract_hbs(force_codes=set(args.codes) if args.codes else None,
                force_all=args.force_all)
