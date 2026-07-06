"""
HBS Extended Indicators – Updated Thresholds + New Variables
=============================================================
Generates ALL HBS indicators using CORRECT thresholds:
  >2× national median  →  indicator _above_2M
  <0.5× national median →  indicator _below_halfM

The original script used median±std, which is both non-standard and
misleadingly named.  This script replaces it with the specification-correct
>2×/< 0.5× median approach, as requested.

New variables vs. the original HBS pipeline
--------------------------------------------
  EUR_HE04  – Total Housing + Utilities share  → HH-HBS-3 (>2×) and HH-HBS-4 (<0.5×)

All existing indicators are also recomputed with the corrected thresholds.

Output
------
Saves `HBS_extended_catalog.csv` in the main HBS output folder with columns:
  code, name, source, code_source, country, year, decile, value,
  already_included, level, threshold, indicator_group

Author: Data for Good – Well-being Team
"""

import os
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Indicator metadata
# ---------------------------------------------------------------------------

EWBI_RETAINED: set = set()        # No HBS indicators are in the current EWBI

# (code, name, source_col, direction, group)
# direction: 'above' = share above 2× median,  'below' = share below 0.5× median
HBS_INDICATORS = [
    # Rent only
    ("HH-HBS-1", "Rent share > 2× national median",                "EUR_HE041", "above", "Energy & Housing"),
    ("HH-HBS-2", "Rent share < 0.5× national median",              "EUR_HE041", "below", "Energy & Housing"),
    # Total housing + utilities
    ("HH-HBS-3", "Housing & utilities share > 2× national median",  "EUR_HE04",  "above", "Energy & Housing"),
    ("HH-HBS-4", "Housing & utilities share < 0.5× national median","EUR_HE04",  "below", "Energy & Housing"),
    # Health
    ("AC-HBS-1", "Health expenditure share > 2× national median",   "EUR_HE06",  "above", "Health"),
    ("AC-HBS-2", "Health expenditure share < 0.5× national median", "EUR_HE06",  "below", "Health"),
    # Food
    ("AE-HBS-1", "Food expenditure share > 2× national median",     "EUR_HE01",  "above", "Equality"),
    ("AE-HBS-2", "Food expenditure share < 0.5× national median",   "EUR_HE01",  "below", "Equality"),
    # Communications
    ("EC-HBS-1", "Communications spend > 2× national median",       "EUR_HJ08",  "above", "Equality"),
    ("EC-HBS-2", "Communications spend < 0.5× national median",     "EUR_HJ08",  "below", "Equality"),
    # Education (restricted to households with children)
    ("IE-HBS-1", "Education expenditure share > 2× national median","EUR_HE10",  "above", "Education"),
    ("IE-HBS-2", "Education expenditure share < 0.5× national median","EUR_HE10","below", "Education"),
    # Transport
    ("TT-HBS-1", "Transport expenditure share > 2× national median","EUR_HE07",  "above", "Equality"),
    ("TT-HBS-2", "Transport expenditure share < 0.5× national median","EUR_HE07","below", "Equality"),
    # Travel & accommodation
    ("TS-HBS-1", "Travel expenditure share > 2× national median",   "EUR_HJ90",  "above", "Equality"),
    ("TS-HBS-2", "Travel expenditure share < 0.5× national median", "EUR_HJ90",  "below", "Equality"),
    # Recreation & culture
    ("IC-HBS-1", "Recreation/culture spend > 2× national median",   "EUR_HE09",  "above", "Equality"),
    ("IC-HBS-2", "Recreation/culture spend < 0.5× national median", "EUR_HE09",  "below", "Equality"),
]

# Build lookup dicts
HBS_CODE_TO_META = {
    code: {'name': name, 'code_source': src, 'direction': dirn, 'indicator_group': grp}
    for code, name, src, dirn, grp in HBS_INDICATORS
}


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def setup_directories() -> dict:
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    EXTERNAL_DATA_DIR = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI"

    dirs = {
        'external_data': EXTERNAL_DATA_DIR,
        'output_base': OUTPUT_DIR,
        'hbs_output':  os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_HBS'),
        'merged_dir':  os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_HBS', '0_merged'),
        'decile_dir':  os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_HBS', '1_income_decile'),
        'final_dir':   os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_HBS'),
    }
    for k, d in dirs.items():
        if k not in ('external_data', 'output_base'):
            os.makedirs(d, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def stack_excels(folder: str, pattern: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder, pattern))
    dfs = []
    for f in files:
        try:
            tmp = pd.read_excel(f)
            tmp['source_file'] = os.path.basename(f)
            dfs.append(tmp)
        except Exception as e:
            print(f"  ⚠ Could not read {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median (handles NaN)."""
    mask = ~np.isnan(values) & ~np.isnan(weights)
    v, w = values[mask], weights[mask]
    if len(v) == 0 or w.sum() == 0:
        return np.nan
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumw = np.cumsum(w)
    return float(v[np.searchsorted(cumw, cumw[-1] / 2.0)])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles) -> np.ndarray:
    mask = ~np.isnan(values) & ~np.isnan(weights)
    v, w = values[mask], weights[mask]
    if len(v) == 0:
        return np.full(len(quantiles), np.nan)
    order = np.argsort(v)
    v, w  = v[order], w[order]
    cumw  = np.cumsum(w)
    return np.interp(quantiles, cumw / cumw[-1], v)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_or_build_combined(dirs: dict):
    """
    Return the combined HBS household dataframe.
    Tries to load from CSV first; falls back to reading Excel files.
    """
    hh_path = os.path.join(dirs['merged_dir'], "HBS_combined_HH.csv")
    if os.path.exists(hh_path):
        print("  Loading pre-combined HH data from CSV …")
        return pd.read_csv(hh_path)

    print("  Pre-combined CSV not found – building from Excel files …")
    paths = {
        '2010': os.path.join(dirs['external_data'], r"0_data/HBS/HBS2010/HBS2010"),
        '2015': os.path.join(dirs['external_data'], r"0_data/HBS/HBS2015/HBS2015"),
        '2020': os.path.join(dirs['external_data'], r"0_data/HBS/HBS2020/HBS2020"),
    }
    patterns = {
        '2010': "*_HBS_hh.xlsx",
        '2015': "*_MFR_hh.xlsx",
        '2020': "HBS_HH_*.xlsx",
    }
    frames = []
    for year, folder in paths.items():
        df = stack_excels(folder, patterns[year])
        if not df.empty:
            df['year'] = year
            frames.append(df)

    if not frames:
        raise RuntimeError("No HBS Excel files found. Check external_data path.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(hh_path, index=False)
    print(f"  Saved combined HH data ({combined.shape}) → {hh_path}")
    return combined


# ---------------------------------------------------------------------------
# Data preparation & decile computation
# ---------------------------------------------------------------------------

def prepare_and_decile(hh_raw: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    """
    Compute equivalised income, assign income deciles, and compute
    expenditure shares as % of equivalised income – all in one pass.

    Returns the enriched household dataframe.
    """
    # ── Try to load from cache ──
    cache_path = os.path.join(dirs['decile_dir'], "HBS_household_data_with_decile_extended.csv")
    if os.path.exists(cache_path):
        print("  Loading cached decile data …")
        return pd.read_csv(cache_path)

    print("  Computing equivalised income and deciles …")
    df = hh_raw.copy()

    # Standardise COUNTRY / YEAR column names
    for old, new in [('COUNTRY', 'COUNTRY'), ('YEAR', 'YEAR')]:
        pass  # already uppercase in raw; keep as-is

    # ── Required columns ──
    needed = ['HA04', 'COUNTRY', 'YEAR', 'HA10', 'EUR_HH095',
              'EUR_HH012', 'EUR_HH023',  # income-in-kind fallback for countries without EUR_HH095
              'HB061',
              'HB075A',
              'EUR_HE01', 'EUR_HE041', 'EUR_HE042', 'EUR_HE043',
              'EUR_HE04',                # total housing+utilities (NEW)
              'EUR_HE045',
              'EUR_HE06', 'EUR_HE10', 'EUR_HE09',
              'EUR_HJ08', 'EUR_HJ90', 'EUR_HE07']

    available = [c for c in needed if c in df.columns]
    df = df[available].copy()

    # Numeric coercion
    for col in df.columns:
        if col not in ('HA04', 'COUNTRY', 'source_file'):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Equivalised income ──
    # Primary: EUR_HH095 (disposable household income).
    # Fallback: EUR_HH012 + EUR_HH023 (income in kind from employment and
    #           non-salaried activities) for countries like Italy where
    #           EUR_HH095 is not reported.
    # Selection is done per country×year so that countries with EUR_HH095
    #  keep using it while only the missing ones use the fallback.
    inc95 = df['EUR_HH095'].fillna(0) if 'EUR_HH095' in df.columns else pd.Series(0.0, index=df.index)
    hh012 = df['EUR_HH012'].fillna(0) if 'EUR_HH012' in df.columns else pd.Series(0.0, index=df.index)
    hh023 = df['EUR_HH023'].fillna(0) if 'EUR_HH023' in df.columns else pd.Series(0.0, index=df.index)
    inc_fallback = hh012 + hh023

    # Flag groups where EUR_HH095 has at least one positive value
    # Cast to bool explicitly: transform() with a scalar-returning lambda can yield float dtype
    if 'EUR_HH095' in df.columns:
        grp_has_95 = df.groupby(['COUNTRY', 'YEAR'])['EUR_HH095'].transform(
            lambda s: (s.fillna(0) > 0).any()
        ).astype(bool)
    else:
        grp_has_95 = pd.Series(False, index=df.index)

    df['equi_disp_inc'] = np.where(grp_has_95, inc95, inc_fallback) / df['HB061'].replace(0, np.nan)

    # Log countries using the fallback income
    fallback_groups = (
        df[~grp_has_95][['COUNTRY', 'YEAR']]
        .dropna(subset=['COUNTRY', 'YEAR'])
        .drop_duplicates()
    )
    if not fallback_groups.empty:
        for _, row_ in fallback_groups.iterrows():
            print(f"  ℹ [{row_['COUNTRY']} {int(row_['YEAR'])}] EUR_HH095 unavailable "
                  f"— using EUR_HH012+EUR_HH023 as income proxy")

    # ── Drop country/year groups with no valid income ──
    # (e.g. Italy where EUR_HH095, EUR_HH012 and EUR_HH023 are all zero)
    # These groups would produce equi_disp_inc=NaN/0 for every household, collapsing
    # all households into decile 1 and making the indicator values meaningless.
    grp_has_income = df.groupby(['COUNTRY', 'YEAR'])['equi_disp_inc'].transform(
        lambda s: s.gt(0).any()
    ).astype(bool)
    no_income = (
        df[~grp_has_income][['COUNTRY', 'YEAR']]
        .dropna(subset=['COUNTRY', 'YEAR'])
        .drop_duplicates()
    )
    if not no_income.empty:
        for _, row_ in no_income.iterrows():
            print(f"  ⚠ [{row_['COUNTRY']} {int(row_['YEAR'])}] no valid income — "
                  f"excluded from HBS indicator catalog")
        df = df[grp_has_income].copy()


    rent_parts = [c for c in ('EUR_HE041', 'EUR_HE042', 'EUR_HE043') if c in df.columns]
    if len(rent_parts) > 1:
        df['EUR_HE041'] = df[rent_parts].fillna(0).sum(axis=1)

    # ── If EUR_HE04 (total housing+utilities) is missing, try to derive it ──
    if 'EUR_HE04' not in df.columns:
        hh_parts = [c for c in ('EUR_HE041', 'EUR_HE042', 'EUR_HE043', 'EUR_HE045')
                    if c in df.columns]
        if hh_parts:
            df['EUR_HE04'] = df[hh_parts].fillna(0).sum(axis=1)
            print("  ℹ EUR_HE04 not in raw data – derived from sub-categories.")
        else:
            df['EUR_HE04'] = np.nan
            print("  ⚠ EUR_HE04 not available and cannot be derived.")

    # ── Expenditure shares (as % of equivalised income per equivalent adult) ──
    exp_cols = [c for c in ('EUR_HE01', 'EUR_HE041', 'EUR_HE04',
                             'EUR_HE045', 'EUR_HE06', 'EUR_HE10',
                             'EUR_HE09', 'EUR_HJ08', 'EUR_HJ90', 'EUR_HE07')
                if c in df.columns]

    for col in exp_cols:
        df[col + '_equiv_share'] = (
            df[col] / df['HB061'] / df['equi_disp_inc'] * 100
        )

    # ── Vectorised decile assignment ──
    quantiles = np.arange(0.1, 1.0, 0.1)

    def compute_group_deciles(group):
        v = group['equi_disp_inc'].to_numpy(dtype=float)
        w = group['HA10'].to_numpy(dtype=float)
        mask = ~np.isnan(v) & ~np.isnan(w)
        if mask.sum() == 0:
            return pd.Series([np.nan] * 9,
                             index=[f'decile_{i}' for i in range(1, 10)])
        return pd.Series(weighted_quantile(v[mask], w[mask], quantiles),
                         index=[f'decile_{i}' for i in range(1, 10)])

    print("  Computing per-country/year decile thresholds …")
    decile_thresholds = (
        df.groupby(['COUNTRY', 'YEAR'])
          .apply(compute_group_deciles)
          .reset_index()
    )

    df = df.merge(decile_thresholds, on=['COUNTRY', 'YEAR'], how='left')

    # Vectorised assignment (same trick as EU-SILC extended script)
    thr_cols = [f'decile_{i}' for i in range(1, 10)]
    available_thr = [c for c in thr_cols if c in df.columns]
    incomes    = df['equi_disp_inc'].to_numpy(dtype=float)
    thresholds = df[available_thr].to_numpy(dtype=float)
    valid = ~np.isnan(incomes) & ~np.any(np.isnan(thresholds), axis=1)
    deciles = np.full(len(incomes), np.nan)
    if valid.any():
        deciles[valid] = (incomes[valid, np.newaxis] > thresholds[valid]).sum(axis=1) + 1
    df['decile'] = deciles

    df.to_csv(cache_path, index=False)
    print(f"  Decile data cached → {cache_path}")
    return df


# ---------------------------------------------------------------------------
# National median computation & threshold flags
# ---------------------------------------------------------------------------

def compute_national_medians(df: pd.DataFrame, share_cols: list) -> pd.DataFrame:
    """
    Compute weighted national median for each expenditure share column,
    per country × year.
    """
    print("  Computing national medians …")

    def group_median(group, col):
        v = group[col].to_numpy(dtype=float)
        w = group['HA10'].to_numpy(dtype=float)
        return weighted_median(v, w)

    records = []
    for (country, year), grp in df.groupby(['COUNTRY', 'YEAR']):
        row = {'COUNTRY': country, 'YEAR': year}
        for col in share_cols:
            row[col + '_national_median'] = group_median(grp, col)
        records.append(row)

    medians_df = pd.DataFrame(records)
    return df.merge(medians_df, on=['COUNTRY', 'YEAR'], how='left')


def add_threshold_flags(df: pd.DataFrame, share_cols: list) -> pd.DataFrame:
    """
    Create binary indicator columns:
      col_above_2M   = 1 if col > 2 × national_median
      col_below_halfM = 1 if col < 0.5 × national_median
    NaN where the share or median is NaN.
    """
    for col in share_cols:
        med_col = col + '_national_median'
        if med_col not in df.columns:
            continue
        share = df[col].to_numpy(dtype=float)
        med   = df[med_col].to_numpy(dtype=float)

        avail = ~np.isnan(share) & ~np.isnan(med)

        above = np.full(len(share), np.nan)
        below = np.full(len(share), np.nan)
        above[avail] = (share[avail] > 2  * med[avail]).astype(float)
        below[avail] = (share[avail] < 0.5 * med[avail]).astype(float)

        df[col + '_above_2M']    = above
        df[col + '_below_halfM'] = below

    return df


# ---------------------------------------------------------------------------
# Fast weighted share computation (same approach as EU-SILC extended)
# ---------------------------------------------------------------------------

def compute_weighted_shares_fast(df: pd.DataFrame,
                                  flag_cols: dict,
                                  weight_col: str,
                                  group_cols: list) -> pd.DataFrame:
    df = df.copy()
    w = df[weight_col].to_numpy(dtype=float)

    num_keys, den_keys = {}, {}
    for code, flag in flag_cols.items():
        f = flag.to_numpy(dtype=float)
        avail = ~np.isnan(f)
        nk = f'_num_{code}'
        dk = f'_den_{code}'
        df[nk] = np.where(avail, f * w, 0.0)
        df[dk] = np.where(avail, w,     0.0)
        num_keys[code] = nk
        den_keys[code] = dk

    agg = {v: 'sum' for kd in (num_keys, den_keys) for v in kd.values()}
    grouped = df.groupby(group_cols, sort=False).agg(agg).reset_index()

    for code in flag_cols:
        nk, dk = num_keys[code], den_keys[code]
        grouped[code] = np.where(
            grouped[dk] > 0,
            grouped[nk] / grouped[dk] * 100,
            np.nan
        )
        grouped.drop(columns=[nk, dk], inplace=True)

    return grouped


def compute_all_levels(df: pd.DataFrame,
                        flag_cols: dict,
                        weight_col: str,
                        base_group_cols: list) -> pd.DataFrame:
    decile_result = compute_weighted_shares_fast(
        df, flag_cols, weight_col, base_group_cols)

    all_group_cols = [c for c in base_group_cols if c != 'decile']
    total_result   = compute_weighted_shares_fast(
        df, flag_cols, weight_col, all_group_cols)
    total_result['decile'] = 'All'

    combined = pd.concat([decile_result, total_result], ignore_index=True)

    id_cols  = [c for c in base_group_cols if c != 'decile'] + ['decile']
    val_cols = list(flag_cols.keys())
    return combined.melt(id_vars=id_cols, value_vars=val_cols,
                          var_name='code', value_name='value')


# ---------------------------------------------------------------------------
# Indicator processing
# ---------------------------------------------------------------------------

def process_hbs_indicators(dirs: dict) -> pd.DataFrame:
    print("\n── Loading and preparing HBS data ──")
    hh_raw = load_or_build_combined(dirs)

    df = prepare_and_decile(hh_raw, dirs)

    # Columns that have equiv_share computed
    share_cols = [c.replace('_equiv_share', '') + '_equiv_share'
                  for c in df.columns if c.endswith('_equiv_share')]

    # ── Zero-suppression: null out (country × year) groups where a share column
    #    has no positive values.  A group that is all-zero (median = 0) would make
    #    the >2× / <0.5× thresholds meaningless.  Applied here so it works even
    #    when loading from the decile cache.
    country_col = 'COUNTRY' if 'COUNTRY' in df.columns else 'country'
    year_col    = 'YEAR'    if 'YEAR'    in df.columns else 'year'
    for share_col in share_cols:
        grp_has_positive = df.groupby([country_col, year_col])[share_col].transform(
            lambda s: s.gt(0).any()
        ).astype(bool)
        mask_suppress = ~grp_has_positive & df[share_col].notna()
        if mask_suppress.any():
            suppressed = (
                df.loc[mask_suppress, [country_col, year_col]]
                .drop_duplicates()
                .sort_values([country_col, year_col])
            )
            raw_col = share_col.replace('_equiv_share', '')
            for _, r in suppressed.iterrows():
                print(f"  ⚠ [{r[country_col]} {int(r[year_col])}] {raw_col} all-zero — suppressed")
            df.loc[mask_suppress, share_col] = np.nan

    print("  Adding threshold flags (>2× and <0.5× national median) …")
    df = compute_national_medians(df, share_cols)
    df = add_threshold_flags(df, share_cols)

    # Map indicator codes to their flag columns
    # source_col → (above_col, below_col)
    src_to_flag: dict[str, tuple] = {}
    for col in share_cols:
        above_col = col + '_above_2M'
        below_col = col + '_below_halfM'
        if above_col in df.columns:
            src_to_flag[col] = (above_col, below_col)

    # Build flag dict from indicator metadata
    flag_cols: dict[str, pd.Series] = {}
    for code, name, src_raw, direction, group in HBS_INDICATORS:
        share_col = src_raw + '_equiv_share'
        if share_col not in src_to_flag:
            print(f"  ⚠ Share column '{share_col}' not found for {code} – skipping")
            continue
        above_col, below_col = src_to_flag[share_col]

        flag_cols[code] = df[above_col if direction == 'above' else below_col]

    # Ensure HA10 (weight) is numeric
    df['HA10'] = pd.to_numeric(df['HA10'], errors='coerce')
    df = df[df['HA10'].notna() & (df['HA10'] > 0)].copy()
    df = df[df['decile'].notna()].copy()
    df['decile'] = df['decile'].astype(int)

    # Rename for consistency
    df.rename(columns={'COUNTRY': 'country', 'YEAR': 'year'}, inplace=True)
    group_cols = ['country', 'year', 'decile']

    # Re-align flag_cols index after rename / filter
    flag_cols = {code: df[col.name if hasattr(col, 'name') else list(flag_cols)[0]]
                 for code, col in flag_cols.items()}

    # Rebuild flag_cols from df columns (safer after filter/rename)
    rebuilt_flags: dict[str, pd.Series] = {}
    for code, name, src_raw, direction, group in HBS_INDICATORS:
        share_col  = src_raw + '_equiv_share'
        suffix     = '_above_2M' if direction == 'above' else '_below_halfM'
        flag_colname = share_col + suffix
        if flag_colname not in df.columns:
            continue

        rebuilt_flags[code] = df[flag_colname]

    print(f"  Computing {len(rebuilt_flags)} indicators over {len(df):,} households …")
    result = compute_all_levels(df, rebuilt_flags, weight_col='HA10',
                                 base_group_cols=group_cols)
    result['source'] = 'HBS'
    result['level']  = 'household'
    return result


# ---------------------------------------------------------------------------
# Catalog assembly
# ---------------------------------------------------------------------------

def build_catalog(long_df: pd.DataFrame) -> pd.DataFrame:
    for col in ('name', 'code_source', 'indicator_group', 'direction'):
        long_df[col] = long_df['code'].map(
            {k: v[col] for k, v in HBS_CODE_TO_META.items()}
        )

    long_df['threshold'] = long_df.apply(
        lambda r: f"> 2× national median of {r['code_source']} share"
        if r['direction'] == 'above'
        else f"< 0.5× national median of {r['code_source']} share",
        axis=1
    )
    long_df.drop(columns=['direction'], inplace=True)

    long_df['already_included'] = long_df['code'].apply(
        lambda c: 'Y' if c in EWBI_RETAINED else 'N'
    )

    first_cols = ['code', 'name', 'source', 'code_source', 'country', 'year',
                  'decile', 'value', 'already_included', 'level',
                  'threshold', 'indicator_group']
    extra = [c for c in long_df.columns if c not in first_cols]
    return long_df[first_cols + extra]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("HBS Extended Indicators – Fast Pipeline")
    print("=" * 60)

    dirs = setup_directories()
    print(f"Output dir: {dirs['final_dir']}")

    result  = process_hbs_indicators(dirs)
    catalog = build_catalog(result)

    out_path = os.path.join(dirs['final_dir'], "HBS_extended_catalog.csv")
    catalog.to_csv(out_path, index=False)

    print(f"\n✅ Catalog saved → {out_path}")
    print(f"   Rows    : {len(catalog):,}")
    print(f"   Codes   : {catalog['code'].nunique()}")
    print(f"   Countries: {catalog['country'].nunique()}")
    print(f"   Years   : {catalog['year'].nunique()}")


if __name__ == '__main__':
    main()
