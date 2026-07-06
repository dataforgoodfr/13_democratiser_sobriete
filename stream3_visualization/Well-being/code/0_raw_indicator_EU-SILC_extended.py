"""
EU-SILC Extended Indicators – Fast Processing Script
=====================================================
Generates ALL EU-SILC indicators (existing retained + new candidates) using fully
vectorised computation, avoiding the slow row-wise apply used in the original script.

Key speed-ups compared to 0_raw_indicator_EU-SILC.py
------------------------------------------------------
1. Vectorised decile assignment via numpy broadcasting (replaces tqdm.pandas apply).
2. Indicator shares computed with a single groupby + sum – no Python loop over groups.
3. Reuses existing intermediate CSVs if present (skips overcrowding & decile recomputation).

New indicators added versus the original pipeline
--------------------------------------------------
EU-SILC household (H-file):
  HQ-SILC-2  – Cannot replace worn-out furniture          (HD080 in [2,3])
  TS-SILC-1  – Cannot afford 1-week holiday               (HS040 == 2)

EU-SILC personal (P-file):
  EL-SILC-1  – Not satisfied with life                    (PW010 < 3)
  AH-SILC-1  – Bad self-perceived health                  (PH010 in [4,5])
  IC-SILC-1  – Cannot regularly participate in leisure    (PD060 in [2,3])
  IC-SILC-2  – Cannot spend small amount on self          (PD070 in [2,3])
  EC-SILC-1  – Cannot meet friends/family monthly         (PD050 in [2,3])

Threshold corrections vs. original script
------------------------------------------
  AN-SILC-1  HS050 == 2  (original code used [1], which measured CAN afford)
  EC-SILC-2  PW191 < 3   (original code used < 4)

Output
------
Saves `EU_SILC_extended_catalog.csv` in the `3_final_merged_df` output folder with
columns: code, name, source, code_source, country, year, decile, value, already_included,
         level, threshold, indicator_group

Author: Data for Good – Well-being Team
"""

import os
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Indicator metadata
# ---------------------------------------------------------------------------

# EWBI-retained indicator codes (from the published retained list)
EWBI_RETAINED = {
    "HQ-SILC-1", "HQ-SILC-2", "HQ-SILC-3", "HQ-SILC-4",
    "HQ-SILC-5", "HQ-SILC-6", "HQ-SILC-7", "HQ-SILC-8",
    "HE-SILC-2",
    "ES-SILC-1", "ES-SILC-2",
    "EC-SILC-2", "EC-SILC-3", "EC-SILC-4",
    "AH-SILC-2", "AH-SILC-3", "AH-SILC-4",
    "AC-SILC-3", "AC-SILC-4",
    "IS-SILC-3", "IS-SILC-4", "IS-SILC-5",
    "RT-SILC-1", "RT-SILC-2",
}

# Metadata for HOUSEHOLD-level indicators
# Each entry: code -> (name, code_source, threshold_description, group)
HH_INDICATOR_META = {
    "HQ-SILC-1": ("Overcrowded dwelling",                        "HH030 (computed)", "overcrowded",      "Energy & Housing"),
    "HQ-SILC-2": ("Cannot replace worn-out furniture",           "HD080",            "in [2,3]",         "Energy & Housing"),
    "HQ-SILC-3": ("Cannot keep dwelling comfortably cool",       "HC070",            "== 2",             "Energy & Housing"),
    "HQ-SILC-4": ("Dwelling too dark",                           "HS160",            "== 1",             "Energy & Housing"),
    "HQ-SILC-5": ("Noise from street or neighbours",             "HS170",            "== 1",             "Energy & Housing"),
    "HQ-SILC-6": ("Leaking roof, damp or rot",                   "HH040",            "== 1",             "Energy & Housing"),
    "HQ-SILC-7": ("Pollution or crime in the area",              "HS180",            "== 1",             "Energy & Housing"),
    "HQ-SILC-8": ("No renovation measures",                      "HC003",            "in [4,99]",        "Energy & Housing"),
    "HE-SILC-1": ("Cannot keep dwelling comfortably warm",       "HC060",            "== 2",             "Energy & Housing"),
    "HE-SILC-2": ("Arrears on utility bills",                    "HS021",            "in [1,2]",         "Equality"),
    "HH-SILC-1": ("Arrears on mortgage or rent",                 "HS011",            "in [1,2]",         "Equality"),
    "AN-SILC-1": ("Cannot afford meat/fish/veg every 2nd day",   "HS050",            "== 2",             "Equality"),
    "ES-SILC-1": ("Cannot face unexpected financial expenses",   "HS060",            "== 2",             "Equality"),
    "ES-SILC-2": ("Hard to make ends meet",                      "HS120",            "in [1,2]",         "Equality"),
    "TS-SILC-1": ("Cannot afford 1-week holiday away from home", "HS040",            "== 2",             "Equality"),
    "EC-SILC-4": ("Persons living alone",                        "HH030 (computed)", "household_size==1","Equality"),
}

# Metadata for PERSONAL-level indicators
PERS_INDICATOR_META = {
    "EL-SILC-1": ("Not satisfied with life",                     "PW010",  "< 3",           "Equality"),
    "AH-SILC-1": ("Bad self-perceived health",                   "PH010",  "in [4,5]",      "Health"),
    "AH-SILC-2": ("Living with a chronic illness",               "PH020",  "== 1",          "Health"),
    "AH-SILC-3": ("Limited by health problems",                  "PH030",  "in [1,2]",      "Health"),
    "AH-SILC-4": ("Unable to work due to long-term illness",     "PL086",  "> 0",           "Health"),
    "AC-SILC-1": ("Could not afford medical care",               "PH050",  "== 1",          "Health"),
    "AC-SILC-3": ("Unmet need for medical examination",          "PH060",  "== 1",          "Health"),
    "AC-SILC-4": ("Unmet need for dental examination",           "PH040",  "== 1",          "Health"),
    "IC-SILC-1": ("Cannot regularly participate in leisure",     "PD060",  "in [2,3]",      "Equality"),
    "IC-SILC-2": ("Cannot spend small amount on self",           "PD070",  "in [2,3]",      "Equality"),
    "EC-SILC-1": ("Cannot meet friends/family monthly",          "PD050",  "in [2,3]",      "Equality"),
    "EC-SILC-2": ("Not trusting others",                         "PW191",  "< 3",           "Equality"),
    "EC-SILC-3": ("Cannot get together with friends/family",     "PD050",  "in [2,3]",      "Equality"),
    "IS-SILC-3": ("No formal education (age > 15)",              "PE041",  "== 0 or NaN",   "Education"),
    "IS-SILC-4": ("Not participating in formal training",        "PE010",  "== 2",          "Education"),
    "IS-SILC-5": ("No secondary education",                      "PE041",  "in [0,100]",    "Education"),
    "RT-SILC-1": ("Adults on fixed-term contracts",              "PL141",  "== 2 (pre-2021) or in [11,12]", "Quality of Jobs"),
    "RT-SILC-2": ("Adults working part-time",                    "PL145",  "== 2",          "Quality of Jobs"),
    "RU-SILC-1": ("Unemployed for 6+ months",                   "PL080",  "> 5",           "Quality of Jobs"),
}


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def setup_directories():
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    EXTERNAL_DATA_DIR = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI"

    dirs = {
        'external_data': EXTERNAL_DATA_DIR,
        'output_base': OUTPUT_DIR,
        'silc_output': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC'),
        'merged_dir':  os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC', '0_merged'),
        'decile_dir':  os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC', '1_income_decile'),
        'overcrowd_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC', '2_overcrowding'),
        'final_merged_dir': os.path.join(OUTPUT_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC', '3_final_merged_df'),
    }
    for k, d in dirs.items():
        if k not in ('external_data', 'output_base'):
            os.makedirs(d, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Fast vectorised helpers
# ---------------------------------------------------------------------------

def assign_deciles_vectorized(equiv_with_deciles: pd.DataFrame) -> pd.Series:
    """
    Vectorised replacement for the slow tqdm.pandas apply in the original script.

    For each household, count how many of the 9 decile thresholds its equivalised
    income exceeds, then add 1 → decile 1–10.  Returns NaN when income or any
    threshold is NaN.
    """
    threshold_cols = [f'decile_{i}' for i in range(1, 10)]
    available = [c for c in threshold_cols if c in equiv_with_deciles.columns]
    if not available:
        return pd.Series(np.nan, index=equiv_with_deciles.index)

    incomes    = equiv_with_deciles['equi_disp_inc'].to_numpy(dtype=float)
    thresholds = equiv_with_deciles[available].to_numpy(dtype=float)          # (N, 9)

    # Rows with any NaN → keep NaN result
    valid = ~np.isnan(incomes) & ~np.any(np.isnan(thresholds), axis=1)

    result = np.full(len(incomes), np.nan)
    # broadcasting: incomes (N,1) > thresholds (N,9) → bool matrix (N,9)
    result[valid] = (incomes[valid, np.newaxis] > thresholds[valid]).sum(axis=1) + 1
    return pd.Series(result, index=equiv_with_deciles.index)


def compute_weighted_shares_fast(df: pd.DataFrame,
                                  flag_cols: dict,
                                  weight_col: str,
                                  group_cols: list) -> pd.DataFrame:
    """
    Compute weighted share (%) of each indicator flag per group in one pass.

    Parameters
    ----------
    df         : Source dataframe (one row per household or person)
    flag_cols  : {indicator_code: flag_series}  where flag_series is float (0/1/NaN).
                 NaN = data not available for that row.
    weight_col : Column name of the survey weight
    group_cols : Columns to group by (e.g. ['year', 'country', 'decile'])

    Returns a wide dataframe with group_cols + one column per indicator code.
    """
    df = df.copy()
    w = df[weight_col].to_numpy(dtype=float)

    num_cols, den_cols = {}, {}
    for code, flag in flag_cols.items():
        f = flag.to_numpy(dtype=float)
        avail = ~np.isnan(f)
        num_key = f'_num_{code}'
        den_key = f'_den_{code}'
        df[num_key] = np.where(avail, f * w, 0.0)
        df[den_key] = np.where(avail, w,     0.0)
        num_cols[code] = num_key
        den_cols[code] = den_key

    agg = {v: 'sum' for v in list(num_cols.values()) + list(den_cols.values())}
    grouped = df.groupby(group_cols, sort=False).agg(agg).reset_index()

    for code in flag_cols:
        nk, dk = num_cols[code], den_cols[code]
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
    """
    Compute shares by decile AND for 'All' (total population per country-year).
    Concatenates both results and returns long-format output.
    """
    # Decile level
    decile_result = compute_weighted_shares_fast(df, flag_cols, weight_col, base_group_cols)

    # Total population (drop decile from grouping)
    all_group_cols = [c for c in base_group_cols if c != 'decile']
    total_result = compute_weighted_shares_fast(df, flag_cols, weight_col, all_group_cols)
    total_result['decile'] = 'All'

    combined = pd.concat([decile_result, total_result], ignore_index=True)

    # Melt to long format
    id_cols   = [c for c in base_group_cols if c != 'decile'] + ['decile']
    val_cols  = list(flag_cols.keys())
    melted = combined.melt(id_vars=id_cols, value_vars=val_cols,
                            var_name='code', value_name='value')
    return melted


# ---------------------------------------------------------------------------
# Load helpers (reuse existing intermediates when available)
# ---------------------------------------------------------------------------

def load_or_abort(path: str, desc: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required intermediate file not found: {path}\n"
            f"Please run 0_raw_indicator_EU-SILC.py first to generate {desc}."
        )
    print(f"  Loading {desc} …")
    return pd.read_csv(path, **kwargs)


def ensure_personal_extra_columns(personal_merged: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    """
    Add PW010 (life satisfaction) and PH010 (self-perceived health) to the
    personal merged dataframe if they are missing.  These columns were not
    collected in the original combine_personal_data step.
    """
    missing = [c for c in ('PW010', 'PH010') if c not in personal_merged.columns]
    if not missing:
        return personal_merged

    print(f"  Columns {missing} not in personal_merged – loading from raw P-files …")
    base_path = os.path.join(
        dirs['external_data'],
        "0_data/EU-SILC/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set"
    )
    if not os.path.isdir(base_path):
        print(f"  ⚠ Raw EU-SILC P-files not found at {base_path}. "
              f"Setting {missing} to NaN.")
        for c in missing:
            personal_merged[c] = np.nan
        return personal_merged

    extra_cols = ['PB010', 'PB020', 'PB030'] + missing
    dfs = []
    for country in os.listdir(base_path):
        cp = os.path.join(base_path, country)
        if not os.path.isdir(cp):
            continue
        for year in os.listdir(cp):
            yp = os.path.join(cp, year)
            if not os.path.isdir(yp):
                continue
            suffix = year[-2:]
            fp = os.path.join(yp, f"UDB_c{country}{suffix}P.csv")
            if not os.path.exists(fp):
                continue
            try:
                sample = pd.read_csv(fp, nrows=2)
                avail  = [c for c in extra_cols if c in sample.columns]
                if len(avail) <= 3:           # only ID cols, no extra
                    continue
                tmp = pd.read_csv(fp, usecols=avail)
                for c in extra_cols:
                    if c not in tmp.columns:
                        tmp[c] = pd.NA
                dfs.append(tmp)
            except Exception as e:
                print(f"    ⚠ Could not read {fp}: {e}")

    if not dfs:
        print(f"  ⚠ No extra column data found. Setting {missing} to NaN.")
        for c in missing:
            personal_merged[c] = np.nan
        return personal_merged

    extra_df = pd.concat(dfs, ignore_index=True)
    extra_df['PB030'] = extra_df['PB030'].fillna(0).astype('int64').astype(str)
    personal_merged['PB030'] = personal_merged['PB030'].fillna(0).astype('int64').astype(str)

    # Merge on year + country + person ID
    # Use all actually-present extra columns (not just `avail` from the last loop iteration)
    on_cols = ['PB010', 'PB020', 'PB030']
    extra_present = on_cols + [c for c in missing if c in extra_df.columns]
    personal_merged = personal_merged.merge(
        extra_df[extra_present],
        on=on_cols, how='left', suffixes=('', '_extra')
    )
    # Resolve suffixed columns
    for c in missing:
        if f'{c}_extra' in personal_merged.columns:
            personal_merged[c] = personal_merged[c].combine_first(personal_merged[f'{c}_extra'])
            personal_merged.drop(columns=[f'{c}_extra'], inplace=True)

    print(f"  ✅ Added {missing} to personal data")
    return personal_merged


# ---------------------------------------------------------------------------
# Household indicator computation
# ---------------------------------------------------------------------------

def build_household_flags(df: pd.DataFrame) -> dict:
    """
    Return {indicator_code: flag_series (float 0/1/NaN)} for all household indicators.
    NaN → variable not collected for that row.
    """
    flags = {}
    year = df['HB010'] if 'HB010' in df.columns else df.get('year', pd.Series(dtype=int))

    def flag(series, cond_true, avail=None):
        """avail: boolean mask of rows where variable was collected."""
        s = series.copy().astype('float64')
        f = cond_true.astype('float64')
        f[series.isna()] = np.nan
        if avail is not None:
            f[~avail] = np.nan
        return f

    # HQ-SILC-1 – overcrowded (pre-computed binary column)
    if 'overcrowded' in df.columns:
        flags['HQ-SILC-1'] = flag(df['overcrowded'], df['overcrowded'] == 1)

    # HQ-SILC-2 – worn-out furniture  (HD080 in [2,3])
    if 'HD080' in df.columns:
        flags['HQ-SILC-2'] = flag(df['HD080'], df['HD080'].isin([2, 3]))

    # HQ-SILC-3 – cannot keep cool  (HC070 == 2)
    if 'HC070' in df.columns:
        flags['HQ-SILC-3'] = flag(df['HC070'], df['HC070'] == 2)

    # HQ-SILC-4 – too dark  (HS160 == 1)
    if 'HS160' in df.columns:
        flags['HQ-SILC-4'] = flag(df['HS160'], df['HS160'] == 1)

    # HQ-SILC-5 – noise  (HS170 == 1)
    if 'HS170' in df.columns:
        flags['HQ-SILC-5'] = flag(df['HS170'], df['HS170'] == 1)

    # HQ-SILC-6 – leaking roof  (HH040 == 1)
    if 'HH040' in df.columns:
        flags['HQ-SILC-6'] = flag(df['HH040'], df['HH040'] == 1)

    # HQ-SILC-7 – pollution/crime  (HS180 == 1)
    if 'HS180' in df.columns:
        flags['HQ-SILC-7'] = flag(df['HS180'], df['HS180'] == 1)

    # HQ-SILC-8 – no renovation  (HC003 in [4, 99])
    if 'HC003' in df.columns:
        flags['HQ-SILC-8'] = flag(df['HC003'], df['HC003'].isin([4, 99]))

    # HE-SILC-1 – cannot keep warm  (HC060 == 2)  [not in retained list]
    if 'HC060' in df.columns:
        f_warm = flag(df['HC060'], df['HC060'] == 2)
        # Known data quality issue: 2016 is unreliable for this variable
        if 'HB010' in df.columns:
            f_warm[df['HB010'] == 2016] = np.nan
        flags['HE-SILC-1'] = f_warm

    # HE-SILC-2 – arrears on utility bills  (HS021 in [1,2] after 2008, [1] before)
    if 'HS021' in df.columns and 'HB010' in df.columns:
        cond = ((df['HB010'] < 2008) & (df['HS021'] == 1)) | \
               ((df['HB010'] >= 2008) & (df['HS021'].isin([1, 2])))
        flags['HE-SILC-2'] = flag(df['HS021'], cond)

    # HH-SILC-1 – arrears on mortgage/rent  (HS011 in [1,2] after 2008, [1] before)
    if 'HS011' in df.columns and 'HB010' in df.columns:
        cond = ((df['HB010'] < 2008) & (df['HS011'] == 1)) | \
               ((df['HB010'] >= 2008) & (df['HS011'].isin([1, 2])))
        flags['HH-SILC-1'] = flag(df['HS011'], cond)

    # AN-SILC-1 – cannot afford meat/fish/veg every 2nd day  (HS050 == 2)
    # NOTE: corrected threshold – original code incorrectly used [1] (=CAN afford)
    if 'HS050' in df.columns:
        flags['AN-SILC-1'] = flag(df['HS050'], df['HS050'] == 2)

    # ES-SILC-1 – cannot face unexpected costs  (HS060 == 2)
    if 'HS060' in df.columns:
        flags['ES-SILC-1'] = flag(df['HS060'], df['HS060'] == 2)

    # ES-SILC-2 – hard to make ends meet  (HS120 in [1,2])
    if 'HS120' in df.columns:
        flags['ES-SILC-2'] = flag(df['HS120'], df['HS120'].isin([1, 2]))

    # TS-SILC-1 – cannot afford 1-week holiday  (HS040 == 2)  [NEW]
    if 'HS040' in df.columns:
        flags['TS-SILC-1'] = flag(df['HS040'], df['HS040'] == 2)

    # EC-SILC-4 – persons living alone (pre-computed binary column)
    # NOTE: ideally uses person-level weights; here household weight used for consistency
    if 'living_alone' in df.columns:
        flags['EC-SILC-4'] = flag(df['living_alone'], df['living_alone'] == 1)

    return flags


def process_household_indicators(dirs: dict) -> pd.DataFrame:
    """Load household merged dataframe and compute all household-level indicators."""
    print("\n── Household indicators ──")
    path = os.path.join(dirs['final_merged_dir'], "EU_SILC_household_final_merged.csv")
    df = load_or_abort(path, "household final merged")

    # Standardise column names
    rename = {'HB010': 'year', 'HB020': 'country'}
    df.rename(columns=rename, inplace=True)

    # Drop rows with no weight
    df = df[df['DB090'].notna() & (df['DB090'] > 0)].copy()

    # Ensure decile is available
    if 'decile' not in df.columns or df['decile'].isna().all():
        raise ValueError("'decile' column missing or all-NaN in household merged file. "
                         "Run 0_raw_indicator_EU-SILC.py first.")

    # Drop rows with no decile assignment
    df = df[df['decile'].notna()].copy()
    df['decile'] = df['decile'].astype(int)

    flag_cols  = build_household_flags(df)
    group_cols = ['year', 'country', 'decile']

    print(f"  Computing {len(flag_cols)} indicators over {len(df):,} households …")
    result = compute_all_levels(df, flag_cols, weight_col='DB090',
                                 base_group_cols=group_cols)
    result['source'] = 'EU-SILC'
    result['level']  = 'household'
    return result


# ---------------------------------------------------------------------------
# Personal indicator computation
# ---------------------------------------------------------------------------

def build_personal_flags(df: pd.DataFrame) -> dict:
    """
    Return {indicator_code: flag_series (float 0/1/NaN)} for all personal indicators.
    """
    flags = {}
    year = df['PB010'] if 'PB010' in df.columns else df.get('year', pd.Series(dtype=int))

    def flag(series, cond_true):
        f = cond_true.astype('float64')
        f[series.isna()] = np.nan
        return f

    # EL-SILC-1 – not satisfied with life  (PW010 < 3)  [NEW]
    if 'PW010' in df.columns:
        pw = pd.to_numeric(df['PW010'], errors='coerce')
        flags['EL-SILC-1'] = flag(pw, pw < 3)

    # AH-SILC-1 – bad self-perceived health  (PH010 in [4,5])  [NEW]
    if 'PH010' in df.columns:
        ph10 = pd.to_numeric(df['PH010'], errors='coerce')
        flags['AH-SILC-1'] = flag(ph10, ph10.isin([4, 5]))

    # AH-SILC-2 – chronic illness  (PH020 == 1)
    if 'PH020' in df.columns:
        flags['AH-SILC-2'] = flag(df['PH020'], df['PH020'] == 1)

    # AH-SILC-3 – limited by health  (PH030 in [1,2])
    if 'PH030' in df.columns:
        flags['AH-SILC-3'] = flag(df['PH030'], df['PH030'].isin([1, 2]))

    # AH-SILC-4 – unable to work due to illness  (PL086 > 0)
    if 'PL086' in df.columns:
        pl86 = pd.to_numeric(df['PL086'], errors='coerce')
        flags['AH-SILC-4'] = flag(pl86, pl86 > 0)

    # AC-SILC-1 – could not afford medical care  (PH050 == 1)
    if 'PH050' in df.columns:
        flags['AC-SILC-1'] = flag(df['PH050'], df['PH050'] == 1)

    # AC-SILC-3 – unmet need for medical examination  (PH060 == 1)
    if 'PH060' in df.columns:
        flags['AC-SILC-3'] = flag(df['PH060'], df['PH060'] == 1)

    # AC-SILC-4 – unmet need for dental examination  (PH040 == 1)
    if 'PH040' in df.columns:
        flags['AC-SILC-4'] = flag(df['PH040'], df['PH040'] == 1)

    # IC-SILC-1 – cannot participate in leisure  (PD060 in [2,3])  [NEW]
    if 'PD060' in df.columns:
        flags['IC-SILC-1'] = flag(df['PD060'], df['PD060'].isin([2, 3]))

    # IC-SILC-2 – cannot spend small amount on self  (PD070 in [2,3])  [NEW]
    if 'PD070' in df.columns:
        flags['IC-SILC-2'] = flag(df['PD070'], df['PD070'].isin([2, 3]))

    # EC-SILC-1 – cannot meet friends/family monthly  (PD050 in [2,3])  [NEW]
    if 'PD050' in df.columns:
        flags['EC-SILC-1'] = flag(df['PD050'], df['PD050'].isin([2, 3]))

    # EC-SILC-2 – not trusting others  (PW191 < 3)
    # NOTE: original code used < 4; corrected to < 3 per specification
    if 'PW191' in df.columns:
        pw191 = pd.to_numeric(df['PW191'], errors='coerce')
        flags['EC-SILC-2'] = flag(pw191, pw191 < 3)

    # EC-SILC-3 – cannot get together with friends/family  (PD050 in [2,3])
    if 'PD050' in df.columns:
        flags['EC-SILC-3'] = flag(df['PD050'], df['PD050'].isin([2, 3]))

    # IS-SILC-3 – no formal education (age > 15 and PE041 == 0 or missing)
    if 'PE041' in df.columns and 'age' in df.columns:
        age = pd.to_numeric(df['age'], errors='coerce')
        pe41 = pd.to_numeric(df['PE041'], errors='coerce')
        cond = (age > 15) & ((pe41 == 0) | pe41.isna())
        avail = age.notna()
        f = cond.astype('float64')
        f[~avail] = np.nan
        flags['IS-SILC-3'] = f

    # IS-SILC-4 – not participating in formal training  (PE010 == 2)
    if 'PE010' in df.columns:
        flags['IS-SILC-4'] = flag(df['PE010'], df['PE010'] == 2)

    # IS-SILC-5 – no secondary education  (PE041 in [0, 100])
    if 'PE041' in df.columns:
        pe41_num = pd.to_numeric(df['PE041'], errors='coerce')
        flags['IS-SILC-5'] = flag(pe41_num, pe41_num.isin([0, 100]))

    # RT-SILC-1 – fixed-term contract (age > 17)
    if 'PL141' in df.columns and 'age' in df.columns:
        age = pd.to_numeric(df['age'], errors='coerce')
        adult = age > 17
        cond = adult & (
            ((year < 2021) & (df['PL141'] == 2)) |
            ((year >= 2021) & (df['PL141'].isin([11, 12])))
        )
        f = cond.astype('float64')
        f[~adult] = np.nan
        f[df['PL141'].isna()] = np.nan
        flags['RT-SILC-1'] = f

    # RT-SILC-2 – part-time (age > 17)
    if 'PL145' in df.columns and 'age' in df.columns:
        age = pd.to_numeric(df['age'], errors='coerce')
        adult = age > 17
        cond = adult & (df['PL145'] == 2)
        f = cond.astype('float64')
        f[~adult] = np.nan
        f[df['PL145'].isna()] = np.nan
        flags['RT-SILC-2'] = f

    # RU-SILC-1 – unemployed for 6+ months  (PL080 > 5)
    if 'PL080' in df.columns:
        pl80 = pd.to_numeric(df['PL080'], errors='coerce')
        flags['RU-SILC-1'] = flag(pl80, pl80 > 5)

    return flags


def process_personal_indicators(dirs: dict) -> pd.DataFrame:
    """Load personal merged dataframe and compute all personal-level indicators."""
    print("\n── Personal indicators ──")
    path = os.path.join(dirs['final_merged_dir'], "EU_SILC_personal_final_merged.csv")
    df = load_or_abort(path, "personal final merged")

    # Add PW010 / PH010 if missing (they were not in the original loading)
    df = ensure_personal_extra_columns(df, dirs)

    # Standardise column names
    rename = {}
    if 'PB010' in df.columns: rename['PB010'] = 'year'
    if 'PB020' in df.columns: rename['PB020'] = 'country'
    df.rename(columns=rename, inplace=True)

    # Weight column
    weight_col = 'RB050'
    df = df[df[weight_col].notna() & (pd.to_numeric(df[weight_col], errors='coerce') > 0)].copy()
    df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')

    if 'decile' not in df.columns or df['decile'].isna().all():
        raise ValueError("'decile' column missing or all-NaN in personal merged file.")

    df = df[df['decile'].notna()].copy()
    df['decile'] = df['decile'].astype(int)

    # Ensure year column is numeric for comparisons in flag building
    if 'year' not in df.columns and 'PB010' in df.columns:
        df['year'] = df['PB010']
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    flag_cols  = build_personal_flags(df)
    group_cols = ['year', 'country', 'decile']

    print(f"  Computing {len(flag_cols)} indicators over {len(df):,} persons …")
    result = compute_all_levels(df, flag_cols, weight_col=weight_col,
                                 base_group_cols=group_cols)
    result['source'] = 'EU-SILC'
    result['level']  = 'personal'
    return result


# ---------------------------------------------------------------------------
# Catalog assembly
# ---------------------------------------------------------------------------

def build_catalog(hh_long: pd.DataFrame, pers_long: pd.DataFrame) -> pd.DataFrame:
    """Attach metadata and produce the unified catalog."""
    combined = pd.concat([hh_long, pers_long], ignore_index=True)

    meta_all = {}
    for code, (name, code_src, threshold, group) in HH_INDICATOR_META.items():
        meta_all[code] = {'name': name, 'code_source': code_src,
                          'threshold': threshold, 'indicator_group': group}
    for code, (name, code_src, threshold, group) in PERS_INDICATOR_META.items():
        meta_all[code] = {'name': name, 'code_source': code_src,
                          'threshold': threshold, 'indicator_group': group}

    for col in ('name', 'code_source', 'threshold', 'indicator_group'):
        combined[col] = combined['code'].map(
            {k: v[col] for k, v in meta_all.items()}
        )

    combined['already_included'] = combined['code'].apply(
        lambda c: 'Y' if c in EWBI_RETAINED else 'N'
    )

    # Reorder columns
    first_cols = ['code', 'name', 'source', 'code_source', 'country', 'year',
                  'decile', 'value', 'already_included', 'level',
                  'threshold', 'indicator_group']
    extra = [c for c in combined.columns if c not in first_cols]
    return combined[first_cols + extra]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("EU-SILC Extended Indicators – Fast Pipeline")
    print("=" * 60)

    dirs = setup_directories()
    print(f"Output dir: {dirs['final_merged_dir']}")

    hh_long   = process_household_indicators(dirs)
    pers_long = process_personal_indicators(dirs)

    catalog = build_catalog(hh_long, pers_long)

    out_path = os.path.join(dirs['final_merged_dir'], "EU_SILC_extended_catalog.csv")
    catalog.to_csv(out_path, index=False)

    print(f"\n✅ Catalog saved → {out_path}")
    print(f"   Rows    : {len(catalog):,}")
    print(f"   Codes   : {catalog['code'].nunique()}")
    print(f"   Countries: {catalog['country'].nunique()}")
    print(f"   Years   : {catalog['year'].nunique()}")
    retained_count = (catalog['already_included'] == 'Y').sum()
    new_count      = (catalog['already_included'] == 'N').sum()
    print(f"   Already in EWBI (Y): {retained_count:,} rows")
    print(f"   New candidates  (N): {new_count:,} rows")


if __name__ == '__main__':
    main()
