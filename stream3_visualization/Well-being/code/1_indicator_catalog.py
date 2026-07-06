"""
Master Indicator Catalog Builder
=================================
Assembles a single wide-format CSV containing ALL computed indicators from:

  1. EU-SILC extended  (0_raw_indicator_EU-SILC_extended.py)   ← preferred
     Falls back to the original pipeline outputs if not yet generated.
  2. HBS extended      (0_raw_indicator_HBS_extended.py)        ← preferred
     Falls back to the original pipeline output if not yet generated.
  3. LFS               (0_raw_indicator_LFS.py output)

Output columns
--------------
  code               – indicator code (e.g. "HQ-SILC-1")
  name               – indicator name in English
  source             – data source ("EU-SILC", "HBS", "LFS")
  code_source        – Eurostat variable code(s) used
  country            – ISO 2-letter country code
  year               – reference year (int)
  decile             – income decile (1–10) or "All"
  value              – share of population (%) meeting the condition
  already_included   – "Y" if in current EWBI retained list, else "N"
  level              – "household" or "personal"
  threshold          – human-readable condition description
  indicator_group    – thematic group

Usage
-----
    python 1_indicator_catalog.py

Author: Data for Good – Well-being Team
"""

import glob
import os
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
SILC_DIR     = os.path.join(BASE_DIR, '0_raw_data_EUROSTAT', '0_EU-SILC', '3_final_merged_df')
HBS_DIR      = os.path.join(BASE_DIR, '0_raw_data_EUROSTAT', '0_HBS')
LFS_DIR      = os.path.join(BASE_DIR, '0_raw_data_EUROSTAT', '0_LFS')
CATALOG_DIR  = os.path.join(BASE_DIR, '1_final_df')
REPORTS_BASE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'reports',
    '3_eu_analysis_with_examples', 'outputs', 'graphs', 'HBS_expense', 'per_country'
))
EHIS_DIR     = os.path.join(BASE_DIR, '0_raw_data_EUROSTAT', '0_EHIS')
os.makedirs(CATALOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# EWBI retained indicators
# ---------------------------------------------------------------------------

EWBI_RETAINED = {
    # Energy & Housing
    "HQ-SILC-1", "HQ-SILC-2", "HQ-SILC-3", "HQ-SILC-4",
    "HQ-SILC-5", "HQ-SILC-6", "HQ-SILC-7", "HQ-SILC-8",
    "HE-SILC-2",
    # Equality
    "ES-SILC-1", "ES-SILC-2",
    "EC-SILC-2", "EC-SILC-3", "EC-SILC-4",
    # Health
    "AH-SILC-2", "AH-SILC-3", "AH-SILC-4",
    "AC-SILC-3", "AC-SILC-4",
    # Education
    "IS-SILC-3", "IS-SILC-4", "IS-SILC-5",
    # Quality of Jobs
    "RT-SILC-1", "RT-SILC-2",
    "RT-LFS-1",  "RT-LFS-2",  "RT-LFS-3",  "RT-LFS-4",
    "RT-LFS-5",  "RT-LFS-6",  "RT-LFS-7",  "RT-LFS-8",
    "RU-LFS-1",
}

# ---------------------------------------------------------------------------
# Indicator metadata lookup
# Covers all codes that appear in any of the three sources.
# Fields: (name, source, code_source, level, threshold, indicator_group)
# ---------------------------------------------------------------------------

INDICATOR_META = {
    # ── EU-SILC Household ────────────────────────────────────────────────
    "HQ-SILC-1": ("Overcrowded dwelling",
                  "EU-SILC", "HH030 (computed)", "household",
                  "overcrowded == 1",         "Energy & Housing"),
    "HQ-SILC-2": ("Cannot replace worn-out furniture",
                  "EU-SILC", "HD080",          "household",
                  "in [2,3]",                 "Energy & Housing"),
    "HQ-SILC-3": ("Cannot keep dwelling comfortably cool",
                  "EU-SILC", "HC070",          "household",
                  "== 2",                     "Energy & Housing"),
    "HQ-SILC-4": ("Dwelling too dark",
                  "EU-SILC", "HS160",          "household",
                  "== 1",                     "Energy & Housing"),
    "HQ-SILC-5": ("Noise from street or neighbours",
                  "EU-SILC", "HS170",          "household",
                  "== 1",                     "Energy & Housing"),
    "HQ-SILC-6": ("Leaking roof, damp or rot",
                  "EU-SILC", "HH040",          "household",
                  "== 1",                     "Energy & Housing"),
    "HQ-SILC-7": ("Pollution or crime in the area",
                  "EU-SILC", "HS180",          "household",
                  "== 1",                     "Energy & Housing"),
    "HQ-SILC-8": ("No renovation measures",
                  "EU-SILC", "HC003",          "household",
                  "in [4,99]",               "Energy & Housing"),
    "HE-SILC-1": ("Cannot keep dwelling comfortably warm",
                  "EU-SILC", "HC060",          "household",
                  "== 2",                     "Energy & Housing"),
    "HE-SILC-2": ("Arrears on utility bills",
                  "EU-SILC", "HS021",          "household",
                  "in [1,2]",                "Energy & Housing"),
    "HH-SILC-1": ("Arrears on mortgage or rent",
                  "EU-SILC", "HS011",          "household",
                  "in [1,2]",                "Equality"),
    "AN-SILC-1": ("Cannot afford meat/fish/veg every 2nd day",
                  "EU-SILC", "HS050",          "household",
                  "== 2",                     "Equality"),
    "ES-SILC-1": ("Cannot face unexpected financial expenses",
                  "EU-SILC", "HS060",          "household",
                  "== 2",                     "Equality"),
    "ES-SILC-2": ("Hard to make ends meet",
                  "EU-SILC", "HS120",          "household",
                  "in [1,2]",                "Equality"),
    "TS-SILC-1": ("Cannot afford 1-week holiday away from home",
                  "EU-SILC", "HS040",          "household",
                  "== 2",                     "Equality"),
    "EC-SILC-4": ("Persons living alone",
                  "EU-SILC", "HH030 (computed)", "household",
                  "household_size == 1",      "Equality"),
    # ── EU-SILC Personal ─────────────────────────────────────────────────
    "EL-SILC-1": ("Not satisfied with life",
                  "EU-SILC", "PW010",          "personal",
                  "< 3",                      "Equality"),
    "AH-SILC-1": ("Bad self-perceived health",
                  "EU-SILC", "PH010",          "personal",
                  "in [4,5]",                "Health"),
    "AH-SILC-2": ("Living with a chronic illness",
                  "EU-SILC", "PH020",          "personal",
                  "== 1",                     "Health"),
    "AH-SILC-3": ("Limited by health problems",
                  "EU-SILC", "PH030",          "personal",
                  "in [1,2]",                "Health"),
    "AH-SILC-4": ("Unable to work due to long-term illness",
                  "EU-SILC", "PL086",          "personal",
                  "> 0",                      "Health"),
    "AC-SILC-1": ("Could not afford medical care",
                  "EU-SILC", "PH050",          "personal",
                  "== 1",                     "Health"),
    "AC-SILC-3": ("Unmet need for medical examination",
                  "EU-SILC", "PH060",          "personal",
                  "== 1",                     "Health"),
    "AC-SILC-4": ("Unmet need for dental examination",
                  "EU-SILC", "PH040",          "personal",
                  "== 1",                     "Health"),
    "IC-SILC-1": ("Cannot regularly participate in leisure activity",
                  "EU-SILC", "PD060",          "personal",
                  "in [2,3]",                "Equality"),
    "IC-SILC-2": ("Cannot spend small amount on self",
                  "EU-SILC", "PD070",          "personal",
                  "in [2,3]",                "Equality"),
    "EC-SILC-1": ("Cannot meet friends/family monthly",
                  "EU-SILC", "PD050",          "personal",
                  "in [2,3]",                "Equality"),
    "EC-SILC-2": ("Not trusting others",
                  "EU-SILC", "PW191",          "personal",
                  "< 3",                      "Equality"),
    "EC-SILC-3": ("Cannot get together with friends/family",
                  "EU-SILC", "PD050",          "personal",
                  "in [2,3]",                "Equality"),
    "IS-SILC-3": ("No formal education (age > 15)",
                  "EU-SILC", "PE041",          "personal",
                  "== 0 or NaN (age > 15)",  "Education"),
    "IS-SILC-4": ("Not participating in formal training",
                  "EU-SILC", "PE010",          "personal",
                  "== 2",                     "Education"),
    "IS-SILC-5": ("No secondary education",
                  "EU-SILC", "PE041",          "personal",
                  "in [0,100]",              "Education"),
    "RT-SILC-1": ("Adults on fixed-term contracts",
                  "EU-SILC", "PL141",          "personal",
                  "== 2 / in [11,12] (age > 17)", "Quality of Jobs"),
    "RT-SILC-2": ("Adults working part-time",
                  "EU-SILC", "PL145",          "personal",
                  "== 2 (age > 17)",          "Quality of Jobs"),
    "RU-SILC-1": ("Unemployed for 6+ months",
                  "EU-SILC", "PL080",          "personal",
                  "> 5",                      "Quality of Jobs"),
    # ── HBS ──────────────────────────────────────────────────────────────
    "HH-HBS-1": ("Rent share > 2× national median",
                 "HBS", "EUR_HE041",           "household",
                 "> 2× nat. median",           "Energy & Housing"),
    "HH-HBS-2": ("Rent share < 0.5× national median",
                 "HBS", "EUR_HE041",           "household",
                 "< 0.5× nat. median",         "Energy & Housing"),
    "HH-HBS-3": ("Housing & utilities share > 2× national median",
                 "HBS", "EUR_HE04",            "household",
                 "> 2× nat. median",           "Energy & Housing"),
    "HH-HBS-4": ("Housing & utilities share < 0.5× national median",
                 "HBS", "EUR_HE04",            "household",
                 "< 0.5× nat. median",         "Energy & Housing"),
    "AC-HBS-1": ("Health expenditure share > 2× national median",
                 "HBS", "EUR_HE06",            "household",
                 "> 2× nat. median",           "Health"),
    "AC-HBS-2": ("Health expenditure share < 0.5× national median",
                 "HBS", "EUR_HE06",            "household",
                 "< 0.5× nat. median",         "Health"),
    "AE-HBS-1": ("Food expenditure share > 2× national median",
                 "HBS", "EUR_HE01",            "household",
                 "> 2× nat. median",           "Equality"),
    "AE-HBS-2": ("Food expenditure share < 0.5× national median",
                 "HBS", "EUR_HE01",            "household",
                 "< 0.5× nat. median",         "Equality"),
    "EC-HBS-1": ("Communications spend > 2× national median",
                 "HBS", "EUR_HJ08",            "household",
                 "> 2× nat. median",           "Equality"),
    "EC-HBS-2": ("Communications spend < 0.5× national median",
                 "HBS", "EUR_HJ08",            "household",
                 "< 0.5× nat. median",         "Equality"),
    "IE-HBS-1": ("Education expenditure share > 2× national median",
                 "HBS", "EUR_HE10",            "household",
                 "> 2× nat. median",           "Education"),
    "IE-HBS-2": ("Education expenditure share < 0.5× national median",
                 "HBS", "EUR_HE10",            "household",
                 "< 0.5× nat. median",         "Education"),
    "TT-HBS-1": ("Transport expenditure share > 2× national median",
                 "HBS", "EUR_HE07",            "household",
                 "> 2× nat. median",           "Equality"),
    "TT-HBS-2": ("Transport expenditure share < 0.5× national median",
                 "HBS", "EUR_HE07",            "household",
                 "< 0.5× nat. median",         "Equality"),
    "TS-HBS-1": ("Travel expenditure share > 2× national median",
                 "HBS", "EUR_HJ90",            "household",
                 "> 2× nat. median",           "Equality"),
    "TS-HBS-2": ("Travel expenditure share < 0.5× national median",
                 "HBS", "EUR_HJ90",            "household",
                 "< 0.5× nat. median",         "Equality"),
    "IC-HBS-1": ("Recreation/culture spend > 2× national median",
                 "HBS", "EUR_HE09",            "household",
                 "> 2× nat. median",           "Equality"),
    "IC-HBS-2": ("Recreation/culture spend < 0.5× national median",
                 "HBS", "EUR_HE09",            "household",
                 "< 0.5× nat. median",         "Equality"),
    # ── LFS ──────────────────────────────────────────────────────────────
    "RT-LFS-1": ("Working multiple jobs",
                 "LFS",  "NUMJOB",             "personal",
                 "NUMJOB in [2,3]",            "Quality of Jobs"),
    "RT-LFS-2": ("Wishing to work more hours",
                 "LFS",  "WISHMORE",           "personal",
                 "WISHMORE == 2",              "Quality of Jobs"),
    "RT-LFS-3": ("Overtime / extra hours worked",
                 "LFS",  "EXTRAHRS",           "personal",
                 "EXTRAHRS > 0",              "Quality of Jobs"),
    "RT-LFS-4": ("No flexibility in working time",
                 "LFS",  "FTPT",               "personal",
                 "FTPT indicates no flex",     "Quality of Jobs"),
    "RT-LFS-5": ("Shift work",
                 "LFS",  "SHIFTWK",            "personal",
                 "SHIFTWK == 1",              "Quality of Jobs"),
    "RT-LFS-6": ("Night work",
                 "LFS",  "NIGHTWK",            "personal",
                 "NIGHTWK == 1",             "Quality of Jobs"),
    "RT-LFS-7": ("Working on Saturdays",
                 "LFS",  "SATWK",              "personal",
                 "SATWK == 1",               "Quality of Jobs"),
    "RT-LFS-8": ("Working on Sundays",
                 "LFS",  "SUNWK",              "personal",
                 "SUNWK == 1",               "Quality of Jobs"),
    "RU-LFS-1": ("Unemployed persons",
                 "LFS",  "ILOSTAT",            "personal",
                 "ILOSTAT == 2",             "Quality of Jobs"),
    "EL-LFS-2": ("No adequate childcare services",
                 "LFS",  "various",            "personal",
                 "computed",                  "Education"),
    # ── HBS Overburden ─────────────────────────────────────────────────────
    "HE-HBS-1": ("Housing and energy expense overburden",
                 "HBS", "housing_energy_overburden_share", "household",
                 "overburden metric",           "Energy & Housing"),
    # ── EHIS ─────────────────────────────────────────────────────────────
    "AN-EHIS-1": ("Poor long-term health / chronic condition",
                  "EHIS", "HA1A",               "personal",
                  "in [2,3,4]",                 "Health"),
    "AE-EHIS-1": ("Low fruit consumption",
                  "EHIS", "FV1",                "personal",
                  "== 4 (less than once a day)", "Health"),
    "AE-EHIS-2": ("Vegetables less than once a week",
                  "EHIS", "DH3",                "personal",
                  "in [4,5]",                   "Health"),
    "AN-EHIS-2": ("Person characterized as obese",
                  "EHIS", "BMI",                "personal",
                  "== 4",                       "Health"),
    "EC-EHIS-1": ("No social support",
                  "EHIS", "SS1",                "personal",
                  "== 1",                       "Equality"),
    "ED-EHIS-1": ("Poor mental health",
                  "EHIS", "HA1B",               "personal",
                  "in [2,3,4]",                 "Health"),
    "AH-EHIS-2": ("No physical activity outside work",
                  "EHIS", "PE6",                "personal",
                  "== 0",                       "Health"),
    "AC-EHIS-1": ("Unmet healthcare need",
                  "EHIS", "UN2C",               "personal",
                  "== 1",                       "Health"),
    "AB-EHIS-1": ("Current smoker",
                  "EHIS", "SK1",                "personal",
                  "== 1",                       "Health"),
    "AB-EHIS-2": ("Hazardous alcohol consumption",
                  "EHIS", "AL1",                "personal",
                  "== 1",                       "Health"),
    "AB-EHIS-3": ("Difficulty affording healthcare",
                  "EHIS", "AC1A",               "personal",
                  "== 1",                       "Health"),
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _normalise(df: pd.DataFrame, source_override: str | None = None) -> pd.DataFrame:
    """
    Bring a dataframe from any pipeline output into the standard catalog schema.
    Expected input columns (old format):  year, country, decile, primary_index, value, database
    Expected input columns (new format):  code, name, source, ... (already normalised)
    """
    # Already in new format
    if 'code' in df.columns and 'source' in df.columns:
        return df

    # Old format: rename primary_index → code,  database → source
    rename = {}
    if 'primary_index' in df.columns:
        rename['primary_index'] = 'code'
    if 'database' in df.columns:
        rename['database'] = 'source'
    df = df.rename(columns=rename)

    if source_override:
        df['source'] = source_override

    return df


def _load_csv(path: str, label: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"  [WARN] Not found: {path}  [{label}]")
        return None
    df = pd.read_csv(path, low_memory=False)
    print(f"  [OK] Loaded {label}: {len(df):,} rows, {df['primary_index' if 'primary_index' in df.columns else 'code'].nunique()} codes")
    return df


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def load_eu_silc(silc_dir: str) -> pd.DataFrame:
    """
    Load EU-SILC indicators, preferring the extended catalog if available.
    Falls back to the two original pipeline summaries (household + personal).
    """
    extended_path = os.path.join(silc_dir, "EU_SILC_extended_catalog.csv")
    if os.path.exists(extended_path):
        print("  Using EU-SILC extended catalog (preferred) …")
        df = pd.read_csv(extended_path, low_memory=False)
        return df

    print("  Extended EU-SILC catalog not found – using original pipeline outputs …")
    frames = []
    for fname in ("EU_SILC_household_final_summary.csv",
                  "EU_SILC_personal_final_summary.csv"):
        p = os.path.join(silc_dir, fname)
        tmp = _load_csv(p, fname)
        if tmp is not None:
            frames.append(tmp)

    if not frames:
        print("  [ERROR] No EU-SILC output found.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = _normalise(df)
    return df


def load_hbs(hbs_dir: str) -> pd.DataFrame:
    """
    Load HBS indicators, preferring the extended catalog.
    Falls back to original HBS summary.
    """
    extended_path = os.path.join(hbs_dir, "HBS_extended_catalog.csv")
    if os.path.exists(extended_path):
        print("  Using HBS extended catalog (preferred) …")
        df = pd.read_csv(extended_path, low_memory=False)
        return df

    print("  Extended HBS catalog not found – using original pipeline output …")
    p = os.path.join(hbs_dir, "HBS_household_final_summary.csv")
    df = _load_csv(p, "HBS_household_final_summary.csv")
    if df is None:
        return pd.DataFrame()
    df = _normalise(df)
    return df


def load_lfs(lfs_dir: str) -> pd.DataFrame:
    """Load LFS indicators (no extended version)."""
    p = os.path.join(lfs_dir, "LFS_household_final_summary.csv")
    df = _load_csv(p, "LFS_household_final_summary.csv")
    if df is None:
        return pd.DataFrame()
    df = _normalise(df, source_override='LFS')
    return df


def load_eu_silc_supplementary(silc_dir: str) -> pd.DataFrame:
    """Load supplementary EU-SILC indicators (SP-SILC-*, GE-SILC-*)."""
    p = os.path.join(silc_dir, "EU_SILC_supplementary_catalog.csv")
    if not os.path.exists(p):
        print(f"  [WARN] Supplementary EU-SILC catalog not found: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    print(f"  [OK] Loaded EU-SILC supplementary: {len(df):,} rows, {df['code'].nunique()} codes")
    return df


def load_hbs_overburden(reports_dir: str) -> pd.DataFrame:
    """
    Load housing and energy overburden data from per-year Excel files.

    Reads all files matching expense_he_overburden_deciles_YEAR.xlsx in
    reports_dir, melts D1-D10 columns into long format, and returns a
    dataframe in the standard catalog schema.
    """
    pattern = os.path.join(reports_dir, "expense_he_overburden_deciles_*.xlsx")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  [WARN] No HBS overburden files found: {pattern}")
        return pd.DataFrame()

    frames = []
    for fpath in files:
        fname = os.path.basename(fpath)
        m = re.search(r'_(\d{4})\.xlsx$', fname)
        if not m:
            continue
        year = int(m.group(1))

        raw = pd.read_excel(fpath)
        val_cols = [c for c in raw.columns if c.startswith('housing_energy_overburden_share_D')]
        melted = raw[['Country'] + val_cols].melt(
            id_vars=['Country'], value_vars=val_cols,
            var_name='decile_col', value_name='value'
        )
        melted['decile'] = (
            melted['decile_col'].str.extract(r'_D(\d+)$').astype(int)
        )
        melted['year']             = year
        melted['country']          = melted['Country']
        melted['code']             = 'HE-HBS-1'
        melted['name']             = 'Housing and energy expense overburden'
        melted['source']           = 'HBS'
        melted['code_source']      = 'housing_energy_overburden_share'
        melted['level']            = 'household'
        melted['threshold']        = 'overburden metric'
        melted['indicator_group']  = 'Energy & Housing'
        melted['already_included'] = 'N'

        keep = ['code', 'name', 'source', 'code_source',
                'country', 'year', 'decile', 'value',
                'already_included', 'level', 'threshold', 'indicator_group']
        frames.append(melted[keep])

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    print(f"  [OK] Loaded HBS overburden (HE-HBS-1): {len(result):,} rows, "
          f"{result['country'].nunique()} countries, {result['year'].nunique()} years")
    return result


def load_ehis(ehis_dir: str) -> pd.DataFrame:
    """Load EHIS indicators and expand quintiles to deciles (Q1→D1+D2, …Q5→D9+D10)."""
    p = os.path.join(ehis_dir, "EHIS_household_final_summary.csv")
    if not os.path.exists(p):
        print(f"  [WARN] EHIS summary not found: {p}")
        return pd.DataFrame()

    df = pd.read_csv(p, low_memory=False)
    df = _normalise(df, source_override='EHIS')

    # Keep only the 3 requested EHIS indicators
    EHIS_KEEP = {'AE-EHIS-2', 'AN-EHIS-2', 'AH-EHIS-2'}
    df = df[df['code'].isin(EHIS_KEEP)]

    # Expand each quintile row into two decile rows: Q→D(2Q-1) and D(2Q)
    QUINTILE_TO_DECILES = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8], 5: [9, 10]}
    numeric_mask = pd.to_numeric(df.get('quintile', pd.Series(dtype=float)), errors='coerce').notna()
    q_df = df[numeric_mask].copy()
    q_df['quintile'] = pd.to_numeric(q_df['quintile'], errors='coerce').astype(int)

    expanded = []
    for q, deciles in QUINTILE_TO_DECILES.items():
        chunk = q_df[q_df['quintile'] == q].copy()
        for d in deciles:
            row = chunk.copy()
            row['decile'] = str(d)
            expanded.append(row)

    # Add an "All" aggregate row as mean across quintiles per country/year/code
    group_cols = ['year', 'country', 'code']
    present_cols = [c for c in group_cols if c in q_df.columns]
    if present_cols == group_cols:
        all_rows = q_df.groupby(group_cols, as_index=False)['value'].mean()
        all_rows['decile'] = 'All'
        all_rows['source'] = 'EHIS'
        if 'flag' in q_df.columns:
            all_rows['flag'] = 'quintile_data'
        expanded.append(all_rows)

    result = pd.concat(expanded, ignore_index=True)
    result = result.drop(columns=['quintile'], errors='ignore')
    print(f"  [OK] Loaded EHIS: {len(result):,} rows, {result['code'].nunique()} codes "
          f"(quintile→decile expansion applied)")
    return result


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------

def enrich_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach name, code_source, level, threshold, and indicator_group columns
    from INDICATOR_META.  Columns already present (from extended catalogs)
    are kept as-is.
    """
    meta_fields = ['name', 'source', 'code_source', 'level', 'threshold', 'indicator_group']

    for field in meta_fields:
        if field not in df.columns:
            df[field] = pd.NA
        # Ensure object dtype so string assignment does not trigger FutureWarning
        df[field] = df[field].astype(object)

    for code, meta in INDICATOR_META.items():
        name, source, code_source, level, threshold, group = meta
        mask = df['code'] == code
        if mask.any():
            # Only fill if the column is still NaN (don't overwrite extended metadata)
            if df.loc[mask, 'name'].isna().all():
                df.loc[mask, 'name']             = name
            if df.loc[mask, 'source'].isna().all():
                df.loc[mask, 'source']           = source
            if df.loc[mask, 'code_source'].isna().all():
                df.loc[mask, 'code_source']      = code_source
            if df.loc[mask, 'level'].isna().all():
                df.loc[mask, 'level']            = level
            if df.loc[mask, 'threshold'].isna().all():
                df.loc[mask, 'threshold']        = threshold
            if df.loc[mask, 'indicator_group'].isna().all():
                df.loc[mask, 'indicator_group']  = group

    df['already_included'] = df['code'].apply(
        lambda c: 'Y' if c in EWBI_RETAINED else 'N'
    )

    return df


# ---------------------------------------------------------------------------
# Final assembly
# ---------------------------------------------------------------------------

def assemble_catalog() -> pd.DataFrame:
    print("\n-- Loading sources --")
    silc_df = load_eu_silc(SILC_DIR)
    hbs_df  = load_hbs(HBS_DIR)
    lfs_df  = load_lfs(LFS_DIR)
    supp_df = load_eu_silc_supplementary(SILC_DIR)
    hbs_ob_df = load_hbs_overburden(REPORTS_BASE)
    ehis_df = load_ehis(EHIS_DIR)

    frames = [f for f in (silc_df, hbs_df, lfs_df, supp_df, hbs_ob_df, ehis_df) if not f.empty]
    if not frames:
        raise RuntimeError("No indicator data could be loaded. Run pipeline scripts first.")

    print("\n-- Combining sources --")
    catalog = pd.concat(frames, ignore_index=True)
    catalog = enrich_metadata(catalog)

    # Coerce types
    catalog['year']   = pd.to_numeric(catalog['year'],  errors='coerce').astype('Int64')
    catalog['value']  = pd.to_numeric(catalog['value'], errors='coerce')

    # decile: keep as string ("1"–"10" or "All")
    catalog['decile'] = catalog['decile'].astype(str).str.strip()
    # Convert numeric floats like "1.0" → "1"
    def clean_decile(d):
        try:
            f = float(d)
            if f == int(f):
                return str(int(f))
        except (ValueError, TypeError):
            pass
        return d
    catalog['decile'] = catalog['decile'].apply(clean_decile)

    # Canonical column order
    first_cols = [
        'code', 'name', 'source', 'code_source',
        'country', 'year', 'decile', 'value',
        'already_included', 'level', 'threshold', 'indicator_group'
    ]
    extra = [c for c in catalog.columns if c not in first_cols]
    catalog = catalog[first_cols + extra]

    # Sort for readability
    catalog.sort_values(['source', 'code', 'country', 'year', 'decile'],
                        ignore_index=True, inplace=True)

    return catalog


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(catalog: pd.DataFrame):
    print("\n" + "=" * 60)
    print("MASTER INDICATOR CATALOG – SUMMARY")
    print("=" * 60)
    print(f"  Total rows         : {len(catalog):,}")
    print(f"  Unique indicators  : {catalog['code'].nunique()}")
    print(f"  Countries          : {catalog['country'].nunique()}")
    years = catalog['year'].dropna().astype(int)
    print(f"  Years              : {years.min()} – {years.max()}")

    print("\n  By source:")
    for src, grp in catalog.groupby('source'):
        print(f"    {src:<10}: {grp['code'].nunique():>3} codes, "
              f"{grp['country'].nunique():>3} countries, "
              f"{grp['year'].nunique():>3} years, "
              f"{len(grp):>8,} rows")

    print("\n  By already_included:")
    vc = catalog.drop_duplicates(subset=['code', 'source'])['already_included'].value_counts()
    for flag, cnt in vc.items():
        label = "Retained in EWBI" if flag == 'Y' else "New / not retained"
        print(f"    {flag} ({label}): {cnt} indicator codes")

    print("\n  EWBI-retained indicators (Y):")
    retained = sorted(catalog[catalog['already_included'] == 'Y']['code'].unique())
    for i, c in enumerate(retained):
        print(f"    {c}", end="\n" if (i + 1) % 6 == 0 else "  ")
    print()

    print("\n  New / candidate indicators (N):")
    new = sorted(catalog[catalog['already_included'] == 'N']['code'].unique())
    for i, c in enumerate(new):
        print(f"    {c}", end="\n" if (i + 1) % 6 == 0 else "  ")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Master Indicator Catalog Builder")
    print("=" * 60)

    catalog = assemble_catalog()

    out_path = os.path.join(CATALOG_DIR, "master_indicator_catalog.csv")
    catalog.to_csv(out_path, index=False)
    print(f"\nMaster catalog saved -> {out_path}")

    print_summary(catalog)


if __name__ == '__main__':
    main()
