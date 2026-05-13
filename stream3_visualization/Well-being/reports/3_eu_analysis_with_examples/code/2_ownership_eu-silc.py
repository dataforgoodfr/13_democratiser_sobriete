"""
2_ownership_eu-silc.py — EU-SILC Ownership Variation for Report Configs
========================================================================
For each report, computes the change in homeownership rate between an
early period and a late period, broken down by age group and income decile.

Grid plot: rows = age groups (youngest → oldest), columns = countries.

Cluster reports (rep_eu, rep_ewbi): 2 countries per cluster, cluster labels.
Focus reports   (rep_fr, rep_ch):   comparison countries side-by-side,
                                    focus country highlighted.

Data Source: EU-SILC Cross-sectional 2004-2023
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os
import openpyxl
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DATA_PATH = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/EU-SILC/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
OUTPUT_BASE = os.path.join(BASE_DIR, 'outputs', 'graphs', 'EU-SILC', 'ownership_variation')
os.makedirs(OUTPUT_BASE, exist_ok=True)

COUNTRY_NAME_MAP = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus',
    'CZ': 'Czechia', 'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia',
    'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland', 'FR': 'France',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta',
    'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania',
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia', 'CH': 'Switzerland',
    'NO': 'Norway', 'IS': 'Iceland',
}

# Cluster definitions (Method 5, same as other scripts)
CLUSTER_CANDIDATES = [
    {
        'label': 'Cluster 0 - Low performer / Low EWBI',
        'candidates': ['FR', 'ES', 'EL', 'IT', 'PT', 'FI'],
    },
    {
        'label': 'Cluster 1 - Low performer / High EWBI',
        'candidates': ['BE', 'NL', 'AT', 'DK', 'IE'],
    },
    {
        'label': 'Cluster 2 - High performer / Low EWBI',
        'candidates': ['LT', 'RO', 'HU', 'BG', 'LV', 'EE'],
    },
    {
        'label': 'Cluster 3 - High performer / High EWBI',
        'candidates': ['DE', 'PL', 'SE', 'CZ', 'SI', 'SK'],
    },
]

# ---------------------------------------------------------------------------
# Time periods
# ---------------------------------------------------------------------------
EARLY_YEARS_DEFAULT = [2004, 2005]
EARLY_YEARS_OVERRIDE = {
    'LT': [2005, 2006],
    'CH': [2007, 2008, 2009],   # 3 years: small CH sample
    'BG': [2006, 2007],
    'RO': [2007, 2008],
    'HR': [2010, 2011],
    'MT': [2007, 2008],
}
LATE_YEARS_DEFAULT = [2022, 2023]
LATE_YEARS_OVERRIDE = {
    'CH': [2020, 2021, 2022],  # 3 years: small CH sample
}

AGE_GROUPS = {
    1: {'label': '18-30',  'range': (18, 30)},
    2: {'label': '31-45',  'range': (31, 45)},
    3: {'label': '46-60',  'range': (46, 60)},
    4: {'label': '61+',    'range': (61, 150)},
}

DECILE_LABELS = [f'D{i}' for i in range(1, 11)]

# ---------------------------------------------------------------------------
# Report configurations
# ---------------------------------------------------------------------------
REPORT_CONFIGS = {
    'rep_eu': {
        'prefix': 'rep_eu',
        'title_suffix': '(EU-27)',
        'show_clusters': True,
        'comparison_countries': None,      # derived from clusters
        'focus_country': None,
    },
    'rep_ewbi': {
        'prefix': 'rep_ewbi',
        'title_suffix': '(EU-27 + EFTA)',
        'show_clusters': True,
        'comparison_countries': None,
        'focus_country': None,
    },
    'rep_fr': {
        'prefix': 'rep_fr',
        'title_suffix': '(France)',
        'show_clusters': False,
        'comparison_countries': ['FR', 'ES', 'IT', 'CH', 'DE', 'BE'],
        'focus_country': 'FR',
    },
    'rep_ch': {
        'prefix': 'rep_ch',
        'title_suffix': '(Switzerland)',
        'show_clusters': False,
        'comparison_countries': ['CH', 'FR', 'IT', 'DE', 'AT'],
        'focus_country': 'CH',
    },
}

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
FOCUS_COLOR_POS = '#2166ac'   # blue for focus country positive bars
FOCUS_COLOR_NEG = '#b2182b'   # dark red for focus country negative bars
NORMAL_COLOR_POS = '#4daf4a'
NORMAL_COLOR_NEG = '#e41a1c'


# ============================================================================
# HELPER FUNCTIONS (from eu_silc_ownership_variation_multi_country.py)
# ============================================================================

def oecd_weight(age):
    if pd.isna(age):
        return 0.5
    try:
        age_val = int(age)
    except Exception:
        return 0.5
    return 0.3 if age_val < 14 else 0.5


def categorize_tenure(tenure_value, year):
    if pd.isna(tenure_value):
        return None
    tenure = int(tenure_value)
    if year >= 2010:
        if tenure in [1, 2]:
            return 'owner'
        elif tenure in [3, 4, 5]:
            return 'renter'
    else:
        if tenure == 1:
            return 'owner'
        elif tenure in [2, 3, 4]:
            return 'renter'
    return None


def get_tenure_column(year):
    return 'HH020' if year < 2010 else 'HH021'


def assign_age_group(age):
    age = int(age)
    if 18 <= age <= 30:
        return 1
    elif 31 <= age <= 45:
        return 2
    elif 46 <= age <= 60:
        return 3
    elif age >= 61:
        return 4
    return None


def file_code(country, year):
    if country == 'EL' and year <= 2007:
        return 'GR'
    return country


# ============================================================================
# DATA LOADING
# ============================================================================

def load_personal_data_with_age(year, country):
    fc = file_code(country, year)
    r_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{fc}{str(year)[-2:]}R.csv"
    if not os.path.exists(r_file_path):
        return None
    try:
        columns_to_try = [
            ['RB010', 'RB020', 'RB030', 'RB080', 'RB100', 'RB110', 'RB220', 'RB230'],
            ['RB010', 'RB020', 'RB030', 'RB080', 'RB100', 'RB220', 'RB230'],
            ['RB010', 'RB020', 'RB030', 'RB080', 'RB220', 'RB230'],
        ]
        df = None
        for cols in columns_to_try:
            try:
                df = pd.read_csv(r_file_path, usecols=cols, on_bad_lines='skip')
                break
            except ValueError:
                continue
        if df is None:
            return None
        df = df[df['RB220'].isna() & df['RB230'].isna()]
        if len(df) == 0:
            return None
        df = df.dropna(subset=['RB080'])
        df['RB080'] = df['RB080'].astype(int)
        if 'RB100' in df.columns and df['RB100'].notna().sum() > 0:
            interview_year = df['RB100'].astype(int)
        else:
            interview_year = year
        df['age'] = interview_year - df['RB080']
        df = df[df['age'] >= 18]
        if len(df) == 0:
            return None
        df['age_group'] = df['age'].apply(assign_age_group)
        return df[['RB010', 'RB020', 'RB030', 'age', 'age_group']]
    except Exception as e:
        print(f"    Error reading R-file {country}/{year}: {e}")
        return None


def load_household_with_tenure(year, country):
    fc = file_code(country, year)
    hh_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{fc}{str(year)[-2:]}H.csv"
    if not os.path.exists(hh_file_path):
        return None
    try:
        tenure_col = get_tenure_column(year)
        df = pd.read_csv(hh_file_path, usecols=['HB010', 'HB020', 'HB030', tenure_col],
                         on_bad_lines='skip')
        df = df.dropna(subset=[tenure_col])
        df['tenure'] = df[tenure_col].apply(lambda x: categorize_tenure(x, year))
        df = df[df['tenure'].notna()]
        return df[['HB010', 'HB020', 'HB030', 'tenure']]
    except Exception as e:
        print(f"    Error reading H-file {country}/{year}: {e}")
        return None


def load_equivalized_income(year, country):
    fc = file_code(country, year)
    hh_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{fc}{str(year)[-2:]}H.csv"
    pr_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{fc}{str(year)[-2:]}R.csv"
    if not os.path.exists(hh_file_path) or not os.path.exists(pr_file_path):
        return None
    try:
        household_df = pd.read_csv(hh_file_path, usecols=['HB010', 'HB020', 'HB030', 'HY020'],
                                   on_bad_lines='skip')
        age_columns_to_try = [
            ['RB010', 'RB020', 'RB030', 'RB081', 'RB082'],
            ['RB010', 'RB020', 'RB030', 'RB082'],
            ['RB010', 'RB020', 'RB030', 'RB081'],
            ['RB010', 'RB020', 'RB030'],
        ]
        personal_df = None
        for cols in age_columns_to_try:
            try:
                personal_df = pd.read_csv(pr_file_path, usecols=cols, on_bad_lines='skip')
                break
            except Exception:
                continue
        if personal_df is None:
            return None

        personal_df['RB030'] = personal_df['RB030'].fillna(0).astype(str)
        personal_df['RB040'] = personal_df['RB030'].str[:-2]
        household_df['HB030'] = household_df['HB030'].fillna(0).astype(str)

        if 'RB081' in personal_df.columns and 'RB082' in personal_df.columns:
            personal_df['age'] = personal_df['RB081'].fillna(personal_df['RB082'])
        elif 'RB081' in personal_df.columns:
            personal_df['age'] = personal_df['RB081']
        elif 'RB082' in personal_df.columns:
            personal_df['age'] = personal_df['RB082']
        else:
            personal_df['age'] = np.nan

        merged_df = personal_df.merge(
            household_df,
            left_on=['RB010', 'RB020', 'RB040'],
            right_on=['HB010', 'HB020', 'HB030'],
            how='left',
        )
        merged_df['oecd_weight'] = merged_df['age'].apply(oecd_weight)
        merged_df.sort_values(
            by=['HB010', 'HB020', 'HB030', 'age'],
            ascending=[True, True, True, False], inplace=True,
        )
        merged_df['person_rank'] = merged_df.groupby(['HB010', 'HB020', 'HB030']).cumcount()
        merged_df['oecd_weight'] = merged_df.apply(
            lambda row: 1.0 if row['person_rank'] == 0 else row['oecd_weight'], axis=1,
        )

        equiv_size_df = (
            merged_df.groupby(['HB010', 'HB020', 'HB030'])['oecd_weight']
            .sum().reset_index()
        )
        equiv_size_df.rename(columns={'oecd_weight': 'equivalent_size'}, inplace=True)

        household_df = household_df.merge(equiv_size_df, on=['HB010', 'HB020', 'HB030'], how='left')
        household_df['equi_disp_inc'] = household_df['HY020'] / household_df['equivalent_size']
        household_df = household_df.dropna(subset=['HY020', 'equi_disp_inc'])
        return household_df
    except Exception as e:
        print(f"    Error computing equivalized income {country}/{year}: {e}")
        return None


def calculate_income_deciles(year, country):
    hh_df = load_equivalized_income(year, country)
    if hh_df is None or len(hh_df) == 0:
        return None
    fc = file_code(country, year)
    db_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{fc}{str(year)[-2:]}D.csv"
    if not os.path.exists(db_file_path):
        return None
    try:
        weights_df = pd.read_csv(db_file_path, usecols=['DB010', 'DB020', 'DB030', 'DB090'],
                                 on_bad_lines='skip')
    except Exception:
        return None

    hh_df['HB030'] = hh_df['HB030'].astype(str)
    weights_df['DB030'] = weights_df['DB030'].astype(str)
    hh_df['HB020'] = hh_df['HB020'].astype(str)
    weights_df['DB020'] = weights_df['DB020'].astype(str)

    hh_df = hh_df.merge(weights_df, left_on=['HB010', 'HB020', 'HB030'],
                        right_on=['DB010', 'DB020', 'DB030'], how='left')
    hh_valid = hh_df.dropna(subset=['equi_disp_inc', 'DB090']).copy()
    if len(hh_valid) == 0:
        return None

    hh_valid = hh_valid.sort_values('equi_disp_inc').reset_index(drop=True)
    hh_valid['cumsum_weight'] = hh_valid['DB090'].cumsum()
    total_weight = hh_valid['DB090'].sum()
    hh_valid['cum_pct'] = hh_valid['cumsum_weight'] / total_weight

    decile_dict = {}
    for decile_pct in range(1, 10):
        target_pct = decile_pct / 10.0
        idx = (hh_valid['cum_pct'] - target_pct).abs().idxmin()
        decile_dict[f'decile_{decile_pct}'] = hh_valid.loc[idx, 'equi_disp_inc']
    return decile_dict


def assign_income_decile(row, decile_thresholds):
    income = row['equi_disp_inc']
    if pd.isna(income) or decile_thresholds is None:
        return np.nan
    try:
        for decile_num in range(1, 10):
            threshold = decile_thresholds.get(f'decile_{decile_num}')
            if pd.isna(threshold):
                return np.nan
            if income <= threshold:
                return decile_num
        return 10
    except Exception:
        return np.nan


def load_d_file_weights(year, country):
    fc = file_code(country, year)
    d_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{fc}{str(year)[-2:]}D.csv"
    if not os.path.exists(d_file_path):
        return None
    try:
        df = pd.read_csv(d_file_path, usecols=['DB010', 'DB020', 'DB030', 'DB090'],
                         on_bad_lines='skip')
        df = df.dropna(subset=['DB090'])
        return df
    except Exception as e:
        print(f"    Error reading D-file {country}/{year}: {e}")
        return None


# ============================================================================
# CORE: pool two years for one country
# ============================================================================

def ownership_pooled(years, country):
    all_merged = []

    for year in years:
        personal_df = load_personal_data_with_age(year, country)
        if personal_df is None or len(personal_df) == 0:
            continue

        household_df = load_household_with_tenure(year, country)
        if household_df is None or len(household_df) == 0:
            continue

        hh_income_df = load_equivalized_income(year, country)
        if hh_income_df is None or len(hh_income_df) == 0:
            continue

        decile_thresholds = calculate_income_deciles(year, country)
        if decile_thresholds is None:
            continue

        hh_income_df['decile'] = hh_income_df.apply(
            lambda row: assign_income_decile(row, decile_thresholds), axis=1)
        hh_income_df = hh_income_df.dropna(subset=['decile'])

        household_df['HB030'] = household_df['HB030'].astype(str)
        hh_income_df['HB030'] = hh_income_df['HB030'].astype(str)
        household_df['HB020'] = household_df['HB020'].astype(str)
        hh_income_df['HB020'] = hh_income_df['HB020'].astype(str)

        household_df = household_df.merge(
            hh_income_df[['HB010', 'HB020', 'HB030', 'decile']],
            on=['HB010', 'HB020', 'HB030'], how='left')
        household_df = household_df.dropna(subset=['decile'])

        weights_df = load_d_file_weights(year, country)
        if weights_df is None:
            continue
        weights_df['DB030'] = weights_df['DB030'].astype(str)
        weights_df['DB020'] = weights_df['DB020'].astype(str)
        household_df = household_df.merge(
            weights_df, left_on=['HB010', 'HB020', 'HB030'],
            right_on=['DB010', 'DB020', 'DB030'], how='left')
        household_df = household_df.dropna(subset=['DB090'])

        personal_df['RB030'] = personal_df['RB030'].astype(str)
        personal_df['household_id'] = personal_df['RB030'].str[:-2]
        household_df['HB030'] = household_df['HB030'].astype(str)
        household_df['HB020'] = household_df['HB020'].astype(str)
        household_df['HB010'] = household_df['HB010'].astype(str)
        personal_df['RB020'] = personal_df['RB020'].astype(str)
        personal_df['RB010'] = personal_df['RB010'].astype(str)

        merged = personal_df.merge(
            household_df,
            left_on=['RB010', 'RB020', 'household_id'],
            right_on=['HB010', 'HB020', 'HB030'],
            how='left')
        merged = merged.dropna(subset=['tenure', 'decile', 'DB090'])
        if len(merged) > 0:
            all_merged.append(merged)

    if not all_merged:
        return None

    combined = pd.concat(all_merged, ignore_index=True)

    results = {}
    for age_id in AGE_GROUPS:
        age_data = combined[combined['age_group'] == age_id]
        results[age_id] = {}
        for decile in range(1, 11):
            dec_data = age_data[age_data['decile'] == decile]
            if len(dec_data) == 0:
                results[age_id][decile] = None
                continue
            total_w = dec_data['DB090'].sum()
            owner_w = dec_data[dec_data['tenure'] == 'owner']['DB090'].sum()
            results[age_id][decile] = (owner_w / total_w * 100) if total_w > 0 else None
    return results


def compute_country_variations(country_code):
    early_years = EARLY_YEARS_OVERRIDE.get(country_code, EARLY_YEARS_DEFAULT)
    late_years = LATE_YEARS_OVERRIDE.get(country_code, LATE_YEARS_DEFAULT)

    print(f"    {country_code}: early={early_years}, late={late_years}")
    early = ownership_pooled(early_years, country_code)
    late = ownership_pooled(late_years, country_code)

    if early is None or late is None:
        return None

    diffs_pp = {}
    diffs_pct = {}
    for age_id in AGE_GROUPS:
        diffs_pp[age_id] = {}
        diffs_pct[age_id] = {}
        for decile in range(1, 11):
            e = early.get(age_id, {}).get(decile)
            l = late.get(age_id, {}).get(decile)
            if e is None or l is None:
                diffs_pp[age_id][decile] = None
                diffs_pct[age_id][decile] = None
                continue
            pp = l - e
            diffs_pp[age_id][decile] = pp
            diffs_pct[age_id][decile] = (pp / e * 100) if e != 0 else None

    return {
        'early_years': early_years,
        'late_years': late_years,
        'pp': diffs_pp,
        'pct': diffs_pct,
    }


# ============================================================================
# COUNTRY SELECTION
# ============================================================================

def select_countries_for_report(cfg, country_cache):
    """
    Returns (ordered_country_codes, cluster_info_or_None).

    cluster reports : picks 2 per cluster, returns cluster list
    focus reports   : uses comparison_countries list, returns None
    """
    if cfg['show_clusters']:
        selected_clusters = []
        ordered = []
        for cluster in CLUSTER_CANDIDATES:
            chosen = []
            for cc in cluster['candidates']:
                if cc not in country_cache:
                    print(f"  Computing {COUNTRY_NAME_MAP.get(cc, cc)} ({cc})...")
                    country_cache[cc] = compute_country_variations(cc)
                if country_cache[cc] is not None:
                    chosen.append(cc)
                if len(chosen) == 2:
                    break
            if len(chosen) < 2:
                print(f"    WARNING: {cluster['label']} — only {len(chosen)} countries")
            selected_clusters.append({'label': cluster['label'], 'countries': chosen})
            ordered.extend(chosen)
        return ordered, selected_clusters
    else:
        ordered = []
        for cc in cfg['comparison_countries']:
            if cc not in country_cache:
                print(f"  Computing {COUNTRY_NAME_MAP.get(cc, cc)} ({cc})...")
                country_cache[cc] = compute_country_variations(cc)
            if country_cache[cc] is not None:
                ordered.append(cc)
            else:
                print(f"    WARNING: {cc} — no data, skipped")
        return ordered, None


# ============================================================================
# PLOTTING
# ============================================================================

def _save_fig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    svg_path = os.path.splitext(path)[0] + '.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"    Saved: {os.path.basename(path)}")
    print(f"    Saved: {os.path.basename(svg_path)}")


def create_variation_grid(all_values, country_codes, country_cache,
                          cfg, selected_clusters, out_dir, value_mode='pp'):
    """
    Grid plot: rows = age groups, columns = countries.
    """
    if not country_codes:
        print("    No data to plot")
        return

    n_cols = len(country_codes)
    n_rows = len(AGE_GROUPS)
    age_order = sorted(AGE_GROUPS.keys())
    focus = cfg.get('focus_country')

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows),
                             squeeze=False, sharey=True)

    value_label = 'percentage points' if value_mode == 'pp' else 'relative variation (%)'
    fig.suptitle(
        f'Change in Homeownership Rate by Age Group and Income Decile\n'
        f'({value_label}) {cfg["title_suffix"]}',
        fontsize=14, fontweight='bold', y=0.995,
    )

    # Global y-range
    all_vals = []
    for cc in country_codes:
        if cc not in all_values:
            continue
        for ag in age_order:
            for d in range(1, 11):
                v = all_values[cc].get(ag, {}).get(d)
                if v is not None:
                    all_vals.append(v)
    if all_vals:
        raw_max = max(all_vals) * 1.10
        raw_min = min(all_vals) * 1.10
        step = 20 if value_mode == 'pp' else 25
        y_top = int(np.ceil(raw_max / step) * step)
        y_bot = int(np.floor(raw_min / step) * step)
        if y_top < step:
            y_top = step
        if y_bot > -step:
            y_bot = -step
    else:
        step = 20 if value_mode == 'pp' else 25
        y_top = 40
        y_bot = -20

    for col_idx, cc in enumerate(country_codes):
        early_yrs = country_cache[cc]['early_years']
        late_yrs = country_cache[cc]['late_years']
        is_focus = (cc == focus)

        for row_idx, age_id in enumerate(age_order):
            ax = axes[row_idx][col_idx]

            values = []
            colors = []
            for decile in range(1, 11):
                v = all_values.get(cc, {}).get(age_id, {}).get(decile)
                if v is None:
                    v = 0
                values.append(v)
                if is_focus:
                    colors.append(FOCUS_COLOR_NEG if v < 0 else FOCUS_COLOR_POS)
                else:
                    colors.append(NORMAL_COLOR_NEG if v < 0 else NORMAL_COLOR_POS)

            x = np.arange(10)
            ax.bar(x, values, color=colors, width=0.7, edgecolor='white', linewidth=0.3)
            ax.axhline(y=0, color='black', linewidth=0.8)

            ax.set_ylim(y_bot, y_top)
            ax.set_yticks(np.arange(y_bot, y_top + 1, step))
            ax.set_xticks(x)
            ax.set_xticklabels(DECILE_LABELS, fontsize=7, rotation=45)
            ax.grid(True, alpha=0.2, axis='y')

            if col_idx == 0:
                ax.set_ylabel(f'{AGE_GROUPS[age_id]["label"]}',
                              fontsize=11, fontweight='bold')

            if row_idx == 0:
                early_str = '/'.join(str(y) for y in early_yrs)
                late_str = '/'.join(str(y) for y in late_yrs)
                title = f"{COUNTRY_NAME_MAP.get(cc, cc)}\n({early_str} → {late_str})"
                weight = 'bold'
                color = FOCUS_COLOR_POS if is_focus else 'black'
                ax.set_title(title, fontsize=10, fontweight=weight, color=color)

            if row_idx == n_rows - 1:
                ax.set_xlabel('Decile', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Cluster labels and separators (cluster reports only)
    if selected_clusters:
        for cluster in selected_clusters:
            cols = [country_codes.index(cc) for cc in cluster['countries'] if cc in country_codes]
            if not cols:
                continue
            left_pos = axes[0][cols[0]].get_position()
            right_pos = axes[0][cols[-1]].get_position()
            center_x = (left_pos.x0 + right_pos.x1) / 2
            fig.text(center_x, 0.955, cluster['label'], ha='center', va='bottom',
                     fontsize=12, fontweight='bold', fontstyle='italic',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#cccccc'))

        col_offset = 0
        for i, cluster in enumerate(selected_clusters[:-1]):
            n_in = sum(1 for cc in cluster['countries'] if cc in country_codes)
            col_offset += n_in
            if col_offset < n_cols:
                right_pos = axes[0][col_offset - 1].get_position()
                left_pos = axes[0][col_offset].get_position()
                line_x = (right_pos.x1 + left_pos.x0) / 2
                fig.add_artist(plt.Line2D([line_x, line_x], [0.02, 0.94],
                                          transform=fig.transFigure, color='#888888',
                                          linewidth=1.5, linestyle='--'))

    suffix = 'pp' if value_mode == 'pp' else 'pct'
    out_png = os.path.join(out_dir, f'{cfg["prefix"]}_ownership_variation_{suffix}.png')
    _save_fig(fig, out_png)
    plt.close(fig)


# ============================================================================
# EXCEL EXPORT
# ============================================================================

def export_excel(all_diffs_pp, all_diffs_pct, country_codes, country_cache, cfg, out_dir):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    for cc in country_codes:
        if cc not in all_diffs_pp:
            continue
        name = COUNTRY_NAME_MAP.get(cc, cc)

        ws = wb.create_sheet(title=f'{cc} - {name}')
        ws.append(['Decile'] + [AGE_GROUPS[ag]['label'] for ag in sorted(AGE_GROUPS.keys())])
        for decile in range(1, 11):
            row = [f'D{decile}']
            for ag in sorted(AGE_GROUPS.keys()):
                val = all_diffs_pp[cc].get(ag, {}).get(decile)
                row.append(round(val, 2) if val is not None else None)
            ws.append(row)

        ws_pct = wb.create_sheet(title=f'{cc} pct')
        ws_pct.append(['Decile'] + [AGE_GROUPS[ag]['label'] for ag in sorted(AGE_GROUPS.keys())])
        for decile in range(1, 11):
            row = [f'D{decile}']
            for ag in sorted(AGE_GROUPS.keys()):
                val = all_diffs_pct[cc].get(ag, {}).get(decile)
                row.append(round(val, 2) if val is not None else None)
            ws_pct.append(row)

    xlsx_path = os.path.join(out_dir, f'{cfg["prefix"]}_ownership_variation_data.xlsx')
    wb.save(xlsx_path)
    print(f"    Saved: {os.path.basename(xlsx_path)}")


# ============================================================================
# SNAPSHOT HEATMAP — EU-27 cross-section (2022-2023)
# ============================================================================

# All EU-27 countries (plus CH/NO/IS if desired) ordered for the heatmap rows
EU27_SNAPSHOT_COUNTRIES = [
    'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES',
    'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT',
    'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK',
]

SNAPSHOT_YEARS = [2022, 2023]   # pooled late period for the snapshot

# Column spec:
#   overall          — weighted mean across all age groups × deciles
#   overall_d10d1    — D10 / D1 pooled across all age groups
#   ag{k}_d10d1      — D10 / D1 within age group k  (k = 1..4)
#   age_ratio        — age-group 4 (61+) / age-group 1 (18-30), pooled across deciles


def _safe_div(a, b):
    """Return a/b, or NaN if either is None/0."""
    if a is None or b is None or b == 0:
        return np.nan
    return a / b


def compute_snapshot_metrics(rates):
    """
    Given rates = {age_id: {decile: ownership_rate%}} (from ownership_pooled),
    return a dict of scalar metrics for the snapshot heatmap.
    """
    # Flatten to lists, ignoring Nones
    all_vals = [
        rates[ag][d]
        for ag in AGE_GROUPS
        for d in range(1, 11)
        if rates.get(ag, {}).get(d) is not None
    ]
    overall = float(np.mean(all_vals)) if all_vals else np.nan

    # Pooled D1 / D10 across age groups
    d1_vals = [rates.get(ag, {}).get(1) for ag in AGE_GROUPS if rates.get(ag, {}).get(1) is not None]
    d10_vals = [rates.get(ag, {}).get(10) for ag in AGE_GROUPS if rates.get(ag, {}).get(10) is not None]
    d1_mean  = float(np.mean(d1_vals))  if d1_vals  else np.nan
    d10_mean = float(np.mean(d10_vals)) if d10_vals else np.nan
    overall_d10d1 = _safe_div(d10_mean, d1_mean)

    # D10/D1 per age group
    per_ag = {}
    for ag in sorted(AGE_GROUPS.keys()):
        d1  = rates.get(ag, {}).get(1)
        d10 = rates.get(ag, {}).get(10)
        per_ag[ag] = _safe_div(d10, d1)

    # Age-group ratio: age 61+ (ag=4) / age 18-30 (ag=1), pooled across deciles
    ag4_vals = [rates.get(4, {}).get(d) for d in range(1, 11) if rates.get(4, {}).get(d) is not None]
    ag1_vals = [rates.get(1, {}).get(d) for d in range(1, 11) if rates.get(1, {}).get(d) is not None]
    ag4_mean = float(np.mean(ag4_vals)) if ag4_vals else np.nan
    ag1_mean = float(np.mean(ag1_vals)) if ag1_vals else np.nan
    age_ratio = _safe_div(ag4_mean, ag1_mean)

    return {
        'overall':       overall,
        'overall_d10d1': overall_d10d1,
        **{f'ag{ag}_d10d1': per_ag[ag] for ag in sorted(AGE_GROUPS.keys())},
        'age_ratio':     age_ratio,
    }


def plot_ownership_snapshot_heatmap(snapshot_cache):
    """
    Build and save the snapshot heatmap.

    Rows    = countries (EU-27, sorted by overall ownership rate descending)
    Columns = [Overall %, Overall D10/D1, D10/D1 18-30, D10/D1 31-45,
               D10/D1 46-60, D10/D1 61+, Age 61+/18-30]

    Each column is colour-normalised independently (min–max → white–colour).
    """
    rows = []
    for cc in EU27_SNAPSHOT_COUNTRIES:
        if snapshot_cache.get(cc) is None:
            continue
        m = snapshot_cache[cc]
        rows.append({
            'Country': cc,
            'Country_Name': COUNTRY_NAME_MAP.get(cc, cc),
            **m,
        })

    if not rows:
        print("  WARNING: no snapshot data to plot")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values('overall', ascending=False).reset_index(drop=True)

    col_keys   = ['overall', 'overall_d10d1',
                  'ag1_d10d1', 'ag2_d10d1', 'ag3_d10d1', 'ag4_d10d1',
                  'age_ratio']
    col_labels = [
        'Overall\nownership %',
        'D10/D1\n(overall)',
        'D10/D1\n18-30',
        'D10/D1\n31-45',
        'D10/D1\n46-60',
        'D10/D1\n61+',
        '61+ / 18-30\n(ownership)',
    ]
    # Colormap per column:
    #   overall % → Greens (more = darker green)
    #   D10/D1 ratios → RdYlGn_r  (higher ratio = more inequality = red)
    #   age_ratio → RdYlGn_r      (higher = older generations own much more)
    col_cmaps = ['Greens', 'RdYlGn_r', 'RdYlGn_r', 'RdYlGn_r', 'RdYlGn_r', 'RdYlGn_r', 'RdYlGn_r']

    n_rows = len(df)
    n_cols = len(col_keys)

    fig, ax = plt.subplots(figsize=(n_cols * 1.6 + 2, n_rows * 0.42 + 1.2))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.set_axis_off()

    # Pre-compute per-column min/max for normalisation
    col_min = {k: df[k].min() for k in col_keys}
    col_max = {k: df[k].max() for k in col_keys}

    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm

    cell_w = 1.0
    cell_h = 1.0

    for row_idx, row in df.iterrows():
        for col_idx, (key, cmap_name) in enumerate(zip(col_keys, col_cmaps)):
            val = row[key]
            cmap = mcm.get_cmap(cmap_name)
            vmin = col_min[key]
            vmax = col_max[key]

            if pd.isna(val) or vmax == vmin:
                face_color = '#e8e8e8'
                text_str = '—'
            else:
                norm_val = (val - vmin) / (vmax - vmin)
                face_color = mcolors.to_hex(cmap(norm_val))
                if key == 'overall':
                    text_str = f'{val:.1f}%'
                else:
                    text_str = f'{val:.2f}'

            rect = plt.Rectangle(
                (col_idx * cell_w, row_idx * cell_h - 0.5),
                cell_w, cell_h,
                facecolor=face_color, edgecolor='white', linewidth=0.8,
            )
            ax.add_patch(rect)

            # Text colour: white on dark backgrounds
            r, g, b, _ = mcolors.to_rgba(face_color)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = 'white' if luminance < 0.45 else 'black'

            ax.text(
                col_idx * cell_w + cell_w / 2,
                row_idx * cell_h,
                text_str,
                ha='center', va='center',
                fontsize=7.5, color=text_color,
            )

    # Country name labels (left)
    for row_idx, row in df.iterrows():
        ax.text(
            -0.12, row_idx * cell_h,
            row['Country_Name'],
            ha='right', va='center', fontsize=8, fontweight='bold',
        )

    # Column headers (top)
    for col_idx, label in enumerate(col_labels):
        ax.text(
            col_idx * cell_w + cell_w / 2,
            -0.65,
            label,
            ha='center', va='bottom', fontsize=8, fontweight='bold',
        )

    ax.set_xlim(-2.5, n_cols * cell_w + 0.1)
    ax.set_ylim(n_rows * cell_h - 0.5, -1.0)

    fig.suptitle(
        f'EU-27 Homeownership Snapshot — {SNAPSHOT_YEARS[0]}–{SNAPSHOT_YEARS[-1]}\n'
        'Rows sorted by overall ownership rate (descending)',
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    out_dir = os.path.join(OUTPUT_BASE, 'snapshot')
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'eu27_ownership_snapshot_heatmap.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(out_png.replace('.png', '.svg'), format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: eu27_ownership_snapshot_heatmap.png / .svg")

    # ---- Excel export ----
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Snapshot'
    header = ['Country', 'Country_Name'] + col_labels
    ws.append(header)
    for _, row in df.iterrows():
        ws.append([
            row['Country'],
            row['Country_Name'],
            *[round(row[k], 4) if not pd.isna(row[k]) else None for k in col_keys],
        ])
    xlsx_path = os.path.join(out_dir, 'eu27_ownership_snapshot_heatmap.xlsx')
    wb.save(xlsx_path)
    print(f"    Saved: eu27_ownership_snapshot_heatmap.xlsx")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(report_key, country_cache):
    cfg = REPORT_CONFIGS[report_key]
    print(f"\n{'=' * 60}")
    print(f"  Report: {cfg['prefix']} {cfg['title_suffix']}")
    print(f"{'=' * 60}")

    out_dir = os.path.join(OUTPUT_BASE, cfg['prefix'])
    os.makedirs(out_dir, exist_ok=True)

    country_codes, selected_clusters = select_countries_for_report(cfg, country_cache)

    if not country_codes:
        print("  ERROR: no countries with valid data")
        return

    print(f"\n  Countries: {', '.join(COUNTRY_NAME_MAP.get(cc, cc) for cc in country_codes)}")

    all_diffs_pp = {
        cc: country_cache[cc]['pp'] for cc in country_codes if country_cache.get(cc) is not None
    }
    all_diffs_pct = {
        cc: country_cache[cc]['pct'] for cc in country_codes if country_cache.get(cc) is not None
    }

    # Plots (pp + pct)
    create_variation_grid(all_diffs_pp, country_codes, country_cache,
                          cfg, selected_clusters, out_dir, value_mode='pp')
    create_variation_grid(all_diffs_pct, country_codes, country_cache,
                          cfg, selected_clusters, out_dir, value_mode='pct')

    # Excel
    export_excel(all_diffs_pp, all_diffs_pct, country_codes, country_cache, cfg, out_dir)

    print(f"\n  All outputs → {out_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 70)
    print('EU-SILC OWNERSHIP VARIATION — REPORT-BASED')
    print(f'Early default: {EARLY_YEARS_DEFAULT}  |  Late default: {LATE_YEARS_DEFAULT}')
    print(f'Overrides early: {EARLY_YEARS_OVERRIDE}')
    print(f'Overrides late:  {LATE_YEARS_OVERRIDE}')
    print('=' * 70)

    # Shared cache: computed once, reused across reports
    country_cache = {}

    # Only FR and CH reports for now
    for report_key in ['rep_fr', 'rep_ch']:
        generate_report(report_key, country_cache)

    # ---- EU-27 snapshot heatmap (2022-2023) ----
    print(f"\n{'=' * 60}")
    print("  EU-27 Snapshot Heatmap")
    print(f"{'=' * 60}")
    snapshot_cache = {}
    for cc in EU27_SNAPSHOT_COUNTRIES:
        if cc not in snapshot_cache:
            print(f"  Loading snapshot for {COUNTRY_NAME_MAP.get(cc, cc)} ({cc})...")
            rates = ownership_pooled(SNAPSHOT_YEARS, cc)
            snapshot_cache[cc] = compute_snapshot_metrics(rates) if rates is not None else None
    plot_ownership_snapshot_heatmap(snapshot_cache)

    print('\n' + '=' * 70)
    print('DONE')
    print('=' * 70)


if __name__ == '__main__':
    main()
