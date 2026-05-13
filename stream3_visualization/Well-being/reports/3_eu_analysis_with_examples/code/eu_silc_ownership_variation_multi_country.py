"""
EU-SILC Ownership Variation by Age Group and Decile — Multi-Country
=====================================================================
For each country × age group × decile, computes:
    ownership_share(2022/2023) − ownership_share(2004/2005)

Produces a grid plot:
    - Rows = age groups (youngest on top, oldest at bottom)
    - Columns = countries (FR, PL, DE, SE, ES, EL)
    - Each cell = bar chart of the percentage-point change per decile

Data Source: EU-SILC Cross-sectional 2004-2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import os
import openpyxl
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DATA_PATH = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/EU-SILC/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "graphs" / "EU-SILC" / "ownership_variation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRY_NAME_MAP = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus',
    'CZ': 'Czechia', 'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia',
    'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland', 'FR': 'France',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta',
    'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania',
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia', 'UK': 'United Kingdom'
}

# New 4-cluster setup (Method 5): select 2 countries per cluster with available data.
# Candidate order prioritizes countries already used in previous visuals where possible.
CLUSTER_CANDIDATES = [
    {
        "label": "Cluster 0 - Low performer / Low EWBI",
        "candidates": ["FR", "ES", "EL", "IT", "PT", "FI"],
    },
    {
        "label": "Cluster 1 - Low performer / High EWBI",
        "candidates": ["BE", "NL", "AT", "DK", "IE"],
    },
    {
        "label": "Cluster 2 - High performer / Low EWBI",
        "candidates": ["LT", "RO", "HU", "BG", "LV", "EE"],
    },
    {
        "label": "Cluster 3 - High performer / High EWBI",
        "candidates": ["DE", "PL", "SE", "CZ", "SI", "SK"],
    },
]

REQUIRED_COUNTRIES = {'FR', 'DE', 'PL'}

EARLY_YEARS_DEFAULT = [2004, 2005]
EARLY_YEARS_OVERRIDE = {
    'LT': [2005, 2006],
}
LATE_YEARS  = [2022, 2023]

AGE_GROUPS = {
    1: {"label": "18-30", "range": (18, 30)},
    2: {"label": "31-45", "range": (31, 45)},
    3: {"label": "46-60", "range": (46, 60)},
    4: {"label": "61+",   "range": (61, 150)},
}

DECILE_LABELS = [f'D{i}' for i in range(1, 11)]


# ============================================================================
# HELPER FUNCTIONS (reused from tenure analysis scripts)
# ============================================================================

def oecd_weight(age):
    if pd.isna(age):
        return 0.5
    try:
        age_val = int(age)
    except:
        return 0.5
    return 0.3 if age_val < 14 else 0.5


def categorize_tenure(tenure_value, year):
    if pd.isna(tenure_value):
        return None
    tenure = int(tenure_value)
    if year >= 2010:
        if tenure in [1, 2]:
            return "owner"
        elif tenure in [3, 4, 5]:
            return "renter"
    else:
        if tenure == 1:
            return "owner"
        elif tenure in [2, 3, 4]:
            return "renter"
    return None


def get_tenure_column(year):
    return "HH020" if year < 2010 else "HH021"


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


# ============================================================================
# DATA LOADING
# ============================================================================

def file_code(country, year):
    """Return the code used inside EU-SILC filenames.
    Greece folder is 'EL' but files are named 'GR' for 2004-2007."""
    if country == 'EL' and year <= 2007:
        return 'GR'
    return country


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
        df = df[df["RB220"].isna() & df["RB230"].isna()]
        if len(df) == 0:
            return None
        df = df.dropna(subset=["RB080"])
        df["RB080"] = df["RB080"].astype(int)
        if 'RB100' in df.columns and df["RB100"].notna().sum() > 0:
            interview_year = df["RB100"].astype(int)
        else:
            interview_year = year
        df["age"] = interview_year - df["RB080"]
        df = df[df["age"] >= 18]
        if len(df) == 0:
            return None
        df["age_group"] = df["age"].apply(assign_age_group)
        return df[["RB010", "RB020", "RB030", "age", "age_group"]]
    except Exception as e:
        print(f"  Error reading R-file {country}/{year}: {e}")
        return None


def load_household_with_tenure(year, country):
    fc = file_code(country, year)
    hh_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{fc}{str(year)[-2:]}H.csv"
    if not os.path.exists(hh_file_path):
        return None
    try:
        tenure_col = get_tenure_column(year)
        df = pd.read_csv(hh_file_path, usecols=["HB010", "HB020", "HB030", tenure_col], on_bad_lines='skip')
        df = df.dropna(subset=[tenure_col])
        df["tenure"] = df[tenure_col].apply(lambda x: categorize_tenure(x, year))
        df = df[df["tenure"].notna()]
        return df[["HB010", "HB020", "HB030", "tenure"]]
    except Exception as e:
        print(f"  Error reading H-file {country}/{year}: {e}")
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
            ['RB010', 'RB020', 'RB030']
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

        personal_df["RB030"] = personal_df["RB030"].fillna(0).astype(str)
        personal_df["RB040"] = personal_df["RB030"].str[:-2]
        household_df["HB030"] = household_df["HB030"].fillna(0).astype(str)

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
            left_on=["RB010", "RB020", "RB040"],
            right_on=["HB010", "HB020", "HB030"],
            how="left"
        )

        merged_df["oecd_weight"] = merged_df["age"].apply(oecd_weight)
        merged_df.sort_values(by=["HB010", "HB020", "HB030", "age"],
                              ascending=[True, True, True, False], inplace=True)
        merged_df["person_rank"] = merged_df.groupby(["HB010", "HB020", "HB030"]).cumcount()
        merged_df["oecd_weight"] = merged_df.apply(
            lambda row: 1.0 if row["person_rank"] == 0 else row["oecd_weight"], axis=1
        )

        equiv_size_df = merged_df.groupby(["HB010", "HB020", "HB030"])["oecd_weight"].sum().reset_index()
        equiv_size_df.rename(columns={"oecd_weight": "equivalent_size"}, inplace=True)

        household_df = household_df.merge(equiv_size_df, on=["HB010", "HB020", "HB030"], how="left")
        household_df["equi_disp_inc"] = household_df["HY020"] / household_df["equivalent_size"]
        household_df = household_df.dropna(subset=['HY020', 'equi_disp_inc'])
        return household_df
    except Exception as e:
        print(f"  Error computing equivalized income {country}/{year}: {e}")
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
    except:
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
    except:
        return np.nan


def load_d_file_weights(year, country):
    fc = file_code(country, year)
    d_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{fc}{str(year)[-2:]}D.csv"
    if not os.path.exists(d_file_path):
        return None
    try:
        df = pd.read_csv(d_file_path, usecols=["DB010", "DB020", "DB030", "DB090"], on_bad_lines='skip')
        df = df.dropna(subset=["DB090"])
        return df
    except Exception as e:
        print(f"  Error reading D-file {country}/{year}: {e}")
        return None


# ============================================================================
# CORE: analyse one year for one country
# ============================================================================

def analyze_year(year, country):
    """Return dict {age_group_id: {decile_int: ownership_pct}} or None."""
    personal_df = load_personal_data_with_age(year, country)
    if personal_df is None or len(personal_df) == 0:
        return None

    household_df = load_household_with_tenure(year, country)
    if household_df is None or len(household_df) == 0:
        return None

    hh_income_df = load_equivalized_income(year, country)
    if hh_income_df is None or len(hh_income_df) == 0:
        return None

    decile_thresholds = calculate_income_deciles(year, country)
    if decile_thresholds is None:
        return None

    hh_income_df['decile'] = hh_income_df.apply(
        lambda row: assign_income_decile(row, decile_thresholds), axis=1)
    hh_income_df = hh_income_df.dropna(subset=['decile'])

    # merge income‒decile onto household tenure
    household_df['HB030'] = household_df['HB030'].astype(str)
    hh_income_df['HB030'] = hh_income_df['HB030'].astype(str)
    household_df['HB020'] = household_df['HB020'].astype(str)
    hh_income_df['HB020'] = hh_income_df['HB020'].astype(str)

    household_df = household_df.merge(
        hh_income_df[['HB010', 'HB020', 'HB030', 'decile']],
        on=['HB010', 'HB020', 'HB030'], how='left')
    household_df = household_df.dropna(subset=['decile'])

    # weights
    weights_df = load_d_file_weights(year, country)
    if weights_df is None:
        return None
    weights_df['DB030'] = weights_df['DB030'].astype(str)
    weights_df['DB020'] = weights_df['DB020'].astype(str)
    household_df = household_df.merge(
        weights_df, left_on=['HB010', 'HB020', 'HB030'],
        right_on=['DB010', 'DB020', 'DB030'], how='left')
    household_df = household_df.dropna(subset=['DB090'])

    # person → household merge
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
    if len(merged) == 0:
        return None

    results = {}
    for age_id in AGE_GROUPS:
        age_data = merged[merged['age_group'] == age_id]
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


# ============================================================================
# POOL TWO YEARS
# ============================================================================

def ownership_pooled(years, country):
    """
    Combine two survey years by summing weights and computing weighted
    ownership rate for each age‑group × decile cell.
    Returns dict {age_id: {decile: ownership_%}} .
    """
    # Collect micro-data across the years
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


# ============================================================================
# VISUALISATION
# ============================================================================

def create_variation_grid(all_values, selected_clusters, country_names, value_mode='pp'):
    """
    all_values: {country_code: {age_id: {decile: value}}}
    """
    # Build ordered country list from clusters
    country_codes = []
    for cluster in selected_clusters:
        country_codes.extend(cc for cc in cluster['countries'] if cc in all_values)
    if not country_codes:
        print("No data to plot")
        return
    n_cols = len(country_codes)
    n_rows = len(AGE_GROUPS)  # 4 age groups

    # age groups ordered youngest (top) → oldest (bottom)
    age_order = sorted(AGE_GROUPS.keys())  # 1, 2, 3, 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows),
                             squeeze=False, sharey=True)
    value_title = 'percentage points' if value_mode == 'pp' else 'relative variation (%)'
    fig.suptitle('Change in Homeownership Rate by Age Group and Income Decile\n'
                 f'({value_title})',
                 fontsize=14, fontweight='bold', y=0.995)

    # find global y-range for consistent scale, rounded to multiples of 20
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
        raw_min = min(all_vals) * 1.10  # min is negative, so *1.10 makes it more negative
        step = 20 if value_mode == 'pp' else 25
        y_top = int(np.ceil(raw_max / step) * step)
        y_bot = int(np.floor(raw_min / step) * step)
        min_bound = step
        if y_top < min_bound:
            y_top = min_bound
        if y_bot > -min_bound:
            y_bot = -min_bound
    else:
        y_top = 40 if value_mode == 'pp' else 50
        y_bot = -20 if value_mode == 'pp' else -50

    for col_idx, cc in enumerate(country_codes):
        early_yrs = EARLY_YEARS_OVERRIDE.get(cc, EARLY_YEARS_DEFAULT)
        for row_idx, age_id in enumerate(age_order):
            ax = axes[row_idx][col_idx]

            values = []
            colors = []
            for decile in range(1, 11):
                v = all_values.get(cc, {}).get(age_id, {}).get(decile)
                if v is None:
                    v = 0
                values.append(v)
                colors.append('#e41a1c' if v < 0 else '#4daf4a')

            x = np.arange(10)
            ax.bar(x, values, color=colors, width=0.7, edgecolor='white', linewidth=0.3)
            ax.axhline(y=0, color='black', linewidth=0.8)

            ax.set_ylim(y_bot, y_top)
            step = 20 if value_mode == 'pp' else 25
            ax.set_yticks(np.arange(y_bot, y_top + 1, step))
            ax.set_xticks(x)
            ax.set_xticklabels(DECILE_LABELS, fontsize=7, rotation=45)
            ax.grid(True, alpha=0.2, axis='y')

            # Row label (age group) on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f'{AGE_GROUPS[age_id]["label"]}',
                              fontsize=11, fontweight='bold')

            # Column label (country + period) on top row
            if row_idx == 0:
                ax.set_title(f"{country_names.get(cc, cc)}\n({early_yrs[0]}/{early_yrs[1]} → {LATE_YEARS[0]}/{LATE_YEARS[1]})",
                             fontsize=10, fontweight='bold')

            # Only show x-label on bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel('Decile', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Cluster labels
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

    # Vertical separators between clusters
    col_offset = 0
    for i, cluster in enumerate(selected_clusters[:-1]):
        n_in_cluster = sum(1 for cc in cluster['countries'] if cc in country_codes)
        col_offset += n_in_cluster
        if col_offset < n_cols:
            right_pos = axes[0][col_offset - 1].get_position()
            left_pos = axes[0][col_offset].get_position()
            line_x = (right_pos.x1 + left_pos.x0) / 2
            fig.add_artist(plt.Line2D([line_x, line_x], [0.02, 0.94],
                                       transform=fig.transFigure, color='#888888',
                                       linewidth=1.5, linestyle='--'))

    if value_mode == 'pp':
        output_path = OUTPUT_DIR / "ownership_variation_multi_country_by_cluster.png"
    else:
        output_path = OUTPUT_DIR / "ownership_variation_multi_country_by_cluster_pct_variation.png"
    svg_path = output_path.with_suffix('.svg')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"\nOK Saved: {output_path}")
    print(f"OK Saved: {svg_path}")
    plt.close()


def compute_country_variations(country_code):
    """Compute pp change and relative % variation for one country."""
    early_years = EARLY_YEARS_OVERRIDE.get(country_code, EARLY_YEARS_DEFAULT)
    early = ownership_pooled(early_years, country_code)
    late = ownership_pooled(LATE_YEARS, country_code)

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
        'pp': diffs_pp,
        'pct': diffs_pct,
    }


def select_countries_by_cluster(country_cache):
    """Pick 2 countries with valid data per cluster using candidate priority."""
    selected_clusters = []
    for cluster in CLUSTER_CANDIDATES:
        chosen = []
        for cc in cluster['candidates']:
            if cc not in country_cache:
                country_cache[cc] = compute_country_variations(cc)
            if country_cache[cc] is not None:
                chosen.append(cc)
            if len(chosen) == 2:
                break

        if len(chosen) < 2:
            print(f"  WARNING: cluster '{cluster['label']}' has only {len(chosen)} countries with data")

        selected_clusters.append({
            'label': cluster['label'],
            'countries': chosen,
        })

    return selected_clusters


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("EU-SILC OWNERSHIP VARIATION — MULTI-COUNTRY")
    print(f"Early period: {EARLY_YEARS_DEFAULT} (default)  |  Late period: {LATE_YEARS}")
    print(f"Overrides: {EARLY_YEARS_OVERRIDE}")
    print("Selection rule: 2 countries per cluster (new 4-cluster setup)")
    print("=" * 70)

    country_cache = {}
    selected_clusters = select_countries_by_cluster(country_cache)
    selected_country_codes = [
        cc for cluster in selected_clusters for cc in cluster['countries']
    ]

    missing_required = [cc for cc in REQUIRED_COUNTRIES if cc not in selected_country_codes]
    if missing_required:
        print(f"WARNING: required countries missing due to unavailable data: {', '.join(missing_required)}")

    print("\nSelected countries:")
    for cluster in selected_clusters:
        names = [COUNTRY_NAME_MAP.get(cc, cc) for cc in cluster['countries']]
        print(f"  {cluster['label']}: {', '.join(names) if names else 'None'}")

    all_diffs_pp = {
        cc: country_cache[cc]['pp']
        for cc in selected_country_codes
        if country_cache.get(cc) is not None
    }
    all_diffs_pct = {
        cc: country_cache[cc]['pct']
        for cc in selected_country_codes
        if country_cache.get(cc) is not None
    }

    # Save CSV
    csv_rows = []
    for cc in selected_country_codes:
        if cc not in all_diffs_pp:
            continue
        for age_id in AGE_GROUPS:
            for decile in range(1, 11):
                csv_rows.append({
                    'Country': cc,
                    'Country_Name': COUNTRY_NAME_MAP.get(cc, cc),
                    'Age_Group': AGE_GROUPS[age_id]['label'],
                    'Decile': decile,
                    'Ownership_Change_PP': all_diffs_pp[cc].get(age_id, {}).get(decile),
                    'Ownership_Change_Pct': all_diffs_pct[cc].get(age_id, {}).get(decile),
                })
    df_csv = pd.DataFrame(csv_rows)
    csv_path = OUTPUT_DIR / "ownership_variation_2004_vs_2022.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"\nOK Data saved: {csv_path}")

    # Excel export: one sheet per country, rows=deciles, columns=age groups
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for cc in selected_country_codes:
        if cc not in all_diffs_pp:
            continue
        ws = wb.create_sheet(title=f"{cc} - {COUNTRY_NAME_MAP.get(cc, cc)}")
        ws.append(['Decile'] + [AGE_GROUPS[ag]['label'] for ag in sorted(AGE_GROUPS.keys())])
        for decile in range(1, 11):
            row = [f'D{decile}']
            for ag in sorted(AGE_GROUPS.keys()):
                val = all_diffs_pp[cc].get(ag, {}).get(decile)
                row.append(round(val, 2) if val is not None else None)
            ws.append(row)

        ws_pct = wb.create_sheet(title=f"{cc} pct")
        ws_pct.append(['Decile'] + [AGE_GROUPS[ag]['label'] for ag in sorted(AGE_GROUPS.keys())])
        for decile in range(1, 11):
            row = [f'D{decile}']
            for ag in sorted(AGE_GROUPS.keys()):
                val = all_diffs_pct[cc].get(ag, {}).get(decile)
                row.append(round(val, 2) if val is not None else None)
            ws_pct.append(row)
    xlsx_path = OUTPUT_DIR / "ownership_variation_data.xlsx"
    wb.save(xlsx_path)
    print(f"OK Excel saved: {xlsx_path}")

    # Create the grid plot
    if all_diffs_pp:
        create_variation_grid(
            all_diffs_pp,
            selected_clusters,
            COUNTRY_NAME_MAP,
            value_mode='pp'
        )
        create_variation_grid(
            all_diffs_pct,
            selected_clusters,
            COUNTRY_NAME_MAP,
            value_mode='pct'
        )
    else:
        print("\nERROR: No data to plot")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
