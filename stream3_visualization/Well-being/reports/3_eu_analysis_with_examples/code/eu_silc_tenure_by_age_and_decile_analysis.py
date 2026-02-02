"""
EU-SILC Tenure Analysis by Age Groups and Income Deciles
========================================================
Analyzes housing ownership by age groups AND income deciles for independent adults.

Combines:
1. Age-based filtering (RB080 for year of birth, 4 age groups)
2. Independent adult filtering (RB220=NaN AND RB230=NaN)
3. Income decile calculation (OECD equivalence scale)
4. Heatmap visualization (years x deciles for each age group)

Data Source: EU-SILC Cross-sectional 2004-2023
Country: Luxembourg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Configuration
BASE_DATA_PATH = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/EU-SILC/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "graphs" / "EU-SILC" / "by_age_and_decile"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRY = "LU"
START_YEAR = 2004
END_YEAR = 2023

# Age group definitions
AGE_GROUPS = {
    1: {"label": "18-30", "range": (18, 30)},
    2: {"label": "31-45", "range": (31, 45)},
    3: {"label": "46-60", "range": (46, 60)},
    4: {"label": "61+", "range": (61, 150)},
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def oecd_weight(age):
    """OECD modified equivalence scale weight"""
    if pd.isna(age):
        return 0.5
    try:
        age_val = int(age)
    except:
        return 0.5
    if age_val < 14:
        return 0.3
    else:
        return 0.5


def weighted_quantile(values, weights, quantiles):
    """Computes weighted quantiles"""
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.full(len(quantiles), np.nan)
    
    mask = ~(np.isnan(values) | np.isnan(weights))
    values_clean = values[mask]
    weights_clean = weights[mask]
    
    if len(values_clean) == 0:
        return np.full(len(quantiles), np.nan)
    
    sorter = np.argsort(values_clean)
    values_sorted = values_clean[sorter]  
    weights_sorted = weights_clean[sorter]

    cumsum_weights = np.cumsum(weights_sorted)
    total_weight = cumsum_weights[-1]
    normalized_weights = cumsum_weights / total_weight

    return np.interp(quantiles, normalized_weights, values_sorted)


def categorize_tenure(tenure_value, year):
    """Categorize tenure as owner or renter"""
    if pd.isna(tenure_value):
        return None
    tenure = int(tenure_value)
    
    if year >= 2010:
        # HH021: 1=owner without mortgage, 2=owner with mortgage
        if tenure in [1, 2]:
            return "owner"
        elif tenure in [3, 4, 5]:
            return "renter"
    else:
        # HH020: 1=owner
        if tenure == 1:
            return "owner"
        elif tenure in [2, 3, 4]:
            return "renter"
    
    return None


def get_tenure_column(year):
    """Determine which tenure column to use"""
    return "HH020" if year < 2010 else "HH021"


def assign_age_group(age):
    """Assign person to age group"""
    age = int(age)
    if 18 <= age <= 30:
        return 1
    elif 31 <= age <= 45:
        return 2
    elif 46 <= age <= 60:
        return 3
    elif age >= 61:
        return 4
    else:
        return None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_personal_data_with_age(year, country=COUNTRY):
    """Load R-file with independent adult filtering and age computation from RB080"""
    r_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{country}{str(year)[-2:]}R.csv"
    
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
        
        # Filter for independent adults: no father AND no mother
        df = df[df["RB220"].isna() & df["RB230"].isna()]
        
        if len(df) == 0:
            return None
        
        # Compute age from RB080
        df = df.dropna(subset=["RB080"])
        df["RB080"] = df["RB080"].astype(int)
        
        if 'RB100' in df.columns and df["RB100"].notna().sum() > 0:
            df["RB100"] = df["RB100"].astype(int)
            interview_year = df["RB100"]
        else:
            interview_year = year
        
        df["age"] = interview_year - df["RB080"]
        df = df[df["age"] >= 0]
        df = df[df["age"] >= 18]
        
        if len(df) == 0:
            return None
        
        # Assign age groups
        df["age_group"] = df["age"].apply(assign_age_group)
        
        return df[["RB010", "RB020", "RB030", "age", "age_group"]]
    
    except Exception as e:
        print(f"  Error reading R-file for {year}: {e}")
        return None


def load_household_with_tenure(year, country=COUNTRY):
    """Load H-file with tenure status"""
    hh_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{country}{str(year)[-2:]}H.csv"
    
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
        print(f"  Error reading H-file for {year}: {e}")
        return None


def load_equivalized_income(year, country=COUNTRY):
    """Load household with equivalized income for decile calculation"""
    hh_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{country}{str(year)[-2:]}H.csv"
    pr_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{country}{str(year)[-2:]}R.csv"
    
    if not os.path.exists(hh_file_path) or not os.path.exists(pr_file_path):
        return None
    
    try:
        household_df = pd.read_csv(hh_file_path, usecols=['HB010', 'HB020', 'HB030', 'HY020'], 
                                   on_bad_lines='skip')
        
        # Load personal register for household composition
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

        # Create age column
        if 'RB081' in personal_df.columns and 'RB082' in personal_df.columns:
            personal_df['age'] = personal_df['RB081'].fillna(personal_df['RB082'])
        elif 'RB081' in personal_df.columns:
            personal_df['age'] = personal_df['RB081']
        elif 'RB082' in personal_df.columns:
            personal_df['age'] = personal_df['RB082']
        else:
            personal_df['age'] = np.nan

        # Merge to get household composition
        merged_df = personal_df.merge(
            household_df,
            left_on=["RB010", "RB020", "RB040"],
            right_on=["HB010", "HB020", "HB030"],
            how="left"
        )

        # Apply OECD weights
        merged_df["oecd_weight"] = merged_df["age"].apply(oecd_weight)
        merged_df.sort_values(by=["HB010", "HB020", "HB030", "age"], 
                            ascending=[True, True, True, False], inplace=True)
        merged_df["person_rank"] = merged_df.groupby(["HB010", "HB020", "HB030"]).cumcount()
        merged_df["oecd_weight"] = merged_df.apply(
            lambda row: 1.0 if row["person_rank"] == 0 else row["oecd_weight"], axis=1
        )

        # Calculate equivalent size per household
        equiv_size_df = merged_df.groupby(["HB010", "HB020", "HB030"])["oecd_weight"].sum().reset_index()
        equiv_size_df.rename(columns={"oecd_weight": "equivalent_size"}, inplace=True)

        household_df = household_df.merge(equiv_size_df, on=["HB010", "HB020", "HB030"], how="left")

        # Compute equivalised income
        household_df["equi_disp_inc"] = household_df["HY020"] / household_df["equivalent_size"]
        household_df = household_df.dropna(subset=['HY020', 'equi_disp_inc'])

        return household_df
        
    except Exception as e:
        print(f"  Error computing equivalized income for {year}: {e}")
        return None


def calculate_income_deciles(year, country=COUNTRY):
    """Calculate income decile thresholds using weighted quantiles"""
    hh_df = load_equivalized_income(year, country)
    
    if hh_df is None or len(hh_df) == 0:
        return None
    
    # Load household weights
    db_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{country}{str(year)[-2:]}D.csv"
    
    if not os.path.exists(db_file_path):
        return None
    
    try:
        weights_df = pd.read_csv(db_file_path, usecols=['DB010', 'DB020', 'DB030', 'DB090'], 
                                on_bad_lines='skip')
    except:
        return None
    
    # Merge weights
    hh_df['HB030'] = hh_df['HB030'].astype(str)
    weights_df['DB030'] = weights_df['DB030'].astype(str)
    hh_df['HB020'] = hh_df['HB020'].astype(str)
    weights_df['DB020'] = weights_df['DB020'].astype(str)
    
    hh_df = hh_df.merge(weights_df, left_on=['HB010', 'HB020', 'HB030'],
                       right_on=['DB010', 'DB020', 'DB030'], how='left')
    
    hh_valid = hh_df.dropna(subset=['equi_disp_inc', 'DB090']).copy()
    
    if len(hh_valid) == 0:
        return None
    
    # Sort by income and calculate cumulative weights
    hh_valid = hh_valid.sort_values('equi_disp_inc').reset_index(drop=True)
    hh_valid['cumsum_weight'] = hh_valid['DB090'].cumsum()
    total_weight = hh_valid['DB090'].sum()
    hh_valid['cum_pct'] = hh_valid['cumsum_weight'] / total_weight
    
    # Find income thresholds
    decile_dict = {}
    for decile_pct in range(1, 10):
        target_pct = decile_pct / 10.0
        idx = (hh_valid['cum_pct'] - target_pct).abs().idxmin()
        decile_dict[f'decile_{decile_pct}'] = hh_valid.loc[idx, 'equi_disp_inc']
    
    return decile_dict


def assign_income_decile(row, decile_thresholds):
    """Assign household to income decile"""
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
        
        return 10  # Richest decile
    except:
        return np.nan


def load_d_file_weights(year, country=COUNTRY):
    """Load D-file with household weights"""
    d_file_path = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{country}{str(year)[-2:]}D.csv"
    
    if not os.path.exists(d_file_path):
        return None
    
    try:
        df = pd.read_csv(d_file_path, usecols=["DB010", "DB020", "DB030", "DB090"], on_bad_lines='skip')
        df = df.dropna(subset=["DB090"])
        return df
    except Exception as e:
        print(f"  Error reading D-file for {year}: {e}")
        return None


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_year(year):
    """Analyze ownership by age group and income decile for a single year"""
    print(f"\nYear {year}:")
    
    # Load personal data (independent adults only)
    personal_df = load_personal_data_with_age(year)
    if personal_df is None or len(personal_df) == 0:
        print(f"  No personal data available")
        return None
    
    # Load household tenure
    household_df = load_household_with_tenure(year)
    if household_df is None or len(household_df) == 0:
        print(f"  No household data available")
        return None
    
    # Load equivalized income and deciles
    hh_income_df = load_equivalized_income(year)
    if hh_income_df is None or len(hh_income_df) == 0:
        print(f"  No income data available")
        return None
    
    decile_thresholds = calculate_income_deciles(year)
    if decile_thresholds is None:
        print(f"  Could not calculate decile thresholds")
        return None
    
    hh_income_df['decile'] = hh_income_df.apply(lambda row: assign_income_decile(row, decile_thresholds), axis=1)
    hh_income_df = hh_income_df.dropna(subset=['decile'])
    
    # Merge income and tenure
    household_df['HB030'] = household_df['HB030'].astype(str)
    hh_income_df['HB030'] = hh_income_df['HB030'].astype(str)
    household_df['HB020'] = household_df['HB020'].astype(str)
    hh_income_df['HB020'] = hh_income_df['HB020'].astype(str)
    
    household_df = household_df.merge(hh_income_df[['HB010', 'HB020', 'HB030', 'decile']], 
                                      on=['HB010', 'HB020', 'HB030'], how='left')
    household_df = household_df.dropna(subset=['decile'])
    
    # Load weights
    weights_df = load_d_file_weights(year)
    if weights_df is None:
        print(f"  No weights available")
        return None
    
    weights_df['DB030'] = weights_df['DB030'].astype(str)
    weights_df['DB020'] = weights_df['DB020'].astype(str)
    household_df = household_df.merge(weights_df, left_on=['HB010', 'HB020', 'HB030'],
                                     right_on=['DB010', 'DB020', 'DB030'], how='left')
    household_df = household_df.dropna(subset=['DB090'])
    
    # Extract household ID from personal data
    personal_df['RB030'] = personal_df['RB030'].astype(str)
    personal_df['household_id'] = personal_df['RB030'].str[:-2]
    
    # Merge personal with household data
    household_df['HB030'] = household_df['HB030'].astype(str)
    household_df['HB020'] = household_df['HB020'].astype(str)
    household_df['HB010'] = household_df['HB010'].astype(str)
    
    personal_df['RB020'] = personal_df['RB020'].astype(str)
    personal_df['RB010'] = personal_df['RB010'].astype(str)
    
    merged = personal_df.merge(
        household_df,
        left_on=['RB010', 'RB020', 'household_id'],
        right_on=['HB010', 'HB020', 'HB030'],
        how='left'
    )
    
    merged = merged.dropna(subset=['tenure', 'decile', 'DB090'])
    
    if len(merged) == 0:
        print(f"  No data after merging")
        return None
    
    print(f"  Persons analyzed: {len(merged)}")
    
    # Calculate ownership by age group and decile
    results = {}
    
    for age_id, age_info in AGE_GROUPS.items():
        age_data = merged[merged['age_group'] == age_id]
        
        if len(age_data) == 0:
            results[age_id] = {}
            continue
        
        results[age_id] = {}
        
        for decile in range(1, 11):
            decile_data = age_data[age_data['decile'] == decile]
            
            if len(decile_data) == 0:
                results[age_id][decile] = None
                continue
            
            # Weighted ownership rate
            total_weight = decile_data['DB090'].sum()
            owner_weight = decile_data[decile_data['tenure'] == 'owner']['DB090'].sum()
            ownership_pct = (owner_weight / total_weight * 100) if total_weight > 0 else None
            
            results[age_id][decile] = ownership_pct
    
    return results


def create_heatmaps():
    """Create heatmap visualizations for each age group"""
    
    # Collect results for all years
    all_results = {}
    
    for year in range(START_YEAR, END_YEAR + 1):
        result = analyze_year(year)
        if result is not None:
            all_results[year] = result
    
    if not all_results:
        print("\nNo results to visualize")
        return
    
    # Create heatmaps for each age group
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Housing Ownership Rate Heatmap by Age Group and Income Decile\nLuxembourg, EU-SILC 2004-2023',
                fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for age_id, age_info in AGE_GROUPS.items():
        ax = axes[age_id - 1]
        
        # Build matrix: rows=deciles, columns=years
        matrix = []
        for decile in range(1, 11):
            row = []
            for year in range(START_YEAR, END_YEAR + 1):
                value = None
                if year in all_results and age_id in all_results[year]:
                    value = all_results[year][age_id].get(decile, None)
                row.append(value)
            matrix.append(row)
        
        matrix = np.array(matrix, dtype=float)
        
        # Create heatmap
        years = list(range(START_YEAR, END_YEAR + 1))
        deciles = [f'D{i}' for i in range(1, 11)]
        
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, rotation=45)
        ax.set_yticks(range(len(deciles)))
        ax.set_yticklabels(deciles)
        
        ax.set_title(f'Age Group {age_info["label"]}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Year')
        ax.set_ylabel('Income Decile')
        
        # Add text annotations
        for i in range(len(deciles)):
            for j in range(len(years)):
                value = matrix[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.0f}',
                                 ha="center", va="center", color="black", fontsize=8)
    
    # Add colorbar to the right side
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Ownership Rate (%)', fontsize=11)
    
    plt.subplots_adjust(right=0.9)
    plt.savefig(OUTPUT_DIR / "heatmap_ownership_by_age_and_decile.png", dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved: {OUTPUT_DIR / 'heatmap_ownership_by_age_and_decile.png'}")
    
    # Save raw data to CSV
    csv_data = []
    for year in range(START_YEAR, END_YEAR + 1):
        if year in all_results:
            for age_id, age_info in AGE_GROUPS.items():
                if age_id in all_results[year]:
                    for decile in range(1, 11):
                        value = all_results[year][age_id].get(decile, None)
                        csv_data.append({
                            'Year': year,
                            'Age_Group': age_info['label'],
                            'Decile': decile,
                            'Ownership_Rate': value
                        })
    
    df_csv = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / "ownership_by_age_and_decile.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"Data saved: {csv_path}")


def main():
    print("="*70)
    print("EU-SILC Tenure Analysis by Age Groups and Income Deciles")
    print("="*70)
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"Country: {COUNTRY}")
    print("Filter: Independent adults (RB220=NaN AND RB230=NaN)")
    print("="*70)
    
    create_heatmaps()
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
