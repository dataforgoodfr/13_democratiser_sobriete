"""
EU-SILC Tenure Status Analysis by Income Decile - Luxembourg
=============================================================

This script analyzes EU-SILC housing tenure data for Luxembourg by income decile.
It creates 10 stacked line graphs (one for each income decile) showing the 
percentage of owner households over time.

Data Variables:
- HB010: Year of survey
- HH021: Tenure status (with coding change in 2010)
  Before 2010: 1=Owner, 2=Tenant at market rate, 3=Tenant at reduced rate, 4=Free accommodation
  From 2010 onwards: 1=Owner without mortgage, 2=Owner with mortgage, 3=Tenant market, 
                     4=Tenant reduced, 5=Free
- HY020: Disposable household income
- HX090: Household cross-sectional weight (for representative estimates)

Income Decile Calculation:
- Uses OECD modified equivalence scale for household composition
- Equivalized income = Disposable income / Equivalent household size
- Weighted quantiles used to calculate income thresholds per year/country
- Assigns each household to 1-10 decile based on equivalized income

Author: Data for Good - Well-being Analysis Team
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# External data directory
EXTERNAL_DATA_DIR = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI"
EU_SILC_BASE_DIR = os.path.join(EXTERNAL_DATA_DIR, "0_data", "EU-SILC", 
                                "_Cross_2004-2023_full_set", "_Cross_2004-2023_full_set")

# EWBI data directory (for pre-computed deciles if available)
EWBI_DATA_DIR = os.path.join(EXTERNAL_DATA_DIR, "1_EWBI_data")

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "graphs", "EU-SILC", "by_decile")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Country code
COUNTRY = "LU"
COUNTRY_NAME = "Luxembourg"

# Year range
START_YEAR = 2004
END_YEAR = 2023

# ============================================================================
# HELPER FUNCTIONS - WEIGHTED QUANTILES & EQUIVALIZATION
# ============================================================================

def weighted_quantile(values, weights, quantiles):
    """
    Computes weighted quantiles. Values and weights must be 1D numpy arrays.
    
    Parameters:
        values (np.array): The data values.
        weights (np.array): The weights for each value.
        quantiles (np.array): The quantiles to compute (0 to 1).
        
    Returns:
        np.array: Weighted quantiles.
    """
    if len(values) == 0 or np.all(np.isnan(values)):
        return np.full(len(quantiles), np.nan)
    
    # Remove NaN values
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


def oecd_weight(age):
    """
    OECD modified equivalence scale weight for household composition.
    
    Args:
        age (int or float): Age of person
    
    Returns:
        float: OECD weight (1.0 for first adult, 0.5 for other adults, 0.3 for children)
    """
    if pd.isna(age):
        return 0.5  # Default weight for missing age
    age = int(age)
    if age < 14:
        return 0.3  # Child
    else:
        return 0.5  # Adult


def calculate_equivalized_income(year, country="LU"):
    """
    Load and calculate equivalized household income using OECD modified equivalence scale.
    
    OECD scale: 1.0 for first adult, 0.5 for other adults, 0.3 for children (<14)
    Matches exactly the methodology from 0_raw_indicator_EU-SILC.py
    
    Args:
        year (int): Year of data
        country (str): Country code
    
    Returns:
        pd.DataFrame: Household data with equivalized income and weights
    """
    # Load household file
    hh_file_path = os.path.join(EU_SILC_BASE_DIR, country, str(year), 
                                f"UDB_c{country}{str(year)[-2:]}H.csv")
    
    # Load personal register file
    pr_file_path = os.path.join(EU_SILC_BASE_DIR, country, str(year), 
                                f"UDB_c{country}{str(year)[-2:]}R.csv")
    
    if not os.path.exists(hh_file_path):
        return None
    
    if not os.path.exists(pr_file_path):
        return None
    
    try:
        # Load household data - try to handle both available and missing columns
        try:
            household_df = pd.read_csv(hh_file_path, usecols=['HB010', 'HB020', 'HB030', 'HY020'], 
                                      on_bad_lines='skip')
        except Exception as e:
            print(f"  WARNING {year}: Could not load HH file: {str(e)[:60]}")
            return None
        
        # Load personal register data with flexible column handling
        # Try different age column combinations for different years
        age_columns_to_try = [
            ['RB010', 'RB020', 'RB030', 'RB081', 'RB082'],
            ['RB010', 'RB020', 'RB030', 'RB082'],
            ['RB010', 'RB020', 'RB030', 'RB081'],
            ['RB010', 'RB020', 'RB030']
        ]
        
        personal_df = None
        for cols_to_load in age_columns_to_try:
            try:
                personal_df = pd.read_csv(pr_file_path, usecols=cols_to_load, on_bad_lines='skip')
                break  # Successfully loaded
            except Exception:
                continue
        
        if personal_df is None:
            print(f"  WARNING {year}: Could not load R-file with any age columns")
            return None

        # Ensure both household and personal IDs are strings
        personal_df["RB030"] = personal_df["RB030"].fillna(0).astype(str)
        personal_df["RB040"] = personal_df["RB030"].str[:-2]
        household_df["HB030"] = household_df["HB030"].fillna(0).astype(str)

        # Create age column - handle all possible column names
        if 'RB081' in personal_df.columns and 'RB082' in personal_df.columns:
            personal_df['age'] = personal_df['RB081'].fillna(personal_df['RB082'])
        elif 'RB081' in personal_df.columns:
            personal_df['age'] = personal_df['RB081']
        elif 'RB082' in personal_df.columns:
            personal_df['age'] = personal_df['RB082']
        else:
            # No age data available - use default equivalent size of 1.0 per person
            personal_df['age'] = np.nan

        # Merge personal and household data
        merged_df = personal_df.merge(
            household_df,
            left_on=["RB010", "RB020", "RB040"],
            right_on=["HB010", "HB020", "HB030"],
            how="left"
        )

        # Define the OECD modified scale weights
        def oecd_weight(age):
            if pd.isna(age):
                return 0.5  # Default weight for missing age
            try:
                age_val = int(age)
            except:
                return 0.5
            if age_val < 14:
                return 0.3
            else:
                return 0.5

        # Apply weights and sum by household
        merged_df["oecd_weight"] = merged_df["age"].apply(oecd_weight)

        # Set weight of first adult to 1.0
        merged_df.sort_values(by=["HB010", "HB020", "HB030", "age"], 
                            ascending=[True, True, True, False], inplace=True)

        # Create a flag for the first person in each household
        merged_df["person_rank"] = merged_df.groupby(["HB010", "HB020", "HB030"]).cumcount()
        merged_df["oecd_weight"] = merged_df.apply(
            lambda row: 1.0 if row["person_rank"] == 0 else row["oecd_weight"], axis=1
        )

        # Calculate equivalent size per household
        equiv_size_df = merged_df.groupby(["HB010", "HB020", "HB030"])["oecd_weight"].sum().reset_index()
        equiv_size_df.rename(columns={"oecd_weight": "equivalent_size"}, inplace=True)

        # Merge equivalent size back to household dataset
        household_df = household_df.merge(equiv_size_df, on=["HB010", "HB020", "HB030"], how="left")

        # Compute equivalised disposable income
        household_df["equi_disp_inc"] = household_df["HY020"] / household_df["equivalent_size"]

        # Remove missing income
        household_df = household_df.dropna(subset=['HY020', 'equi_disp_inc'])

        return household_df
        
    except Exception as e:
        print(f"  ERROR {year}: {str(e)[:100]}")
        return None


def calculate_income_deciles(year, country="LU"):
    """
    Calculate income decile thresholds for a given year using weighted quantiles.
    
    Creates 10 equal-population groups by:
    1. Computing equivalized income per household
    2. Sorting households by income
    3. Using cumulative weights to find 10%, 20%, ..., 90% cutpoints
    
    Args:
        year (int): Year of data
        country (str): Country code
    
    Returns:
        dict: Dictionary with decile thresholds (decile_1 to decile_9)
    """
    hh_df = calculate_equivalized_income(year, country)
    
    if hh_df is None or len(hh_df) == 0:
        return None
    
    # Load household weights from D-file
    db_file_path = os.path.join(EU_SILC_BASE_DIR, country, str(year), 
                                f"UDB_c{country}{str(year)[-2:]}D.csv")
    
    if not os.path.exists(db_file_path):
        return None
    
    try:
        weights_df = pd.read_csv(db_file_path, usecols=['DB010', 'DB020', 'DB030', 'DB090'], 
                                on_bad_lines='skip')
    except:
        return None
    
    # Ensure consistent data types for merge
    hh_df['HB030'] = hh_df['HB030'].astype(str)
    weights_df['DB030'] = weights_df['DB030'].astype(str)
    hh_df['HB020'] = hh_df['HB020'].astype(str)
    weights_df['DB020'] = weights_df['DB020'].astype(str)
    hh_df['HB010'] = hh_df['HB010'].astype(int)
    weights_df['DB010'] = weights_df['DB010'].astype(int)
    
    # Merge weights
    hh_df = hh_df.merge(weights_df, left_on=['HB010', 'HB020', 'HB030'],
                       right_on=['DB010', 'DB020', 'DB030'], how='left')
    
    # Remove rows with missing income or weights
    hh_valid = hh_df.dropna(subset=['equi_disp_inc', 'DB090']).copy()
    
    if len(hh_valid) == 0:
        return None
    
    # Sort by equivalized income (ascending - poorest to richest)
    hh_valid = hh_valid.sort_values('equi_disp_inc').reset_index(drop=True)
    
    # Calculate cumulative weights
    hh_valid['cumsum_weight'] = hh_valid['DB090'].cumsum()
    total_weight = hh_valid['DB090'].sum()
    hh_valid['cum_pct'] = hh_valid['cumsum_weight'] / total_weight
    
    # Find income thresholds at each decile boundary
    decile_dict = {}
    for decile_pct in range(1, 10):  # 10%, 20%, ..., 90%
        target_pct = decile_pct / 10.0
        
        # Find the household closest to this cumulative percentage
        idx = (hh_valid['cum_pct'] - target_pct).abs().idxmin()
        decile_dict[f'decile_{decile_pct}'] = hh_valid.loc[idx, 'equi_disp_inc']
    
    return decile_dict


def assign_income_decile(row, decile_thresholds):
    """
    Assign a household to an income decile based on equivalized income.
    
    Decile 1 = poorest (income <= threshold_1)
    Decile 2 = income <= threshold_2
    ...
    Decile 10 = richest (income > threshold_9)
    
    Args:
        row: DataFrame row with equivalized income
        decile_thresholds (dict): Dictionary with decile thresholds
    
    Returns:
        int: Decile (1-10) or NaN if missing
    """
    income = row['equi_disp_inc']
    
    if pd.isna(income) or decile_thresholds is None:
        return np.nan
    
    try:
        # Check each decile threshold in order (1-9)
        for decile_num in range(1, 10):
            threshold = decile_thresholds.get(f'decile_{decile_num}')
            if pd.isna(threshold):
                return np.nan
            if income <= threshold:
                return decile_num
        
        # If income is above all thresholds, it's in decile 10 (richest)
        return 10
    except:
        return np.nan


def load_household_with_decile(year, country="LU"):
    """
    Load household data with tenure status, income, and income decile assignment.
    
    Args:
        year (int): Year of data
        country (str): Country code
    
    Returns:
        pd.DataFrame: Household data with all variables
    """
    # Get equivalized income
    hh_df = calculate_equivalized_income(year, country)
    
    if hh_df is None or len(hh_df) == 0:
        return None
    
    # Get tenure status - try both pre-2010 and post-2010 variable names
    hh_file_path = os.path.join(EU_SILC_BASE_DIR, country, str(year), 
                                f"UDB_c{country}{str(year)[-2:]}H.csv")
    
    if year < 2010:
        tenure_cols_to_try = ['HH020', 'HH021']  # Try both
    else:
        tenure_cols_to_try = ['HH021', 'HH020']  # Try both
    
    tenure_loaded = False
    for tenure_col in tenure_cols_to_try:
        try:
            tenure_df = pd.read_csv(hh_file_path, usecols=['HB010', 'HB020', 'HB030', tenure_col], 
                                   on_bad_lines='skip')
            # Ensure consistent data types BEFORE merge
            tenure_df['HB010'] = tenure_df['HB010'].astype(int)
            tenure_df['HB020'] = tenure_df['HB020'].astype(str)
            tenure_df['HB030'] = tenure_df['HB030'].fillna(0).astype(str)
            
            hh_df['HB010'] = hh_df['HB010'].astype(int)
            hh_df['HB020'] = hh_df['HB020'].astype(str)
            hh_df['HB030'] = hh_df['HB030'].astype(str)
            
            # Merge tenure data
            hh_df = hh_df.merge(tenure_df, on=['HB010', 'HB020', 'HB030'], how='left')
            hh_df['HH021'] = hh_df[tenure_col]  # Standardize column name
            tenure_loaded = True
            break
        except Exception as e:
            continue
    
    if not tenure_loaded:
        print(f"  WARNING {year}: Could not load tenure status")
        hh_df['HH021'] = np.nan
    
    # Load weights for decile calculation
    db_file_path = os.path.join(EU_SILC_BASE_DIR, country, str(year), 
                                f"UDB_c{country}{str(year)[-2:]}D.csv")
    
    try:
        weights_df = pd.read_csv(db_file_path, usecols=['DB010', 'DB020', 'DB030', 'DB090'], 
                                on_bad_lines='skip')
        weights_df['DB010'] = weights_df['DB010'].astype(int)
        weights_df['DB020'] = weights_df['DB020'].astype(str)
        weights_df['DB030'] = weights_df['DB030'].fillna(0).astype(str)
        
        hh_df = hh_df.merge(weights_df, left_on=['HB010', 'HB020', 'HB030'],
                           right_on=['DB010', 'DB020', 'DB030'], how='left')
    except:
        pass
    
    # Calculate and assign deciles
    decile_thresholds = calculate_income_deciles(year, country)
    
    if decile_thresholds is not None:
        hh_df['income_decile'] = hh_df.apply(lambda row: assign_income_decile(row, decile_thresholds), axis=1)
    else:
        hh_df['income_decile'] = np.nan
    
    return hh_df


# ============================================================================
# TENURE ANALYSIS FUNCTIONS
# ============================================================================

def categorize_tenure(row):
    """
    Categorize tenure status into Owner vs Tenant.
    
    Args:
        row: DataFrame row with 'HB010' and 'HH021' columns
    
    Returns:
        str: "Owner" or "Tenant" or np.nan
    """
    if pd.isna(row['HH021']):
        return np.nan
    
    year = row['HB010']
    tenure = int(row['HH021'])
    
    if year < 2010:
        if tenure == 1:
            return "Owner"
        elif tenure in [2, 3, 4]:
            return "Tenant"
    else:
        if tenure in [1, 2]:
            return "Owner"
        elif tenure in [3, 4, 5]:
            return "Tenant"
    
    return np.nan


def calculate_ownership_by_decile(df_year):
    """
    Calculate ownership percentage by income decile for a given year.
    
    Args:
        df_year (pd.DataFrame): Household data for a single year
    
    Returns:
        dict: Dictionary with ownership percentages for each decile
    """
    df_year = df_year.copy()
    df_year['tenure_category'] = df_year.apply(categorize_tenure, axis=1)
    
    df_valid = df_year.dropna(subset=['tenure_category', 'income_decile', 'DB090'])
    
    if len(df_valid) == 0:
        return None
    
    results = {}
    
    for decile in range(1, 11):
        df_decile = df_valid[df_valid['income_decile'] == decile]
        
        if len(df_decile) == 0:
            results[f'decile_{decile}'] = np.nan
            continue
        
        # Calculate weighted ownership percentage
        weighted_stats = df_decile.groupby('tenure_category')['DB090'].sum()
        total_weight = weighted_stats.sum()
        owner_weight = weighted_stats.get('Owner', 0)
        owner_pct = (owner_weight / total_weight * 100) if total_weight > 0 else np.nan
        
        results[f'decile_{decile}'] = owner_pct
    
    return results


def process_country_data_by_decile(country="LU", country_name="Luxembourg", 
                                   start_year=2004, end_year=2023):
    """
    Process tenure data by income decile for multiple years.
    
    Args:
        country (str): Country code
        country_name (str): Full country name
        start_year (int): Starting year
        end_year (int): Ending year
    
    Returns:
        pd.DataFrame: Time series with ownership percentages by decile
    """
    print(f"\n{'='*70}")
    print(f"Processing EU-SILC Data by Income Decile for {country_name}")
    print(f"{'='*70}")
    
    results = []
    
    for year in range(start_year, end_year + 1):
        print(f"\nYear {year}:")
        
        try:
            df = load_household_with_decile(year, country)
            
            if df is None or len(df) == 0:
                print(f"  Skipping {year} - no data available")
                continue
            
            print(f"  OK Loaded {len(df):,} households")
            
            # Calculate statistics by decile
            decile_stats = calculate_ownership_by_decile(df)
            
            if decile_stats is not None:
                row_data = {'year': year}
                row_data.update(decile_stats)
                results.append(row_data)
                
                # Print summary
                non_nan_deciles = [v for v in decile_stats.values() if not pd.isna(v)]
                if non_nan_deciles:
                    print(f"  Decile statistics computed: {len(non_nan_deciles)}/10 deciles")
                    print(f"    Range: {min(non_nan_deciles):.1f}% - {max(non_nan_deciles):.1f}%")
        
        except Exception as e:
            print(f"  ERROR Processing {year}: {e}")
            continue
    
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n{'='*70}")
        print(f"Summary: {len(df_results)} years of data processed")
        print(f"{'='*70}")
        return df_results
    else:
        print(f"\nNo data could be processed for {country_name}")
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_decile_stacked_graphs(df_results, country_name="Luxembourg", output_dir=None):
    """
    Create 10 stacked line graphs (one per income decile) showing tenure trends.
    
    Args:
        df_results (pd.DataFrame): Time series data with decile columns
        country_name (str): Country name for titles
        output_dir (str): Directory to save graphs
    """
    if df_results is None or len(df_results) == 0:
        print("  WARNING No data available for visualization")
        return
    
    # Define colors for each decile (from poorest to richest)
    colors = [
        '#d73027',  # Red (decile 1 - poorest)
        '#fc8d59',
        '#fee090',
        '#e0f3f8',
        '#91bfdb',
        '#4575b4',  # Blue
        '#4575b4',
        '#2166ac',
        '#1a9850',
        '#006837'   # Dark green (decile 10 - richest)
    ]
    
    fig, axes = plt.subplots(5, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    df_plot = df_results.sort_values('year')
    
    for decile in range(1, 11):
        ax = axes[decile - 1]
        col_name = f'decile_{decile}'
        
        # Get data for this decile
        if col_name not in df_plot.columns:
            ax.text(0.5, 0.5, f'No data for Decile {decile}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Decile {decile}', fontweight='bold')
            continue
        
        decile_data = df_plot[['year', col_name]].copy()
        decile_data.columns = ['year', 'ownership']
        decile_data = decile_data.dropna(subset=['ownership'])
        
        if len(decile_data) == 0:
            ax.text(0.5, 0.5, f'No data for Decile {decile}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Decile {decile}', fontweight='bold')
            continue
        
        # Plot line
        ax.plot(decile_data['year'], decile_data['ownership'], 
               marker='o', linewidth=2.5, markersize=6, 
               color=colors[decile - 1], label=f'Decile {decile}')
        
        # Customize subplot
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Owner Households (%)', fontsize=10)
        
        # Decile labels
        decile_labels = {
            1: 'D1 (Poorest)', 2: 'D2', 3: 'D3', 4: 'D4', 5: 'D5 (Median)',
            6: 'D6', 7: 'D7', 8: 'D8', 9: 'D9', 10: 'D10 (Richest)'
        }
        ax.set_title(f'{decile_labels[decile]}', fontweight='bold', fontsize=11)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        
        # Add value labels
        for idx, row in decile_data.iterrows():
            ax.text(row['year'], row['ownership'] + 1.5, 
                   f"{row['ownership']:.0f}%", 
                   ha='center', va='bottom', fontsize=8)
    
    # Overall title
    fig.suptitle(f'Housing Ownership by Income Decile - {country_name} (2004-2023)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        graph_path = os.path.join(output_dir, f"{country_name.lower()}_tenure_by_decile.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        print(f"  OK Graph saved to: {graph_path}")
    
    return fig


def create_decile_summary_table(df_results, country_name="Luxembourg", output_dir=None):
    """
    Create summary statistics table and save as CSV.
    
    Args:
        df_results (pd.DataFrame): Time series data with decile columns
        country_name (str): Country name
        output_dir (str): Directory to save CSV
    """
    if df_results is None or len(df_results) == 0:
        return
    
    # Create summary
    df_summary = df_results.copy()
    df_summary = df_summary.sort_values('year')
    
    # Save as CSV
    if output_dir:
        csv_path = os.path.join(output_dir, f"{country_name.lower()}_tenure_by_decile.csv")
        df_summary.to_csv(csv_path, index=False)
        print(f"  OK Statistics saved to: {csv_path}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"Summary Statistics - Housing Ownership by Income Decile")
    print(f"Country: {country_name}")
    print(f"{'='*70}")
    
    decile_labels = {
        1: 'D1 (Poorest)', 2: 'D2', 3: 'D3', 4: 'D4', 5: 'D5 (Median)',
        6: 'D6', 7: 'D7', 8: 'D8', 9: 'D9', 10: 'D10 (Richest)'
    }
    
    print(f"\nAverage Ownership Rate by Decile:")
    for decile in range(1, 11):
        col = f'decile_{decile}'
        if col in df_summary.columns:
            avg = df_summary[col].mean()
            min_val = df_summary[col].min()
            max_val = df_summary[col].max()
            if not pd.isna(avg):
                print(f"  {decile_labels[decile]:15s}: {avg:6.1f}% (range: {min_val:5.1f}% - {max_val:5.1f}%)")
    
    print(f"\n{'='*70}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("EU-SILC TENURE STATUS BY INCOME DECILE ANALYSIS")
    print("="*70)
    print(f"Country: {COUNTRY_NAME} ({COUNTRY})")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*70 + "\n")
    
    # Process data
    df_results = process_country_data_by_decile(COUNTRY, COUNTRY_NAME, START_YEAR, END_YEAR)
    
    if df_results is None or len(df_results) == 0:
        print("\n  ERROR No data available for analysis")
        return
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Create stacked graphs
    create_decile_stacked_graphs(df_results, COUNTRY_NAME, OUTPUT_DIR)
    
    # Create summary statistics
    create_decile_summary_table(df_results, COUNTRY_NAME, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("  OK Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
