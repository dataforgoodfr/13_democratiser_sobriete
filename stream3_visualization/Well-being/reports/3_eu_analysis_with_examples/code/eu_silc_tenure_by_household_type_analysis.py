"""
EU-SILC Tenure Status Analysis by Household Type - Luxembourg
==============================================================

This script analyzes EU-SILC housing tenure data for Luxembourg by household type.
It creates 7 line graphs (one for each household type) showing the percentage of 
owner households over time.

Data Variables:
- HB010: Year of survey
- HB110: Household type
  1 = One-person household
  2 = Lone parent with at least one child aged less than 25
  3 = Lone parent with all children aged 25 or more
  4 = Couple without any child(ren)
  5 = Couple with at least one child aged less than 25
  6 = Couple with all children aged 25 or more
  7 = Other type of household
- HH020/HH021: Housing tenure status (with coding change in 2010)
  Before 2010: 1=Owner, 2=Tenant at market rate, 3=Tenant at reduced rate, 4=Free accommodation
  From 2010 onwards: 1=Owner without mortgage, 2=Owner with mortgage, 3=Tenant market, 
                     4=Tenant reduced, 5=Free
- DB090: Household weight

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
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# External data directory
EXTERNAL_DATA_DIR = r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI"
EU_SILC_BASE_DIR = os.path.join(EXTERNAL_DATA_DIR, "0_data", "EU-SILC", 
                                "_Cross_2004-2023_full_set", "_Cross_2004-2023_full_set")

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "graphs", "EU-SILC", "by_household_type")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Country code
COUNTRY = "LU"
COUNTRY_NAME = "Luxembourg"

# Year range
START_YEAR = 2004
END_YEAR = 2023

# Household type definitions
HOUSEHOLD_TYPES = {
    1: 'One-person',
    2: 'Lone parent (<25)',
    3: 'Lone parent (25+)',
    4: 'Couple no children',
    5: 'Couple with child (<25)',
    6: 'Couple with child (25+)',
    7: 'Other type'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_household_with_type(year, country="LU"):
    """
    Load household data with type and tenure status.
    
    Args:
        year (int): Year of data
        country (str): Country code
    
    Returns:
        pd.DataFrame or None: Household data with type, tenure, and weights
    """
    hh_file_path = os.path.join(EU_SILC_BASE_DIR, country, str(year), 
                                f"UDB_c{country}{str(year)[-2:]}H.csv")
    
    db_file_path = os.path.join(EU_SILC_BASE_DIR, country, str(year), 
                                f"UDB_c{country}{str(year)[-2:]}D.csv")
    
    if not os.path.exists(hh_file_path) or not os.path.exists(db_file_path):
        return None
    
    try:
        # Determine tenure variable based on year
        tenure_cols_to_try = ['HH020', 'HH021'] if year < 2010 else ['HH021', 'HH020']
        
        tenure_col_found = None
        hh_df = None
        
        for tenure_col in tenure_cols_to_try:
            try:
                hh_df = pd.read_csv(hh_file_path, usecols=['HB010', 'HB020', 'HB030', 'HB110', tenure_col], 
                                   on_bad_lines='skip')
                tenure_col_found = tenure_col
                break
            except Exception:
                continue
        
        if hh_df is None or tenure_col_found is None:
            return None
        
        # Standardize tenure column name
        hh_df.rename(columns={tenure_col_found: 'HH021'}, inplace=True)
        
        # Load household weights from D-file
        weights_df = pd.read_csv(db_file_path, usecols=['DB010', 'DB020', 'DB030', 'DB090'], 
                                on_bad_lines='skip')
        
        # Ensure consistent data types for merge
        hh_df['HB010'] = hh_df['HB010'].astype(int)
        hh_df['HB020'] = hh_df['HB020'].astype(str)
        hh_df['HB030'] = hh_df['HB030'].fillna(0).astype(str)
        
        weights_df['DB010'] = weights_df['DB010'].astype(int)
        weights_df['DB020'] = weights_df['DB020'].astype(str)
        weights_df['DB030'] = weights_df['DB030'].fillna(0).astype(str)
        
        # Merge tenure and weights
        merged = hh_df.merge(weights_df, 
                            left_on=['HB010', 'HB020', 'HB030'],
                            right_on=['DB010', 'DB020', 'DB030'],
                            how='left')
        
        # Remove rows with missing household type, tenure, or weights
        merged = merged.dropna(subset=['HB110', 'HH021', 'DB090'])
        
        if len(merged) == 0:
            return None
        
        return merged
    except Exception as e:
        return None


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


def calculate_ownership_by_household_type(df_year):
    """
    Calculate ownership percentage by household type for a given year.
    
    Args:
        df_year (pd.DataFrame): Household data for a single year
    
    Returns:
        dict: Dictionary with ownership percentages for each household type
    """
    df_year = df_year.copy()
    
    # Categorize tenure
    df_year['tenure_category'] = df_year.apply(categorize_tenure, axis=1)
    
    # Remove rows with missing data
    df_valid = df_year.dropna(subset=['HB110', 'tenure_category', 'DB090'])
    
    if len(df_valid) == 0:
        return None
    
    results = {}
    
    for hh_type in range(1, 8):
        df_type = df_valid[df_valid['HB110'] == hh_type]
        
        if len(df_type) == 0:
            results[f'hh_type_{hh_type}'] = np.nan
            continue
        
        # Calculate weighted ownership percentage
        weighted_stats = df_type.groupby('tenure_category')['DB090'].sum()
        total_weight = weighted_stats.sum()
        owner_weight = weighted_stats.get('Owner', 0)
        owner_pct = (owner_weight / total_weight * 100) if total_weight > 0 else np.nan
        
        results[f'hh_type_{hh_type}'] = owner_pct
    
    return results


def process_country_data_by_household_type(country="LU", country_name="Luxembourg", 
                                          start_year=2004, end_year=2023):
    """
    Process tenure data by household type for multiple years.
    
    Args:
        country (str): Country code
        country_name (str): Full country name
        start_year (int): Starting year
        end_year (int): Ending year
    
    Returns:
        pd.DataFrame: Time series with ownership percentages by household type
    """
    print(f"\n{'='*70}")
    print(f"Processing EU-SILC Data by Household Type for {country_name}")
    print(f"{'='*70}")
    
    results = []
    
    for year in range(start_year, end_year + 1):
        print(f"\nYear {year}:")
        
        try:
            # Load household data
            df = load_household_with_type(year, country)
            
            if df is None or len(df) == 0:
                print(f"  Skipping {year} - no data available")
                continue
            
            print(f"  Loaded {len(df):,} households")
            
            # Calculate statistics by household type
            hh_stats = calculate_ownership_by_household_type(df)
            
            if hh_stats is not None:
                row_data = {'year': year}
                row_data.update(hh_stats)
                results.append(row_data)
                
                # Print summary
                non_nan_types = [v for v in hh_stats.values() if not pd.isna(v)]
                if non_nan_types:
                    print(f"  Household type statistics computed: {len(non_nan_types)}/7 types")
                    print(f"    Range: {min(non_nan_types):.1f}% - {max(non_nan_types):.1f}%")
        
        except Exception as e:
            print(f"  Error processing {year}: {e}")
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

def create_household_type_graphs(df_results, country_name="Luxembourg", output_dir=None):
    """
    Create 7 line graphs (one per household type) showing tenure trends.
    
    Args:
        df_results (pd.DataFrame): Time series data with household type columns
        country_name (str): Country name for titles
        output_dir (str): Directory to save graphs
    """
    if df_results is None or len(df_results) == 0:
        print("  WARNING No data available for visualization")
        return
    
    # Define colors for each household type
    colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#a65628',  # Brown
        '#f781bf'   # Pink
    ]
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    df_plot = df_results.sort_values('year')
    
    for hh_type in range(1, 8):
        ax = axes[hh_type - 1]
        col_name = f'hh_type_{hh_type}'
        
        # Get data for this household type
        if col_name not in df_plot.columns:
            ax.text(0.5, 0.5, f'No data for Type {hh_type}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Type {hh_type}: {HOUSEHOLD_TYPES[hh_type]}', fontweight='bold')
            continue
        
        type_data = df_plot[['year', col_name]].copy()
        type_data.columns = ['year', 'ownership']
        type_data = type_data.dropna(subset=['ownership'])
        
        if len(type_data) == 0:
            ax.text(0.5, 0.5, f'No data for Type {hh_type}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Type {hh_type}: {HOUSEHOLD_TYPES[hh_type]}', fontweight='bold')
            continue
        
        # Plot line
        ax.plot(type_data['year'], type_data['ownership'], 
               marker='o', linewidth=2.5, markersize=6, 
               color=colors[hh_type - 1], label=f'Type {hh_type}')
        
        # Customize subplot
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Owner Households (%)', fontsize=10)
        
        ax.set_title(f'Type {hh_type}: {HOUSEHOLD_TYPES[hh_type]}', fontweight='bold', fontsize=11)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        
        # Add value labels
        for idx, row in type_data.iterrows():
            ax.text(row['year'], row['ownership'] + 1.5, 
                   f"{row['ownership']:.0f}%", 
                   ha='center', va='bottom', fontsize=8)
    
    # Hide the 8th subplot
    axes[7].axis('off')
    
    # Overall title
    fig.suptitle(f'Housing Ownership by Household Type - {country_name} (2004-2023)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        graph_path = os.path.join(output_dir, f"{country_name.lower()}_tenure_by_household_type.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        print(f"  OK Graph saved to: {graph_path}")
    
    return fig


def create_household_type_summary_table(df_results, country_name="Luxembourg", output_dir=None):
    """
    Create summary statistics table and save as CSV.
    
    Args:
        df_results (pd.DataFrame): Time series data with household type columns
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
        csv_path = os.path.join(output_dir, f"{country_name.lower()}_tenure_by_household_type.csv")
        df_summary.to_csv(csv_path, index=False)
        print(f"  OK Statistics saved to: {csv_path}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"Summary Statistics - Housing Ownership by Household Type")
    print(f"Country: {country_name}")
    print(f"{'='*70}")
    
    print(f"\nAverage Ownership Rate by Household Type:")
    for hh_type in range(1, 8):
        col = f'hh_type_{hh_type}'
        if col in df_summary.columns:
            avg = df_summary[col].mean()
            min_val = df_summary[col].min()
            max_val = df_summary[col].max()
            if not pd.isna(avg):
                print(f"  {hh_type}: {HOUSEHOLD_TYPES[hh_type]:30s}: {avg:6.1f}% (range: {min_val:5.1f}% - {max_val:5.1f}%)")
    
    print(f"\n{'='*70}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("EU-SILC TENURE STATUS BY HOUSEHOLD TYPE ANALYSIS")
    print("="*70)
    print(f"Country: {COUNTRY_NAME} ({COUNTRY})")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nHousehold Types:")
    for hh_type, name in HOUSEHOLD_TYPES.items():
        print(f"  Type {hh_type}: {name}")
    print("="*70 + "\n")
    
    # Process data
    df_results = process_country_data_by_household_type(COUNTRY, COUNTRY_NAME, START_YEAR, END_YEAR)
    
    if df_results is None or len(df_results) == 0:
        print("\n  ERROR No data available for analysis")
        return
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Create household type graphs
    create_household_type_graphs(df_results, COUNTRY_NAME, OUTPUT_DIR)
    
    # Create summary statistics
    create_household_type_summary_table(df_results, COUNTRY_NAME, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("  OK Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
