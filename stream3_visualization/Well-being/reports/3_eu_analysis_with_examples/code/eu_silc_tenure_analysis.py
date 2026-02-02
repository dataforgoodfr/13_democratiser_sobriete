"""
EU-SILC Tenure Status Analysis - Luxembourg
============================================

This script analyzes EU-SILC housing tenure data for Luxembourg.
It creates visualizations showing the percentage of owner households over time,
handling the change in HH021 coding that occurred in 2010.

Data Variables:
- HB010: Year of survey
- HH021: Tenure status (with coding change in 2010)
  Before 2010: 1=Owner, 2=Tenant at market rate, 3=Tenant at reduced rate, 4=Free accommodation
  From 2010 onwards: 1=Owner without mortgage, 2=Owner with mortgage, 3=Tenant market, 
                     4=Tenant reduced, 5=Free
- HX090: Household cross-sectional weight (for representative estimates)

Weight Usage:
The HX090 weight is used to calculate population-representative percentages.
When aggregating households, the sum of weights gives the population estimate.

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

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "graphs", "EU-SILC")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Country code
COUNTRY = "LU"
COUNTRY_NAME = "Luxembourg"

# Year range
START_YEAR = 2004
END_YEAR = 2023

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_household_data(year, country="LU"):
    """
    Load EU-SILC household data for a specific year and country.
    Handles tenure variable name change:
    - Pre-2010: Uses HH020 (tenure status)
    - 2010+: Uses HH021 (tenure status)
    
    Args:
        year (int): Year of data
        country (str): Country code (e.g., "LU" for Luxembourg)
    
    Returns:
        pd.DataFrame or None: Household data with required columns or None if file not found
    """
    file_path = os.path.join(EU_SILC_BASE_DIR, country, str(year), 
                            f"UDB_c{country}{str(year)[-2:]}H.csv")
    
    if not os.path.exists(file_path):
        print(f"  ⚠️  File not found: {file_path}")
        return None
    
    try:
        # Determine which tenure variable to use based on year
        if year < 2010:
            # Pre-2010: uses HH020 for tenure status
            tenure_col = 'HH020'
            columns_to_use = ['HB010', tenure_col, 'HX090']
        else:
            # 2010+: uses HH021 for tenure status
            tenure_col = 'HH021'
            columns_to_use = ['HB010', tenure_col, 'HX090']
        
        # Try to load with appropriate columns
        df = pd.read_csv(file_path, usecols=columns_to_use)
        
        # Standardize tenure column name for consistent processing
        if tenure_col != 'HH021':
            df['HH021'] = df[tenure_col]
            df = df.drop(columns=[tenure_col])
        
        print(f"  ✅ Loaded {len(df):,} households from {year} (tenure var: {tenure_col})")
        return df
    except Exception as e:
        print(f"  ❌ Error loading {year}: {e}")
        return None


def categorize_tenure(row):
    """
    Categorize tenure status into Owner vs Tenant.
    
    Handles different coding schemes:
    
    PRE-2010 (HH020):
    - 1 = Owner
    - 2 = Tenant or subtenant paying rent at market rate
    - 3 = Accommodation rented at reduced rate
    - 4 = Accommodation provided free
    
    FROM 2010 ONWARDS (HH021):
    - 1 = Owner without outstanding mortgage
    - 2 = Owner with outstanding mortgage
    - 3 = Tenant, rent at market price
    - 4 = Tenant, rent at reduced price
    - 5 = Tenant, rent free
    
    Owner Definition:
    - Pre-2010: Category 1 only
    - From 2010: Categories 1 + 2 (all owners, with or without mortgage)
    
    Args:
        row: DataFrame row with 'HB010' (year) and 'HH021' (tenure) columns
    
    Returns:
        str: "Owner" or "Tenant" or np.nan if missing
    """
    if pd.isna(row['HH021']):
        return np.nan
    
    year = row['HB010']
    tenure = int(row['HH021'])
    
    if year < 2010:
        # Pre-2010 coding (HH020)
        if tenure == 1:
            return "Owner"
        elif tenure in [2, 3, 4]:  # 2=market rent, 3=reduced rent, 4=free
            return "Tenant"
    else:
        # From 2010 onwards coding (HH021)
        if tenure in [1, 2]:  # 1=without mortgage, 2=with mortgage
            return "Owner"
        elif tenure in [3, 4, 5]:  # 3=market, 4=reduced, 5=free
            return "Tenant"
    
    return np.nan


def calculate_ownership_statistics(df_year):
    """
    Calculate weighted percentage of owner households.
    
    Args:
        df_year (pd.DataFrame): Household data for a single year
    
    Returns:
        dict: Statistics including percentage of owners and sample sizes
    """
    # Categorize tenure
    df_year = df_year.copy()
    df_year['tenure_category'] = df_year.apply(categorize_tenure, axis=1)
    
    # Remove rows with missing tenure data
    df_valid = df_year.dropna(subset=['tenure_category', 'HX090'])
    
    if len(df_valid) == 0:
        return None
    
    # Calculate weighted statistics
    # Group by tenure category and sum weights
    weighted_stats = df_valid.groupby('tenure_category')['HX090'].sum()
    total_weight = weighted_stats.sum()
    
    # Calculate percentages
    owner_weight = weighted_stats.get('Owner', 0)
    tenant_weight = weighted_stats.get('Tenant', 0)
    
    owner_pct = (owner_weight / total_weight * 100) if total_weight > 0 else np.nan
    
    return {
        'owner_pct': owner_pct,
        'owner_weight': owner_weight,
        'tenant_weight': tenant_weight,
        'total_weight': total_weight,
        'n_households': len(df_valid),
        'n_households_raw': len(df_year)
    }


def process_country_data(country="LU", country_name="Luxembourg", 
                        start_year=2004, end_year=2023):
    """
    Process EU-SILC data for a country and calculate ownership percentages over time.
    
    Args:
        country (str): Country code
        country_name (str): Full country name
        start_year (int): Starting year
        end_year (int): Ending year
    
    Returns:
        pd.DataFrame: Time series data with ownership statistics
    """
    print(f"\n{'='*70}")
    print(f"Processing EU-SILC Data for {country_name}")
    print(f"{'='*70}")
    
    results = []
    
    for year in range(start_year, end_year + 1):
        print(f"\nYear {year}:")
        df = load_household_data(year, country)
        
        if df is None or len(df) == 0:
            print(f"  Skipping {year} - no data available")
            continue
        
        # Calculate statistics
        stats = calculate_ownership_statistics(df)
        
        if stats is not None:
            print(f"  Owner households: {stats['owner_pct']:.2f}%")
            print(f"  Sample size: {stats['n_households']:,} households (valid)")
            
            results.append({
                'year': year,
                'owner_pct': stats['owner_pct'],
                'owner_weight': stats['owner_weight'],
                'tenant_weight': stats['tenant_weight'],
                'total_weight': stats['total_weight'],
                'n_households': stats['n_households'],
                'n_households_raw': stats['n_households_raw']
            })
        else:
            print(f"  Unable to calculate statistics for {year}")
    
    # Convert to DataFrame
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

def create_ownership_trend_graph(df_results, country_name="Luxembourg", 
                                output_path=None):
    """
    Create a line graph showing the trend in owner household percentage over time.
    
    Args:
        df_results (pd.DataFrame): Time series results
        country_name (str): Country name for title
        output_path (str): Path to save the graph
    """
    if df_results is None or len(df_results) == 0:
        print("⚠️  No data available for visualization")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort by year
    df_plot = df_results.sort_values('year')
    
    # Plot line
    ax.plot(df_plot['year'], df_plot['owner_pct'], 
            marker='o', linewidth=2.5, markersize=8, 
            color='#2E86AB', label='Owner Households')
    
    # Add a confidence band (optional - using unweighted std as proxy)
    ax.fill_between(df_plot['year'], 
                    df_plot['owner_pct'] - 2,  # Approximate margin
                    df_plot['owner_pct'] + 2, 
                    alpha=0.2, color='#2E86AB')
    
    # Customize plot
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Owner Households (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Housing Tenure Status in {country_name}\nPercentage of Owner Households (2004-2023)', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    
    # Add data labels on points
    for idx, row in df_plot.iterrows():
        ax.text(row['year'], row['owner_pct'] + 1.5, 
               f"{row['owner_pct']:.1f}%", 
               ha='center', va='bottom', fontsize=9)
    
    # Add legend and metadata
    ax.legend(fontsize=11, loc='best')
    
    # Add metadata at the bottom
    n_years = len(df_plot)
    min_val = df_plot['owner_pct'].min()
    max_val = df_plot['owner_pct'].max()
    avg_val = df_plot['owner_pct'].mean()
    
    metadata_text = (f"Data Points: {n_years} years | "
                    f"Range: {min_val:.1f}% - {max_val:.1f}% | "
                    f"Average: {avg_val:.1f}%")
    fig.text(0.5, 0.02, metadata_text, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graph saved to: {output_path}")
    
    return fig


def create_detailed_statistics_table(df_results, country_name="Luxembourg", 
                                    output_path=None):
    """
    Create a detailed statistics table and save as image and CSV.
    
    Args:
        df_results (pd.DataFrame): Time series results
        country_name (str): Country name
        output_path (str): Base path for output files
    """
    if df_results is None or len(df_results) == 0:
        return
    
    # Create summary statistics
    df_summary = df_results.copy()
    df_summary = df_summary.round(2)
    df_summary = df_summary.sort_values('year')
    
    # Save as CSV
    csv_path = output_path.replace('.png', '.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"✅ Statistics saved to: {csv_path}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"Summary Statistics for {country_name} - Housing Ownership")
    print(f"{'='*70}")
    print(df_summary.to_string(index=False))
    print(f"\nOverall Statistics:")
    print(f"  Average ownership: {df_summary['owner_pct'].mean():.2f}%")
    print(f"  Min ownership: {df_summary['owner_pct'].min():.2f}% (Year {df_summary.loc[df_summary['owner_pct'].idxmin(), 'year']:.0f})")
    print(f"  Max ownership: {df_summary['owner_pct'].max():.2f}% (Year {df_summary.loc[df_summary['owner_pct'].idxmax(), 'year']:.0f})")
    print(f"  Trend: {df_summary['owner_pct'].iloc[-1] - df_summary['owner_pct'].iloc[0]:+.2f} percentage points")
    print(f"{'='*70}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("EU-SILC TENURE STATUS ANALYSIS")
    print("="*70)
    print(f"Country: {COUNTRY_NAME} ({COUNTRY})")
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*70 + "\n")
    
    # Process data
    df_results = process_country_data(COUNTRY, COUNTRY_NAME, START_YEAR, END_YEAR)
    
    if df_results is None or len(df_results) == 0:
        print("\n❌ No data available for analysis")
        return
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Main trend graph
    graph_path = os.path.join(OUTPUT_DIR, f"{COUNTRY}_housing_ownership_trend.png")
    create_ownership_trend_graph(df_results, COUNTRY_NAME, graph_path)
    
    # Detailed statistics
    create_detailed_statistics_table(df_results, COUNTRY_NAME, graph_path)
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
