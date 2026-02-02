#!/usr/bin/env python3
"""
EU-27 Construction Migration Analysis by Regional Groups
Analyzes share of first-generation construction workers by region of origin
across all EU-27 countries.

NOTE: Individual country-of-birth data is NOT available in COUNTRYB column.
COUNTRYB contains regional aggregates only (EU27, European, Asian, African regions).
This script analyzes construction migration patterns by these regional groups.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

YEAR = 2023

# EU-27 member states
EU27_COUNTRIES = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 
                   'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 
                   'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']

CONSTRUCTION_NACE = 'F'

# Country name mappings
COUNTRY_NAME_MAP = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus',
    'CZ': 'Czechia', 'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia',
    'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland', 'FR': 'France',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta',
    'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania',
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia'
}

# Regional groups mapping (based on COUNTRYB values discovered in test)
REGIONAL_GROUPS = {
    'EU27_2020': 'EU-27 Citizens',
    'EUR_NEU27_2020_NEFTA': 'Non-EU Europe/EFTA',
    'ASI_NME': 'Asia (Near/Middle East)',
    'ASI_E': 'Asia (East)',
    'ASI_SSE': 'Asia (South/Southeast)',
    'AFR_N': 'Africa (North)',
    'AFR_OTH': 'Africa (Other)',
    'AME_N': 'Americas (North)',
    'AME_C_CRB': 'Americas (Central/Caribbean)',
    'AME_S': 'Americas (South)'
}

# ============================================================================
# SETUP
# ============================================================================

def setup_directories():
    """Set up directory paths."""
    EXTERNAL_DATA_DIR = Path(r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI")
    
    dirs = {
        'raw_data': EXTERNAL_DATA_DIR / "0_data" / "LFS" / "LFS_1983-2023_YEARLY_full_set-002" / "LFS_1983-2023_YEARLY_full_set",
        'output': Path(r"c:\Users\valentin.stuhlfauth\Desktop\Git D4G\13_democratiser_sobriete\stream3_visualization\Well-being\reports\3_eu_analysis_with_examples\outputs\EU_Maps")
    }
    
    dirs['output'].mkdir(parents=True, exist_ok=True)
    return dirs

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_lfs_data(country_code, data_dir):
    """Load LFS data for a specific country."""
    year_file = data_dir / f"{country_code}{YEAR}_y.csv"
    
    if not year_file.exists():
        print(f"  WARNING: {year_file} not found")
        return None
    
    try:
        df = pd.read_csv(year_file, low_memory=False)
        return df
    except Exception as e:
        print(f"  ERROR loading {country_code}: {e}")
        return None

def derive_migstat(df):
    """
    Derive migration status from country of birth and parent country of birth.
    
    MIGSTAT codes:
    0 = Native with both parents native
    1 = Native with one parent born abroad
    2 = Native with both parents born abroad
    3 = Foreign-born (first-generation)
    4 = Unknown
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # Default to unknown (4)
    df['MIGSTAT'] = 4
    
    # Skip if age > 74 (code 9)
    df.loc[df['AGE'] == 9, 'MIGSTAT'] = 9
    
    # Foreign-born (COUNTRYB != 'NAT')
    df.loc[df['COUNTRYB'] != 'NAT', 'MIGSTAT'] = 3
    
    # Native (COUNTRYB == 'NAT')
    mask_native = df['COUNTRYB'] == 'NAT'
    
    # Native with both parents native
    df.loc[mask_native & (df['COBFATH'] == 'NAT') & (df['COBMOTH'] == 'NAT'), 'MIGSTAT'] = 0
    
    # Native with one parent born abroad
    mask_one_parent = mask_native & ((df['COBFATH'] != 'NAT') | (df['COBMOTH'] != 'NAT')) & \
                      ~((df['COBFATH'] != 'NAT') & (df['COBMOTH'] != 'NAT'))
    df.loc[mask_one_parent, 'MIGSTAT'] = 1
    
    # Native with both parents born abroad
    df.loc[mask_native & (df['COBFATH'] != 'NAT') & (df['COBMOTH'] != 'NAT'), 'MIGSTAT'] = 2
    
    return df

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_construction_migration_by_region():
    """
    Analyze first-generation construction workers by regional origin.
    
    For each EU-27 country:
    - Count first-gen (MIGSTAT=3) construction (NACE='F') workers
    - Group by COUNTRYB regional codes
    - Show distribution of origin regions
    """
    dirs = setup_directories()
    data_dir = dirs['raw_data']
    
    print("\n" + "="*80)
    print("EU-27 CONSTRUCTION MIGRATION ANALYSIS BY REGIONAL GROUPS")
    print("="*80)
    
    results = []
    total_construction_workers = {}
    firstgen_by_region = {}
    
    # Process each EU-27 country
    for country_code in tqdm(EU27_COUNTRIES, desc="Processing countries"):
        df = load_lfs_data(country_code, data_dir)
        
        if df is None or df.empty:
            print(f"  Skipping {country_code}: no data")
            continue
        
        # Derive migration status
        df = derive_migstat(df)
        
        # Get COEFFY column (household weights) if exists
        if 'COEFFY' not in df.columns:
            df['COEFFY'] = 1.0
        
        # Count total workers in construction sector
        construction_workers = df[df['NACE'] == CONSTRUCTION_NACE]
        total_construction_workers[country_code] = construction_workers['COEFFY'].sum()
        
        # Filter for first-generation construction workers
        firstgen_construction = construction_workers[construction_workers['MIGSTAT'] == 3]
        
        if len(firstgen_construction) == 0:
            firstgen_by_region[country_code] = {}
            continue
        
        # Group by country of birth (regional codes)
        countryb_dist = firstgen_construction.groupby('COUNTRYB')['COEFFY'].sum()
        firstgen_by_region[country_code] = countryb_dist.to_dict()
        
        # Calculate percentages for this country's first-gen construction workers
        total_firstgen = countryb_dist.sum()
        
        for region_code, count in countryb_dist.items():
            region_name = REGIONAL_GROUPS.get(region_code, region_code)
            pct = (count / total_firstgen * 100) if total_firstgen > 0 else 0
            
            results.append({
                'Country': country_code,
                'Country_Name': COUNTRY_NAME_MAP.get(country_code, country_code),
                'Region_Code': region_code,
                'Region_Name': region_name,
                'First_Gen_Construction': int(count),
                'Percentage': pct,
                'Total_Construction': int(total_construction_workers[country_code])
            })
    
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("\nNo first-generation construction workers found!")
        return None
    
    print(f"\nProcessed {len(total_construction_workers)} countries")
    print(f"Total construction workers: {sum(total_construction_workers.values()):,.0f}")
    print(f"Total first-gen construction workers: {df_results['First_Gen_Construction'].sum():,.0f}")
    
    return df_results, total_construction_workers

def create_visualizations(df_results, total_workers, dirs):
    """Create visualizations of regional migration patterns."""
    
    if df_results is None or df_results.empty:
        print("No data to visualize")
        return
    
    # 1. Distribution of first-gen construction workers by region (overall)
    print("\nCreating visualizations...")
    
    plt.figure(figsize=(14, 8))
    
    # Top subplot: Overall distribution by region
    ax1 = plt.subplot(2, 1, 1)
    region_dist = df_results.groupby('Region_Name')['First_Gen_Construction'].sum().sort_values(ascending=False)
    
    colors = sns.color_palette("husl", len(region_dist))
    bars = ax1.barh(region_dist.index, region_dist.values, color=colors)
    
    # Add value labels
    for i, (region, value) in enumerate(region_dist.items()):
        ax1.text(value + 50, i, f'{int(value):,}', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Number of First-Generation Construction Workers', fontsize=11, fontweight='bold')
    ax1.set_title('EU-27: First-Generation Construction Workers by Regional Origin', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    # Bottom subplot: Number of countries with first-gen construction by region
    ax2 = plt.subplot(2, 1, 2)
    countries_by_region = df_results.groupby('Region_Name')['Country'].nunique().sort_values(ascending=False)
    
    bars = ax2.barh(countries_by_region.index, countries_by_region.values, color=colors)
    
    # Add value labels
    for i, (region, value) in enumerate(countries_by_region.items()):
        ax2.text(value + 0.2, i, f'{int(value)}', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Number of EU-27 Countries', fontsize=11, fontweight='bold')
    ax2.set_title('EU-27 Countries with First-Generation Construction Workers from Each Region', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(0, len(EU27_COUNTRIES) + 1)
    
    plt.tight_layout()
    output_file = dirs['output'] / 'EU27_construction_migration_by_region.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # 2. Heatmap: Top countries by first-gen construction from each region
    print("Creating heatmap...")
    
    # Pivot: Countries x Regions
    pivot_data = df_results.pivot_table(
        values='First_Gen_Construction',
        index='Country_Name',
        columns='Region_Name',
        aggfunc='sum',
        fill_value=0
    )
    
    # Select top regions by total count
    top_regions = df_results.groupby('Region_Name')['First_Gen_Construction'].sum().nlargest(6).index
    pivot_data = pivot_data[top_regions]
    
    # Sort countries by total
    pivot_data = pivot_data.loc[pivot_data.sum(axis=1).sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Count'},
                ax=ax, linewidths=0.5, linecolor='gray')
    
    ax.set_title('EU-27: First-Generation Construction Workers by Country and Region of Origin', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Region of Birth', fontsize=11, fontweight='bold')
    ax.set_ylabel('Country', fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_file = dirs['output'] / 'EU27_construction_migration_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # 3. Stacked bar chart: Countries with origin region breakdown
    print("Creating stacked bar chart...")
    
    # Top 12 countries by total construction migration
    top_countries = df_results.groupby('Country_Name')['First_Gen_Construction'].sum().nlargest(12).index
    df_top = df_results[df_results['Country_Name'].isin(top_countries)]
    
    pivot_stack = df_top.pivot_table(
        values='First_Gen_Construction',
        index='Country_Name',
        columns='Region_Name',
        aggfunc='sum',
        fill_value=0
    )
    
    # Reorder regions
    pivot_stack = pivot_stack[top_regions]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    pivot_stack.plot(kind='barh', stacked=True, ax=ax, 
                     colormap='tab20', edgecolor='white', linewidth=1.5)
    
    ax.set_title('Top 12 EU-27 Countries: First-Generation Construction Workers by Region of Origin',
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Number of First-Generation Construction Workers', fontsize=11, fontweight='bold')
    ax.set_ylabel('Country', fontsize=11, fontweight='bold')
    ax.legend(title='Region of Birth', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = dirs['output'] / 'EU27_construction_migration_stacked.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

def save_detailed_results(df_results, dirs):
    """Save detailed results to CSV."""
    output_file = dirs['output'] / 'EU27_construction_migration_by_region.csv'
    df_results.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    
    # Summary table
    summary = df_results.groupby('Region_Name').agg({
        'First_Gen_Construction': 'sum',
        'Country': 'nunique'
    }).rename(columns={'Country': 'Countries_with_Workers'}).sort_values('First_Gen_Construction', ascending=False)
    
    summary_file = dirs['output'] / 'EU27_construction_migration_summary.csv'
    summary.to_csv(summary_file)
    print(f"  Saved: {summary_file}")
    
    print("\nSummary by Region:")
    print(summary)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("EU-27 CONSTRUCTION SECTOR MIGRATION ANALYSIS")
    print("By Regional Groups (Individual country codes not available in COUNTRYB)")
    print("="*80)
    
    dirs = setup_directories()
    print(f"\nOutput directory: {dirs['output']}")
    
    # Analyze migration patterns
    df_results, total_workers = analyze_construction_migration_by_region()
    
    if df_results is not None:
        # Create visualizations
        create_visualizations(df_results, total_workers, dirs)
        
        # Save results
        save_detailed_results(df_results, dirs)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
    else:
        print("\nAnalysis failed - no data available")
