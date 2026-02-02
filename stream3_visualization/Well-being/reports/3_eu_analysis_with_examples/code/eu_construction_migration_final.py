#!/usr/bin/env python3
"""
EU-27 Construction Workers Migration Analysis

Since CITIZENSHIP and COUNTRYB are AGGREGATED by region (not individual countries),
we analyze:
1. Total construction workers by migration status (NAT vs first-gen)
2. Cross-border workers (comparing residence vs workplace)
3. First-gen EU-27 workers working in construction across EU

Key finding: Individual country codes (like 'RO' for Romania) are NOT available
in CITIZENSHIP or COUNTRYB columns. These columns use regional aggregates:
- NAT = Native to residence country
- EU27_2020 = EU-27 citizen (specific country unknown)
- AFR_N, AFR_OTH = African regions
- ASI_NME, ASI_E, ASI_SSE = Asian regions
- AME_LAT, AME_N_OCE = American regions
- EUR_NEU27_2020 = Non-EU European (Norway, Iceland, etc.)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

YEAR = 2023
CONSTRUCTION_NACE = 'F'

# EU-27 member states
EU27_COUNTRIES = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 
                   'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 
                   'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']

COUNTRY_NAME_MAP = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus',
    'CZ': 'Czechia', 'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia',
    'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland', 'FR': 'France',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta',
    'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania',
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia'
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
# DATA LOADING
# ============================================================================

def get_available_countries(data_dir):
    """Get list of available countries from folder structure."""
    countries = []
    for folder in data_dir.iterdir():
        if folder.is_dir():
            match = re.match(r"([A-Z]{2})_YEAR", folder.name)
            if match:
                countries.append(match.group(1))
    return sorted(countries)

def load_lfs_data(country_code, year, data_dir):
    """Load LFS data for a specific country and year."""
    year_file = data_dir / f"{country_code}_YEAR" / f"{country_code}{year}_y.csv"
    
    if not year_file.exists():
        return None
    
    try:
        df = pd.read_csv(year_file, low_memory=False)
        return df
    except Exception as e:
        return None

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_construction_migration():
    """
    Analyze EU construction workers migration.
    
    Since individual country codes not available, we analyze:
    1. Native vs EU27 citizens vs other first-gen in construction
    2. Cross-border workers (COUNTRYW != COUNTRY)
    3. First-gen EU-27 citizens working in construction
    """
    dirs = setup_directories()
    data_dir = dirs['raw_data']
    
    print("\n" + "="*80)
    print("EU-27 CONSTRUCTION WORKERS MIGRATION ANALYSIS (2023)")
    print("="*80)
    print("\nNOTE: CITIZENSHIP and COUNTRYB contain REGIONAL AGGREGATES, not individual countries")
    print("Analysis focus: Migration by region of origin and cross-border patterns")
    print("="*80)
    
    # Get available countries
    available_countries = get_available_countries(data_dir)
    available_eu27 = [c for c in available_countries if c in EU27_COUNTRIES]
    
    print(f"\nAvailable EU-27 countries: {len(available_eu27)}/{len(EU27_COUNTRIES)}")
    print(f"Countries: {', '.join(available_eu27)}")
    
    results = {
        'by_country': [],
        'cross_border': [],
        'migration_by_origin': []
    }
    
    print(f"\n--- ANALYZING CONSTRUCTION WORKERS BY COUNTRY ---\n")
    
    for country_code in tqdm(available_eu27, desc="Processing EU-27 countries"):
        df = load_lfs_data(country_code, YEAR, data_dir)
        
        if df is None or df.empty or 'NACE2_1D' not in df.columns:
            continue
        
        if 'COEFFY' not in df.columns:
            df['COEFFY'] = 1.0
        
        # Filter construction workers
        construction = df[df['NACE2_1D'] == CONSTRUCTION_NACE].copy()
        
        if len(construction) == 0:
            continue
        
        total_construction = construction['COEFFY'].sum()
        
        # Analyze by citizenship/origin
        if 'CITIZENSHIP' in construction.columns:
            citizenship_dist = construction.groupby('CITIZENSHIP')['COEFFY'].sum()
            
            native = citizenship_dist.get('NAT', 0)
            eu27_citizen = citizenship_dist.get('EU27_2020', 0)
            other_first_gen = citizenship_dist.sum() - native - eu27_citizen
            
            results['by_country'].append({
                'Country': country_code,
                'Country_Name': COUNTRY_NAME_MAP.get(country_code, country_code),
                'Total_Construction': int(total_construction),
                'Native': int(native),
                'EU27_Citizen': int(eu27_citizen),
                'Other_First_Gen': int(other_first_gen),
                'Pct_Native': (native / total_construction * 100) if total_construction > 0 else 0,
                'Pct_EU27': (eu27_citizen / total_construction * 100) if total_construction > 0 else 0,
                'Pct_Other_Gen': (other_first_gen / total_construction * 100) if total_construction > 0 else 0
            })
            
            # Track migration by origin region
            for origin, count in citizenship_dist.items():
                if origin != 'NAT':
                    results['migration_by_origin'].append({
                        'Host_Country': country_code,
                        'Host_Country_Name': COUNTRY_NAME_MAP.get(country_code, country_code),
                        'Origin': origin,
                        'Construction_Workers': int(count)
                    })
        
        # Cross-border analysis
        if 'COUNTRYW' in construction.columns and 'COUNTRY' in construction.columns:
            cross_border = construction[
                (construction['COUNTRYW'].notna()) & 
                (construction['COUNTRYW'] != construction['COUNTRY'])
            ]
            
            if len(cross_border) > 0:
                results['cross_border'].append({
                    'Residence_Country': country_code,
                    'Residence_Country_Name': COUNTRY_NAME_MAP.get(country_code, country_code),
                    'Cross_Border_Workers': int(cross_border['COEFFY'].sum())
                })
    
    return results, dirs

def create_visualizations(results, dirs):
    """Create visualizations."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Total construction workers
    if results['by_country']:
        ax1 = fig.add_subplot(gs[0, :])
        df_const = pd.DataFrame(results['by_country']).sort_values('Total_Construction', ascending=True).tail(20)
        ax1.barh(df_const['Country_Name'], df_const['Total_Construction'], color='#3498db')
        ax1.set_xlabel('Number of Construction Workers (Weighted)', fontsize=10, fontweight='bold')
        ax1.set_title('Top 20 EU-27 Countries: Total Construction Workers (2023)', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Stacked - Native vs EU27 vs Other
    if results['by_country']:
        ax2 = fig.add_subplot(gs[1, :])
        df_const = pd.DataFrame(results['by_country']).sort_values('Total_Construction', ascending=False).head(15)
        
        x = np.arange(len(df_const))
        width = 0.6
        
        ax2.bar(x, df_const['Native'], width, label='Native', color='#2ecc71')
        ax2.bar(x, df_const['EU27_Citizen'], width, bottom=df_const['Native'], 
               label='EU27 Citizen', color='#e74c3c')
        ax2.bar(x, df_const['Other_First_Gen'], width, 
               bottom=df_const['Native'] + df_const['EU27_Citizen'],
               label='Other First-Gen', color='#f39c12')
        
        ax2.set_ylabel('Number of Workers', fontsize=10, fontweight='bold')
        ax2.set_title('Top 15 Countries: Construction Workers by Migration Status', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_const['Country'], rotation=45, ha='right')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Percentage stacked
    if results['by_country']:
        ax3 = fig.add_subplot(gs[2, 0])
        df_const = pd.DataFrame(results['by_country']).sort_values('Total_Construction', ascending=False).head(15)
        
        ax3.barh(df_const['Country_Name'], df_const['Pct_Native'], 
                label='Native', color='#2ecc71')
        ax3.barh(df_const['Country_Name'], df_const['Pct_EU27'], 
                left=df_const['Pct_Native'],
                label='EU27 Citizen', color='#e74c3c')
        ax3.barh(df_const['Country_Name'], df_const['Pct_Other_Gen'],
                left=df_const['Pct_Native'] + df_const['Pct_EU27'],
                label='Other First-Gen', color='#f39c12')
        
        ax3.set_xlabel('Percentage (%)', fontsize=10, fontweight='bold')
        ax3.set_title('Top 15: Composition by Migration Status (%)', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(axis='x', alpha=0.3)
    
    # Plot 4: Cross-border workers
    if results['cross_border']:
        ax4 = fig.add_subplot(gs[2, 1])
        df_cb = pd.DataFrame(results['cross_border']).sort_values('Cross_Border_Workers', ascending=True).tail(15)
        ax4.barh(df_cb['Residence_Country_Name'], df_cb['Cross_Border_Workers'], color='#9b59b6')
        ax4.set_xlabel('Number of Cross-Border Workers', fontsize=10, fontweight='bold')
        ax4.set_title('Top 15: Cross-Border Construction Workers\n(Residents working abroad)', fontsize=11, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
    
    plt.suptitle('EU-27 Construction Workers: Migration & Cross-Border Analysis (2023)', 
                fontsize=14, fontweight='bold', y=0.995)
    
    output_file = dirs['output'] / 'EU27_construction_migration_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")
    plt.close()

def save_results(results, dirs):
    """Save detailed results."""
    
    if results['by_country']:
        df = pd.DataFrame(results['by_country']).sort_values('Total_Construction', ascending=False)
        file_path = dirs['output'] / 'EU27_construction_by_country.csv'
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")
        print("\n" + "="*80)
        print("CONSTRUCTION WORKERS BY COUNTRY")
        print("="*80)
        print(df[['Country_Name', 'Total_Construction', 'Pct_Native', 'Pct_EU27', 'Pct_Other_Gen']].to_string(index=False))
    
    if results['migration_by_origin']:
        df = pd.DataFrame(results['migration_by_origin']).sort_values('Construction_Workers', ascending=False)
        file_path = dirs['output'] / 'EU27_construction_by_migration_origin.csv'
        df.to_csv(file_path, index=False)
        print(f"\nSaved: {file_path}")
    
    if results['cross_border']:
        df = pd.DataFrame(results['cross_border']).sort_values('Cross_Border_Workers', ascending=False)
        file_path = dirs['output'] / 'EU27_construction_cross_border.csv'
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("EU-27 CONSTRUCTION WORKERS MIGRATION ANALYSIS")
    print("="*80)
    
    results, dirs = analyze_construction_migration()
    
    if results['by_country']:
        create_visualizations(results, dirs)
        save_results(results, dirs)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
    else:
        print("\nNo data available")
