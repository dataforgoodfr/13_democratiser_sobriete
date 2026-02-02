#!/usr/bin/env python3
"""
EU-27 Construction Sector Migration Analysis
Maps the share of each country's population working in construction as first-gen immigrants in other EU countries.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

YEAR = 2023

# EU-27 member states (alphabetical by code)
EU27_COUNTRIES = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 
                   'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 
                   'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']

CONSTRUCTION_NACE = 'F'

# Country name mappings for mapping
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
        'output': Path(__file__).parent.parent / 'outputs' / 'graphs' / 'EU_Maps'
    }
    
    dirs['output'].mkdir(parents=True, exist_ok=True)
    
    return dirs


def load_lfs_data(country_code, year, data_path):
    """Load LFS data for a specific country and year."""
    country_folder = data_path / f"{country_code}_YEAR"
    year_file = country_folder / f"{country_code}{year}_y.csv"
    
    if not year_file.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(year_file)
        return df
    except Exception as e:
        print(f"ERROR loading {year_file}: {e}")
        return pd.DataFrame()


# ============================================================================
# ANALYSIS
# ============================================================================

def derive_migstat(df):
    """Derive migration status from country of birth data."""
    df = df.copy()
    
    # Convert to string for comparison
    df['COUNTRYB'] = df['COUNTRYB'].astype(str).str.strip().str.upper()
    df['COBFATH'] = df['COBFATH'].astype(str).str.strip().str.upper()
    df['COBMOTH'] = df['COBMOTH'].astype(str).str.strip().str.upper()
    
    # Initialize MIGSTAT
    df['MIGSTAT'] = np.nan
    
    # First, identify natives vs foreign-born
    native_birth = df['COUNTRYB'] == 'NAT'
    foreign_birth = ~native_birth & (df['COUNTRYB'] != 'NO ANSWER')
    unknown = df['COUNTRYB'] == 'NO ANSWER'
    
    # For foreign-born: MIGSTAT = 3 (first generation)
    df.loc[foreign_birth, 'MIGSTAT'] = 3
    df.loc[unknown, 'MIGSTAT'] = 4
    
    # For native-born, check parents
    native_both_parents = (df['COBFATH'] == 'NAT') & (df['COBMOTH'] == 'NAT')
    native_one_parent = ((df['COBFATH'] == 'NAT') & (df['COBMOTH'] != 'NAT') & (df['COBMOTH'] != 'NO ANSWER')) | \
                        ((df['COBFATH'] != 'NAT') & (df['COBFATH'] != 'NO ANSWER') & (df['COBMOTH'] == 'NAT'))
    foreign_both_parents = (df['COBFATH'] != 'NAT') & (df['COBMOTH'] != 'NAT') & \
                           (df['COBFATH'] != 'NO ANSWER') & (df['COBMOTH'] != 'NO ANSWER')
    
    df.loc[native_birth & native_both_parents, 'MIGSTAT'] = 0
    df.loc[native_birth & native_one_parent, 'MIGSTAT'] = 1
    df.loc[native_birth & foreign_both_parents, 'MIGSTAT'] = 2
    
    return df


def analyze_construction_immigration():
    """Analyze construction sector first-gen immigrants by origin country."""
    print("\n" + "="*80)
    print("EU-27 CONSTRUCTION SECTOR MIGRATION ANALYSIS")
    print("="*80)
    
    dirs = setup_directories()
    
    # Load population data (estimate from overall employment)
    population_shares = {}
    
    # Dictionary to store results: {origin_country: {destination_country: share}}
    results = {}
    
    print(f"\n--- LOADING DATA FOR {len(EU27_COUNTRIES)} COUNTRIES ---")
    
    # Load all LFS data
    all_lfs_data = {}
    for country_code in tqdm(EU27_COUNTRIES, desc="Loading LFS data"):
        df = load_lfs_data(country_code, YEAR, dirs['raw_data'])
        if not df.empty:
            # Derive MIGSTAT
            df = derive_migstat(df)
            all_lfs_data[country_code] = df
            print(f"  {country_code}: {len(df)} records loaded")
    
    print(f"\nSuccessfully loaded data for {len(all_lfs_data)} countries")
    
    # For each country (destination), identify first-gen immigrants from other EU-27
    print(f"\n--- ANALYZING CONSTRUCTION IMMIGRATION PATTERNS ---")
    
    # Store first-gen construction workers by origin country
    firstgen_construction_by_origin = {}
    
    for dest_country in tqdm(EU27_COUNTRIES, desc="Analyzing destination countries"):
        if dest_country not in all_lfs_data:
            continue
        
        df_dest = all_lfs_data[dest_country].copy()
        
        # Filter for construction workers
        df_construction = df_dest[df_dest['NACE2_1D'] == CONSTRUCTION_NACE].copy()
        
        if len(df_construction) == 0:
            continue
        
        # Filter for first-generation immigrants (MIGSTAT = 3)
        df_firstgen = df_construction[
            (pd.to_numeric(df_construction['MIGSTAT'], errors='coerce') == 3)
        ].copy()
        
        if len(df_firstgen) == 0:
            continue
        
        # Apply weights
        weights = df_firstgen['COEFFY'] if 'COEFFY' in df_firstgen.columns else np.ones(len(df_firstgen))
        
        # Get country of birth for each first-gen immigrant
        df_firstgen['COUNTRYB'] = df_firstgen['COUNTRYB'].astype(str).str.strip()
        
        # Count by origin country (only EU-27)
        for origin_country in EU27_COUNTRIES:
            if origin_country == dest_country:
                continue
            
            # Try different country code formats
            mask = (df_firstgen['COUNTRYB'] == origin_country) | \
                   (df_firstgen['COUNTRYB'].str.upper() == origin_country) | \
                   (df_firstgen['COUNTRYB'].str.startswith(origin_country[:2]))
            
            weighted_count = weights[mask].sum()
            
            if weighted_count > 0:
                if origin_country not in firstgen_construction_by_origin:
                    firstgen_construction_by_origin[origin_country] = {}
                firstgen_construction_by_origin[origin_country][dest_country] = weighted_count
    
    print(f"\nIdentified first-gen construction workers from {len(firstgen_construction_by_origin)} origin countries")
    
    # Calculate share of origin country population (estimated from total workers)
    # For each origin country: sum all first-gen construction workers in other EU countries
    # Divide by total workers in origin country (as proxy for population)
    
    final_results = {}
    
    for origin_country in tqdm(EU27_COUNTRIES, desc="Calculating shares"):
        if origin_country not in all_lfs_data:
            continue
        
        df_origin = all_lfs_data[origin_country].copy()
        
        # Estimate population from total workers
        weights_origin = df_origin['COEFFY'] if 'COEFFY' in df_origin.columns else np.ones(len(df_origin))
        total_workers = weights_origin.sum()
        
        if total_workers == 0:
            continue
        
        # Sum first-gen construction workers working in other EU countries
        firstgen_abroad = 0
        if origin_country in firstgen_construction_by_origin:
            firstgen_abroad = sum(firstgen_construction_by_origin[origin_country].values())
        
        # Calculate share (first-gen construction abroad / total workers at home)
        share = (firstgen_abroad / total_workers) * 100 if total_workers > 0 else 0
        
        final_results[origin_country] = {
            'share': share,
            'firstgen_construction_abroad': firstgen_abroad,
            'total_workers_home': total_workers
        }
        
        print(f"{origin_country}: {share:.3f}% ({int(firstgen_abroad)} firstgen construction workers in other EU)")
    
    return final_results, dirs


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_map(results, output_dir):
    """Create choropleth map of EU-27."""
    print(f"\n--- CREATING MAP ---")
    
    # Prepare data for mapping
    data_for_map = []
    for country_code, data in results.items():
        data_for_map.append({
            'country': COUNTRY_NAME_MAP.get(country_code, country_code),
            'country_code': country_code,
            'share': data['share']
        })
    
    df_map_data = pd.DataFrame(data_for_map)
    
    # Try to load world map
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        
        # Filter for EU27
        eu27_names = [COUNTRY_NAME_MAP[c] for c in EU27_COUNTRIES]
        world_eu = world[world['name'].isin(eu27_names)].copy()
        
        # Merge with our data
        world_eu = world_eu.merge(df_map_data, left_on='name', right_on='country', how='left')
        
        # Create map
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Plot background
        world.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc')
        
        # Plot EU27 with choropleth
        world_eu.plot(column='share', ax=ax, legend=True, cmap='YlOrRd',
                     edgecolor='#333333', linewidth=0.5,
                     legend_kwds={'label': 'Share of Population (%)',
                                 'orientation': 'horizontal',
                                 'shrink': 0.8})
        
        # Add country labels
        for idx, row in world_eu.iterrows():
            if pd.notna(row['share']):
                ax.text(row['geometry'].centroid.x, row['geometry'].centroid.y,
                       f"{row['share']:.2f}%", fontsize=8, ha='center', va='center',
                       fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_xlim([-10, 40])
        ax.set_ylim([35, 70])
        ax.set_title('Share of Population Working in Construction as First-Gen Immigrants in Other EU-27 Countries\nFrance 2023',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'EU27_construction_migration_map.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"OK Saved: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"ERROR creating map: {e}")
        print("Creating simple bar chart instead...")
        
        # Create bar chart as fallback
        df_sorted = df_map_data.sort_values('share', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.barh(df_sorted['country_code'], df_sorted['share'], color='#fc8d62', edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Share of Population (%)', fontsize=12, fontweight='bold')
        ax.set_title('Share of Population Working in Construction as First-Gen Immigrants in Other EU-27 Countries',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(df_sorted['share']):
            ax.text(v + 0.01, i, f'{v:.2f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'EU27_construction_migration_chart.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"OK Saved: {output_file}")
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    results, dirs = analyze_construction_immigration()
    create_map(results, dirs['output'])
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80 + "\n")
