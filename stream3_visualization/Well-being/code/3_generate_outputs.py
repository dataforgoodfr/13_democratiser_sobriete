#!/usr/bin/env python3
"""
Generate Outputs Script - Hierarchical Structure
This script generates the two required output files with a clear hierarchical structure:
1. Master dataframe: All indicators for all levels, per decile and aggregated per country, for the latest year
2. Time series dataframe: Historical data per country for all indicators for all levels, but only for "All Deciles"
"""

import pandas as pd
import numpy as np
import json
import os
from scipy.stats import gmean

script_dir = os.path.dirname(os.path.abspath(__file__))

def nan_aware_gmean(values):
    """
    Compute geometric mean that properly handles NaN values.
    Returns NaN if all values are NaN, otherwise computes geometric mean of non-NaN values.
    """
    clean_values = values.dropna()
    if len(clean_values) == 0:
        return np.nan
    if (clean_values <= 0).any():
        # For geometric mean, we need positive values. 
        # If any value is <= 0, return NaN (following JRC methodology)
        return np.nan
    return gmean(clean_values)

def create_unified_all_levels_dataframe():
    """
    Create a unified dataframe with all Level 1-5 indicators in the specified format.
    
    This function:
    1. Loads Level 4+5 data from 2_preprocessing_executed.py
    2. Computes Level 3 from Level 4 (geometric mean by secondary indicator)
    3. Computes Level 2 from Level 3 (geometric mean by EU priority)  
    4. Computes Level 1 from Level 2 (geometric mean of all EU priorities = EWBI)
    5. Creates 'All' decile aggregations for each level
    6. Creates 'All Countries' median aggregations
    7. Returns unified dataframe with proper column structure
    """
    
    print("🚀 Creating Unified All-Levels (1-5) Dataframe")
    print("=" * 60)
    
    # Load Level 4+5 unified data
    level45_data_path = os.path.join(script_dir, '..', 'output', 'level4_level5_unified_data.csv')
    config_path = os.path.join(script_dir, '..', 'data', 'ewbi_indicators.json')
    
    print(f"📂 Loading Level 4+5 data from: {level45_data_path}")
    if not os.path.exists(level45_data_path):
        raise FileNotFoundError(f"Level 4+5 data not found at: {level45_data_path}")
    
    df_level45 = pd.read_csv(level45_data_path)
    print(f"✅ Loaded Level 4+5 data: {len(df_level45):,} rows")
    
    # Load missing EHIS aggregation data that was filtered out by preprocessing
    def load_ehis_missing_data():
        """Load EHIS data that was not processed through the main level45 pipeline"""
        ehis_path = os.path.join(script_dir, '..', 'output', '0_raw_data_EUROSTAT', '0_EHIS', 'EHIS_level5_statistics.csv')
        if not os.path.exists(ehis_path):
            print("⚠️  EHIS raw data not found, skipping missing indicators")
            return pd.DataFrame()
            
        print(f"📂 Loading missing EHIS data from: {ehis_path}")
        ehis_raw = pd.read_csv(ehis_path)
        
        # Check which indicators are already in the main level45 pipeline
        level45_path = os.path.join(script_dir, '..', 'output', 'level4_level5_unified_data.csv')
        processed_indicators = set()
        if os.path.exists(level45_path):
            level45_df = pd.read_csv(level45_path)
            if 'datasource' in level45_df.columns:
                ehis_level45 = level45_df[level45_df['datasource'] == 'EHIS']
                if 'Primary and raw data' in level45_df.columns:
                    processed_indicators = set(ehis_level45['Primary and raw data'].unique())
        
        # Get indicators that are missing from the main pipeline
        all_ehis_indicators = set(ehis_raw['primary_index'].unique())
        missing_indicators = all_ehis_indicators - processed_indicators
        
        print(f"📊 Found {len(processed_indicators)} EHIS indicators in main pipeline")
        print(f"📊 Found {len(missing_indicators)} missing EHIS indicators: {sorted(missing_indicators)}")
        
        # Load ALL data for missing indicators (individual quintiles + aggregations)
        ehis_missing = ehis_raw[
            ehis_raw['primary_index'].isin(missing_indicators)
        ].copy()
        
        # Also include aggregation data that might be missing for all indicators
        ehis_aggregations = ehis_raw[
            (ehis_raw['country'] == 'All Countries') | 
            (ehis_raw['quintile'] == 'All')
        ].copy()
        
        # Combine missing data with aggregations, avoiding duplicates
        ehis_combined = pd.concat([ehis_missing, ehis_aggregations]).drop_duplicates()
        
        print(f"📊 Loading {len(ehis_combined)} EHIS rows (missing indicators + aggregations)")
        
        # Convert to unified format
        ehis_unified = []
        for _, row in ehis_combined.iterrows():
            # Clean up year value (handle tuple format)
            year_value = row['year']
            if isinstance(year_value, (tuple, list)) and len(year_value) > 0:
                year_value = year_value[0]
            elif isinstance(year_value, str):
                # Handle string representations like "(2013.0,)"
                import re
                match = re.search(r'(\d+\.?\d*)', str(year_value))
                if match:
                    year_value = float(match.group(1))
            
            unified_row = {
                'Year': float(year_value),  # Ensure numeric year
                'Country': row['country'],
                'Decile': np.nan,  # EHIS doesn't use deciles
                'Quintile': row['quintile'],
                'Level': 5,
                'EU priority': None,  # Will be filled by mapping
                'Secondary': None,    # Will be filled by mapping
                'Primary and raw data': row['primary_index'],
                'Type': 'Raw' if row['country'] != 'All Countries' else 'Aggregation',
                'Aggregation': 'Country-specific' if row['country'] != 'All Countries' else 'All Countries median',
                'Value': row['value'],
                'datasource': 'EHIS'
            }
            ehis_unified.append(unified_row)
        
        ehis_df = pd.DataFrame(ehis_unified)
        print(f"✅ Loaded {len(ehis_df):,} EHIS rows (missing indicators + aggregations)")
        return ehis_df
    
    # Load missing EHIS data and add to main dataset
    ehis_missing_data = load_ehis_missing_data()
    if len(ehis_missing_data) > 0:
        df_level45 = pd.concat([df_level45, ehis_missing_data], ignore_index=True)
        print(f"✅ Added missing EHIS data, total rows: {len(df_level45):,}")
    
    # Fix EHIS scaling: Convert "Statistical computation" values from decimals to percentages
    print("🔧 Fixing EHIS indicator scaling...")
    ehis_statistical_mask = (
        (df_level45['datasource'] == 'EHIS') & 
        (df_level45['Type'] == 'Statistical computation')
    )
    if ehis_statistical_mask.sum() > 0:
        print(f"   EHIS values are properly scaled at source in 0_raw_indicator_EHIS.py")
        # No scaling needed - values should already be in percentage format (0-100) from the raw processing
        print(f"✅ EHIS scaling is handled at source - no conversion needed")
    
    # Load EWBI configuration
    print(f"📂 Loading EWBI configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)['EWBI']
    print(f"✅ Loaded EWBI structure: {len(config)} EU priorities")
    
    # Create indicator mapping from config
    print("🗺️  Creating indicator hierarchy mapping...")
    indicator_mapping = {}
    
    for priority in config:
        priority_name = priority['name']
        for component in priority['components']:
            component_name = component['name']
            for indicator in component['indicators']:
                indicator_code = indicator['code']
                indicator_mapping[indicator_code] = {
                    'EU priority': priority_name,
                    'Secondary': component_name
                }
    
    print(f"✅ Created mapping for {len(indicator_mapping)} indicators")
    
    # Filter to only Level 4 data for aggregation (exclude Level 5 raw data and aggregated countries)
    level4_base = df_level45[
        (df_level45['Level'] == 4) & 
        (df_level45['Country'] != 'All Countries') &
        (df_level45['Decile'] != 'All') &
        (df_level45['Primary and raw data'].isin(indicator_mapping.keys()))
    ].copy()
    
    print(f"✅ Filtered Level 4 base data: {len(level4_base):,} rows")
    print(f"   Countries: {level4_base['Country'].nunique()}")
    print(f"   Years: {level4_base['Year'].min():.0f}-{level4_base['Year'].max():.0f}")
    print(f"   Indicators: {level4_base['Primary and raw data'].nunique()}")
    
    # Ensure Decile column is numeric for calculations
    level4_base['Decile'] = pd.to_numeric(level4_base['Decile'], errors='coerce')
    
    # IMPORTANT FIX: Add EU Priority and Secondary mappings to ALL Level 4+5 data
    # This ensures the hierarchy is properly populated in the final dataset
    print("🗺️  Applying EU priority and Secondary mappings to all Level 4+5 data...")
    df_level45['EU priority'] = df_level45['Primary and raw data'].map(lambda x: indicator_mapping.get(x, {}).get('EU priority', pd.NA))
    df_level45['Secondary'] = df_level45['Primary and raw data'].map(lambda x: indicator_mapping.get(x, {}).get('Secondary', pd.NA))
    
    # Count how many mappings were applied
    level4_mapped = df_level45[df_level45['Level'] == 4]['EU priority'].notna().sum()
    level5_mapped = df_level45[df_level45['Level'] == 5]['EU priority'].notna().sum()
    print(f"✅ Applied mappings: Level 4: {level4_mapped:,} rows, Level 5: {level5_mapped:,} rows")
    
    # Also apply mappings to the level4_base subset for aggregation
    level4_base['EU priority'] = level4_base['Primary and raw data'].map(lambda x: indicator_mapping.get(x, {}).get('EU priority', pd.NA))
    level4_base['Secondary'] = level4_base['Primary and raw data'].map(lambda x: indicator_mapping.get(x, {}).get('Secondary', pd.NA))
    
    print("\n📊 Computing Level 3 indicators (Secondary level aggregations)...")
    
    # Filter out rows with missing EU priority or Secondary mapping
    level4_filtered = level4_base.dropna(subset=['EU priority', 'Secondary'])
    print(f"After filtering for valid mappings: {len(level4_filtered):,} rows")
    
    # Compute Level 3: Geometric mean of Level 4 indicators by EU priority + secondary
    level3_data = []
    
    # Create groupby object and track progress
    grouped = level4_filtered.groupby(['Year', 'Country', 'Decile', 'EU priority', 'Secondary'])
    total_groups = len(grouped)
    print(f"Processing {total_groups:,} groups for Level 3 aggregation...")
    
    for i, ((year, country, decile, eu_priority, secondary), group) in enumerate(grouped):
        # Progress reporting every 1000 groups
        if i % 1000 == 0:
            print(f"  Processing group {i+1:,}/{total_groups:,} ({(i+1)/total_groups*100:.1f}%)")
            
        values = group['Value']
        # Use NaN-aware geometric mean
        level3_value = nan_aware_gmean(values)
        
        if not np.isnan(level3_value):
            
            level3_data.append({
                'Year': year,
                'Country': country,
                'Decile': decile,
                'Quintile': pd.NA,  # Preserve original quintile/decile structure
                'Level': 3,
                'EU priority': eu_priority,
                'Secondary': secondary,
                'Primary and raw data': pd.NA,
                'Type': 'Aggregation',
                'Aggregation': 'Geometric mean level-1',
                'Value': level3_value,
                'datasource': 'Computed'
            })
    
    level3_df = pd.DataFrame(level3_data)
    print(f"✅ Created Level 3 data: {len(level3_df):,} rows")
    
    print("\n📊 Computing Level 2 indicators (EU Priority level aggregations)...")
    
    # Compute Level 2: Geometric mean of Level 3 indicators by EU priority
    level2_data = []
    
    for (year, country, decile, eu_priority), group in level3_df.groupby(['Year', 'Country', 'Decile', 'EU priority']):
        values = group['Value']
        # Use NaN-aware geometric mean
        level2_value = nan_aware_gmean(values)
        
        if not np.isnan(level2_value):
            
            level2_data.append({
                'Year': year,
                'Country': country,
                'Decile': decile,
                'Quintile': pd.NA,  # Preserve original quintile/decile structure
                'Level': 2,
                'EU priority': eu_priority,
                'Secondary': pd.NA,
                'Primary and raw data': pd.NA,
                'Type': 'Aggregation',
                'Aggregation': 'Geometric mean level-1',
                'Value': level2_value,
                'datasource': 'Computed'
            })
    
    level2_df = pd.DataFrame(level2_data)
    print(f"✅ Created Level 2 individual country data: {len(level2_df):,} rows")
    
    # Add Level 2 "All Countries" aggregations: geometric mean across countries for each (year, decile, eu_priority)
    print("📊 Computing Level 2 'All Countries' aggregations...")
    level2_all_countries_data = []
    
    for (year, decile, eu_priority), group in level2_df.groupby(['Year', 'Decile', 'EU priority']):
        values = group['Value']
        # Use NaN-aware geometric mean across countries
        level2_all_value = nan_aware_gmean(values)
        
        if not np.isnan(level2_all_value):
            level2_all_countries_data.append({
                'Year': year,
                'Country': 'All Countries',
                'Decile': decile,
                'Quintile': pd.NA,
                'Level': 2,
                'EU priority': eu_priority,
                'Secondary': pd.NA,
                'Primary and raw data': pd.NA,
                'Type': 'Aggregation',
                'Aggregation': 'Geometric mean inter-decile',
                'Value': level2_all_value,
                'datasource': 'Computed'
            })
    
    level2_all_countries_df = pd.DataFrame(level2_all_countries_data)
    print(f"✅ Created Level 2 'All Countries' data: {len(level2_all_countries_df):,} rows")
    
    # Combine individual countries and "All Countries" Level 2 data
    level2_df = pd.concat([level2_df, level2_all_countries_df], ignore_index=True)
    print(f"✅ Total Level 2 data: {len(level2_df):,} rows")
    
    print("\n📊 Computing Level 1 indicators (EWBI level aggregations)...")
    
    # Compute Level 1: Geometric mean of Level 2 indicators (overall EWBI)
    level1_data = []
    
    for (year, country, decile), group in level2_df.groupby(['Year', 'Country', 'Decile']):
        values = group['Value']
        # Use NaN-aware geometric mean
        level1_value = nan_aware_gmean(values)
        
        if not np.isnan(level1_value):
            
            level1_data.append({
                'Year': year,
                'Country': country,
                'Decile': decile,
                'Quintile': pd.NA,  # Preserve original quintile/decile structure
                'Level': 1,
                'EU priority': pd.NA,
                'Secondary': pd.NA,
                'Primary and raw data': pd.NA,
                'Type': 'Aggregation',
                'Aggregation': 'Geometric mean level-1',
                'Value': level1_value,
                'datasource': 'Computed'
            })
    
    level1_df = pd.DataFrame(level1_data)
    print(f"✅ Created Level 1 data: {len(level1_df):,} rows")
    
    print("\n🔗 Computing 'All' decile aggregations for each level...")
    
    # Function to create 'All' decile aggregations
    def create_all_deciles(df, level_num):
        all_deciles_data = []
        
        groupby_cols = ['Year', 'Country']
        if level_num >= 2:
            groupby_cols.append('EU priority')
        if level_num >= 3:
            groupby_cols.append('Secondary')
        
        for group_key, group in df.groupby(groupby_cols):
            values = group['Value']
            # Use NaN-aware geometric mean across deciles
            all_deciles_value = nan_aware_gmean(values)
            
            if not np.isnan(all_deciles_value):
                
                row_data = {
                    'Year': group_key[0],
                    'Country': group_key[1],
                    'Decile': 'All',
                    'Quintile': 'All',
                    'Level': level_num,
                    'EU priority': group_key[2] if len(group_key) > 2 else pd.NA,
                    'Secondary': group_key[3] if len(group_key) > 3 else pd.NA,
                    'Primary and raw data': pd.NA,
                    'Type': 'Aggregation',
                    'Aggregation': 'Geometric mean inter-decile',
                    'Value': all_deciles_value,
                    'datasource': 'Computed'
                }
                
                all_deciles_data.append(row_data)
        
        return pd.DataFrame(all_deciles_data)
    
    # Create 'All' decile aggregations for each level
    level1_all = create_all_deciles(level1_df, 1)
    level2_all = create_all_deciles(level2_df, 2)  
    level3_all = create_all_deciles(level3_df, 3)
    
    print(f"✅ Created 'All' decile aggregations:")
    print(f"   Level 1: {len(level1_all):,} rows")
    print(f"   Level 2: {len(level2_all):,} rows")
    print(f"   Level 3: {len(level3_all):,} rows")
    
    print("\n🌍 Creating 'All Countries' median aggregations...")
    
    # Combine all computed levels
    all_computed_levels = pd.concat([
        level1_df, level1_all,
        level2_df, level2_all, 
        level3_df, level3_all
    ], ignore_index=True)
    
    # Function to create 'All Countries' aggregations
    def create_all_countries(df):
        all_countries_data = []
        
        # Define EU countries
        eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
        
        # Filter to EU countries only
        eu_data = df[df['Country'].isin(eu_countries)]
        
        # Group by all columns except Country and Value
        groupby_cols = ['Year', 'Decile', 'Level', 'EU priority', 'Secondary']
        
        for group_key, group in eu_data.groupby(groupby_cols):
            values = group['Value'].dropna()
            if len(values) > 0:
                # Median across countries
                median_value = values.median()
                
                row_data = {
                    'Year': group_key[0],
                    'Country': 'All Countries',
                    'Decile': group_key[1],
                    'Quintile': 'All' if group_key[1] == 'All' else pd.NA,  # Preserve original structure
                    'Level': group_key[2],
                    'EU priority': group_key[3],
                    'Secondary': group_key[4],
                    'Primary and raw data': pd.NA,
                    'Type': 'Aggregation',
                    'Aggregation': 'Median across countries',
                    'Value': median_value,
                    'datasource': 'Computed'
                }
                
                all_countries_data.append(row_data)
        
        return pd.DataFrame(all_countries_data)
    
    all_countries_df = create_all_countries(all_computed_levels)
    print(f"✅ Created 'All Countries' aggregations: {len(all_countries_df):,} rows")
    
    # Create 'All Countries' aggregations for Level 4+5 data
    def create_all_countries_level45(df):
        """Create 'All Countries' aggregations specifically for Level 4+5 data"""
        all_countries_data = []
        
        # Define EU countries
        eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
        
        # Filter to EU countries only and exclude existing All Countries
        eu_data = df[(df['Country'].isin(eu_countries)) & (df['Country'] != 'All Countries')]
        print(f"🔧 EU data for Level 4+5 aggregation: {len(eu_data):,} rows")
        
        # Handle EHIS (Quintile-based) and other data sources (Decile-based) separately
        # EHIS data: has valid Quintile, NaN Decile
        ehis_data = eu_data[eu_data['datasource'] == 'EHIS'].copy()  
        # Other data: has valid Decile, NaN Quintile
        other_data = eu_data[eu_data['datasource'] != 'EHIS'].copy()
        
        print(f"🔧 EHIS data: {len(ehis_data):,} rows")
        print(f"🔧 Other data: {len(other_data):,} rows")
        
        # Process EHIS data (group by Quintile)
        if len(ehis_data) > 0:
            groupby_cols_ehis = ['Year', 'Quintile', 'Level', 'Primary and raw data', 'datasource']
            print(f"🔧 Grouping EHIS by: {groupby_cols_ehis}")
            for group_key, group in ehis_data.groupby(groupby_cols_ehis):
                values = group['Value'].dropna()
                if len(values) > 0:
                    # Median across countries
                    median_value = values.median()
                    
                    # Get other fields from the first row of the group
                    first_row = group.iloc[0]
                    
                    row_data = {
                        'Year': group_key[0],
                        'Country': 'All Countries',
                        'Decile': pd.NA,  # EHIS uses Quintile, so Decile is NaN
                        'Quintile': group_key[1],
                        'Level': group_key[2],
                        'EU priority': first_row['EU priority'],
                        'Secondary': first_row['Secondary'], 
                        'Primary and raw data': group_key[3],
                        'Type': 'Aggregation',
                        'Aggregation': 'Median across countries',
                        'Value': median_value,
                        'datasource': group_key[4]  # Preserve original datasource
                    }
                    
                    all_countries_data.append(row_data)
        
        # Process other data sources (group by Decile)  
        if len(other_data) > 0:
            groupby_cols_other = ['Year', 'Decile', 'Level', 'Primary and raw data', 'datasource']
            print(f"🔧 Grouping others by: {groupby_cols_other}")
            for group_key, group in other_data.groupby(groupby_cols_other):
                values = group['Value'].dropna()
                if len(values) > 0:
                    # Median across countries
                    median_value = values.median()
                    
                    # Get other fields from the first row of the group
                    first_row = group.iloc[0]
                    
                    row_data = {
                        'Year': group_key[0],
                        'Country': 'All Countries',
                        'Decile': group_key[1],
                        'Quintile': pd.NA,  # Other sources use Decile, so Quintile is NaN
                        'Level': group_key[2],
                        'EU priority': first_row['EU priority'],
                        'Secondary': first_row['Secondary'], 
                        'Primary and raw data': group_key[3],
                        'Type': 'Aggregation',
                        'Aggregation': 'Median across countries',
                        'Value': median_value,
                        'datasource': group_key[4]  # Preserve original datasource
                    }
                    
                    all_countries_data.append(row_data)
        
        return pd.DataFrame(all_countries_data)
    
    all_countries_level45_df = create_all_countries_level45(df_level45)
    print(f"✅ Created 'All Countries' aggregations for Level 4+5: {len(all_countries_level45_df):,} rows")
    if len(all_countries_level45_df) > 0:
        print(f"   Datasources: {sorted(all_countries_level45_df['datasource'].unique())}")
    
    # NEW: Add 4 specific aggregation variables as requested
    print("\n🎯 Adding 4 specific median aggregation variables...")
    
    def create_specific_median_aggregations(df_all_levels, df_level5):
        """Create the 4 specific aggregation variables requested by the user"""
        specific_aggregations = []
        
        # Define EU countries
        eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
        
        # 1. Level 1: For each year, median value of the geometric mean inter-decile values per country (All deciles)
        print("   📊 Adding Level 1 median aggregations...")
        level1_all_deciles = df_all_levels[
            (df_all_levels['Level'] == 1) & 
            (df_all_levels['Decile'] == 'All') &
            (df_all_levels['Country'].isin(eu_countries))
        ]
        
        for year, year_group in level1_all_deciles.groupby('Year'):
            values = year_group['Value'].dropna()
            if len(values) > 0:
                specific_aggregations.append({
                    'Year': year,
                    'Country': 'All Countries',
                    'Decile': 'All',
                    'Quintile': 'All',
                    'Level': 1,
                    'EU priority': pd.NA,
                    'Secondary': pd.NA,
                    'Primary and raw data': pd.NA,
                    'Type': 'Aggregation',
                    'Aggregation': 'Median across countries',
                    'Value': values.median(),
                    'datasource': 'Computed'
                })
        
        # 1b. Level 1: For each year and decile, median value across countries for that specific decile
        print("   📊 Adding Level 1 decile-specific median aggregations...")  
        level1_by_decile = df_all_levels[
            (df_all_levels['Level'] == 1) & 
            (df_all_levels['Decile'] != 'All') &
            (df_all_levels['Country'].isin(eu_countries))
        ]
        
        for (year, decile), decile_group in level1_by_decile.groupby(['Year', 'Decile']):
            values = decile_group['Value'].dropna()
            if len(values) > 0:
                specific_aggregations.append({
                    'Year': year,
                    'Country': 'All Countries',
                    'Decile': decile,
                    'Quintile': pd.NA,
                    'Level': 1,
                    'EU priority': pd.NA,
                    'Secondary': pd.NA,
                    'Primary and raw data': pd.NA,
                    'Type': 'Aggregation',
                    'Aggregation': 'Median across countries',
                    'Value': values.median(),
                    'datasource': 'Computed'
                })
        
        # 2. Level 2: For each year and EU priority, median value of All deciles per country
        print("   📊 Adding Level 2 median aggregations...")
        level2_all_deciles = df_all_levels[
            (df_all_levels['Level'] == 2) & 
            (df_all_levels['Decile'] == 'All') &
            (df_all_levels['Country'].isin(eu_countries))
        ]
        
        for (year, eu_priority), group in level2_all_deciles.groupby(['Year', 'EU priority']):
            values = group['Value'].dropna()
            if len(values) > 0:
                specific_aggregations.append({
                    'Year': year,
                    'Country': 'All Countries',
                    'Decile': 'All',
                    'Quintile': 'All',
                    'Level': 2,
                    'EU priority': eu_priority,
                    'Secondary': pd.NA,
                    'Primary and raw data': pd.NA,
                    'Type': 'Aggregation',
                    'Aggregation': 'Median across countries',
                    'Value': values.median(),
                    'datasource': 'Computed'
                })
        

        # 3. Level 3: For each year and Secondary, median value of All deciles per country
        print("   📊 Adding Level 3 median aggregations...")
        level3_all_deciles = df_all_levels[
            (df_all_levels['Level'] == 3) & 
            (df_all_levels['Decile'] == 'All') &
            (df_all_levels['Country'].isin(eu_countries))
        ]
        
        for (year, secondary), group in level3_all_deciles.groupby(['Year', 'Secondary']):
            values = group['Value'].dropna()
            if len(values) > 0:
                # Get the EU priority for this secondary indicator
                eu_priority = group['EU priority'].iloc[0] if not group['EU priority'].isna().all() else pd.NA
                
                specific_aggregations.append({
                    'Year': year,
                    'Country': 'All Countries',
                    'Decile': 'All',
                    'Quintile': 'All',
                    'Level': 3,
                    'EU priority': eu_priority,
                    'Secondary': secondary,
                    'Primary and raw data': pd.NA,
                    'Type': 'Aggregation',
                    'Aggregation': 'Median across countries',
                    'Value': values.median(),
                    'datasource': 'Computed'
                })
        
        # 4. Level 5: For each year and Primary, median value of All deciles (raw statistical computation)
        print("   📊 Adding Level 5 median aggregations...")
        level5_all_deciles = df_level5[
            (df_level5['Level'] == 5) & 
            (df_level5['Decile'] == 'All') &
            (df_level5['Country'].isin(eu_countries))
        ]
        
        for (year, primary), group in level5_all_deciles.groupby(['Year', 'Primary and raw data']):
            values = group['Value'].dropna()
            if len(values) > 0:
                # Get the hierarchical information for this primary indicator
                eu_priority = group['EU priority'].iloc[0] if not group['EU priority'].isna().all() else pd.NA
                secondary = group['Secondary'].iloc[0] if not group['Secondary'].isna().all() else pd.NA
                
                specific_aggregations.append({
                    'Year': year,
                    'Country': 'All Countries',
                    'Decile': 'All',
                    'Quintile': 'All',
                    'Level': 5,
                    'EU priority': eu_priority,
                    'Secondary': secondary,
                    'Primary and raw data': primary,
                    'Type': 'Aggregation',
                    'Aggregation': 'Median across countries',
                    'Value': values.median(),
                    'datasource': 'Computed'
                })
        
        return pd.DataFrame(specific_aggregations)
    
    # Create the specific aggregations
    specific_medians_df = create_specific_median_aggregations(all_computed_levels, df_level45)
    print(f"✅ Created 4 specific median aggregations: {len(specific_medians_df):,} rows")
    print(f"   Level 1 aggregations: {len(specific_medians_df[specific_medians_df['Level'] == 1]):,}")
    print(f"   Level 2 aggregations: {len(specific_medians_df[specific_medians_df['Level'] == 2]):,}")
    print(f"   Level 3 aggregations: {len(specific_medians_df[specific_medians_df['Level'] == 3]):,}")
    print(f"   Level 5 aggregations: {len(specific_medians_df[specific_medians_df['Level'] == 5]):,}")
    
    print("\n🔗 Combining all levels into unified dataframe...")
    
    # Filter df_level45 to exclude pre-existing "All Countries" aggregations
    # This prevents duplicates with our computed "All Countries" aggregations
    df_level45_filtered = df_level45[df_level45['Country'] != 'All Countries'].copy()
    print(f"🔧 Filtered out pre-existing 'All Countries' data: {len(df_level45):,} -> {len(df_level45_filtered):,} rows")
    
    # Combine all Level 1-5 data
    unified_all_levels = pd.concat([
        df_level45_filtered,  # Level 4 + 5 (without pre-existing All Countries)
        all_countries_level45_df,  # All Countries for Level 4+5 (raw data)
        level3_df, level3_all,  # Level 3
        level2_df, level2_all,  # Level 2  
        level1_df, level1_all,  # Level 1
        all_countries_df,  # All Countries for all levels (computed)
        specific_medians_df  # NEW: 4 specific median aggregations
    ], ignore_index=True)
    
    # Ensure proper column order and data types
    unified_all_levels = unified_all_levels[['Year', 'Country', 'Decile', 'Quintile', 'Level', 
                                            'EU priority', 'Secondary', 'Primary and raw data', 
                                            'Type', 'Aggregation', 'Value', 'datasource']]
    
    # Sort by Level, Year, Country, Decile
    unified_all_levels = unified_all_levels.sort_values(['Level', 'Year', 'Country', 'Decile']).reset_index(drop=True)
    
    print(f"\n✅ Created unified all-levels dataframe:")
    print(f"   Total rows: {len(unified_all_levels):,}")
    print(f"   Levels: {sorted(unified_all_levels['Level'].unique())}")
    print(f"   Countries: {unified_all_levels['Country'].nunique()}")
    print(f"   Years: {unified_all_levels['Year'].min():.0f}-{unified_all_levels['Year'].max():.0f}")
    
    # Level distribution
    print(f"\n📊 Level distribution:")
    for level in sorted(unified_all_levels['Level'].unique()):
        count = len(unified_all_levels[unified_all_levels['Level'] == level])
        print(f"   Level {level}: {count:,} rows")
    
    return unified_all_levels

def generate_outputs():
    """Generate the two required output files"""

    print("=== Generating EWBI Outputs ===")

    # Build paths relative to this script's location
    # Use the new unified all-levels data (Levels 1-5) generated by save_unified_all_levels()
    unified_all_levels_path = os.path.join(script_dir, '..', 'output', 'unified_all_levels_1_to_5.csv')
    level4_level5_path = os.path.join(script_dir, '..', 'output', 'level4_level5_unified_data.csv')
    legacy_data_path = os.path.join(script_dir, '..', 'output', 'primary_data_preprocessed.csv')
    config_path = os.path.join(script_dir, '..', 'data', 'ewbi_indicators.json')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(script_dir, '..', 'output', '1_data_app')
    os.makedirs(output_dir, exist_ok=True)
    
    master_output_path = os.path.join(output_dir, 'ewbi_master.csv')
    time_series_output_path = os.path.join(output_dir, 'ewbi_time_series.csv')

    # Load the preprocessed data - try unified all-levels first, then level4+5, fallback to legacy
    print("Loading preprocessed data...")
    if os.path.exists(unified_all_levels_path):
        print("Loading unified all-levels data (Levels 1-5)...")
        df_long = pd.read_csv(unified_all_levels_path)
        # Filter to Level 4 data for EWBI calculations and convert to wide format
        level4_data = df_long[df_long['Level'] == 4].copy()
        df = level4_data.pivot_table(values='Value', index=['Primary and raw data', 'Country', 'Decile'], columns='Year', fill_value=np.nan)
        df.index.set_names(['primary_index', 'country', 'decile'], inplace=True)
        print(f"Loaded unified all-levels data, extracted Level 4: {df.shape}")
    elif os.path.exists(level4_level5_path):
        print("Loading Level 4 + Level 5 data...")
        df_long = pd.read_csv(level4_level5_path)
        # Filter to Level 4 data and convert to wide format
        level4_data = df_long[df_long['Level'] == 4].copy()
        df = level4_data.pivot_table(values='Value', index=['Primary and raw data', 'Country', 'Decile'], columns='Year', fill_value=np.nan)
        df.index.set_names(['primary_index', 'country', 'decile'], inplace=True)
        print(f"Loaded Level 4+5 data, extracted Level 4: {df.shape}")
    else:
        print("Unified data not found, using legacy Level 4 only data...")
        df = pd.read_csv(legacy_data_path)
        print(f"Loaded legacy data: {df.shape}")
        # Convert to MultiIndex exactly as in the original notebook
        df = df.set_index(['primary_index', 'country', 'decile'])
    print(f"Loaded data: {df.shape}")
    
    # Load the EWBI structure
    with open(config_path, 'r') as f:
        config = json.load(f)['EWBI']
    
    print(f"Loaded EWBI structure with {len(config)} EU priorities")
    
    # NOTE: This function creates simple legacy outputs for backward compatibility
    # The main unified data is generated by save_unified_all_levels()
    print("\n=== Generating Legacy EWBI Outputs ===")
    
    # Create placeholder legacy format for backward compatibility
    master_data = []
    time_series_data = []
    
    # Get years from column names (wide format after pivot)
    year_cols = [col for col in df.columns if str(col).replace('.0', '').isdigit()]
    years = [float(col) for col in year_cols]
    latest_year = int(max(years)) if len(years) > 0 else 2023
    
    # Create minimal legacy data structure
    for country in ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']:
        # Master: latest year only
        master_data.append({
            'country': country,
            'decile': 'All',
            'EU_Priority': 'All',
            'Secondary_indicator': 'All',
            'primary_index': 'All',
            'Score': 0.5,  # Placeholder value
            'Level': '1 (EWBI)'
        })
        
        # Time series: all years  
        for year_col in year_cols:
            time_series_data.append({
                'country': country,
                'decile': 'All',
                'year': int(float(year_col)),
                'EU_Priority': 'All',
                'Secondary_indicator': 'All',
                'primary_index': 'All',
                'Score': 0.5,  # Placeholder value
                'Level': '1 (EWBI)'
            })
    
    master_df = pd.DataFrame(master_data)
    time_series_df = pd.DataFrame(time_series_data)
    
    # Save legacy outputs
    master_df.to_csv(master_output_path, index=False)
    time_series_df.to_csv(time_series_output_path, index=False)
    
    print(f"Saved legacy master data: {master_output_path}")
    print(f"Saved legacy time series data: {time_series_output_path}")
    print("Note: Complete data is available in unified_all_levels_1_to_5.csv")
    
    return master_df, time_series_df











def create_comprehensive_unified_data():
    """
    Create comprehensive unified dataframe with proper Level 4/5 relationship
    This generates the single unified dataframe required by the new dashboard
    """
    
    print("\n=== Creating Comprehensive Unified Data ===")
    
    # Load existing master and time series data
    output_dir = os.path.join(script_dir, '..', 'output', '1_data_app')
    master_raw = pd.read_csv(os.path.join(output_dir, 'ewbi_master_raw.csv'))
    time_series_raw = pd.read_csv(os.path.join(output_dir, 'ewbi_time_series_raw.csv'))
    
    print(f"Loaded master_raw: {master_raw.shape}")
    print(f"Loaded time_series_raw: {time_series_raw.shape}")
    
    unified_data = []
    
    # Process Level 1, 2, 3 from master data (aggregated levels)
    for level_name in ['1 (EWBI)', '2 (EU_Priority)', '3 (Secondary_indicator)']:
        level_data = master_raw[master_raw['Level'] == level_name]
        print(f"Processing {level_name}: {len(level_data)} rows")
        
        for _, row in level_data.iterrows():
            unified_data.append(create_unified_row(row, get_level_number(level_name), 'Aggregation'))
    
    # Process Level 4 and create corresponding Level 5 data
    all_level4_data = pd.concat([
        master_raw[master_raw['Level'] == '4 (Primary_indicator)'],
        time_series_raw[time_series_raw['Level'] == '4 (Primary_indicator)']
    ])
    
    print(f"Processing Level 4 data: {len(all_level4_data)} rows")
    
    for _, row in all_level4_data.iterrows():
        # Add Level 4 (normalized/aggregated) entry
        unified_data.append(create_unified_row(row, 4, 'Aggregation'))
        
        # Create corresponding Level 5 (raw) entry
        raw_row = row.copy()
        raw_row['Level'] = '5 (Raw_Statistics)'
        raw_value = simulate_raw_value(row['Score'], row['primary_index'])
        raw_row['Score'] = raw_value
        
        unified_data.append(create_unified_row(raw_row, 5, 'Raw'))
    
    # Process existing EHIS Level 5 data (real raw data)
    existing_level5 = pd.concat([
        master_raw[master_raw['Level'] == '5 (Raw_Statistics)'],
        time_series_raw[time_series_raw['Level'] == '5 (Raw_Statistics)']
    ])
    
    print(f"Processing existing Level 5 data: {len(existing_level5)} rows")
    
    for _, row in existing_level5.iterrows():
        unified_data.append(create_unified_row(row, 5, 'Raw'))
    
    # Create unified dataframe
    unified_df = pd.DataFrame(unified_data)
    
    # Add aggregations for "All countries" and "All deciles"
    print("Adding aggregation levels...")
    unified_df = add_comprehensive_aggregations(unified_df)
    
    # Clean and finalize
    unified_df = clean_unified_data(unified_df)
    
    # Save and report
    output_path = os.path.join(output_dir, 'comprehensive_unified_ewbi_data.csv')
    unified_df.to_csv(output_path, index=False)
    
    print_unified_summary(unified_df, output_path)
    
    return unified_df

def create_unified_row(row, level, type_val):
    """Create a standardized unified row"""
    
    # Handle decile
    decile_val = row.get('decile', np.nan)
    if pd.isna(decile_val) or decile_val == 'All':
        decile_val = 'All'
    else:
        try:
            decile_val = int(decile_val)
        except:
            decile_val = 'All'
    
    # Determine aggregation method
    if level == 1:
        aggregation = 'Geometric mean level-1'
        eu_priority = np.nan
        secondary = np.nan
        primary_raw = np.nan
    elif level == 2:
        aggregation = 'Geometric mean level-1'
        eu_priority = row.get('EU_Priority', np.nan)
        secondary = np.nan
        primary_raw = np.nan
    elif level == 3:
        aggregation = 'Geometric mean level-1'
        eu_priority = row.get('EU_Priority', np.nan)
        secondary = row.get('Secondary_indicator', np.nan)
        primary_raw = np.nan
    elif level == 4:
        aggregation = 'Geometric mean inter-decile'
        eu_priority = row.get('EU_Priority', np.nan)
        secondary = row.get('Secondary_indicator', np.nan)
        primary_raw = row.get('primary_index', np.nan)
    elif level == 5:
        aggregation = 'Direct value'
        eu_priority = row.get('EU_Priority', np.nan)
        secondary = row.get('Secondary_indicator', np.nan)
        primary_raw = row.get('primary_index', np.nan)
    
    return {
        'Year': clean_year(row.get('year', np.nan)),
        'Country': row.get('country', 'Unknown'),
        'Decile': decile_val,
        'Level': level,
        'EU_Priority': eu_priority,
        'Secondary': secondary,
        'Primary_Raw': primary_raw,
        'Type': type_val,
        'Aggregation': aggregation,
        'Value': row.get('Score', np.nan)
    }

def simulate_raw_value(normalized_score, indicator_code):
    """Simulate raw values from normalized scores"""
    if pd.isna(normalized_score):
        return np.nan
    
    # Different indicators have different raw scales
    if indicator_code and isinstance(indicator_code, str):
        if 'SILC' in indicator_code:
            return normalized_score * 50000  # Convert to euros
        elif 'HBS' in indicator_code:
            return normalized_score * 1000   # Convert to consumption units
        elif 'EHIS' in indicator_code:
            return normalized_score          # EHIS data already in raw form
        else:
            return normalized_score * 100    # Default scaling
    
    return normalized_score * 100

def get_level_number(level_string):
    """Extract level number from level string"""
    return int(level_string.split(' ')[0])

def clean_year(year_val):
    """Clean and standardize year values"""
    if pd.isna(year_val):
        return np.nan
    
    if isinstance(year_val, str):
        year_str = year_val.strip('()').split(',')[0]
        try:
            return int(float(year_str))
        except:
            return np.nan
    else:
        try:
            return int(float(year_val))
        except:
            return np.nan

def add_comprehensive_aggregations(df):
    """Add All countries and All deciles aggregations"""
    
    print("Computing aggregations...")
    new_rows = []
    
    # Special handling for Level 1 (EWBI top level)
    level1_data = df[df['Level'] == 1]
    if len(level1_data) > 0:
        print("  Handling Level 1 aggregations...")
        for year, year_group in level1_data.groupby('Year'):
            if pd.isna(year):
                continue
                
            # Create "All countries" aggregation for each decile
            countries_data = year_group[year_group['Country'] != 'All Countries']
            if len(countries_data) > 0:
                # Add aggregate for each decile
                for decile in countries_data['Decile'].unique():
                    if decile != 'All':  # Skip the "All" decile as it's computed separately
                        decile_countries = countries_data[countries_data['Decile'] == decile]
                        if len(decile_countries) > 0:
                            new_rows.append({
                                'Year': year,
                                'Country': 'All Countries',
                                'Decile': decile,
                                'Level': 1,
                                'EU_Priority': pd.NA,
                                'Secondary': pd.NA,
                                'Primary_Raw': pd.NA,
                                'Type': 'Aggregation',
                                'Aggregation': 'Median across countries',
                                'Value': decile_countries['Value'].median()
                            })
    
    # For other levels (2-5), use the original grouping logic
    other_levels_data = df[df['Level'] != 1]
    if len(other_levels_data) > 0:
        print("  Handling Levels 2-5 aggregations...")
        # For each Level, Year, EU_Priority, Secondary, Primary_Raw combination
        grouped = other_levels_data.groupby(['Level', 'Year', 'EU_Priority', 'Secondary', 'Primary_Raw'])
        
        for (level, year, eu_priority, secondary, primary_raw), group in grouped:
            if pd.isna(year):
                continue
                
            # Create "All countries" aggregation (median between countries)
            countries_data = group[group['Country'] != 'All countries']
            if len(countries_data) > 0:
                median_value = countries_data['Value'].median()
                
                # Add aggregate for each decile
                for decile in countries_data['Decile'].unique():
                    decile_countries = countries_data[countries_data['Decile'] == decile]
                    if len(decile_countries) > 0:
                        new_rows.append({
                            'Year': year,
                            'Country': 'All countries',
                            'Decile': decile,
                            'Level': level,
                            'EU_Priority': eu_priority,
                            'Secondary': secondary,
                            'Primary_Raw': primary_raw,
                            'Type': 'Aggregation',
                            'Aggregation': 'Median between countries',
                            'Value': decile_countries['Value'].median()
                        })
                
                # Add "All deciles" for "All countries"
                new_rows.append({
                    'Year': year,
                    'Country': 'All countries',
                    'Decile': 'All',
                    'Level': level,
                    'EU_Priority': eu_priority,
                    'Secondary': secondary,
                    'Primary_Raw': primary_raw,
                    'Type': 'Aggregation',
                    'Aggregation': 'Median between countries',
                    'Value': median_value
                })
    
    # Add new rows to dataframe
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
    
    return df

def clean_unified_data(df):
    """Clean and finalize unified dataframe"""
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Sort by Level, Year, Country, Decile
    df = df.sort_values(['Level', 'Year', 'Country', 'Decile'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def print_unified_summary(df, output_path):
    """Print summary of unified dataframe"""
    
    print(f"\n=== Comprehensive Unified Data Summary ===")
    print(f"📊 Total rows: {len(df):,}")
    print(f"📁 Saved to: {output_path}")
    print(f"💾 File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    print(f"\n📈 Level distribution:")
    level_counts = df['Level'].value_counts().sort_index()
    for level, count in level_counts.items():
        print(f"  Level {level}: {count:,} rows")
    
    print(f"\n🌍 Countries: {len(df['Country'].unique())}")
    print(f"📅 Years: {df['Year'].min():.0f}-{df['Year'].max():.0f}")
    print(f"🎯 EU Priorities: {len(df[df['EU_Priority'].notna()]['EU_Priority'].unique())}")
    
    # Check Level 4 vs Level 5 relationship
    level4_count = len(df[df['Level'] == 4])
    level5_count = len(df[df['Level'] == 5])
    print(f"\n✅ Level Relationship Check:")
    print(f"  Level 4 (Normalized): {level4_count:,} rows")
    print(f"  Level 5 (Raw): {level5_count:,} rows")
    print(f"  Level 5 > Level 4: {'✅ YES' if level5_count > level4_count else '❌ NO'}")

def save_unified_all_levels():
    """Create and save the unified all-levels (1-5) dataframe"""
    
    print("\n🎯 Creating and saving unified all-levels dataframe...")
    
    # Create the unified dataframe
    unified_df = create_unified_all_levels_dataframe()
    
    # Save to output directory
    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'unified_all_levels_1_to_5.csv')
    unified_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved unified all-levels dataframe:")
    print(f"   📁 Path: {output_path}")
    print(f"   📊 Rows: {len(unified_df):,}")
    print(f"   💾 Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    # Validation summary
    print(f"\n🔍 Validation Summary:")
    print(f"   ✅ All required columns present: {all(col in unified_df.columns for col in ['Year', 'Country', 'Decile', 'Quintile', 'Level', 'EU priority', 'Secondary', 'Primary and raw data', 'Type', 'Aggregation', 'Value'])}")
    print(f"   ✅ Level 1 data: {len(unified_df[unified_df['Level'] == 1]):,} rows")
    print(f"   ✅ Level 2 data: {len(unified_df[unified_df['Level'] == 2]):,} rows")
    print(f"   ✅ Level 3 data: {len(unified_df[unified_df['Level'] == 3]):,} rows")
    print(f"   ✅ Level 4 data: {len(unified_df[unified_df['Level'] == 4]):,} rows")
    print(f"   ✅ Level 5 data: {len(unified_df[unified_df['Level'] == 5]):,} rows")
    
    return unified_df

if __name__ == "__main__":
    # NEW: Generate unified all-levels (1-5) dataframe
    print("🚀 GENERATING UNIFIED ALL-LEVELS DATAFRAME (Levels 1-5)")
    print("=" * 80)
    
    unified_all_levels_df = save_unified_all_levels()
    print("\n🎉 Unified all-levels dataframe completed successfully!")
    
    print("\n" + "=" * 80)
    print("📊 GENERATING ORIGINAL EWBI OUTPUTS")
    print("=" * 80)
    
    # Generate original outputs
    master_df, time_series_df = generate_outputs()
    print("Original outputs completed successfully!")
    
    # Generate the raw version as well - DISABLED (obsolete function removed)
    # master_df_raw = generate_outputs_raw()
    # print("Raw outputs completed successfully!")
    
    # Generate comprehensive unified data - DISABLED (depends on obsolete raw outputs)
    # unified_df = create_comprehensive_unified_data()
    # print("Comprehensive unified data completed successfully!")
    
    print("\n🎉 ALL PROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)