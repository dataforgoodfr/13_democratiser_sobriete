#!/usr/bin/env python3
"""
DIRECT PATCH: Fix Level 4 'All Countries' missing data issue

This script directly patches the unified dataset by adding the missing 
Level 4 'All Countries' aggregations that should have been created by the main script.
"""

import pandas as pd
import numpy as np
import json

def patch_level4_all_countries():
    """Directly add missing Level 4 'All Countries' data to unified dataset"""
    
    print("PATCHING: Adding missing Level 4 'All Countries' data")
    print("=" * 60)
    
    # Load the current unified dataset
    unified_path = 'output/unified_all_levels_1_to_5.csv'
    unified_df = pd.read_csv(unified_path, low_memory=False)
    print(f"Loaded current unified dataset: {len(unified_df):,} rows")
    
    # Verify Level 4 'All Countries' is missing
    level4_all = unified_df[(unified_df['Level'] == 4) & (unified_df['Country'] == 'All Countries')]
    print(f"Current Level 4 'All Countries' rows: {len(level4_all):,}")
    
    if len(level4_all) > 0:
        print("Level 4 'All Countries' data already exists. No patch needed.")
        return True
    
    # Create the missing Level 4 'All Countries' data
    print("Creating missing Level 4 'All Countries' aggregations...")
    
    # Load base Level 4+5 data and apply mappings
    df_level45 = pd.read_csv('output/level4_level5_unified_data.csv', low_memory=False)
    
    # Load and apply indicator mappings
    with open('data/ewbi_indicators.json', 'r') as f:
        config = json.load(f)['EWBI']

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

    df_level45['EU priority'] = df_level45['Primary and raw data'].map(lambda x: indicator_mapping.get(x, {}).get('EU priority', pd.NA))
    df_level45['Secondary'] = df_level45['Primary and raw data'].map(lambda x: indicator_mapping.get(x, {}).get('Secondary', pd.NA))
    
    # Create Level 4 'All Countries' aggregations
    eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    
    # Filter for EU Level 4 data with complete mappings
    eu_level4 = df_level45[
        (df_level45['Level'] == 4) &
        (df_level45['Country'].isin(eu_countries)) &
        (df_level45['EU priority'].notna()) &
        (df_level45['Secondary'].notna())
    ]
    
    print(f"EU Level 4 data for aggregation: {len(eu_level4):,} rows")
    
    # Create aggregations by grouping
    grouped = eu_level4.groupby(['Year', 'Decile', 'Level', 'EU priority', 'Secondary', 'Primary and raw data'])
    
    all_countries_data = []
    for (year, decile, level, eu_priority, secondary, indicator), group in grouped:
        values = group['Value'].dropna()
        
        if len(values) > 0:
            median_value = values.median()
            
            all_countries_data.append({
                'Year': year,
                'Country': 'All Countries',
                'Decile': decile,
                'Quintile': pd.NA,
                'Level': level,
                'EU priority': eu_priority,
                'Secondary': secondary,
                'Primary and raw data': indicator,
                'Type': 'relative',
                'Aggregation': 'median',
                'Value': median_value,
                'datasource': group['datasource'].iloc[0]
            })
    
    all_countries_df = pd.DataFrame(all_countries_data)
    print(f"Created Level 4 'All Countries' aggregations: {len(all_countries_df):,} rows")
    
    # Verify AE-EHIS-1 is included
    aehis1_level4 = all_countries_df[all_countries_df['Primary and raw data'] == 'AE-EHIS-1']
    print(f"AE-EHIS-1 Level 4 'All Countries' created: {len(aehis1_level4):,} rows")
    
    if len(aehis1_level4) == 0:
        print("ERROR: Failed to create AE-EHIS-1 Level 4 'All Countries' data")
        return False
    
    # Add the Level 4 'All Countries' data to the unified dataset
    print("Adding Level 4 'All Countries' data to unified dataset...")
    
    # Ensure column alignment
    all_countries_df = all_countries_df.reindex(columns=unified_df.columns, fill_value=pd.NA)
    
    # Combine
    patched_unified_df = pd.concat([unified_df, all_countries_df], ignore_index=True)
    
    # Sort by Level, Year, Country, Decile
    patched_unified_df = patched_unified_df.sort_values(['Level', 'Year', 'Country', 'Decile']).reset_index(drop=True)
    
    print(f"Patched unified dataset: {len(patched_unified_df):,} rows (added {len(all_countries_df):,})")
    
    # Verify the patch worked
    level4_all_patched = patched_unified_df[(patched_unified_df['Level'] == 4) & (patched_unified_df['Country'] == 'All Countries')]
    print(f"Level 4 'All Countries' after patch: {len(level4_all_patched):,} rows")
    
    aehis1_patched = level4_all_patched[level4_all_patched['Primary and raw data'] == 'AE-EHIS-1']
    print(f"AE-EHIS-1 Level 4 'All Countries' after patch: {len(aehis1_patched):,} rows")
    
    if len(aehis1_patched) > 0:
        print("SUCCESS: Patch created the needed AE-EHIS-1 Level 4 'All Countries' data!")
        
        # Save the patched dataset
        backup_path = 'output/unified_all_levels_1_to_5_backup.csv'
        print(f"Backing up original to: {backup_path}")
        unified_df.to_csv(backup_path, index=False)
        
        print(f"Saving patched dataset to: {unified_path}")
        patched_unified_df.to_csv(unified_path, index=False)
        
        print("\nSample AE-EHIS-1 Level 4 'All Countries' data:")
        sample = aehis1_patched.head(3)
        for _, row in sample.iterrows():
            print(f"  {row['Year']:.0f}, Decile {row['Decile']}: {row['Value']:.4f}")
        
        return True
    else:
        print("ERROR: Patch failed to create AE-EHIS-1 data")
        return False

if __name__ == "__main__":
    success = patch_level4_all_countries()
    
    if success:
        print("\n" + "=" * 60)
        print("PATCH COMPLETED SUCCESSFULLY!")
        print("Both original issues should now be resolved:")
        print("1. EHIS 'Nutrition expense' secondary indicator will show data")
        print("2. Agriculture and Food EU priority will show 'All Countries' in decile graphs")
        print("\nThe unified dataset has been updated with the missing Level 4 'All Countries' data.")
    else:
        print("\nPATCH FAILED!")
        print("Manual intervention may be required.")