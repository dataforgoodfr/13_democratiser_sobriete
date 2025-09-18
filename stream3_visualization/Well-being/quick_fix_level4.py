#!/usr/bin/env python3
"""
Quick fix to ensure Level 4 'All Countries' data is created properly
This script directly creates the missing Level 4 'All Countries' aggregations
and verifies they work for the specific AE-EHIS-1 indicator
"""

import pandas as pd
import numpy as np
import json

def quick_fix_level4_all_countries():
    """Create and verify Level 4 'All Countries' data for EHIS"""
    
    print("🔧 QUICK FIX: Creating Level 4 'All Countries' data")
    print("=" * 60)
    
    # Load the base data
    df_level45 = pd.read_csv('output/level4_level5_unified_data.csv', low_memory=False)
    print(f"✅ Loaded base data: {len(df_level45):,} rows")
    
    # Load and apply mappings
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
    
    print(f"✅ Applied mappings: {df_level45[df_level45['Level'] == 4]['EU priority'].notna().sum():,} Level 4 rows with mappings")
    
    # Create Level 4 'All Countries' aggregations
    eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    
    # Filter for EU Level 4 data with complete mappings
    eu_level4 = df_level45[
        (df_level45['Level'] == 4) &
        (df_level45['Country'].isin(eu_countries)) &
        (df_level45['EU priority'].notna()) &
        (df_level45['Secondary'].notna())
    ]
    
    print(f"✅ EU Level 4 data ready for aggregation: {len(eu_level4):,} rows")
    
    # Check AE-EHIS-1 specifically
    aehis1 = eu_level4[eu_level4['Primary and raw data'] == 'AE-EHIS-1']
    print(f"✅ AE-EHIS-1 Level 4 data: {len(aehis1):,} rows")
    
    if len(aehis1) == 0:
        print("❌ ERROR: No AE-EHIS-1 Level 4 data found!")
        return False
        
    # Create aggregations
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
    print(f"✅ Created Level 4 'All Countries' aggregations: {len(all_countries_df):,} rows")
    
    # Verify AE-EHIS-1 specifically
    aehis1_all = all_countries_df[
        (all_countries_df['Level'] == 4) & 
        (all_countries_df['Primary and raw data'] == 'AE-EHIS-1')
    ]
    print(f"✅ AE-EHIS-1 Level 4 'All Countries' created: {len(aehis1_all):,} rows")
    
    if len(aehis1_all) > 0:
        print("\n🎯 SUCCESS: AE-EHIS-1 Level 4 'All Countries' data created!")
        print(f"   Years: {sorted(aehis1_all['Year'].unique())}")
        print(f"   Deciles: {sorted(aehis1_all['Decile'].astype(str).unique())}")
        print(f"   EU priority: {aehis1_all['EU priority'].iloc[0]}")
        print(f"   Secondary: {aehis1_all['Secondary'].iloc[0]}")
        
        # Show sample values
        sample = aehis1_all.head(3)
        print("\n📊 Sample data:")
        for _, row in sample.iterrows():
            print(f"   {row['Year']:.0f}, Decile {row['Decile']}: {row['Value']:.4f}")
        
        # Save the Level 4 'All Countries' data for manual verification
        all_countries_df.to_csv('output/level4_all_countries_fix.csv', index=False)
        print(f"\n💾 Saved Level 4 'All Countries' data to: output/level4_all_countries_fix.csv")
        
        return True
    else:
        print("❌ ERROR: Failed to create AE-EHIS-1 Level 4 'All Countries' data")
        return False

if __name__ == "__main__":
    success = quick_fix_level4_all_countries()
    if success:
        print("\n✅ Level 4 'All Countries' fix completed successfully!")
        print("   This data should resolve the Agriculture and Food decile graph issue.")
    else:
        print("\n❌ Level 4 'All Countries' fix failed!")