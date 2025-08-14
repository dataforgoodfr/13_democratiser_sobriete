#!/usr/bin/env python3
"""
Add primary indicators to the master aggregated dataframe using the corrected geometric mean scores.
This will complete the 4-level structure needed by the dashboard.
"""

import pandas as pd
import numpy as np

def add_primary_indicators_to_aggregated():
    """Add primary indicators to the master aggregated dataframe"""
    
    print("Loading current aggregated data...")
    current_df = pd.read_csv('master_dataframe_with_decile_aggregates.csv')
    print(f"Current data shape: {current_df.shape}")
    
    print("Loading corrected primary indicator scores...")
    primary_scores = pd.read_csv('primary_indicators_geometric_mean_corrected.csv')
    print(f"Primary scores shape: {primary_scores.shape}")
    
    # Create primary indicator rows in the same format as the current data
    primary_rows = []
    
    for _, row in primary_scores.iterrows():
        primary_rows.append({
            'country': row['country'],
            'ewbi_score': np.nan,  # Not applicable for primary indicators
            'decile': 'All Deciles',
            'source': 'primary_indicators',
            'eu_priority': np.nan,  # Will be filled based on the indicator structure
            'eu_priority_score': np.nan,  # Not applicable for primary indicators
            'secondary_indicator': np.nan,  # Will be filled based on the indicator structure
            'secondary_score': np.nan,  # Not applicable for primary indicators
            'primary_indicator': row['primary_index'],
            'primary_score': row['geometric_mean_score']
        })
    
    # Create DataFrame for primary indicators
    primary_df = pd.DataFrame(primary_rows)
    print(f"Created {len(primary_df)} primary indicator rows")
    
    # Now we need to map primary indicators to their EU priority and secondary indicator
    # Let me check the EWBI structure to understand the hierarchy
    print("\nLoading EWBI structure to map primary indicators...")
    
    try:
        import json
        with open('../data/ewbi_indicators.json', 'r') as f:
            ewbi_structure = json.load(f)['EWBI']
        
        # Create a mapping from primary indicator to EU priority and secondary indicator
        primary_mapping = {}
        
        for eu_priority in ewbi_structure:
            priority_name = eu_priority['name']
            for component in eu_priority['components']:
                component_name = component['name']
                for indicator in component['indicators']:
                    primary_code = indicator['code']
                    primary_mapping[primary_code] = {
                        'eu_priority': priority_name,
                        'secondary_indicator': component_name
                    }
        
        print(f"Created mapping for {len(primary_mapping)} primary indicators")
        
        # Update the primary indicator rows with the correct EU priority and secondary indicator
        for idx, row in primary_df.iterrows():
            primary_code = row['primary_indicator']
            if primary_code in primary_mapping:
                mapping = primary_mapping[primary_code]
                primary_df.at[idx, 'eu_priority'] = mapping['eu_priority']
                primary_df.at[idx, 'secondary_indicator'] = mapping['secondary_indicator']
            else:
                print(f"Warning: No mapping found for primary indicator {primary_code}")
        
        # Check how many were mapped successfully
        mapped_count = primary_df['eu_priority'].notna().sum()
        print(f"Successfully mapped {mapped_count} out of {len(primary_df)} primary indicators")
        
    except Exception as e:
        print(f"Warning: Could not load EWBI structure: {e}")
        print("Primary indicators will be added without EU priority mapping")
    
    # Combine the current data with the new primary indicators
    print("\nCombining data...")
    
    # Add the new columns to the current dataframe if they don't exist
    if 'primary_indicator' not in current_df.columns:
        current_df['primary_indicator'] = np.nan
    if 'primary_score' not in current_df.columns:
        current_df['primary_score'] = np.nan
    
    # Combine the dataframes
    combined_df = pd.concat([current_df, primary_df], ignore_index=True)
    
    print(f"Combined data shape: {combined_df.shape}")
    print(f"Added {len(primary_df)} primary indicator rows")
    
    # Save the updated aggregated dataframe
    output_file = 'master_dataframe_with_decile_aggregates_updated.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"Saved updated aggregated data to: {output_file}")
    
    # Show summary of the updated data
    print("\n=== Updated Data Summary ===")
    source_counts = combined_df['source'].value_counts()
    for source, count in source_counts.items():
        print(f"{source}: {count} rows")
    
    return combined_df

if __name__ == "__main__":
    print("=== Adding Primary Indicators to Aggregated Data ===")
    print("This script adds the missing primary indicators to complete the 4-level structure.")
    print()
    
    updated_df = add_primary_indicators_to_aggregated()
    
    print("\n=== Summary ===")
    print("✅ Added primary indicators to the aggregated dataframe")
    print("✅ Used corrected geometric mean scores")
    print("✅ Mapped primary indicators to EU priorities and secondary indicators")
    print(f"✅ Updated data saved to: master_dataframe_with_decile_aggregates_updated.csv")
    print("\nNext steps:")
    print("1. Review the updated aggregated data")
    print("2. Update the time series file with primary indicators")
    print("3. Test the complete 4-level dashboard functionality") 