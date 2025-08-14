#!/usr/bin/env python3
"""
Fix aggregation issue: recalculate all country scores using geometric mean across deciles
instead of arithmetic mean, which is mathematically correct for EWBI scores (ratios/percentages).
"""

import pandas as pd
import numpy as np
import os

def geometric_mean(values):
    """Calculate geometric mean, handling zeros and negative values"""
    # Filter out zeros and negative values (they would make geometric mean undefined)
    valid_values = [v for v in values if v > 0]
    if not valid_values:
        return np.nan
    return np.exp(np.mean(np.log(valid_values)))

def recalculate_aggregates_with_geometric_mean():
    """Recalculate all aggregated scores using geometric mean across deciles"""
    
    print("Loading disaggregated data...")
    
    # Load the original disaggregated data
    primary_data = pd.read_csv('primary_data_preprocessed.csv')
    
    # Get the latest year (2023)
    latest_year = '2023'
    
    print(f"Using data from year: {latest_year}")
    print(f"Data shape: {primary_data.shape}")
    
    # Filter for the latest year and get the scores
    primary_data_latest = primary_data[['primary_index', 'country', 'decile', latest_year]].copy()
    primary_data_latest = primary_data_latest.rename(columns={latest_year: 'score'})
    
    # Remove rows with missing scores
    primary_data_latest = primary_data_latest.dropna(subset=['score'])
    
    print(f"Data after filtering: {primary_data_latest.shape}")
    
    # Calculate geometric mean across deciles for each country and primary indicator
    print("Calculating geometric mean across deciles...")
    
    aggregated_scores = []
    
    for (primary_index, country), group in primary_data_latest.groupby(['primary_index', 'country']):
        if len(group) > 0:
            # Calculate geometric mean across all available deciles
            decile_scores = group['score'].values
            geometric_avg = geometric_mean(decile_scores)
            
            if not np.isnan(geometric_avg):
                aggregated_scores.append({
                    'primary_index': primary_index,
                    'country': country,
                    'geometric_mean_score': geometric_avg,
                    'deciles_count': len(group),
                    'min_score': group['score'].min(),
                    'max_score': group['score'].max()
                })
    
    aggregated_df = pd.DataFrame(aggregated_scores)
    
    print(f"Calculated geometric means for {len(aggregated_df)} country-indicator combinations")
    print(f"Sample results:")
    print(aggregated_df.head())
    
    # Save the corrected aggregated scores
    output_file = 'primary_indicators_geometric_mean_corrected.csv'
    aggregated_df.to_csv(output_file, index=False)
    print(f"Saved corrected scores to: {output_file}")
    
    # Now let's compare with the current aggregated data to see the difference
    print("\nComparing with current aggregated data...")
    
    try:
        current_aggregated = pd.read_csv('master_dataframe_with_decile_aggregates.csv')
        print(f"Current aggregated data shape: {current_aggregated.shape}")
        
        # Check if we have any primary indicators in the current data
        if 'primary_indicator' in current_aggregated.columns:
            print("Primary indicators found in current aggregated data")
        else:
            print("No primary indicators found in current aggregated data")
            
    except FileNotFoundError:
        print("Current aggregated data file not found")
    
    return aggregated_df

if __name__ == "__main__":
    print("=== Fixing Aggregation: Geometric Mean vs Arithmetic Mean ===")
    print("This script recalculates aggregated scores using geometric mean")
    print("which is mathematically correct for EWBI scores (ratios/percentages).")
    print()
    
    corrected_scores = recalculate_aggregates_with_geometric_mean()
    
    print("\n=== Summary ===")
    print(f"✅ Recalculated {len(corrected_scores)} country-indicator combinations")
    print(f"✅ Used geometric mean across deciles (mathematically correct)")
    print(f"✅ Results saved to: primary_indicators_geometric_mean_corrected.csv")
    print("\nNext steps:")
    print("1. Review the corrected scores")
    print("2. Update the master aggregated dataframe")
    print("3. Add primary indicators to both aggregated files") 