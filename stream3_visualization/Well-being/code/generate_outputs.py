#!/usr/bin/env python3
"""
Generate Outputs Script
This script generates the two required output files:
1. Master dataframe: All indicators for all levels, per decile and aggregated per country, for the latest year
2. Time series dataframe: Historical data per country for all indicators for all levels, but only for "All Deciles"
"""

import pandas as pd
import numpy as np
import json
import os

def generate_outputs():
    """Generate the two required output files"""
    
    print("=== Generating EWBI Outputs ===")
    
    # Load the preprocessed data
    print("Loading preprocessed data...")
    df = pd.read_csv('../output/primary_data_preprocessed.csv')
    print(f"Loaded data: {df.shape}")
    
    # Convert to MultiIndex exactly as in the original notebook
    df = df.set_index(['country', 'primary_index', 'decile'])
    print(f"Converted to MultiIndex: {df.shape}")
    
    # Load the EWBI structure
    with open('../data/ewbi_indicators.json', 'r') as f:
        config = json.load(f)['EWBI']
    
    print(f"Loaded EWBI structure with {len(config)} EU priorities")
    
    # Filter out economic good indicators (only keep satisfiers) - exactly as in original
    economic_indicators_to_remove = [
        'AN-SILC-1',
        'AE-HBS-1', 'AE-HBS-2',
        'HQ-SILC-2',
        'HH-SILC-1', 'HH-HBS-1', 'HH-HBS-2', 'HH-HBS-3', 'HH-HBS-4',
        'EC-HBS-1', 'EC-HBS-2',
        'ED-ICT-1', 'ED-EHIS-1',
        'AC-SILC-1', 'AC-SILC-2', 'AC-HBS-1', 'AC-HBS-2', 'AC-EHIS-1',
        'IE-HBS-1', 'IE-HBS-2',
        'IC-SILC-1', 'IC-SILC-2', 'IC-HBS-1', 'IC-HBS-2',
        'TT-SILC-1', 'TT-SILC-2', 'TT-HBS-1', 'TT-HBS-2',
        'TS-SILC-1', 'TS-HBS-1', 'TS-HBS-2'
    ]
    
    print(f"Filtering out {len(economic_indicators_to_remove)} economic indicators")
    print(f"Initial data shape: {df.shape}")
    
    # Remove economic indicators
    df_filtered = df[~df.index.get_level_values('primary_index').isin(economic_indicators_to_remove)]
    
    print(f"After filtering: {df_filtered.shape}")
    print(f"Removed {df.shape[0] - df_filtered.shape[0]} rows")
    
    # Use filtered data for the rest of the computation
    df = df_filtered
    
    # Get the latest year (2023)
    year_cols = [col for col in df.columns if str(col).isdigit()]
    latest_year = max(year_cols)
    print(f"Using latest year: {latest_year}")
    
    # 1. Generate Master Dataframe (latest year, all deciles)
    print("\n=== Generating Master Dataframe ===")
    
    # For now, let's create a simple structure that matches what the dashboard expects
    # We'll use the latest year data and create the basic structure
    
    master_data = []
    
    for country in df.index.get_level_values('country').unique():
        for decile in df.index.get_level_values('decile').unique():
            country_decile_data = df.loc[(country, slice(None), decile)]
            
            if not country_decile_data.empty:
                # Get the latest year values for all primary indicators
                latest_values = country_decile_data[latest_year]
                
                # Create a row for this country-decile combination
                row = {
                    'country': country,
                    'decile': decile,
                    'latest_year': latest_year
                }
                
                # Add primary indicator values
                for primary_index in latest_values.index:
                    row[primary_index] = latest_values[primary_index]
                
                master_data.append(row)
    
    master_df = pd.DataFrame(master_data)
    print(f"Generated master dataframe: {master_df.shape}")
    
    # 2. Generate Time Series Dataframe (all years, All Deciles only)
    print("\n=== Generating Time Series Dataframe ===")
    
    # For time series, we want country-level aggregated data (All Deciles)
    # We'll aggregate across deciles for each country-year-primary_indicator combination
    
    time_series_data = []
    
    for country in df.index.get_level_values('country').unique():
        for primary_index in df.index.get_level_values('primary_index').unique():
            # Get all deciles for this country-primary_indicator combination
            country_primary_data = df.loc[(country, primary_index, slice(None))]
            
            if not country_primary_data.empty:
                # Aggregate across deciles for each year
                for year in year_cols:
                    year_values = country_primary_data[year]
                    if not year_values.isna().all():  # Only add if we have some data
                        # Calculate average across deciles
                        avg_value = year_values.mean()
                        
                        time_series_data.append({
                            'country': country,
                            'decile': 'All Deciles',
                            'year': int(year),
                            'primary_index': primary_index,
                            'primary_score': avg_value
                        })
    
    time_series_df = pd.DataFrame(time_series_data)
    print(f"Generated time series dataframe: {time_series_df.shape}")
    
    # Save the outputs
    print("\n=== Saving Outputs ===")
    
    # Save master dataframe
    master_output_path = '../output/ewbi_master.csv'
    master_df.to_csv(master_output_path, index=False)
    print(f"Saved master dataframe to: {master_output_path}")
    
    # Save time series dataframe
    time_series_output_path = '../output/ewbi_time_series.csv'
    time_series_df.to_csv(time_series_output_path, index=False)
    print(f"Saved time series dataframe to: {time_series_output_path}")
    
    print("\n=== Output Generation Complete ===")
    print(f"Master dataframe: {master_df.shape}")
    print(f"Time series dataframe: {time_series_df.shape}")
    
    return master_df, time_series_df

if __name__ == "__main__":
    master_df, time_series_df = generate_outputs()
    print("Script completed successfully!") 