#!/usr/bin/env python3
"""
Recreate Preprocessing Script
This script recreates the primary_data_preprocessed.csv file that was deleted
"""

import pandas as pd
import numpy as np
import os

def recreate_preprocessing():
    """Recreate the primary_data_preprocessed.csv file"""
    
    # Load the original data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = pd.read_csv(os.path.join(data_dir, '2025-06-05_df_final_EWBI.csv'))
    
    print(f"Loaded original data: {df.shape}")
    
    # Data cleaning
    df = df.drop(columns=['database'])
    
    # Convert value column to numeric, handling any string values
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Drop rows with NaN values in the value column
    df = df.dropna(subset=['value'])
    
    print(f"Data cleaned: {df.shape}")
    
    # Pivot the data to have years as columns
    pivot_df = df.pivot_table(
        index=['country', 'decile', 'primary_index'],
        columns='year',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    print(f"Pivoted data: {pivot_df.shape}")
    
    # Fill missing values with forward fill and backward fill
    # Use the newer pandas methods
    filled = pivot_df.ffill(axis=1).bfill(axis=1)
    
    print(f"Filled missing values: {filled.shape}")
    
    # Normalize the data over countries for each indicator-decile combination
    res = []
    for (ind, decile), grouped in filled.groupby(['primary_index', 'decile']):
        data = grouped.copy()
        
        # Get only the year columns (exclude country, decile, primary_index)
        year_cols = [col for col in data.columns if col not in ['country', 'decile', 'primary_index']]
        
        if len(year_cols) > 0:
            # Ensure year columns are numeric
            year_data = data[year_cols].apply(pd.to_numeric, errors='coerce')
            
            # Drop rows where all year values are NaN
            year_data = year_data.dropna(how='all')
            
            if not year_data.empty and len(year_data.columns) > 0:
                # normalize the data over countries, so that the best-performing country has value 1 and the worst 0
                # values are negative in the sense that the best-performing country is the one with the lowest initial value and vice-versa
                min_vals = year_data.min(axis=0)
                max_vals = year_data.max(axis=0)
                
                # Avoid division by zero
                range_vals = max_vals - min_vals
                range_vals = range_vals.replace(0, 1)  # Replace 0 ranges with 1 to avoid division by zero
                
                norm = 1 - (year_data - min_vals) / range_vals
                
                # replace 0 values with 0.001 as well as all values in between
                norm[norm < 0.001] = 0.001
                
                # Add back the index columns
                norm['country'] = data.loc[year_data.index, 'country'].values
                norm['decile'] = data.loc[year_data.index, 'decile'].values
                norm['primary_index'] = data.loc[year_data.index, 'primary_index'].values
                
                res.append(norm)
    
    if res:
        preprocessed = pd.concat(res)
        print(f"Preprocessed data: {preprocessed.shape}")
        
        # Save the preprocessed data
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        output_path = os.path.join(output_dir, 'primary_data_preprocessed.csv')
        
        # Reorder columns to have country, decile, primary_index first, then years
        year_cols = [col for col in preprocessed.columns if col not in ['country', 'decile', 'primary_index']]
        year_cols.sort()  # Sort years chronologically
        
        final_cols = ['country', 'decile', 'primary_index'] + year_cols
        preprocessed = preprocessed[final_cols]
        
        preprocessed.to_csv(output_path, index=False)
        print(f"Saved preprocessed data to: {output_path}")
        
        return preprocessed
    else:
        print("No preprocessed data generated!")
        return None

if __name__ == "__main__":
    preprocessed_df = recreate_preprocessing()
    if preprocessed_df is not None:
        print("Preprocessing completed successfully!")
    else:
        print("Preprocessing failed!") 