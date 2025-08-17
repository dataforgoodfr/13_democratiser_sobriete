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
    
    # Data cleaning - exactly as in the original
    df = df.drop(columns=['database'])
    df['value'] = df['value'].str.replace(',', '.')  # some commas appear as decile separators
    df['value'] = df['value'].astype(float)
    
    print(f"Data cleaned: {df.shape}")
    
    # Pivot the data exactly as in the original
    wide = df.pivot_table(values='value', index=['primary_index', 'decile', 'country'], columns='year')
    
    print(f"Pivoted data: {wide.shape}")
    
    # Fill missing values exactly as in the original
    filled = wide.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
    
    print(f"Filled missing values: {filled.shape}")
    
    # Normalize the data exactly as in the original
    res = []
    for (ind, decile), grouped in filled.groupby(['primary_index', 'decile']):
        data = grouped.copy()
        
        # normalize the data over countries, so that the best-performing country has value 1 and the worst 0
        # values are negative in the sense that the best-performing country is the one with the lowest initial value and vice-versa
        norm = 1 - (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        
        # replace 0 values with 0.001 as well as all values in between
        norm[norm < 0.001] = 0.001
        res.append(norm)
    
    preprocessed = pd.concat(res)
    
    print(f"Preprocessed data: {preprocessed.shape}")
    
    # Save exactly as in the original
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    output_path = os.path.join(output_dir, 'primary_data_preprocessed.csv')
    
    preprocessed.swaplevel(1, 2).sort_index().to_csv(output_path)
    print(f"Saved preprocessed data to: {output_path}")
    
    return preprocessed

if __name__ == "__main__":
    preprocessed_df = recreate_preprocessing()
    print("Preprocessing completed successfully!") 