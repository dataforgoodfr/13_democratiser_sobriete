#!/usr/bin/env python3
"""
Recreate Preprocessing Script
This script recreates the primary_data_preprocessed.csv file that was deleted
"""

import pandas as pd
import numpy as np
import os
import json

def recreate_preprocessing():
    """Recreate the primary_data_preprocessed.csv file"""
    
    # Load the original data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = pd.read_csv(os.path.join(data_dir, '2025-06-05_df_final_EWBI.csv'))
    
    print(f"Loaded original data: {df.shape}")
    
    # Load the EWBI config to get valid primary indicator codes
    with open(os.path.join(data_dir, 'ewbi_indicators.json'), 'r') as f:
        config = json.load(f)['EWBI']
    
    # Extract all valid primary indicator codes from the config
    valid_primary_codes = set()
    for eu_priority in config:
        for secondary in eu_priority['components']:
            for primary in secondary['indicators']:
                valid_primary_codes.add(primary['code'])
    
    print(f"Found {len(valid_primary_codes)} valid primary indicator codes in config")
    
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
    df = df[~df['primary_index'].isin(economic_indicators_to_remove)]
    
    print(f"After filtering out economic indicators: {df.shape}")
    
    # Filter the data to only include valid primary indicators from config
    df = df[df['primary_index'].isin(valid_primary_codes)]
    print(f"After filtering to valid indicators from config: {df.shape}")
    
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
    
    # Output exactly as in the original - preserve MultiIndex structure
    preprocessed.swaplevel(0, 1).sort_index().to_csv(output_path)
    print(f"Saved preprocessed data to: {output_path}")
    
    return preprocessed

if __name__ == "__main__":
    preprocessed_df = recreate_preprocessing()
    print("Preprocessing completed successfully!") 