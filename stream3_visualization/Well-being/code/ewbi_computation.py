#!/usr/bin/env python3
"""
EWBI Computation Script
Converts primary indicator data to secondary indicators, EU priorities, and EWBI scores.
This script calculates all indicator levels with proper aggregation methods.
"""

import json
import numpy as np
import pandas as pd
import os

def simple_average(data):
    """
    Simple straight average of all indicators (ignoring weights)
    Args:
        data: list of tuples (values, weight) where values is a pandas Series
    Returns:
        pandas Series with the average values
    """
    all_values = []
    for values, weight in data:
        all_values.append(values)
    
    # Concatenate all series and calculate mean
    if all_values:
        combined = pd.concat(all_values, axis=1)
        return combined.mean(axis=1)
    else:
        return pd.Series(dtype=float)

def weighted_geometric_mean(data):
    """
    Calculate weighted geometric mean of indicators
    Args:
        data: list of tuples (values, weight) where values is a pandas Series
    Returns:
        pandas Series with the weighted geometric mean values
    """
    if not data:
        return pd.Series(dtype=float)
    
    # Extract values and weights
    values_list = [item[0] for item in data]
    weights = [item[1] for item in data]
    
    # Normalize weights to sum to 1
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Calculate weighted geometric mean
    # For each position, calculate: exp(sum(weight * log(value)))
    result = pd.Series(index=values_list[0].index, dtype=float)
    
    for idx in result.index:
        # Get values at this index from all series
        idx_values = [series.loc[idx] for series in values_list]
        
        # Filter out non-positive values (geometric mean requires positive numbers)
        valid_pairs = [(val, weight) for val, weight in zip(idx_values, normalized_weights) if val > 0]
        
        if valid_pairs:
            # Calculate weighted geometric mean
            log_values = [np.log(val) for val, _ in valid_pairs]
            log_weights = [weight for _, weight in valid_pairs]
            
            # Weighted sum of log values
            weighted_log_sum = sum(log_val * weight for log_val, weight in zip(log_values, log_weights))
            result.loc[idx] = np.exp(weighted_log_sum)
        else:
            result.loc[idx] = np.nan
    
    return result

def calculate_secondary_indicators(df, config):
    """
    Calculate secondary indicators using simple averages from primary indicators
    Args:
        df: DataFrame with primary indicator data indexed by (country, primary_index, decile)
        config: EWBI configuration structure
    Returns:
        DataFrame with secondary indicator scores
    """
    print("Calculating secondary indicators...")
    
    secondary = {}
    missing = {}
    
    # Separate countries as indicators aren't all available for all countries
    for country, cdf in df.groupby('country'):
        cdf = cdf.loc[country]
        for prio in config:
            for component in prio['components']:
                factors = []
                for ind in component['indicators']:
                    code = ind['code']
                    if code in cdf.index:
                        factors.append((cdf.loc[code], 1))  # weight set to 1 since we ignore it
                    elif code not in {'IS-SILC-2', 'IS-SILC-1', 'RU-LFS-1'}:
                        print(f"{country},{code}")
                
                if factors:
                    secondary[country, prio['name'], component['name']] = simple_average(factors)
                else:
                    # print('Missing', country, component['name'])
                    pass
    
    # Convert to DataFrame
    secondary_df = pd.concat(secondary, names=('country', 'eu_priority', 'secondary_indicator'))
    
    print(f"Calculated {len(secondary_df)} secondary indicator scores")
    return secondary_df

def calculate_eu_priorities(secondary_df, config):
    """
    Calculate EU priority scores using weighted geometric mean from secondary indicators
    Args:
        secondary_df: DataFrame with secondary indicator scores
        config: EWBI configuration structure
    Returns:
        DataFrame with EU priority scores
    """
    print("Calculating EU priorities...")
    
    priorities = {}
    
    for country, cdf in secondary_df.groupby('country'):
        cdf = cdf.loc[country]
        for prio in config:
            pname = prio['name']
            if pname in cdf.index:
                cpdf = cdf.loc[pname]
                factors = []
                for c in prio['components']:
                    name = c['name']
                    weight = c['weight']
                    try:
                        weight = float(weight)
                    except ValueError:
                        numerator, denominator = map(int, weight.split('/'))
                        weight = float(numerator) / denominator

                    if name in cpdf.index and weight != 0:
                        factors.append((cpdf.loc[name], weight))

                if factors:
                    priorities[country, pname] = weighted_geometric_mean(factors)
                else:
                    print('Missing', country, pname)
    
    # Convert to DataFrame
    priorities_df = pd.concat(priorities, names=['country', 'eu_priority'])
    
    print(f"Calculated {len(priorities_df)} EU priority scores")
    return priorities_df

def calculate_ewbi_scores(priorities_df):
    """
    Calculate EWBI scores using simple average from EU priorities
    Args:
        priorities_df: DataFrame with EU priority scores
    Returns:
        DataFrame with EWBI scores
    """
    print("Calculating EWBI scores...")
    
    ewbi = {}
    
    for country, cdf in priorities_df.groupby('country'):
        cdf = cdf.loc[country]
        factors = [(cdf.loc[prio], 1) for prio in cdf.index.get_level_values('eu_priority')]
        ewbi[country] = simple_average(factors)
    
    # Convert to DataFrame
    ewbi_df = pd.concat(ewbi, names=['country'])
    
    print(f"Calculated {len(ewbi_df)} EWBI scores")
    return ewbi_df

def add_decile_level_calculations(primary_df, secondary_df, priorities_df, ewbi_df, config):
    """
    Add decile-level calculations for all indicator levels
    Args:
        primary_df: DataFrame with primary indicator data
        secondary_df: DataFrame with secondary indicator scores
        priorities_df: DataFrame with EU priority scores
        ewbi_df: DataFrame with EWBI scores
        config: EWBI configuration structure
    Returns:
        Tuple of DataFrames with decile-level scores
    """
    print("Adding decile-level calculations...")
    
    # For now, we'll create placeholder decile-level DataFrames
    # The actual decile-level calculations would need to be implemented
    # based on how the primary data is structured
    
    print("Note: Decile-level calculations need to be implemented based on data structure")
    
    return secondary_df, priorities_df, ewbi_df

def main():
    """Main function to run the EWBI computation"""
    print("=== EWBI Computation Script ===")
    
    # Load the data
    print("Loading primary indicators data...")
    df = pd.read_csv('../output/primary_data_preprocessed.csv')
    df = df.set_index(['country', 'primary_index', 'decile'])
    
    print(f"Initial data shape: {df.shape}")
    
    # Filter out economic good indicators (only keep satisfiers)
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
    
    # Remove economic indicators
    df_filtered = df[~df.index.get_level_values('primary_index').isin(economic_indicators_to_remove)]
    
    print(f"After filtering: {df_filtered.shape}")
    print(f"Removed {df.shape[0] - df_filtered.shape[0]} rows")
    
    # Use filtered data for the rest of the computation
    df = df_filtered
    
    # Load EWBI structure
    print("Loading EWBI structure...")
    with open('../data/ewbi_indicators.json') as f:
        config = json.load(f)['EWBI']
    
    # Get all expected codes from config
    all_codes = set()
    for priority in config:
        for component in priority['components']:
            for indicator in component['indicators']:
                all_codes.add(indicator['code'])
    
    print(f"Expected {len(all_codes)} primary indicators from config")
    
    # Check which indicators are missing
    present_codes = set(df.index.get_level_values('primary_index'))
    print("Present in json file but not in index:", all_codes.difference(present_codes))
    print("Present in index but not in json file:", present_codes.difference(all_codes))
    
    # Calculate secondary indicators
    secondary_df = calculate_secondary_indicators(df, config)
    
    # Calculate EU priorities
    priorities_df = calculate_eu_priorities(secondary_df, config)
    
    # Calculate EWBI scores
    ewbi_df = calculate_ewbi_scores(priorities_df)
    
    # Add decile-level calculations (placeholder for now)
    secondary_df_decile, priorities_df_decile, ewbi_df_decile = add_decile_level_calculations(
        df, secondary_df, priorities_df, ewbi_df, config
    )
    
    # Save results
    print("Saving results...")
    secondary_df.to_csv('../output/secondary_indicators.csv')
    priorities_df.to_csv('../output/eu_priorities.csv')
    ewbi_df.to_csv('../output/ewbi_results.csv')
    
    print("=== Computation Complete ===")
    print(f"Secondary indicators: {len(secondary_df)} scores")
    print(f"EU priorities: {len(priorities_df)} scores")
    print(f"EWBI scores: {len(ewbi_df)} scores")
    print("\nFiles saved:")
    print("- ../output/secondary_indicators.csv")
    print("- ../output/eu_priorities.csv")
    print("- ../output/ewbi_results.csv")

if __name__ == "__main__":
    main() 