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
        data: list of tuples (values, weight) where values is a pandas Series or numpy float
    Returns:
        pandas Series with the average values, or float if all inputs are floats
    """
    all_values = []
    for values, weight in data:
        all_values.append(values)
    
    # Check if all values are numeric (not Series)
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in all_values):
        # All values are numeric, return simple average
        return np.mean(all_values)
    
    # Some values are Series, concatenate and calculate mean
    if all_values:
        combined = pd.concat(all_values, axis=1)
        return combined.mean(axis=1)
    else:
        return pd.Series(dtype=float)

def geometric_mean(data):
    """
    Calculate geometric mean of indicators (treating all values equally)
    Args:
        data: list of tuples (values, weight) where values is a pandas Series
    Returns:
        pandas Series with the geometric mean values
    """
    if not data:
        return pd.Series(dtype=float)
    
    # Extract values (ignore weights, treat all equally)
    values_list = [item[0] for item in data]
    
    # Calculate geometric mean
    # For each position, calculate: exp(mean(log(values)))
    result = pd.Series(index=values_list[0].index, dtype=float)
    
    for idx in result.index:
        # Get values at this index from all series
        idx_values = [series.loc[idx] for series in values_list]
        
        # Filter out non-positive values (geometric mean requires positive numbers)
        positive_values = [v for v in idx_values if v > 0]
        
        if positive_values:
            # Calculate geometric mean: exp(mean(log(values)))
            log_values = np.log(positive_values)
            geometric_mean_val = np.exp(np.mean(log_values))
            result.loc[idx] = geometric_mean_val
        else:
            result.loc[idx] = np.nan
    
    return result

def calculate_secondary_indicators(primary_data, config):
    """
    Calculate secondary indicators from primary indicators using arithmetic mean
    Args:
        primary_data: DataFrame with primary indicator data
        config: EWBI configuration structure
    Returns:
        DataFrame with secondary indicator scores
    """
    print("Calculating secondary indicators...")
    
    secondary_data = []
    
    for eu_priority in config:
        eu_priority_name = eu_priority['name']
        for secondary in eu_priority['components']:
            secondary_name = secondary['name']
            primary_indicators = secondary['indicators']
            
            print(f"Processing {eu_priority_name} - {secondary_name} ({len(primary_indicators)} indicators)")
            
            for country in primary_data['country'].unique():
                for decile in primary_data['decile'].unique():
                    # Get data for this country and decile
                    country_data = primary_data[
                        (primary_data['country'] == country) & 
                        (primary_data['decile'] == decile)
                    ]
                    
                    if not country_data.empty:
                        # Collect values for this secondary indicator's primary indicators
                        factors = []
                        for primary in primary_indicators:
                            primary_code = primary['code']
                            primary_values = country_data[country_data['primary_index'] == primary_code]
                            
                            if not primary_values.empty:
                                # Get the latest year value
                                year_cols = [col for col in primary_values.columns if col.isdigit()]
                                if year_cols:
                                    latest_year = max(year_cols)
                                    values = primary_values[latest_year].iloc[0]
                                    weight = primary.get('weight', 1.0)
                                    factors.append((values, weight))
                        
                        if factors:
                            # Calculate secondary indicator score using arithmetic mean
                            secondary_score = simple_average(factors)
                            if not pd.isna(secondary_score):
                                secondary_data.append({
                                    'country': country,
                                    'eu_priority': eu_priority_name,
                                    'secondary_indicator': secondary_name,
                                    'decile': decile,
                                    'secondary_score': secondary_score
                                })
    
    secondary_df = pd.DataFrame(secondary_data)
    print(f"Created {len(secondary_df)} secondary indicator scores")
    return secondary_df

def calculate_eu_priorities(secondary_df, config):
    """
    Calculate EU priorities from secondary indicators using arithmetic mean
    Args:
        secondary_df: DataFrame with secondary indicator scores
        config: EWBI configuration structure
    Returns:
        DataFrame with EU priority scores
    """
    print("Calculating EU priorities...")
    
    priorities_data = []
    
    for eu_priority in config:
        eu_priority_name = eu_priority['name']
        secondary_indicators = [comp['name'] for comp in eu_priority['components']]
        
        print(f"Processing EU priority: {eu_priority_name} ({len(secondary_indicators)} components)")
        
        for country in secondary_df['country'].unique():
            for decile in secondary_df['decile'].unique():
                # Get secondary indicator scores for this EU priority
                eu_data = secondary_df[
                    (secondary_df['country'] == country) & 
                    (secondary_df['decile'] == decile) & 
                    (secondary_df['eu_priority'] == eu_priority_name)
                ]
                
                if not eu_data.empty:
                    # Collect values for this EU priority's secondary indicators
                    factors = []
                    for _, row in eu_data.iterrows():
                        values = row['secondary_score']
                        weight = 1.0  # Equal weight for all secondary indicators
                        factors.append((values, weight))
                    
                    if factors:
                        # Calculate EU priority score using arithmetic mean
                        eu_priority_score = simple_average(factors)
                        if not pd.isna(eu_priority_score):
                            priorities_data.append({
                                'country': country,
                                'eu_priority': eu_priority_name,
                                'decile': decile,
                                'eu_priority_score': eu_priority_score
                            })
    
    priorities_df = pd.DataFrame(priorities_data)
    print(f"Created {len(priorities_df)} EU priority scores")
    return priorities_df

def calculate_ewbi(priorities_df, config):
    """
    Calculate EWBI scores from EU priorities using arithmetic mean
    Args:
        priorities_df: DataFrame with EU priority scores
        config: EWBI configuration structure
    Returns:
        DataFrame with EWBI scores
    """
    print("Calculating EWBI scores...")
    
    ewbi_data = []
    
    for country in priorities_df['country'].unique():
        for decile in priorities_df['decile'].unique():
            # Get EU priority scores for this country and decile
            country_data = priorities_df[
                (priorities_df['country'] == country) & 
                (priorities_df['decile'] == decile)
            ]
            
            if not country_data.empty:
                # Collect values for all EU priorities
                factors = []
                for _, row in country_data.iterrows():
                    values = row['eu_priority_score']
                    weight = 1.0  # Equal weight for all EU priorities
                    factors.append((values, weight))
                
                if factors:
                    # Calculate EWBI score using arithmetic mean
                    ewbi_score = simple_average(factors)
                    if not pd.isna(ewbi_score):
                        ewbi_data.append({
                            'country': country,
                            'decile': decile,
                            'ewbi_score': ewbi_score
                        })
    
    ewbi_df = pd.DataFrame(ewbi_data)
    print(f"Created {len(ewbi_df)} EWBI scores")
    return ewbi_df

def create_all_deciles_aggregates(secondary_df, priorities_df, ewbi_df):
    """
    Create "All Deciles" aggregates using geometric mean across deciles for all indicator levels
    Args:
        secondary_df: DataFrame with secondary indicator scores by decile
        priorities_df: DataFrame with EU priority scores by decile
        ewbi_df: DataFrame with EWBI scores by decile
    Returns:
        Tuple of DataFrames with "All Deciles" aggregates
    """
    print("Creating 'All Deciles' aggregates using geometric mean...")
    
    def geometric_mean_across_deciles(df, value_col):
        """Calculate geometric mean across deciles for each country-indicator combination"""
        aggregates = []
        
        # Group by all columns except decile and the value column
        group_cols = [col for col in df.columns if col not in ['decile', value_col]]
        
        for name, group in df.groupby(group_cols):
            if len(group) > 0:
                # Get values for this group
                values = group[value_col].values
                # Filter out non-positive values (geometric mean requires positive numbers)
                positive_values = [v for v in values if isinstance(v, (int, float)) and v > 0]
                
                if positive_values:
                    # Calculate geometric mean
                    geometric_mean = np.exp(np.mean(np.log(positive_values)))
                    
                    # Create record with identifier values
                    if isinstance(name, tuple):
                        record = dict(zip(group_cols, name))
                    else:
                        record = {group_cols[0]: name}
                    
                    record['decile'] = 'All Deciles'
                    record[value_col] = geometric_mean
                    aggregates.append(record)
        
        return pd.DataFrame(aggregates)
    
    # Create aggregates for each level
    secondary_aggregates = geometric_mean_across_deciles(secondary_df, 'secondary_score')
    priorities_aggregates = geometric_mean_across_deciles(priorities_df, 'eu_priority_score')
    ewbi_aggregates = geometric_mean_across_deciles(ewbi_df, 'ewbi_score')
    
    print(f"Created {len(secondary_aggregates)} secondary indicator aggregates")
    print(f"Created {len(priorities_aggregates)} EU priority aggregates")
    print(f"Created {len(ewbi_aggregates)} EWBI aggregates")
    
    return secondary_aggregates, priorities_aggregates, ewbi_aggregates

def run_ewbi_computation():
    """
    Main function to run the complete EWBI computation
    Returns:
        Tuple of (secondary_df, priorities_df, ewbi_df, secondary_with_aggregates, priorities_with_aggregates, ewbi_with_aggregates)
    """
    print("=== Starting EWBI Computation ===")
    
    # Load primary indicator data
    print("Loading primary indicator data...")
    primary_data = pd.read_csv('../output/primary_data_preprocessed.csv')
    
    # Load EWBI structure
    with open('../data/ewbi_indicators.json', 'r') as f:
        config = json.load(f)['EWBI']
    
    print(f"Loaded {len(primary_data)} primary indicator records")
    print(f"EWBI structure has {len(config)} EU priorities")
    
    # Calculate all indicator levels
    secondary_df = calculate_secondary_indicators(primary_data, config)
    priorities_df = calculate_eu_priorities(secondary_df, config)
    ewbi_df = calculate_ewbi(priorities_df, config)
    
    # Create "All Deciles" aggregates
    secondary_aggregates, priorities_aggregates, ewbi_aggregates = create_all_deciles_aggregates(
        secondary_df, priorities_df, ewbi_df
    )
    
    # Combine decile-level data with aggregates
    secondary_with_aggregates = pd.concat([secondary_df.reset_index(), secondary_aggregates], ignore_index=True)
    priorities_with_aggregates = pd.concat([priorities_df.reset_index(), priorities_aggregates], ignore_index=True)
    ewbi_with_aggregates = pd.concat([ewbi_df.reset_index(), ewbi_aggregates], ignore_index=True)
    
    print("=== Computation Complete ===")
    print(f"Secondary indicators (deciles only): {len(secondary_df)} scores")
    print(f"Secondary indicators (with aggregates): {len(secondary_with_aggregates)} scores")
    print(f"EU priorities (deciles only): {len(priorities_df)} scores")
    print(f"EU priorities (with aggregates): {len(priorities_with_aggregates)} scores")
    print(f"EWBI (deciles only): {len(ewbi_df)} scores")
    print(f"EWBI (with aggregates): {len(ewbi_with_aggregates)} scores")
    
    return (secondary_df, priorities_df, ewbi_df, 
            secondary_with_aggregates, priorities_with_aggregates, ewbi_with_aggregates)

# Main execution (only runs when script is executed directly)
if __name__ == "__main__":
    # Run the computation
    (secondary_df, priorities_df, ewbi_df, 
     secondary_with_aggregates, priorities_with_aggregates, ewbi_with_aggregates) = run_ewbi_computation()
    
    # Save results
    print("Saving results...")
    
    # Convert Series to DataFrames for proper CSV output
    secondary_df_csv = secondary_df.reset_index()
    priorities_df_csv = priorities_df.reset_index()
    ewbi_df_csv = ewbi_df.reset_index()
    
    # Save decile-level data
    secondary_df_csv.to_csv('../output/secondary_indicators_deciles.csv', index=False)
    priorities_df_csv.to_csv('../output/eu_priorities_deciles.csv', index=False)
    ewbi_df_csv.to_csv('../output/ewbi_results_deciles.csv', index=False)
    
    # Save data with aggregates
    secondary_with_aggregates.to_csv('../output/secondary_indicators.csv', index=False)
    priorities_with_aggregates.to_csv('../output/eu_priorities.csv', index=False)
    ewbi_with_aggregates.to_csv('../output/ewbi_results.csv', index=False)
    
    print("All results saved successfully!") 