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
            
            for country in primary_data.index.get_level_values('country').unique():
                for decile in primary_data.index.get_level_values('decile').unique():
                    # Get data for this country and decile using MultiIndex
                    country_data = primary_data.loc[(country, slice(None), decile)]
                    
                    if not country_data.empty:
                        # Collect values for this secondary indicator's primary indicators
                        factors = []
                        for primary in primary_indicators:
                            primary_code = primary['code']
                            # Get data for this primary indicator
                            if primary_code in country_data.index.get_level_values('primary_index'):
                                primary_values = country_data.loc[primary_code]
                                
                                if not primary_values.empty:
                                    # Get the latest year value
                                    year_cols = [col for col in primary_values.index if str(col).isdigit()]
                                    if year_cols:
                                        latest_year = max(year_cols)
                                        values = primary_values[latest_year]
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

def create_master_dataframe(secondary_df, priorities_df, ewbi_df, primary_data, config):
    """
    Create the master dataframe with all indicator levels in the format expected by the dashboard
    Args:
        secondary_df: DataFrame with secondary indicator scores
        priorities_df: DataFrame with EU priority scores
        ewbi_df: DataFrame with EWBI scores
        primary_data: DataFrame with primary indicator data
        config: EWBI configuration structure
    Returns:
        DataFrame with the complete master structure
    """
    print("Creating master dataframe...")
    
    # Create indicator mappings
    primary_to_secondary = {}
    secondary_to_eu_priority = {}
    
    for eu_priority in config:
        eu_priority_name = eu_priority['name']
        for secondary in eu_priority['components']:
            secondary_name = secondary['name']
            secondary_to_eu_priority[secondary_name] = eu_priority_name
            
            for primary in secondary['indicators']:
                primary_code = primary['code']
                primary_to_secondary[primary_code] = secondary_name
    
    # Start with the base structure (country, decile combinations)
    base_df = ewbi_df[['country', 'decile']].copy()
    
    # Add EWBI scores
    master_df = base_df.merge(ewbi_df, on=['country', 'decile'], how='left')
    
    # Add EU priority scores
    eu_priorities_pivot = priorities_df.pivot_table(
        values='eu_priority_score', 
        index=['country', 'decile'], 
        columns='eu_priority', 
        aggfunc='first'
    ).reset_index()
    
    master_df = master_df.merge(eu_priorities_pivot, on=['country', 'decile'], how='left')
    
    # Add secondary indicator scores
    secondary_pivot = secondary_df.pivot_table(
        values='secondary_score', 
        index=['country', 'decile'], 
        columns=['eu_priority', 'secondary_indicator'], 
        aggfunc='first'
    ).reset_index()
    
    # Flatten the multi-level columns
    secondary_pivot.columns = ['country', 'decile'] + [
        f"{eu_priority}_{secondary_indicator}".replace(' ', '_').replace(',', '') 
        for eu_priority, secondary_indicator in secondary_pivot.columns[2:]
    ]
    
    master_df = master_df.merge(secondary_pivot, on=['country', 'decile'], how='left')
    
    # Add primary indicator scores (latest year available)
    print("Processing primary indicators...")
    
    # Get the latest year available for each country-decile combination
    year_cols = [col for col in primary_data.columns if col.isdigit()]
    latest_year = max(year_cols)
    
    # Pivot primary indicators for the latest year
    primary_latest = primary_data[['country', 'primary_index', 'decile', latest_year]].copy()
    primary_latest.columns = ['country', 'primary_index', 'decile', 'primary_score']
    
    # Map primary indicators to their corresponding secondary indicators
    primary_latest['secondary_indicator'] = primary_latest['primary_index'].map(primary_to_secondary)
    primary_latest['eu_priority'] = primary_latest['secondary_indicator'].map(secondary_to_eu_priority)
    
    # Pivot primary indicators
    primary_pivot = primary_latest.pivot_table(
        values='primary_score', 
        index=['country', 'decile'], 
        columns='primary_index', 
        aggfunc='first'
    ).reset_index()
    
    # Rename primary indicator columns to avoid confusion
    primary_pivot.columns = ['country', 'decile'] + [f'primary_{col}' for col in primary_pivot.columns[2:]]
    
    # Add primary indicator scores to master dataframe
    master_df = master_df.merge(primary_pivot, on=['country', 'decile'], how='left')
    
    # Add metadata columns
    master_df['latest_year'] = latest_year
    master_df['data_level'] = 'complete'
    
    # Reorder columns for better readability
    ewbi_cols = ['country', 'decile', 'ewbi_score', 'latest_year', 'data_level']
    eu_priority_cols = [col for col in master_df.columns if col not in ewbi_cols and not col.startswith('primary_') and not col.startswith('secondary_')]
    secondary_cols = [col for col in master_df.columns if col.startswith('secondary_')]
    primary_cols = [col for col in master_df.columns if col.startswith('primary_')]
    
    final_cols = ewbi_cols + eu_priority_cols + secondary_cols + primary_cols
    master_df = master_df[final_cols]
    
    return master_df

def create_time_series_dataframe(primary_data, primary_to_secondary, secondary_to_eu_priority):
    """
    Create time series dataframe with all 4 levels, but only country-level data (no individual deciles)
    Args:
        primary_data: DataFrame with primary indicator data
        primary_to_secondary: Mapping from primary to secondary indicators
        secondary_to_eu_priority: Mapping from secondary to EU priority indicators
    Returns:
        DataFrame with time series data for all 4 levels at country level
    """
    print("Creating time series dataframe (country-level only)...")
    
    # Get all years available
    year_cols = [col for col in primary_data.columns if col.isdigit()]
    year_cols.sort()
    
    # Create a comprehensive time series with all levels
    time_series_data = []
    
    for year in year_cols:
        year_int = int(year)
        print(f"Processing year {year_int}...")
        
        # Get data for this year
        year_data = primary_data[['country', 'primary_index', 'decile', year]].copy()
        year_data.columns = ['country', 'primary_index', 'decile', 'primary_score']
        year_data['year'] = year_int
        
        # Map to secondary and EU priority levels
        year_data['secondary_indicator'] = year_data['primary_index'].map(primary_to_secondary)
        year_data['eu_priority'] = year_data['secondary_indicator'].map(secondary_to_eu_priority)
        
        # For time series, we only want country-level aggregated data
        # So we'll aggregate across deciles for each country-primary_index-year combination
        aggregated_data = year_data.groupby(['country', 'primary_index', 'year', 'secondary_indicator', 'eu_priority']).agg({
            'primary_score': 'mean'  # Use arithmetic mean for primary indicators
        }).reset_index()
        
        # Set decile to "All Deciles" for time series
        aggregated_data['decile'] = 'All Deciles'
        
        time_series_data.append(aggregated_data)
    
    # Combine all years
    all_primary_data = pd.concat(time_series_data, ignore_index=True)
    print(f"Created {len(all_primary_data)} primary indicator records for time series")
    
    # Now create the comprehensive time series with all levels
    comprehensive_time_series = []
    
    for country in all_primary_data['country'].unique():
        for year in all_primary_data['year'].unique():
            print(f"Processing {country} - {year}...")
            
            # Get data for this country and year
            country_year_data = all_primary_data[
                (all_primary_data['country'] == country) & 
                (all_primary_data['year'] == year)
            ]
            
            if not country_year_data.empty:
                # 1. Add EWBI level (one row per country-year)
                # Calculate EWBI as average of all primary scores for this country-year
                ewbi_score = country_year_data['primary_score'].mean()
                
                comprehensive_time_series.append({
                    'country': country,
                    'decile': 'All Deciles',
                    'year': year,
                    'data_level': 'EWBI',
                    'ewbi_score': ewbi_score,
                    'eu_priority_score': np.nan,
                    'secondary_score': np.nan,
                    'primary_score': np.nan,
                    'eu_priority': '',
                    'secondary_indicator': '',
                    'primary_index': ''
                })
                
                # 2. Add EU Priority level (one row per EU priority per country-year)
                for eu_priority in country_year_data['eu_priority'].unique():
                    if pd.notna(eu_priority):  # Skip NaN values
                        # Get all primary indicators for this EU priority
                        eu_priority_data = country_year_data[country_year_data['eu_priority'] == eu_priority]
                        if not eu_priority_data.empty:
                            eu_priority_score = eu_priority_data['primary_score'].mean()
                            
                            comprehensive_time_series.append({
                                'country': country,
                                'decile': 'All Deciles',
                                'year': year,
                                'data_level': 'EU Priority',
                                'ewbi_score': np.nan,
                                'eu_priority_score': eu_priority_score,
                                'secondary_score': np.nan,
                                'primary_score': np.nan,
                                'eu_priority': eu_priority,
                                'secondary_indicator': '',
                                'primary_index': ''
                            })
                
                # 3. Add Secondary Indicator level (one row per secondary indicator per country-year)
                for secondary_indicator in country_year_data['secondary_indicator'].unique():
                    if pd.notna(secondary_indicator):  # Skip NaN values
                        # Get all primary indicators for this secondary indicator
                        secondary_data = country_year_data[country_year_data['secondary_indicator'] == secondary_indicator]
                        if not secondary_data.empty:
                            secondary_score = secondary_data['primary_score'].mean()
                            
                            # Get the corresponding EU priority
                            eu_priority = secondary_to_eu_priority.get(secondary_indicator, '')
                            
                            comprehensive_time_series.append({
                                'country': country,
                                'decile': 'All Deciles',
                                'year': year,
                                'data_level': 'Secondary Indicator',
                                'ewbi_score': np.nan,
                                'eu_priority_score': np.nan,
                                'secondary_score': secondary_score,
                                'primary_score': np.nan,
                                'eu_priority': eu_priority,
                                'secondary_indicator': secondary_indicator,
                                'primary_index': ''
                            })
                
                # 4. Add Primary Indicator level (one row per primary indicator per country-year)
                for _, row in country_year_data.iterrows():
                    comprehensive_time_series.append({
                        'country': country,
                        'decile': 'All Deciles',
                        'year': year,
                        'data_level': 'Primary Indicator',
                        'ewbi_score': np.nan,
                        'eu_priority_score': np.nan,
                        'secondary_score': np.nan,
                        'primary_score': row['primary_score'],
                        'eu_priority': row['eu_priority'],
                        'secondary_indicator': row['secondary_indicator'],
                        'primary_index': row['primary_index']
                    })
    
    # Create the final dataframe
    final_time_series_df = pd.DataFrame(comprehensive_time_series)
    
    print(f"Final comprehensive time series shape: {final_time_series_df.shape}")
    print(f"Data levels: {final_time_series_df['data_level'].value_counts().to_dict()}")
    
    return final_time_series_df

def add_eu_and_all_countries_aggregates(master_df):
    """
    Add EU Countries Average and All Countries Average to the master dataframe
    Args:
        master_df: DataFrame with country-level data
    Returns:
        DataFrame with added aggregate rows
    """
    print("Adding EU and All Countries aggregates...")
    
    # Define EU countries
    eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    
    # Check if this is a time series dataframe (has 'year' column) or master dataframe
    is_time_series = 'year' in master_df.columns
    
    if is_time_series:
        # For time series, group by year and decile
        group_cols = ['year', 'decile']
    else:
        # For master dataframe, group by decile only
        group_cols = ['decile']
    
    # EU Countries aggregate
    eu_only_df = master_df[master_df['country'].isin(eu_countries)]
    
    # Dynamically determine which numeric columns to aggregate
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove grouping columns from numeric columns
    numeric_cols = [col for col in numeric_cols if col not in group_cols]
    
    # Create aggregation dictionary
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = 'mean'
    
    eu_aggregate = eu_only_df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Add other columns that might exist
    other_cols = [col for col in master_df.columns if col not in group_cols + numeric_cols]
    for col in other_cols:
        if col in master_df.columns:
            eu_aggregate[col] = eu_only_df.groupby(group_cols)[col].first().values
    
    eu_aggregate['country'] = 'EU Countries Average'
    
    # All Countries aggregate
    all_countries_aggregate = master_df.groupby(group_cols).agg(agg_dict).reset_index()
    
    for col in other_cols:
        if col in master_df.columns:
            all_countries_aggregate[col] = master_df.groupby(group_cols)[col].first().values
    
    all_countries_aggregate['country'] = 'All Countries Average'
    
    # Add both aggregates to master dataframe
    master_df_with_aggregates = pd.concat([master_df, eu_aggregate, all_countries_aggregate], ignore_index=True)
    
    return master_df_with_aggregates

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
    
    # Convert to MultiIndex exactly as in the original notebook
    primary_data = primary_data.set_index(['country', 'primary_index', 'decile'])
    
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
    
    # Create master dataframe
    master_df = create_master_dataframe(secondary_df, priorities_df, ewbi_df, primary_data, config)
    
    # Create indicator mappings for time series
    primary_to_secondary = {}
    secondary_to_eu_priority = {}
    
    for eu_priority in config:
        eu_priority_name = eu_priority['name']
        for secondary in eu_priority['components']:
            secondary_name = secondary['name']
            secondary_to_eu_priority[secondary_name] = eu_priority_name
            
            for primary in secondary['indicators']:
                primary_code = primary['code']
                primary_to_secondary[primary_code] = secondary_name
    
    # Create time series dataframe
    time_series_df = create_time_series_dataframe(primary_data, primary_to_secondary, secondary_to_eu_priority)
    
    # Add EU and All Countries aggregates to master and time series
    master_df_with_aggregates = add_eu_and_all_countries_aggregates(master_df)
    time_series_df_with_aggregates = add_eu_and_all_countries_aggregates(time_series_df)
    
    print("=== Computation Complete ===")
    print(f"Secondary indicators (deciles only): {len(secondary_df)} scores")
    print(f"Secondary indicators (with aggregates): {len(secondary_with_aggregates)} scores")
    print(f"EU priorities (deciles only): {len(priorities_df)} scores")
    print(f"EU priorities (with aggregates): {len(priorities_with_aggregates)} scores")
    print(f"EWBI (deciles only): {len(ewbi_df)} scores")
    print(f"EWBI (with aggregates): {len(ewbi_with_aggregates)} scores")
    print(f"Master dataframe (deciles only): {len(master_df)} rows")
    print(f"Master dataframe (with aggregates): {len(master_df_with_aggregates)} rows")
    print(f"Time series dataframe (deciles only): {len(time_series_df)} rows")
    print(f"Time series dataframe (with aggregates): {len(time_series_df_with_aggregates)} rows")
    
    return (secondary_df, priorities_df, ewbi_df, 
            secondary_with_aggregates, priorities_with_aggregates, ewbi_with_aggregates,
            master_df, time_series_df, master_df_with_aggregates, time_series_df_with_aggregates)

# Main execution (only runs when script is executed directly)
if __name__ == "__main__":
    # Run the computation
    (secondary_df, priorities_df, ewbi_df, 
     secondary_with_aggregates, priorities_with_aggregates, ewbi_with_aggregates,
     master_df, time_series_df, master_df_with_aggregates, time_series_df_with_aggregates) = run_ewbi_computation()
    
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

    # Save master and time series data
    master_df.to_csv('../output/ewbi_master_deciles.csv', index=False)
    master_df_with_aggregates.to_csv('../output/ewbi_master.csv', index=False)
    time_series_df.to_csv('../output/ewbi_time_series_deciles.csv', index=False)
    time_series_df_with_aggregates.to_csv('../output/ewbi_time_series.csv', index=False)
    
    print("All results saved successfully!") 