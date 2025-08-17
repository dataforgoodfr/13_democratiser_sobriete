#!/usr/bin/env python3
"""
Fix Time Series Data Script
This script populates the missing columns in ewbi_time_series.csv with proper data
"""

import pandas as pd
import numpy as np
import os

def fix_time_series_data():
    """Fix the time series data by populating missing columns"""
    
    # Load the existing data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    master_df = pd.read_csv(os.path.join(data_dir, 'ewbi_master.csv'))
    time_series_df = pd.read_csv(os.path.join(data_dir, 'ewbi_time_series.csv'))
    
    print(f"Loaded master dataframe: {master_df.shape}")
    print(f"Loaded time series dataframe: {time_series_df.shape}")
    
    # Load the EWBI structure to understand the hierarchy
    with open(os.path.join(data_dir, '..', 'data', 'ewbi_indicators.json'), 'r') as f:
        import json
        config = json.load(f)['EWBI']
    
    # Create mappings
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
    
    print(f"Created mappings: {len(primary_to_secondary)} primary->secondary, {len(secondary_to_eu_priority)} secondary->eu_priority")
    
    # Now create a comprehensive time series with all levels
    comprehensive_time_series = []
    
    # Get unique countries and years from the existing time series
    countries = time_series_df['country'].unique()
    years = time_series_df['year'].unique()
    
    print(f"Processing {len(countries)} countries for {len(years)} years")
    
    for country in countries:
        for year in years:
            print(f"Processing {country} - {year}...")
            
            # Get the existing EWBI score for this country-year
            existing_row = time_series_df[(time_series_df['country'] == country) & (time_series_df['year'] == year)]
            if existing_row.empty:
                continue
                
            ewbi_score = existing_row['ewbi_score'].iloc[0]
            
            # 1. Add EWBI level (one row per country-year)
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
            for eu_priority in config:
                eu_priority_name = eu_priority['name']
                
                # Get the EU priority score from the master dataframe
                if eu_priority_name in master_df.columns:
                    # Find the row for this country in the master dataframe
                    country_master_data = master_df[master_df['country'] == country]
                    if not country_master_data.empty:
                        eu_priority_score = country_master_data[eu_priority_name].iloc[0]
                        
                        comprehensive_time_series.append({
                            'country': country,
                            'decile': 'All Deciles',
                            'year': year,
                            'data_level': 'EU Priority',
                            'ewbi_score': np.nan,
                            'eu_priority_score': eu_priority_score,
                            'secondary_score': np.nan,
                            'primary_score': np.nan,
                            'eu_priority': eu_priority_name,
                            'secondary_indicator': '',
                            'primary_index': ''
                        })
            
            # 3. Add Secondary Indicator level (one row per secondary indicator per country-year)
            for eu_priority in config:
                eu_priority_name = eu_priority['name']
                for secondary in eu_priority['components']:
                    secondary_name = secondary['name']
                    
                    # Get the secondary indicator score from the master dataframe
                    secondary_col = f"{eu_priority_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}_{secondary_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}"
                    
                    if secondary_col in master_df.columns:
                        country_master_data = master_df[master_df['country'] == country]
                        if not country_master_data.empty:
                            secondary_score = country_master_data[secondary_col].iloc[0]
                            
                            comprehensive_time_series.append({
                                'country': country,
                                'decile': 'All Deciles',
                                'year': year,
                                'data_level': 'Secondary Indicator',
                                'ewbi_score': np.nan,
                                'eu_priority_score': np.nan,
                                'secondary_score': secondary_score,
                                'primary_score': np.nan,
                                'eu_priority': eu_priority_name,
                                'secondary_indicator': secondary_name,
                                'primary_index': ''
                            })
            
            # 4. Add Primary Indicator level (one row per primary indicator per country-year)
            # Get primary indicator columns from master dataframe
            primary_cols = [col for col in master_df.columns if col.startswith('primary_')]
            
            for primary_col in primary_cols:
                primary_index = primary_col
                
                # Find the corresponding secondary indicator and EU priority
                secondary_indicator = primary_to_secondary.get(primary_index, '')
                eu_priority = secondary_to_eu_priority.get(secondary_indicator, '')
                
                if secondary_indicator and eu_priority:
                    country_master_data = master_df[master_df['country'] == country]
                    if not country_master_data.empty:
                        primary_score = country_master_data[primary_col].iloc[0]
                        
                        comprehensive_time_series.append({
                            'country': country,
                            'decile': 'All Deciles',
                            'year': year,
                            'data_level': 'Primary Indicator',
                            'ewbi_score': np.nan,
                            'eu_priority_score': np.nan,
                            'secondary_score': np.nan,
                            'primary_score': primary_score,
                            'eu_priority': eu_priority,
                            'secondary_indicator': secondary_indicator,
                            'primary_index': primary_index
                        })
    
    # Create the final dataframe
    final_time_series_df = pd.DataFrame(comprehensive_time_series)
    
    print(f"Final comprehensive time series shape: {final_time_series_df.shape}")
    print(f"Data levels: {final_time_series_df['data_level'].value_counts().to_dict()}")
    
    # Save the fixed time series data
    output_path = os.path.join(data_dir, 'ewbi_time_series_fixed.csv')
    final_time_series_df.to_csv(output_path, index=False)
    print(f"Saved fixed time series to: {output_path}")
    
    # Also save a backup of the original
    backup_path = os.path.join(data_dir, 'ewbi_time_series_backup.csv')
    time_series_df.to_csv(backup_path, index=False)
    print(f"Saved backup of original to: {backup_path}")
    
    return final_time_series_df

if __name__ == "__main__":
    fixed_df = fix_time_series_data()
    print("Time series data fixed successfully!") 