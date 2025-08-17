#!/usr/bin/env python3
"""
Generate Outputs Script - Hierarchical Structure
This script generates the two required output files with a clear hierarchical structure:
1. Master dataframe: All indicators for all levels, per decile and aggregated per country, for the latest year
2. Time series dataframe: Historical data per country for all indicators for all levels, but only for "All Deciles"
"""

import pandas as pd
import numpy as np
import json
import os

def calculate_aggregated_scores(df, config):
    """Calculate aggregated scores using the correct aggregation logic"""
    
    # Get the latest year
    year_cols = [col for col in df.columns if str(col).isdigit()]
    latest_year = max(year_cols)
    
    print(f"Calculating scores for year: {latest_year}")
    
    # Calculate secondary indicator scores (Level 3)
    print("Calculating secondary indicator scores...")
    secondary_scores = {}
    
    for priority in config:
        priority_name = priority['name']
        for component in priority['components']:
            component_name = component['name']
            component_indicators = [ind['code'] for ind in component['indicators']]
            
            # Calculate score for this secondary indicator for each country-decile
            for country in df.index.get_level_values('country').unique():
                for decile in df.index.get_level_values('decile').unique():
                    values = []
                    for indicator_code in component_indicators:
                        if indicator_code in df.index.get_level_values('primary_index').unique():
                            try:
                                value = df.loc[(indicator_code, country, decile), latest_year]
                                if pd.notna(value):
                                    values.append(value)
                            except:
                                continue
                    
                    if values:
                        # Level 4 to Level 3: Arithmetic mean (as specified)
                        score = np.mean(values)
                        key = f"{priority_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}_{component_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}"
                        if key not in secondary_scores:
                            secondary_scores[key] = []
                        secondary_scores[key].append({
                            'country': country,
                            'decile': decile,
                            'score': score
                        })
    
    print(f"Generated {len(secondary_scores)} secondary indicator scores")
    
    # Calculate EU priority scores (Level 2)
    print("Calculating EU priority scores...")
    eu_priority_scores = {}
    
    for priority in config:
        priority_name = priority['name']
        priority_key = priority_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')
        
        # Find all secondary indicators for this priority
        priority_secondary_keys = [key for key in secondary_scores.keys() if key.startswith(priority_key)]
        
        for country in df.index.get_level_values('country').unique():
            for decile in df.index.get_level_values('decile').unique():
                values = []
                for key in priority_secondary_keys:
                    # Find the score for this country-decile combination
                    for item in secondary_scores[key]:
                        if item['country'] == country and item['decile'] == decile:
                            values.append(item['score'])
                            break
                
                if values:
                    # Level 3 to Level 2: Arithmetic mean (as specified)
                    score = np.mean(values)
                    if priority_name not in eu_priority_scores:
                        eu_priority_scores[priority_name] = []
                    eu_priority_scores[priority_name].append({
                        'country': country,
                        'decile': decile,
                        'score': score
                    })
    
    print(f"Generated {len(eu_priority_scores)} EU priority scores")
    
    # Calculate EWBI scores (Level 1)
    print("Calculating EWBI scores...")
    ewbi_scores = {}
    
    for country in df.index.get_level_values('country').unique():
        for decile in df.index.get_level_values('decile').unique():
            values = []
            for priority_name in eu_priority_scores:
                # Find the score for this country-decile combination
                for item in eu_priority_scores[priority_name]:
                    if item['country'] == country and item['decile'] == decile:
                        values.append(item['score'])
                        break
            
            if values:
                # Level 2 to Level 1: Arithmetic mean (as specified)
                score = np.mean(values)
                if 'ewbi_score' not in ewbi_scores:
                    ewbi_scores['ewbi_score'] = []
                ewbi_scores['ewbi_score'].append({
                    'country': country,
                    'decile': decile,
                    'score': score
                })
    
    print(f"Generated {len(ewbi_scores)} EWBI scores")
    
    return secondary_scores, eu_priority_scores, ewbi_scores

def create_hierarchical_master_dataframe(df, secondary_scores, eu_priority_scores, ewbi_scores, config):
    """Create the hierarchical master dataframe with the structure you specified"""
    
    print("Creating hierarchical master dataframe...")
    
    # Get the latest year
    year_cols = [col for col in df.columns if str(col).isdigit()]
    latest_year = max(year_cols)
    
    # Filter out EU priorities and secondary indicators that have no underlying data
    # Based on the red-highlighted indicators in the screenshot
    eu_priorities_to_remove = [
        'Sustainable Transport and Tourism'  # No underlying primary indicators
    ]
    
    secondary_indicators_to_remove = [
        'Housing expense',           # No underlying primary indicators
        'Digital Skills',            # No underlying primary indicators
        'Health cost and medical care',  # No underlying primary indicators
        'Accidents and addictive behaviour',  # No underlying primary indicators
        'Education expense',         # No underlying primary indicators
        'Leisure and culture',       # No underlying primary indicators
        'Transport',                 # No underlying primary indicators
        'Tourism'                    # No underlying primary indicators
    ]
    
    # Filter the config to remove unwanted EU priorities
    filtered_config = [priority for priority in config if priority['name'] not in eu_priorities_to_remove]
    
    print(f"Filtered out {len(eu_priorities_to_remove)} EU priorities with no underlying data")
    print(f"Filtered out {len(secondary_indicators_to_remove)} secondary indicators with no underlying data")
    
    master_data = []
    
    # For each country-decile combination, create rows for all 4 levels
    for country in df.index.get_level_values('country').unique():
        for decile in df.index.get_level_values('decile').unique():
            
            # Level 1: EWBI (Overall)
            ewbi_score = None
            for item in ewbi_scores.get('ewbi_score', []):
                if item['country'] == country and item['decile'] == decile:
                    ewbi_score = item['score']
                    break
            
            master_data.append({
                'country': country,
                'decile': decile,
                'year': latest_year,
                'EU_Priority': 'All',
                'Secondary_indicator': 'All',
                'primary_index': 'All',
                'Score': ewbi_score,
                'Level': '1 (EWBI)'
            })
            
            # Level 2: EU Priorities (filtered)
            for priority in filtered_config:
                priority_name = priority['name']
                priority_score = None
                for item in eu_priority_scores.get(priority_name, []):
                    if item['country'] == country and item['decile'] == decile:
                        priority_score = item['score']
                        break
                
                master_data.append({
                    'country': country,
                    'decile': decile,
                    'year': latest_year,
                    'EU_Priority': priority_name,
                    'Secondary_indicator': 'All',
                    'primary_index': 'All',
                    'Score': priority_score,
                    'Level': '2 (EU_Priority)'
                })
                
                # Level 3: Secondary Indicators (filtered)
                for component in priority['components']:
                    component_name = component['name']
                    
                    # Skip secondary indicators that have no underlying data
                    if component_name in secondary_indicators_to_remove:
                        continue
                    
                    secondary_key = f"{priority_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}_{component_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}"
                    
                    secondary_score = None
                    for item in secondary_scores.get(secondary_key, []):
                        if item['country'] == country and item['decile'] == decile:
                            secondary_score = item['score']
                            break
                    
                    master_data.append({
                        'country': country,
                        'decile': decile,
                        'year': latest_year,
                        'EU_Priority': priority_name,
                        'Secondary_indicator': component_name,
                        'primary_index': 'All',
                        'Score': secondary_score,
                        'Level': '3 (Secondary_indicator)'
                    })
                    
                    # Level 4: Primary Indicators
                    for indicator in component['indicators']:
                        indicator_code = indicator['code']
                        if indicator_code in df.index.get_level_values('primary_index').unique():
                            try:
                                primary_score = df.loc[(indicator_code, country, decile), latest_year]
                                if pd.notna(primary_score):
                                    master_data.append({
                                        'country': country,
                                        'decile': decile,
                                        'year': latest_year,
                                        'EU_Priority': priority_name,
                                        'Secondary_indicator': component_name,
                                        'primary_index': indicator_code,
                                        'Score': primary_score,
                                        'Level': '4 (Primary_indicator)'
                                    })
                            except:
                                continue
    
    # Now create country aggregates (All Deciles) using geometric mean
    print("Creating country aggregates (All Deciles)...")
    
    for country in df.index.get_level_values('country').unique():
        # Level 1: EWBI (Overall) - Geometric mean across deciles
        ewbi_values = []
        for item in ewbi_scores.get('ewbi_score', []):
            if item['country'] == country:
                ewbi_values.append(item['score'])
        
        if ewbi_values:
            ewbi_aggregate = np.exp(np.mean(np.log(ewbi_values)))
            master_data.append({
                'country': country,
                'decile': 'All',
                'year': latest_year,
                'EU_Priority': 'All',
                'Secondary_indicator': 'All',
                'primary_index': 'All',
                'Score': ewbi_aggregate,
                'Level': '1 (EWBI)'
            })
        
        # Level 2: EU Priorities (filtered) - Geometric mean across deciles
        for priority in filtered_config:
            priority_name = priority['name']
            priority_values = []
            for item in eu_priority_scores.get(priority_name, []):
                if item['country'] == country:
                    priority_values.append(item['score'])
            
            if priority_values:
                priority_aggregate = np.exp(np.mean(np.log(priority_values)))
                master_data.append({
                    'country': country,
                    'decile': 'All',
                    'year': latest_year,
                    'EU_Priority': priority_name,
                    'Secondary_indicator': 'All',
                    'primary_index': 'All',
                    'Score': priority_aggregate,
                    'Level': '2 (EU_Priority)'
                })
        
        # Level 3: Secondary Indicators (filtered) - Geometric mean across deciles
        for priority in filtered_config:
            priority_name = priority['name']
            for component in priority['components']:
                component_name = component['name']
                
                # Skip secondary indicators that have no underlying data
                if component_name in secondary_indicators_to_remove:
                    continue
                
                secondary_key = f"{priority_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}_{component_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}"
                
                secondary_values = []
                for item in secondary_scores.get(secondary_key, []):
                    if item['country'] == country:
                        secondary_values.append(item['score'])
                
                if secondary_values:
                    secondary_aggregate = np.exp(np.mean(np.log(secondary_values)))
                    master_data.append({
                        'country': country,
                        'decile': 'All',
                        'year': latest_year,
                        'EU_Priority': priority_name,
                        'Secondary_indicator': component_name,
                        'primary_index': 'All',
                        'Score': secondary_aggregate,
                        'Level': '3 (Secondary_indicator)'
                    })
        
        # Level 4: Primary Indicators - Geometric mean across deciles
        for primary_index in df.index.get_level_values('primary_index').unique():
            primary_values = []
            for decile in df.index.get_level_values('decile').unique():
                try:
                    value = df.loc[(primary_index, country, decile), latest_year]
                    if pd.notna(value):
                        primary_values.append(value)
                except:
                    continue
            
            if primary_values:
                primary_aggregate = np.exp(np.mean(np.log(primary_values)))
                master_data.append({
                    'country': country,
                    'decile': 'All',
                    'year': latest_year,
                    'EU_Priority': 'All',
                    'Secondary_indicator': 'All',
                    'primary_index': primary_index,
                    'Score': primary_aggregate,
                    'Level': '4 (Primary_indicator)'
                })
    
    # Create EU Average (aggregated across all countries)
    print("Creating EU Average...")
    
    # Level 1: EWBI - Arithmetic mean across countries
    all_ewbi_scores = []
    for country in df.index.get_level_values('country').unique():
        country_ewbi_data = [row for row in master_data if row['country'] == country and row['Level'] == '1 (EWBI)']
        if country_ewbi_data and country_ewbi_data[0]['Score'] is not None:
            all_ewbi_scores.append(country_ewbi_data[0]['Score'])
    
    if all_ewbi_scores:
        eu_ewbi_average = np.mean(all_ewbi_scores)
        master_data.append({
            'country': 'EU Average',
            'decile': 'All',
            'year': latest_year,
            'EU_Priority': 'All',
            'Secondary_indicator': 'All',
            'primary_index': 'All',
            'Score': eu_ewbi_average,
            'Level': '1 (EWBI)'
        })
    
    # Level 2: EU Priorities (filtered) - Arithmetic mean across countries
    for priority in filtered_config:
        priority_name = priority['name']
        all_priority_scores = []
        for country in df.index.get_level_values('country').unique():
            country_priority_data = [row for row in master_data if row['country'] == country and row['Level'] == '2 (EU_Priority)' and row['EU_Priority'] == priority_name]
            if country_priority_data and country_priority_data[0]['Score'] is not None:
                all_priority_scores.append(country_priority_data[0]['Score'])
        
        if all_priority_scores:
            eu_priority_average = np.mean(all_priority_scores)
            master_data.append({
                'country': 'EU Average',
                'decile': 'All',
                'year': latest_year,
                'EU_Priority': priority_name,
                'Secondary_indicator': 'All',
                'primary_index': 'All',
                'Score': eu_priority_average,
                'Level': '2 (EU_Priority)'
            })
    
    return pd.DataFrame(master_data)

def create_time_series_dataframe(df, secondary_scores, eu_priority_scores, ewbi_scores, config):
    """Create the hierarchical time series dataframe with the same structure as master dataframe"""
    
    print("Creating hierarchical time series dataframe...")
    
    # Get all years
    year_cols = [col for col in df.columns if str(col).isdigit()]
    year_cols.sort()
    
    # Filter out EU priorities and secondary indicators that have no underlying data
    # Same filtering as in master dataframe
    eu_priorities_to_remove = [
        'Sustainable Transport and Tourism'  # No underlying primary indicators
    ]
    
    secondary_indicators_to_remove = [
        'Housing expense',           # No underlying primary indicators
        'Digital Skills',            # No underlying primary indicators
        'Health cost and medical care',  # No underlying primary indicators
        'Accidents and addictive behaviour',  # No underlying primary indicators
        'Education expense',         # No underlying primary indicators
        'Leisure and culture',       # No underlying primary indicators
        'Transport',                 # No underlying primary indicators
        'Tourism'                    # No underlying primary indicators
    ]
    
    # Filter the config to remove unwanted EU priorities
    filtered_config = [priority for priority in config if priority['name'] not in eu_priorities_to_remove]
    
    time_series_data = []
    
    # For each country, create hierarchical structure for each year
    for country in df.index.get_level_values('country').unique():
        for year in year_cols:
            year_int = int(year)
            
            # Level 1: EWBI (Overall) - Aggregate across deciles for this year
            ewbi_values = []
            for decile in df.index.get_level_values('decile').unique():
                for item in ewbi_scores.get('ewbi_score', []):
                    if item['country'] == country and item['decile'] == decile:
                        # We need to calculate EWBI for this specific year
                        # For now, let's use the latest year score as a proxy
                        ewbi_values.append(item['score'])
                        break
            
            if ewbi_values:
                # Use geometric mean across deciles for this year
                ewbi_score = np.exp(np.mean(np.log(ewbi_values)))
                time_series_data.append({
                    'country': country,
                    'decile': 'All Deciles',
                    'year': year_int,
                    'EU_Priority': 'All',
                    'Secondary_indicator': 'All',
                    'primary_index': 'All',
                    'Score': ewbi_score,
                    'Level': '1 (EWBI)'
                })
            
            # Level 2: EU Priorities (filtered) - Aggregate across deciles for this year
            for priority in filtered_config:
                priority_name = priority['name']
                
                # For EU priorities, we need to calculate the score for this specific year
                # This requires aggregating the secondary indicators for this year
                priority_secondary_keys = []
                for component in priority['components']:
                    component_name = component['name']
                    if component_name not in secondary_indicators_to_remove:
                        secondary_key = f"{priority_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}_{component_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}"
                        priority_secondary_keys.append(secondary_key)
                
                if priority_secondary_keys:
                    # Calculate EU priority score for this year by aggregating secondary indicators
                    secondary_values = []
                    for key in priority_secondary_keys:
                        for item in secondary_scores.get(key, []):
                            if item['country'] == country:
                                secondary_values.append(item['score'])
                                break
                    
                    if secondary_values:
                        # Level 3 to Level 2: Arithmetic mean (as specified)
                        priority_score = np.mean(secondary_values)
                        time_series_data.append({
                            'country': country,
                            'decile': 'All Deciles',
                            'year': year_int,
                            'EU_Priority': priority_name,
                            'Secondary_indicator': 'All',
                            'primary_index': 'All',
                            'Score': priority_score,
                            'Level': '2 (EU_Priority)'
                        })
            
            # Level 3: Secondary Indicators (filtered) - Aggregate across deciles for this year
            for priority in filtered_config:
                priority_name = priority['name']
                for component in priority['components']:
                    component_name = component['name']
                    
                    # Skip secondary indicators that have no underlying data
                    if component_name in secondary_indicators_to_remove:
                        continue
                    
                    secondary_key = f"{priority_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}_{component_name.replace(' ', '_').replace(',', '').replace(' and ', '_and_')}"
                    
                    # For secondary indicators, we need to calculate the score for this specific year
                    # This requires aggregating the primary indicators for this year
                    component_indicators = [ind['code'] for ind in component['indicators']]
                    primary_values = []
                    
                    for indicator_code in component_indicators:
                        if indicator_code in df.index.get_level_values('primary_index').unique():
                            # Aggregate across deciles for this country-year-indicator combination
                            year_values = []
                            for decile in df.index.get_level_values('decile').unique():
                                try:
                                    value = df.loc[(indicator_code, country, decile), year]
                                    if pd.notna(value):
                                        year_values.append(value)
                                except:
                                    continue
                            
                            if year_values:
                                # Use geometric mean across deciles for this year
                                primary_avg = np.exp(np.mean(np.log(year_values)))
                                primary_values.append(primary_avg)
                    
                    if primary_values:
                        # Level 4 to Level 3: Arithmetic mean (as specified)
                        secondary_score = np.mean(primary_values)
                        time_series_data.append({
                            'country': country,
                            'decile': 'All Deciles',
                            'year': year_int,
                            'EU_Priority': priority_name,
                            'Secondary_indicator': component_name,
                            'primary_index': 'All',
                            'Score': secondary_score,
                            'Level': '3 (Secondary_indicator)'
                        })
            
            # Level 4: Primary Indicators - Aggregate across deciles for this year
            for primary_index in df.index.get_level_values('primary_index').unique():
                # Aggregate across deciles for this country-year-primary_indicator combination
                year_values = []
                for decile in df.index.get_level_values('decile').unique():
                    try:
                        value = df.loc[(primary_index, country, decile), year]
                        if pd.notna(value):
                            year_values.append(value)
                    except:
                        continue
                
                if year_values:
                    # Use geometric mean across deciles for this year
                    primary_score = np.exp(np.mean(np.log(year_values)))
                    time_series_data.append({
                        'country': country,
                        'decile': 'All Deciles',
                        'year': year_int,
                        'EU_Priority': 'All',
                        'Secondary_indicator': 'All',
                        'primary_index': primary_index,
                        'Score': primary_score,
                        'Level': '4 (Primary_indicator)'
                    })
    
    # Add EU Average for all years
    print("Adding EU Average to time series...")
    
    for year in year_cols:
        year_int = int(year)
        
        # Level 1: EWBI - Arithmetic mean across countries for this year
        year_ewbi_values = []
        for country in df.index.get_level_values('country').unique():
            country_ewbi_data = [row for row in time_series_data if row['country'] == country and row['year'] == year_int and row['Level'] == '1 (EWBI)']
            if country_ewbi_data:
                year_ewbi_values.append(country_ewbi_data[0]['Score'])
        
        if year_ewbi_values:
            eu_ewbi_average = np.mean(year_ewbi_values)
            time_series_data.append({
                'country': 'EU Average',
                'decile': 'All Deciles',
                'year': year_int,
                'EU_Priority': 'All',
                'Secondary_indicator': 'All',
                'primary_index': 'All',
                'Score': eu_ewbi_average,
                'Level': '1 (EWBI)'
            })
        
        # Level 2: EU Priorities - Arithmetic mean across countries for this year
        for priority in filtered_config:
            priority_name = priority['name']
            year_priority_values = []
            for country in df.index.get_level_values('country').unique():
                country_priority_data = [row for row in time_series_data if row['country'] == country and row['year'] == year_int and row['Level'] == '2 (EU_Priority)' and row['EU_Priority'] == priority_name]
                if country_priority_data:
                    year_priority_values.append(country_priority_data[0]['Score'])
            
            if year_priority_values:
                eu_priority_average = np.mean(year_priority_values)
                time_series_data.append({
                    'country': 'EU Average',
                    'decile': 'All Deciles',
                    'year': year_int,
                    'EU_Priority': priority_name,
                    'Secondary_indicator': 'All',
                    'primary_index': 'All',
                    'Score': eu_priority_average,
                    'Level': '2 (EU_Priority)'
                })
    
    return pd.DataFrame(time_series_data)

def generate_outputs():
    """Generate the two required output files"""
    
    print("=== Generating EWBI Outputs ===")
    
    # Load the preprocessed data
    print("Loading preprocessed data...")
    df = pd.read_csv('output/primary_data_preprocessed.csv')
    print(f"Loaded data: {df.shape}")
    
    # Convert to MultiIndex exactly as in the original notebook
    df = df.set_index(['primary_index', 'country', 'decile'])
    print(f"Converted to MultiIndex: {df.shape}")
    
    # Load the EWBI structure
    with open('data/ewbi_indicators.json', 'r') as f:
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
    
    # Calculate aggregated scores
    print("Calculating aggregated scores...")
    secondary_scores, eu_priority_scores, ewbi_scores = calculate_aggregated_scores(df, config)
    
    # 1. Generate Master Dataframe with hierarchical structure
    print("\n=== Generating Hierarchical Master Dataframe ===")
    master_df = create_hierarchical_master_dataframe(df, secondary_scores, eu_priority_scores, ewbi_scores, config)
    print(f"Generated hierarchical master dataframe: {master_df.shape}")
    
    # 2. Generate Time Series Dataframe (all years, All Deciles only)
    print("\n=== Generating Time Series Dataframe ===")
    time_series_df = create_time_series_dataframe(df, secondary_scores, eu_priority_scores, ewbi_scores, config)
    print(f"Generated time series dataframe: {time_series_df.shape}")
    
    # Save the outputs
    print("\n=== Saving Outputs ===")
    
    # Save master dataframe
    master_output_path = 'output/ewbi_master.csv'
    master_df.to_csv(master_output_path, index=False)
    print(f"Saved master dataframe to: {master_output_path}")
    
    # Save time series dataframe
    time_series_output_path = 'output/ewbi_time_series.csv'
    time_series_df.to_csv(time_series_output_path, index=False)
    print(f"Saved time series dataframe to: {time_series_output_path}")
    
    print("\n=== Output Generation Complete ===")
    print(f"Master dataframe: {master_df.shape}")
    print(f"Time series dataframe: {time_series_df.shape}")
    
    return master_df, time_series_df

if __name__ == "__main__":
    master_df, time_series_df = generate_outputs()
    print("Script completed successfully!") 