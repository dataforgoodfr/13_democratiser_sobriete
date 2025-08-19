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
        # Level 4: Primary Indicators - Geometric mean across deciles
        # Must maintain proper hierarchical relationships
        for priority in filtered_config:
            priority_name = priority['name']
            for component in priority['components']:
                component_name = component['name']
                
                # Skip secondary indicators that have no underlying data
                if component_name in secondary_indicators_to_remove:
                    continue
                
                for indicator in component['indicators']:
                    indicator_code = indicator['code']
                    if indicator_code in df.index.get_level_values('primary_index').unique():
                        primary_values = []
                        for decile in df.index.get_level_values('decile').unique():
                            try:
                                value = df.loc[(indicator_code, country, decile), latest_year]
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
                                'EU_Priority': priority_name,  # Correct parent EU Priority
                                'Secondary_indicator': component_name,  # Correct parent Secondary indicator
                                'primary_index': indicator_code,
                                'Score': primary_aggregate,
                                'Level': '4 (Primary_indicator)'
                            })
    
    # Now calculate Level 3 "All Deciles" correctly as geometric mean of Level 3 individual decile scores
    print("Calculating Level 3 (Secondary) 'All Deciles' as geometric mean of individual decile scores...")
    
    # Get the Level 3 individual decile scores that were calculated earlier
    temp_df = pd.DataFrame(master_data)
    level3_individual = temp_df[temp_df['Level'] == '3 (Secondary_indicator)']
    
    for country in df.index.get_level_values('country').unique():
        country_level3_individual = level3_individual[level3_individual['country'] == country]
        
        for priority in filtered_config:
            priority_name = priority['name']
            for component in priority['components']:
                component_name = component['name']
                
                # Skip secondary indicators that have no underlying data
                if component_name in secondary_indicators_to_remove:
                    continue
                
                # Find all Level 3 individual decile scores for this secondary indicator
                component_scores = country_level3_individual[
                    (country_level3_individual['EU_Priority'] == priority_name) & 
                    (country_level3_individual['Secondary_indicator'] == component_name)
                ]
                
                if not component_scores.empty:
                    # Get Level 3 individual decile scores and filter out NaN values
                    individual_scores = component_scores['Score'].values
                    valid_scores = individual_scores[~pd.isna(individual_scores)]
                    
                    if len(valid_scores) > 0:
                        # Calculate Level 3 "All Deciles" as geometric mean of individual decile scores
                        level3_all_deciles_score = np.exp(np.mean(np.log(valid_scores)))
                        master_data.append({
                            'country': country,
                            'decile': 'All',
                            'year': latest_year,
                            'EU_Priority': priority_name,
                            'Secondary_indicator': component_name,
                            'primary_index': 'All',
                            'Score': level3_all_deciles_score,
                            'Level': '3 (Secondary_indicator)'
                        })
                    else:
                        print(f"Warning: No valid Level 3 individual scores for {country} - {priority_name} - {component_name}, skipping Level 3 'All Deciles' calculation")
    
    # Now calculate Level 2 (EU Priority) from Level 3 (Secondary) using arithmetic mean
    print("Calculating Level 2 (EU Priority) from Level 3 (Secondary) using arithmetic mean...")
    
    # Convert current master_data to DataFrame for easier processing
    temp_df = pd.DataFrame(master_data)
    level3_data = temp_df[temp_df['Level'] == '3 (Secondary_indicator)']
    

    
    print("Calculating Level 2 (EU Priority) from Level 3 (Secondary) using arithmetic mean...")
    
    for country in df.index.get_level_values('country').unique():
        country_level3 = level3_data[level3_data['country'] == country]
        
        # Level 2: EU Priorities - Arithmetic mean of Level 3 (Secondary) scores
        for priority in filtered_config:
            priority_name = priority['name']
            
            # Find all Level 3 indicators for this EU priority
            priority_secondaries = country_level3[country_level3['EU_Priority'] == priority_name]
            
            if not priority_secondaries.empty:
                # Get secondary scores and filter out NaN values
                secondary_scores = priority_secondaries['Score'].values
                valid_scores = secondary_scores[~pd.isna(secondary_scores)]
                
                if len(valid_scores) > 0:
                    # Arithmetic mean of valid secondary indicator scores only
                    priority_score = np.mean(valid_scores)
                    master_data.append({
                        'country': country,
                        'decile': 'All',
                        'year': latest_year,
                        'EU_Priority': priority_name,
                        'Secondary_indicator': 'All',
                        'primary_index': 'All',
                        'Score': priority_score,
                        'Level': '2 (EU_Priority)'
                    })
                else:
                    print(f"Warning: No valid Secondary scores for {country} - {priority_name}, skipping EU Priority calculation")
    
    # Update temp_df to include Level 2 data
    temp_df = pd.DataFrame(master_data)
    level2_data = temp_df[temp_df['Level'] == '2 (EU_Priority)']
    
    print("Calculating Level 1 (EWBI) from Level 2 (EU Priority) using arithmetic mean...")
    
    for country in df.index.get_level_values('country').unique():
        country_level2 = level2_data[level2_data['country'] == country]
        
        # Level 1: EWBI - Arithmetic mean of Level 2 (EU Priority) scores
        if not country_level2.empty:
            # Get EU priority scores and filter out NaN values
            eu_priority_scores = country_level2['Score'].values
            valid_scores = eu_priority_scores[~pd.isna(eu_priority_scores)]
            
            if len(valid_scores) > 0:
                # Arithmetic mean of valid EU priority scores only
                ewbi_score = np.mean(valid_scores)
                master_data.append({
                    'country': country,
                    'decile': 'All',
                    'year': latest_year,
                    'EU_Priority': 'All',
                    'Secondary_indicator': 'All',
                    'primary_index': 'All',
                    'Score': ewbi_score,
                    'Level': '1 (EWBI)'
                })
            else:
                print(f"Warning: No valid EU Priority scores for {country}, skipping EWBI calculation")
    
    # Create EU Average (comprehensive across all levels and deciles)
    print("Creating comprehensive EU Average...")
    
    # Define EU countries (same as in original ewbi_computation.py)
    eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    
    # Convert master_data to DataFrame for easier processing
    temp_df = pd.DataFrame(master_data)
    
    # Filter for EU countries only
    eu_data = temp_df[temp_df['country'].isin(eu_countries)]
    
    if not eu_data.empty:
        # Group by all combinations of decile, EU_Priority, Secondary_indicator, primary_index, Level
        # and calculate mean Score across EU countries
        grouping_cols = ['decile', 'year', 'EU_Priority', 'Secondary_indicator', 'primary_index', 'Level']
        
        eu_averages = eu_data.groupby(grouping_cols)['Score'].mean().reset_index()
        eu_averages['country'] = 'EU Average'
        
        # Convert back to list of dictionaries and add to master_data
        for _, row in eu_averages.iterrows():
            master_data.append({
                'country': row['country'],
                'decile': row['decile'], 
                'year': row['year'],
                'EU_Priority': row['EU_Priority'],
                'Secondary_indicator': row['Secondary_indicator'],
                'primary_index': row['primary_index'],
                'Score': row['Score'],
                'Level': row['Level']
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
    
    print("Using optimized approach: Level 4 deciles → roll-up to higher levels")
    
    # For each country and year, calculate hierarchically  
    for country in df.index.get_level_values('country').unique():
        for year in year_cols:
            year_int = int(year)
            
            # Step 1: Calculate Level 4 (Primary) country aggregates for this year
            # This is the only level where we need decile calculations
            level4_country_scores = {}  # {indicator_code: score}
            
            for priority in filtered_config:
                priority_name = priority['name']
                for component in priority['components']:
                    component_name = component['name']
                    
                    # Skip secondary indicators that have no underlying data
                    if component_name in secondary_indicators_to_remove:
                        continue
                    
                    for indicator in component['indicators']:
                        indicator_code = indicator['code']
                        if indicator_code in df.index.get_level_values('primary_index').unique():
                            decile_values = []
                            for decile in df.index.get_level_values('decile').unique():
                                try:
                                    value = df.loc[(indicator_code, country, decile), year]
                                    if pd.notna(value):
                                        decile_values.append(value)
                                except:
                                    continue
                            
                            if decile_values:
                                # Geometric mean across deciles (deciles → country aggregate)
                                primary_score = np.exp(np.mean(np.log(decile_values)))
                                level4_country_scores[indicator_code] = primary_score
                                
                                # Add to time series
                                time_series_data.append({
                                    'country': country,
                                    'decile': 'All Deciles',
                                    'year': year_int,
                                    'EU_Priority': priority_name,
                                    'Secondary_indicator': component_name,
                                    'primary_index': indicator_code,
                                    'Score': primary_score,
                                    'Level': '4 (Primary_indicator)'
                                })
            
            # Step 2: Calculate Level 3 (Secondary) from Level 4 country aggregates
            level3_country_scores = {}  # {(priority_name, component_name): score}
            
            for priority in filtered_config:
                priority_name = priority['name']
                for component in priority['components']:
                    component_name = component['name']
                    
                    # Skip secondary indicators that have no underlying data
                    if component_name in secondary_indicators_to_remove:
                        continue
                    
                    # Find all Level 4 scores for this secondary indicator
                    primary_scores = []
                    for indicator in component['indicators']:
                        indicator_code = indicator['code']
                        if indicator_code in level4_country_scores:
                            primary_scores.append(level4_country_scores[indicator_code])
                    
                    if primary_scores:
                        # Arithmetic mean (Level 4 → Level 3)
                        secondary_score = np.mean(primary_scores)
                        level3_country_scores[(priority_name, component_name)] = secondary_score
                        
                        # Add to time series
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
            
            # Step 3: Calculate Level 2 (EU Priority) from Level 3 country aggregates
            level2_country_scores = {}  # {priority_name: score}
            
            for priority in filtered_config:
                priority_name = priority['name']
                
                # Find all Level 3 scores for this EU priority
                secondary_scores = []
                for component in priority['components']:
                    component_name = component['name']
                    if component_name not in secondary_indicators_to_remove:
                        key = (priority_name, component_name)
                        if key in level3_country_scores:
                            secondary_scores.append(level3_country_scores[key])
                
                if secondary_scores:
                    # Arithmetic mean (Level 3 → Level 2)
                    priority_score = np.mean(secondary_scores)
                    level2_country_scores[priority_name] = priority_score
                    
                    # Add to time series
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
            
            # Step 4: Calculate Level 1 (EWBI) from Level 2 country aggregates
            if level2_country_scores:
                # Get Level 2 scores and filter out any NaN values
                level2_scores = list(level2_country_scores.values())
                valid_scores = [score for score in level2_scores if not pd.isna(score)]
                
                if valid_scores:
                    # Arithmetic mean (Level 2 → Level 1)
                    ewbi_score = np.mean(valid_scores)
                    
                    # Add to time series
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
            # Must maintain proper hierarchical relationships
            for priority in filtered_config:
                priority_name = priority['name']
                for component in priority['components']:
                    component_name = component['name']
                    
                    # Skip secondary indicators that have no underlying data
                    if component_name in secondary_indicators_to_remove:
                        continue
                    
                    for indicator in component['indicators']:
                        indicator_code = indicator['code']
                        if indicator_code in df.index.get_level_values('primary_index').unique():
                            # Aggregate across deciles for this country-year-primary_indicator combination
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
                                primary_score = np.exp(np.mean(np.log(year_values)))
                                time_series_data.append({
                                    'country': country,
                                    'decile': 'All Deciles',
                                    'year': year_int,
                                    'EU_Priority': priority_name,  # Correct parent EU Priority
                                    'Secondary_indicator': component_name,  # Correct parent Secondary indicator
                                    'primary_index': indicator_code,
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