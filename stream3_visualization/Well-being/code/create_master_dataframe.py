import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_ewbi_structure():
    """Load the EWBI indicator structure from JSON"""
    with open('../data/ewbi_indicators.json', 'r') as f:
        config = json.load(f)['EWBI']
    return config

def create_indicator_mapping(config):
    """Create mappings between different indicator levels"""
    # Map primary indicators to secondary indicators
    primary_to_secondary = {}
    # Map secondary indicators to EU priorities
    secondary_to_eu_priority = {}
    
    for eu_priority in config:
        eu_priority_name = eu_priority['name']
        for secondary in eu_priority['components']:
            secondary_name = secondary['name']
            secondary_to_eu_priority[secondary_name] = eu_priority_name
            
            for primary in secondary['indicators']:
                primary_code = primary['code']
                primary_to_secondary[primary_code] = secondary_name
    
    return primary_to_secondary, secondary_to_eu_priority

def load_and_process_data():
    """Load and process all data files"""
    print("Loading data files...")
    
    # Load EWBI results
    ewbi_df = pd.read_csv('../output/ewbi_results.csv')
    ewbi_df.columns = ['country', 'decile', 'ewbi_score']
    
    # Load EU priorities
    eu_priorities_df = pd.read_csv('../output/eu_priorities.csv')
    eu_priorities_df.columns = ['country', 'eu_priority', 'decile', 'eu_priority_score']
    
    # Load secondary indicators
    secondary_df = pd.read_csv('../output/secondary_indicators.csv')
    secondary_df.columns = ['country', 'eu_priority', 'secondary_indicator', 'decile', 'secondary_score']
    
    # Load primary indicators (this is a large file)
    print("Loading primary indicators data...")
    primary_df = pd.read_csv('../output/primary_data_preprocessed.csv')
    # The structure is: primary_index, country, decile, year1, year2, ...
    # Get actual years from the data instead of hardcoding
    actual_years = [col for col in primary_df.columns if col.isdigit()]
    actual_years.sort()
    print(f"Available years in data: {min(actual_years)} to {max(actual_years)}")
    
    # Rename columns properly
    primary_df.columns = ['primary_index', 'country', 'decile'] + actual_years
    
    return ewbi_df, eu_priorities_df, secondary_df, primary_df

def create_master_dataframe(ewbi_df, eu_priorities_df, secondary_df, primary_df, 
                           primary_to_secondary, secondary_to_eu_priority):
    """Create the master dataframe with all indicator levels"""
    print("Creating master dataframe...")
    
    # Start with the base structure (country, decile combinations)
    base_df = ewbi_df[['country', 'decile']].copy()
    
    # Add EWBI scores
    master_df = base_df.merge(ewbi_df, on=['country', 'decile'], how='left')
    
    # Add EU priority scores
    eu_priorities_pivot = eu_priorities_df.pivot_table(
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
    primary_years = [col for col in primary_df.columns if col.isdigit()]
    latest_year = max(primary_years)
    
    # Pivot primary indicators for the latest year
    primary_latest = primary_df[['country', 'primary_index', 'decile', latest_year]].copy()
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

def add_time_series_data(master_df, primary_df, primary_to_secondary, secondary_to_eu_priority):
    """Add time series data for primary indicators with EU and All Countries averages"""
    print("Adding time series data...")
    
    # Get all years available
    year_cols = [col for col in primary_df.columns if col.isdigit()]
    year_cols.sort()
    
    # Create a time series version
    time_series_data = []
    
    for year in year_cols:
        year_data = primary_df[['country', 'primary_index', 'decile', year]].copy()
        year_data.columns = ['country', 'primary_index', 'decile', 'value']
        year_data['year'] = int(year)
        
        # Map to secondary and EU priority levels
        year_data['secondary_indicator'] = year_data['primary_index'].map(primary_to_secondary)
        year_data['eu_priority'] = year_data['secondary_indicator'].map(secondary_to_eu_priority)
        
        time_series_data.append(year_data)
    
    time_series_df = pd.concat(time_series_data, ignore_index=True)
    
    # Create a separate list for additional aggregate data
    additional_data = []
    
    # Now add EU Countries Average and All Countries Average for secondary indicators
    print("Adding EU and All Countries averages for secondary indicators...")
    
    # Define EU countries
    eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    
    # Get unique secondary indicators
    secondary_indicators = time_series_df['secondary_indicator'].dropna().unique()
    
    for year in year_cols:
        year_int = int(year)
        
        # For each secondary indicator, calculate EU and All Countries averages
        for secondary_indicator in secondary_indicators:
            if pd.isna(secondary_indicator):
                continue
                
            # Get data for this secondary indicator and year
            indicator_data = time_series_df[
                (time_series_df['secondary_indicator'] == secondary_indicator) & 
                (time_series_df['year'] == year_int)
            ]
            
            if not indicator_data.empty:
                # Calculate EU Countries Average
                eu_data = indicator_data[indicator_data['country'].isin(eu_countries)]
                if not eu_data.empty:
                    # Average across countries and deciles for this secondary indicator
                    eu_avg_value = eu_data['value'].mean()
                    
                    # Add EU Countries Average record
                    additional_data.append({
                        'country': 'EU Countries Average',
                        'primary_index': None,  # Not applicable for secondary indicators
                        'decile': None,  # Not applicable for secondary indicators
                        'value': eu_avg_value,
                        'year': year_int,
                        'secondary_indicator': secondary_indicator,
                        'eu_priority': secondary_to_eu_priority.get(secondary_indicator, None)
                    })
                
                # Calculate All Countries Average
                all_avg_value = indicator_data['value'].mean()
                
                # Add All Countries Average record
                additional_data.append({
                    'country': 'All Countries Average',
                    'primary_index': None,  # Not applicable for secondary indicators
                    'decile': None,  # Not applicable for secondary indicators
                    'value': all_avg_value,
                    'year': year_int,
                        'secondary_indicator': secondary_indicator,
                        'eu_priority': secondary_to_eu_priority.get(secondary_indicator, None)
                })
    
    # Now add EU Countries Average and All Countries Average for primary indicators
    print("Adding EU and All Countries averages for primary indicators...")
    
    # Get unique primary indicators
    primary_indicators = time_series_df['primary_index'].dropna().unique()
    
    for year in year_cols:
        year_int = int(year)
        
        # For each primary indicator, calculate EU and All Countries averages
        for primary_indicator in primary_indicators:
            if pd.isna(primary_indicator):
                continue
                
            # Get data for this primary indicator and year
            indicator_data = time_series_df[
                (time_series_df['primary_index'] == primary_indicator) & 
                (time_series_df['year'] == year_int)
            ]
            
            if not indicator_data.empty:
                # Calculate EU Countries Average
                eu_data = indicator_data[indicator_data['country'].isin(eu_countries)]
                if not eu_data.empty:
                    # Average across countries and deciles for this primary indicator
                    eu_avg_value = eu_data['value'].mean()
                    
                    # Add EU Countries Average record
                    additional_data.append({
                        'country': 'EU Countries Average',
                        'primary_index': primary_indicator,
                        'decile': None,  # Not applicable for averages
                        'value': eu_avg_value,
                        'year': year_int,
                        'secondary_indicator': primary_to_secondary.get(primary_indicator, None),
                        'eu_priority': secondary_to_eu_priority.get(primary_to_secondary.get(primary_indicator, None), None)
                    })
                
                # Calculate All Countries Average
                all_avg_value = indicator_data['value'].mean()
                
                # Add All Countries Average record
                additional_data.append({
                    'country': 'All Countries Average',
                    'primary_index': primary_indicator,
                    'decile': None,  # Not applicable for averages
                    'value': all_avg_value,
                    'year': year_int,
                    'secondary_indicator': primary_to_secondary.get(primary_indicator, None),
                    'eu_priority': secondary_to_eu_priority.get(primary_to_secondary.get(primary_indicator, None), None)
                })
    
    # Convert additional data to DataFrame and concatenate with original time series
    if additional_data:
        additional_df = pd.DataFrame(additional_data)
        final_time_series_df = pd.concat([time_series_df, additional_df], ignore_index=True)
        print(f"Added {len(additional_data)} EU and All Countries average records")
    else:
        final_time_series_df = time_series_df
    
    print(f"Final time series shape: {final_time_series_df.shape}")
    
    return final_time_series_df

def calculate_historical_ewbi_scores(time_series_df, config):
    """Calculate historical EWBI scores for all countries over time"""
    print("Calculating historical EWBI scores...")
    
    # Create a mapping from primary indicators to EU priorities
    primary_to_eu_priority = {}
    for priority in config:
        for component in priority['components']:
            for indicator in component['indicators']:
                primary_to_eu_priority[indicator['code']] = priority['name']
    
    print(f"Created mapping for {len(primary_to_eu_priority)} primary indicators")
    
    # Calculate historical EWBI scores
    historical_ewbi = []
    
    for country in time_series_df['country'].unique():
        print(f"Processing {country}...")
        
        country_data = time_series_df[time_series_df['country'] == country]
        
        for year in sorted(country_data['year'].unique()):
            year_data = country_data[country_data['year'] == year]
            
            for decile in sorted(year_data['decile'].unique()):
                decile_data = year_data[year_data['decile'] == decile]
                
                # Calculate EU priority scores for this decile and year
                eu_priority_scores = {}
                
                for priority in config:
                    priority_name = priority['name']
                    priority_indicators = []
                    
                    # Get all primary indicators for this priority
                    for component in priority['components']:
                        for indicator in component['indicators']:
                            indicator_code = indicator['code']
                            if indicator_code in decile_data['primary_index'].values:
                                indicator_values = decile_data[decile_data['primary_index'] == indicator_code]['value'].values
                                if len(indicator_values) > 0:
                                    # Take the mean if multiple values exist
                                    priority_indicators.extend(indicator_values)
                    
                    if priority_indicators:
                        # Calculate the score for this priority (geometric mean)
                        eu_priority_scores[priority_name] = np.exp(np.mean(np.log(priority_indicators)))
                
                # Calculate overall EWBI score (geometric mean of EU priorities)
                if eu_priority_scores:
                    ewbi_score = np.exp(np.mean(np.log(list(eu_priority_scores.values()))))
                    
                    historical_ewbi.append({
                        'country': country,
                        'year': year,  # Use the actual year
                        'decile': decile,
                        'ewbi_score': ewbi_score,
                        **eu_priority_scores
                    })
    
    # Convert to DataFrame
    historical_ewbi_df = pd.DataFrame(historical_ewbi)
    
    print(f"Calculated {len(historical_ewbi_df)} historical EWBI records")
    
    # Create a simplified version with just country-year averages (no deciles)
    country_year_ewbi = historical_ewbi_df.groupby(['country', 'year']).agg({
        'ewbi_score': 'mean'
    }).reset_index()
    
    # Add EU Countries Average for EWBI
    eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    
    eu_ewbi = historical_ewbi_df[historical_ewbi_df['country'].isin(eu_countries)].groupby('year')['ewbi_score'].mean().reset_index()
    eu_ewbi['country'] = 'EU Countries Average'
    
    # Also add EU Countries Average for each EU priority
    eu_priority_cols = [col for col in historical_ewbi_df.columns if col not in ['country', 'year', 'decile', 'ewbi_score']]
    
    for year in sorted(historical_ewbi_df['year'].unique()):
        year_data = historical_ewbi_df[historical_ewbi_df['year'] == year]
        eu_year_data = year_data[year_data['country'].isin(eu_countries)]
        
        if not eu_year_data.empty:
            # Calculate EU average for each priority for this year
            eu_priority_averages = {}
            for priority_col in eu_priority_cols:
                eu_priority_averages[priority_col] = eu_year_data[priority_col].mean()
            
            # Add EU Countries Average record for this year
            eu_priority_averages['country'] = 'EU Countries Average'
            eu_priority_averages['year'] = year
            eu_priority_averages['ewbi_score'] = eu_ewbi[eu_ewbi['year'] == year]['ewbi_score'].iloc[0]
            
            historical_ewbi.append(eu_priority_averages)
    
    # Convert back to DataFrame and recreate the simplified version
    historical_ewbi_df = pd.DataFrame(historical_ewbi)
    
    # Recreate simplified version with EU Countries Average included
    country_year_ewbi = historical_ewbi_df.groupby(['country', 'year']).agg({
        'ewbi_score': 'mean'
    }).reset_index()
    
    # Combine individual countries and EU average
    final_ewbi = pd.concat([country_year_ewbi, eu_ewbi], ignore_index=True)
    
    print(f"Created simplified EWBI evolution with {len(final_ewbi)} records")
    
    return historical_ewbi_df, final_ewbi

def main():
    """Main function to create the master dataframe"""
    print("Starting creation of master dataframe...")
    
    # Load EWBI structure
    config = load_ewbi_structure()
    
    # Create indicator mappings
    primary_to_secondary, secondary_to_eu_priority = create_indicator_mapping(config)
    
    # Load data
    ewbi_df, eu_priorities_df, secondary_df, primary_df = load_and_process_data()
    
    # Create master dataframe
    master_df = create_master_dataframe(
        ewbi_df, eu_priorities_df, secondary_df, primary_df,
        primary_to_secondary, secondary_to_eu_priority
    )
    
    # Calculate aggregates and add to master dataframe
    print("Calculating aggregates...")
    
    # Define EU countries (excluding non-EU countries like UK, CH, NO, etc.)
    eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    
    # EU Countries aggregate
    eu_only_df = master_df[master_df['country'].isin(eu_countries)]
    eu_aggregate = eu_only_df.groupby('decile').agg({
        'ewbi_score': 'mean',
        'latest_year': 'first',
        'data_level': 'first'
    }).reset_index()
    
    # Add EU priority averages
    eu_priority_cols = [col for col in master_df.columns if col not in ['country', 'decile', 'ewbi_score', 'latest_year', 'data_level'] and not col.startswith('primary_') and not col.startswith('secondary_')]
    for col in eu_priority_cols:
        eu_aggregate[col] = eu_only_df.groupby('decile')[col].mean().values
    
    eu_aggregate['country'] = 'EU Countries Average'
    
    # All Countries aggregate
    all_countries_aggregate = master_df.groupby('decile').agg({
        'ewbi_score': 'mean',
        'latest_year': 'first',
        'data_level': 'first'
    }).reset_index()
    
    for col in eu_priority_cols:
        all_countries_aggregate[col] = master_df.groupby('decile')[col].mean().values
    
    all_countries_aggregate['country'] = 'All Countries Average'
    
    # Add primary indicator averages to EU aggregate
    primary_cols = [col for col in master_df.columns if col.startswith('primary_')]
    for col in primary_cols:
        eu_aggregate[col] = eu_only_df.groupby('decile')[col].mean().values
    
    # Add primary indicator averages to All Countries aggregate
    for col in primary_cols:
        all_countries_aggregate[col] = master_df.groupby('decile')[col].mean().values
    
    # Add both aggregates to master dataframe
    master_df_with_aggregates = pd.concat([master_df, eu_aggregate, all_countries_aggregate], ignore_index=True)
    
    # Create time series version
    time_series_df = add_time_series_data(master_df, primary_df, primary_to_secondary, secondary_to_eu_priority)
    
    # Calculate historical EWBI scores
    historical_ewbi_df, ewbi_evolution_df = calculate_historical_ewbi_scores(time_series_df, config)
    
    # Save all versions
    print("Saving master dataframe and historical EWBI data...")
    master_df_with_aggregates.to_csv('../output/master_dataframe.csv', index=False)
    time_series_df.to_csv('../output/master_dataframe_time_series.csv', index=False)
    historical_ewbi_df.to_csv('../output/historical_ewbi_scores.csv', index=False)
    ewbi_evolution_df.to_csv('../output/ewbi_evolution_over_time.csv', index=False)
    
    # Print summary statistics
    print("\n=== MASTER DATAFRAME SUMMARY ===")
    print(f"Shape: {master_df_with_aggregates.shape}")
    print(f"Countries: {master_df_with_aggregates['country'].nunique()} (including 2 aggregates)")
    print(f"Deciles: {master_df_with_aggregates['decile'].nunique()}")
    print(f"EU Priorities: {len([col for col in master_df_with_aggregates.columns if col not in ['country', 'decile', 'ewbi_score', 'latest_year', 'data_level'] and not col.startswith('primary_') and not col.startswith('secondary_')])}")
    print(f"Secondary Indicators: {len([col for col in master_df_with_aggregates.columns if col.startswith('secondary_')])}")
    print(f"Primary Indicators: {len([col for col in master_df_with_aggregates.columns if col.startswith('primary_')])}")
    
    print("\n=== TIME SERIES DATAFRAME SUMMARY ===")
    print(f"Shape: {time_series_df.shape}")
    print(f"Years: {time_series_df['year'].nunique()}")
    print(f"Countries: {time_series_df['country'].nunique()}")
    print(f"Deciles: {time_series_df['decile'].nunique()}")
    print(f"Primary Indicators: {time_series_df['primary_index'].nunique()}")
    
    print("\n=== HISTORICAL EWBI SUMMARY ===")
    print(f"Historical EWBI records: {len(historical_ewbi_df)}")
    print(f"Simplified EWBI evolution: {len(ewbi_evolution_df)}")
    print(f"Years covered: {sorted(ewbi_evolution_df['year'].unique())}")
    print(f"Countries: {sorted(ewbi_evolution_df['country'].unique())}")
    
    print("\nMaster dataframe created successfully!")
    print("Files saved:")
    print("- ../output/master_dataframe.csv (cross-sectional view)")
    print("- ../output/master_dataframe_time_series.csv (time series view)")
    print("- ../output/historical_ewbi_scores.csv (historical EWBI with deciles)")
    print("- ../output/ewbi_evolution_over_time.csv (EWBI evolution over time)")

if __name__ == "__main__":
    main() 