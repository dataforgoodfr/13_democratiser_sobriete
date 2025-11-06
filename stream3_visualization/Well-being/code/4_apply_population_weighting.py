#!/usr/bin/env python3
"""
Apply population-weighted aggregations to existing PCA output file.
This script modifies 'Median across countries' aggregations to use population weighting 
for Level 1 and Level 2 indicators only.
"""

import pandas as pd
import numpy as np
import os
from population_data_transform import get_population_weights

def compute_population_weighted_geometric_means(df, population_data, eu_countries):
    """
    Compute population-weighted geometric means for Level 1 & 2 aggregations.
    
    Two-step process:
    1. Compute population-weighted geometric mean per decile across countries 
    2. Compute "All" decile value as geometric mean across those decile values
    
    Args:
        df (pd.DataFrame): Input dataframe
        population_data (pd.DataFrame): Population data
        eu_countries (list): List of EU country codes
    
    Returns:
        pd.DataFrame: Modified dataframe with new population-weighted geometric means
    """
    print("🔄 Computing population-weighted geometric means...")
    
    df_modified = df.copy()
    new_rows = []
    
    # Get unique combinations of Level, Year, EU priority for Level 1 & 2
    level_1_2_data = df_modified[
        (df_modified['Level'].isin([1, 2])) & 
        (df_modified['Country'].isin(eu_countries)) &
        (df_modified['Decile'] != 'All') &
        (df_modified['Value'].notna())
    ]
    
    if level_1_2_data.empty:
        print("⚠️  No Level 1 & 2 individual country data found")
        return df_modified
    
    # Group by relevant dimensions
    if not level_1_2_data.empty:
        # For Level 1: Group by Year, Level, Decile
        level_1_groups = level_1_2_data[level_1_2_data['Level'] == 1].groupby(['Year', 'Level', 'Decile'])
        
        for (year, level, decile), group in level_1_groups:
            if len(group) < 2:  # Need at least 2 countries for meaningful aggregation
                continue
                
            countries = group['Country'].tolist()
            values = group['Value'].values
            
            # Compute population-weighted geometric mean for this decile
            weighted_geom_mean = compute_population_weighted_geom_mean(values, countries, year, population_data)
            
            if not np.isnan(weighted_geom_mean):
                # Create new row for this decile
                new_row = group.iloc[0].copy()
                new_row['Country'] = 'All Countries'
                new_row['Value'] = weighted_geom_mean
                new_row['Aggregation'] = 'Population-weighted geometric mean'
                new_rows.append(new_row)
        
        # For Level 2: Group by Year, Level, EU priority, Decile  
        level_2_groups = level_1_2_data[level_1_2_data['Level'] == 2].groupby(['Year', 'Level', 'EU priority', 'Decile'])
        
        for (year, level, eu_priority, decile), group in level_2_groups:
            if len(group) < 2:  # Need at least 2 countries for meaningful aggregation
                continue
                
            countries = group['Country'].tolist()
            values = group['Value'].values
            
            # Compute population-weighted geometric mean for this decile
            weighted_geom_mean = compute_population_weighted_geom_mean(values, countries, year, population_data)
            
            if not np.isnan(weighted_geom_mean):
                # Create new row for this decile
                new_row = group.iloc[0].copy()
                new_row['Country'] = 'All Countries'
                new_row['Value'] = weighted_geom_mean
                new_row['Aggregation'] = 'Population-weighted geometric mean'
                new_rows.append(new_row)
    
    # Add the new decile-specific rows
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df_modified = pd.concat([df_modified, new_df], ignore_index=True)
        print(f"✅ Added {len(new_rows)} population-weighted geometric mean rows (per decile)")
    
    # Now compute "All" decile values as geometric mean across deciles
    all_decile_rows = []
    
    # For Level 1: Compute "All" decile from decile 1-10
    pwgm_level1 = df_modified[
        (df_modified['Level'] == 1) & 
        (df_modified['Country'] == 'All Countries') &
        (df_modified['Aggregation'] == 'Population-weighted geometric mean') &
        (df_modified['Decile'] != 'All') &
        (df_modified['Value'].notna())
    ]
    
    if not pwgm_level1.empty:
        level1_all_groups = pwgm_level1.groupby(['Year', 'Level'])
        
        for (year, level), group in level1_all_groups:
            values = group['Value'].values
            if len(values) > 0:
                # Compute geometric mean across deciles
                geom_mean_all = np.exp(np.mean(np.log(values)))
                
                # Create "All" decile row
                all_row = group.iloc[0].copy()
                all_row['Decile'] = 'All'
                all_row['Value'] = geom_mean_all
                all_decile_rows.append(all_row)
    
    # For Level 2: Compute "All" decile from decile 1-10 for each EU priority
    pwgm_level2 = df_modified[
        (df_modified['Level'] == 2) & 
        (df_modified['Country'] == 'All Countries') &
        (df_modified['Aggregation'] == 'Population-weighted geometric mean') &
        (df_modified['Decile'] != 'All') &
        (df_modified['Value'].notna())
    ]
    
    if not pwgm_level2.empty:
        level2_all_groups = pwgm_level2.groupby(['Year', 'Level', 'EU priority'])
        
        for (year, level, eu_priority), group in level2_all_groups:
            values = group['Value'].values
            if len(values) > 0:
                # Compute geometric mean across deciles
                geom_mean_all = np.exp(np.mean(np.log(values)))
                
                # Create "All" decile row
                all_row = group.iloc[0].copy()
                all_row['Decile'] = 'All'
                all_row['Value'] = geom_mean_all
                all_decile_rows.append(all_row)
    
    # Add the "All" decile rows
    if all_decile_rows:
        all_df = pd.DataFrame(all_decile_rows)
        df_modified = pd.concat([df_modified, all_df], ignore_index=True)
        print(f"✅ Added {len(all_decile_rows)} 'All' decile population-weighted geometric mean rows")
    
    return df_modified

def compute_population_weighted_geom_mean(values, countries, year, population_data):
    """Compute population-weighted geometric mean for a set of values and countries."""
    try:
        if len(values) == 0 or len(values) != len(countries):
            return np.nan
            
        # Get population weights
        weights_dict = get_population_weights(countries, year, population_data)
        
        # Create weights series aligned with values
        weights = np.array([weights_dict.get(country, 0.0) for country in countries])
        
        # Remove entries with zero weights or invalid values
        valid_mask = (weights > 0) & (values > 0) & ~np.isnan(values)
        if not valid_mask.any():
            return np.exp(np.mean(np.log(values[values > 0])))  # Fallback to simple geometric mean
        
        valid_values = values[valid_mask]
        valid_weights = weights[valid_mask]
        
        # Normalize weights to sum to 1
        normalized_weights = valid_weights / valid_weights.sum()
        
        # Compute weighted geometric mean: exp(sum(w_i * log(v_i)))
        log_values = np.log(valid_values)
        weighted_log_mean = np.sum(normalized_weights * log_values)
        
        return np.exp(weighted_log_mean)
        
    except Exception as e:
        print(f"⚠️  Error computing population-weighted geometric mean for year {year}: {e}")
        if len(values[values > 0]) > 0:
            return np.exp(np.mean(np.log(values[values > 0])))  # Fallback to simple geometric mean
        return np.nan

def apply_population_weighting():
    """Apply population weighting to existing median aggregations for Level 1 & 2."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, '..', 'output', 'unified_all_levels_1_to_5_pca.csv')
    output_path = os.path.join(script_dir, '..', 'output', 'unified_all_levels_1_to_5_pca_weighted.csv')
    
    print("🔄 Loading existing PCA output file...")
    try:
        df = pd.read_csv(input_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} rows")
    except Exception as e:
        print(f"❌ Could not load input file: {e}")
        return
    
    # Load population data
    print("📊 Loading population data...")
    try:
        pop_data_path = os.path.join(script_dir, '..', 'data', 'population_transformed.csv')
        population_data = pd.read_csv(pop_data_path)
        print(f"✅ Loaded population data: {len(population_data):,} rows")
    except Exception as e:
        print(f"❌ Could not load population data: {e}")
        return
    
    # Function to compute population-weighted average
    def population_weighted_average(values, countries, year):
        """Compute population-weighted average for a given year."""
        if len(values) == 0:
            return np.nan
        
        try:
            # Convert to numpy arrays to avoid pandas indexing issues
            values_array = np.array(values)
            countries_array = np.array(countries)
            
            # Get population weights
            weights_dict = get_population_weights(countries_array.tolist(), year, population_data)
            
            # Create weights array aligned with values
            weights_array = np.array([weights_dict.get(country, 0.0) for country in countries_array])
            
            # Remove entries with zero weights
            valid_mask = weights_array > 0
            if not valid_mask.any():
                return values_array.mean()  # Fallback to simple mean
            
            valid_values = values_array[valid_mask]
            valid_weights = weights_array[valid_mask]
            
            # Normalize weights to sum to 1
            normalized_weights = valid_weights / valid_weights.sum()
            
            # Compute weighted average
            return np.sum(normalized_weights * valid_values)
        
        except Exception as e:
            print(f"⚠️  Error computing population-weighted average for year {year}: {e}")
            return np.array(values).mean()  # Fallback to simple mean
    
    # Create a copy of the dataframe
    df_modified = df.copy()
    
    # Define EU countries
    eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    
    # Find median aggregations for 'All Countries' at Level 1 and 2
    condition = (
        (df_modified['Country'] == 'All Countries') &
        (df_modified['Aggregation'] == 'Median across countries') &
        (df_modified['Level'].isin([1, 2]))
    )
    
    median_agg_rows = df_modified[condition]
    print(f"📊 Found {len(median_agg_rows)} Level 1 & 2 median aggregations to convert")
    
    if len(median_agg_rows) == 0:
        print("⚠️  No Level 1 & 2 median aggregations found for 'All Countries'")
    else:
        print(f"🔄 Computing population-weighted averages for {len(median_agg_rows)} median aggregations...")
        
        # Group the data we need to re-aggregate
        updates_made = 0
        
        for idx, row in median_agg_rows.iterrows():
            year = row['Year']
            level = row['Level']
            decile = row['Decile']
            eu_priority = row['EU priority']
            
            # Find the individual country data that was used to create this aggregation
            source_condition = (
                (df_modified['Year'] == year) &
                (df_modified['Level'] == level) &
                (df_modified['Decile'] == decile) &
                (df_modified['Country'].isin(eu_countries)) &
                (df_modified['Country'] != 'All Countries')
            )
            
            # For Level 2, also match EU priority
            if level == 2 and pd.notna(eu_priority):
                source_condition &= (df_modified['EU priority'] == eu_priority)
            
            source_data = df_modified[source_condition]
            
            if len(source_data) > 0:
                values = source_data['Value'].dropna()
                countries = source_data['Country']
                
                if len(values) > 0:
                    # Compute population-weighted average
                    weighted_avg = population_weighted_average(values, countries, year)
                    
                    if not np.isnan(weighted_avg):
                        # Update the row
                        df_modified.loc[idx, 'Value'] = weighted_avg
                        df_modified.loc[idx, 'Aggregation'] = 'Population-weighted average'
                        updates_made += 1
        
        print(f"✅ Updated {updates_made} aggregations to use population weighting")
    
    # Always compute population-weighted geometric means for Level 1 & 2
    print("🔄 Computing population-weighted geometric means for Level 1 & 2...")
    df_modified = compute_population_weighted_geometric_means(df_modified, population_data, eu_countries)
    
    # Save the modified dataframe
    print(f"💾 Saving modified data to {output_path}")
    df_modified.to_csv(output_path, index=False)
    
    # Show summary of changes
    pop_weighted_avg_count = len(df_modified[df_modified['Aggregation'] == 'Population-weighted average'])
    pop_weighted_geom_count = len(df_modified[df_modified['Aggregation'] == 'Population-weighted geometric mean'])
    median_count = len(df_modified[df_modified['Aggregation'] == 'Median across countries'])
    
    print(f"\n📊 Final counts:")
    print(f"  Population-weighted averages: {pop_weighted_avg_count:,}")
    print(f"  Population-weighted geometric means: {pop_weighted_geom_count:,}")
    print(f"  Median aggregations: {median_count:,}")
    
    # Show some examples
    if pop_weighted_geom_count > 0:
        print(f"\n📋 Sample population-weighted geometric means:")
        sample_geom = df_modified[df_modified['Aggregation'] == 'Population-weighted geometric mean'].head(5)
        cols_to_show = ['Year', 'Country', 'Level', 'EU priority', 'Decile', 'Value', 'Aggregation']
        print(sample_geom[cols_to_show].to_string(index=False))
    
    if pop_weighted_avg_count > 0:
        print(f"\n📋 Sample population-weighted averages:")
        sample_avg = df_modified[df_modified['Aggregation'] == 'Population-weighted average'].head(5)
        cols_to_show = ['Year', 'Country', 'Level', 'EU priority', 'Decile', 'Value', 'Aggregation']
        print(sample_avg[cols_to_show].to_string(index=False))

if __name__ == "__main__":
    print("🚀 Applying Population-Weighted Aggregations to PCA Output")
    print("=" * 60)
    
    apply_population_weighting()
    
    print("\n✅ Population weighting application completed!")