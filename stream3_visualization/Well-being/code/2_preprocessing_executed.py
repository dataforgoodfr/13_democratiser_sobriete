#!/usr/bin/env python
# coding: utf-8

# ## Computation of EWBI and wellbeing sub-indicators

# In[1]:


import pandas as pd
import numpy as np
import os

# Configure pandas to suppress FutureWarnings about downcasting
pd.set_option('future.no_silent_downcasting', True)

# In[2]:


# Build path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the unified dataframe from 1_final_df.py output which contains Level 5 indicators
unified_data_path = os.path.join(script_dir, '..', 'output', '0_raw_data_EUROSTAT', '1_final_df', 'unified_app_data.csv')

print("🚀 Loading unified Level 5 data from 1_final_df.py...")
print(f"📂 Data path: {unified_data_path}")

# Load the unified dataframe with proper column structure
df_unified = pd.read_csv(unified_data_path)
print(f"✅ Loaded unified data: {len(df_unified):,} rows, {len(df_unified.columns)} columns")
print(f"📊 Columns: {list(df_unified.columns)}")

# For backward compatibility, also create the old structure if needed
# Convert unified structure to old structure for existing processing
df = df_unified.rename(columns={
    'Year': 'year',
    'Country': 'country', 
    'Decile': 'decile',
    'Quintile': 'quintile',
    'Primary and raw data': 'primary_index',
    'Value': 'value'
}).copy()

# Add datasource column from the original unified data
if 'datasource' in df_unified.columns:
    df['datasource'] = df_unified['datasource']
elif 'database' in df_unified.columns:
    df['datasource'] = df_unified['database']

print(f"✅ Converted to processing format: {len(df)} rows")

# Filter Level 5 data to exclude aggregated data that shouldn't be normalized
print("\n🔍 Filtering Level 5 data for Level 4 processing...")
print(f"Before filtering: {len(df):,} rows")

# Exclude "All Countries" data and any other aggregated data
# Include EHIS data even if decile is NaN (they have quintile data instead)
level5_for_processing = df[
    (df['country'] != 'All Countries') &  # Exclude multi-country aggregations
    (df['decile'] != 'All') &  # Exclude total population data
    (df['quintile'] != 'All') &  # Exclude total population quintile data
    (df['decile'].astype(str) != 'All') &  # Ensure string 'All' is also excluded
    (df['quintile'].astype(str) != 'All') &  # Ensure string 'All' is also excluded
    ((df['decile'].notna()) | (df['quintile'].notna()))  # Must have either decile or quintile values
].copy()

print(f"After filtering: {len(level5_for_processing):,} rows")
print(f"Unique countries: {level5_for_processing['country'].nunique()}")
print(f"Unique deciles: {sorted([str(d) for d in level5_for_processing['decile'].unique()])}")
print(f"Unique quintiles: {sorted([str(q) for q in level5_for_processing['quintile'].unique()])}")
print(f"Unique indicators: {level5_for_processing['primary_index'].nunique()}")

# Keep original full dataset for final stacking
df_original = df_unified.copy()

# Use filtered data for Level 4 processing
df = level5_for_processing

print(f"✅ Ready for Level 4 processing: {len(df)} rows")

# Completeness analysis (optional)
import pandas as pd
import os

# Completeness
output_dir = os.path.join(script_dir, '..', 'output')
os.makedirs(output_dir, exist_ok=True)
excel_path = os.path.join(output_dir, 'completeness.xlsx')

with pd.ExcelWriter(excel_path) as writer:
    for primary_index in df['primary_index'].unique():
        sub = df[df['primary_index'] == primary_index]
        # Create a pivot table: index=country, columns=year, values=number of unique deciles
        completeness = sub.groupby(['country', 'year'])['decile'].nunique().unstack(fill_value=0)
        # Ensure all countries and years are present
        all_countries = df['country'].unique()
        all_years = df['year'].unique()
        completeness = completeness.reindex(index=all_countries, columns=sorted(all_years), fill_value=0)
        completeness.to_excel(writer, sheet_name=str(primary_index)[:31])  # Excel sheet names max 31 chars


# ## Preprocessing
# ### Data cleaning

def create_provisional_database_for_normalization(df):
    """
    Create provisional database (1) for Level 4 normalization:
    - Filter only non-"All" values from 1_final_df.py output
    - Convert EHIS quintiles to deciles (quintile 1 -> deciles 1&2, etc.)
    - Set quintile values to NaN after conversion
    """
    print("\n🔄 Creating provisional database (1) for Level 4 normalization...")
    print(f"Initial data from 1_final_df.py: {len(df):,}")
    
    # Step 1: Filter for non-"All" values only
    print("Step 1: Filtering for non-'All' values only...")
    provisional_db = df[
        (df['decile'] != 'All') & 
        (df['quintile'] != 'All') &
        (df['decile'].notna() | df['quintile'].notna())  # Must have either decile or quintile
    ].copy()
    
    print(f"After filtering non-'All' values: {len(provisional_db):,}")
    
    # Step 2: Process EHIS quintile to decile conversion
    print("Step 2: Converting EHIS quintiles to deciles...")
    
    # Convert decile column to numeric for processing
    provisional_db['decile'] = pd.to_numeric(provisional_db['decile'], errors='coerce')
    provisional_db['quintile'] = pd.to_numeric(provisional_db['quintile'], errors='coerce')
    
    # Identify EHIS quintile data that needs conversion
    if 'datasource' in provisional_db.columns:
        ehis_quintile_mask = (
            (provisional_db['datasource'] == 'EHIS') & 
            (provisional_db['quintile'].notna()) & 
            (provisional_db['quintile'].isin([1.0, 2.0, 3.0, 4.0, 5.0]))
        )
        
        ehis_quintile_data = provisional_db[ehis_quintile_mask].copy()
        non_ehis_data = provisional_db[~ehis_quintile_mask].copy()
        
        print(f"EHIS quintile data to convert: {len(ehis_quintile_data):,}")
        print(f"Other data (already with deciles): {len(non_ehis_data):,}")
        
        if len(ehis_quintile_data) > 0:
            # Convert EHIS quintiles to deciles
            # Quintile 1 -> Deciles 1&2, Quintile 2 -> Deciles 3&4, etc.
            ehis_converted = []
            
            for _, row in ehis_quintile_data.iterrows():
                quintile_val = row['quintile']
                
                # Create two decile rows for each quintile
                decile_1 = (quintile_val * 2) - 1  # Q1->D1, Q2->D3, Q3->D5, etc.
                decile_2 = quintile_val * 2        # Q1->D2, Q2->D4, Q3->D6, etc.
                
                for decile in [decile_1, decile_2]:
                    new_row = row.copy()
                    new_row['decile'] = decile
                    new_row['quintile'] = pd.NA  # Set quintile to NaN after conversion
                    ehis_converted.append(new_row)
            
            # Convert to DataFrame
            ehis_converted_df = pd.DataFrame(ehis_converted)
            print(f"EHIS data after quintile->decile conversion: {len(ehis_converted_df):,}")
            
            # Combine converted EHIS data with other data
            provisional_db_final = pd.concat([non_ehis_data, ehis_converted_df], ignore_index=True)
        else:
            print("No EHIS quintile data found to convert")
            provisional_db_final = provisional_db.copy()
    else:
        print("Warning: 'datasource' column not found - cannot identify EHIS data")
        provisional_db_final = provisional_db.copy()
    
    # Step 3: Ensure quintile values are NaN for all data in provisional database
    print("Step 3: Setting all quintile values to NaN in provisional database...")
    provisional_db_final['quintile'] = pd.NA
    
    # Step 4: Ensure only valid deciles (1-10) remain
    valid_deciles = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    provisional_db_final = provisional_db_final[
        provisional_db_final['decile'].isin(valid_deciles)
    ].copy()
    
    print(f"Final provisional database (1): {len(provisional_db_final):,}")
    print(f"Decile values in provisional DB: {sorted(provisional_db_final['decile'].unique())}")
    print(f"Quintile values in provisional DB: {provisional_db_final['quintile'].unique()}")  # Should be all NaN
    
    return provisional_db_final

# In[3]:

# Create provisional database (1) for Level 4 normalization
# Keep datasource column needed to identify EHIS data for conversion
provisional_db = create_provisional_database_for_normalization(df)


# In[4]:


# Use provisional database for Level 4 processing
df_for_level4 = provisional_db.copy()

# Check if value column is already numeric or needs string processing
if df_for_level4['value'].dtype == 'object':
    df_for_level4['value'] = df_for_level4['value'].str.replace(',', '.') # some commas appear as decile separators
    df_for_level4['value'] = df_for_level4['value'].astype(float)
else:
    # Value column is already numeric
    print(f"Value column is already numeric with dtype: {df_for_level4['value'].dtype}")

# Ensure year is integer
df_for_level4['year'] = df_for_level4['year'].astype(int)
print(f"Years in provisional database: {sorted(df_for_level4['year'].unique())}")

# Drop datasource column after processing
df_for_level4 = df_for_level4.drop(columns=['datasource'], errors='ignore')





# ### Prepare provisional database for Level 4 normalization processing
# The provisional database contains non-"All" values with EHIS quintiles converted to deciles

print("\n📊 Using provisional database with converted EHIS deciles for Level 4 normalization...")

# IMPORTANT FIX: Use the provisional database that has EHIS quintiles converted to deciles
# instead of loading from file which still has EHIS data with quintiles
level5_data_for_normalization = df_for_level4.copy()

print(f"✅ Using provisional database with {len(level5_data_for_normalization):,} records for normalization")
print("   This includes EHIS data with quintiles properly converted to deciles")


# ### Fill missing values and prepare for Level 4 normalization
# The EU JRC methodology tells us to fill missing values (NaNs) for each indicator using forward fill, 
# and if absent the next available one. This is preferred to ignoring indicators for the years they're not available.

def forward_fill_missing_data(df):
    """
    Forward fill missing data according to EU JRC methodology:
    - For each indicator/country/decile combination, fill missing years with forward fill
    - If no previous data exists, use backward fill
    - Ensure complete data coverage until the maximum year
    """
    print("\n🔄 Applying forward fill to complete missing data...")
    print(f"Input data: {len(df):,} records")
    
    # Convert year to int to avoid float comparison issues
    df = df.copy()
    df['year'] = df['year'].astype(int)
    
    # Get all unique years as a complete sequence
    all_years = sorted(df['year'].unique())
    max_year = max(all_years)
    min_year = min(all_years)
    complete_year_range = list(range(min_year, max_year + 1))
    
    print(f"Year range: {min_year} to {max_year} ({len(complete_year_range)} years)")
    print(f"Creating complete timeline with forward fill...")
    
    # Use pandas groupby and apply for efficiency
    def fill_group(group):
        # Get the first year where this indicator has data
        first_data_year = group['year'].min()
        
        # Create a complete year index for this group
        complete_index = pd.DataFrame({'year': complete_year_range})
        
        # Merge with existing data
        merged = complete_index.merge(group, on='year', how='left')
        
        # Forward fill ONLY from the first data point onward (JRC methodology)
        # Keep NaN values for years before first data point
        merged.loc[merged['year'] >= first_data_year] = merged.loc[merged['year'] >= first_data_year].ffill()
        
        # Remove rows that are still NaN AFTER the last available year
        # But preserve NaN rows BEFORE the first available year
        last_data_year = group['year'].max()
        mask = (merged['year'] < first_data_year) | (merged['year'] <= last_data_year) | merged['value'].notna()
        
        return merged[mask]
    
    # Group by the key columns and apply forward fill
    print("Applying forward fill by group...")
    completed_df = df.groupby(['primary_index', 'country', 'decile']).apply(fill_group).reset_index(drop=True)
    
    print(f"✅ Forward fill completed: {len(completed_df):,} records")
    print(f"Original records: {len(df):,}")
    print(f"Added records: {len(completed_df) - len(df):,}")
    
    return completed_df

print("\n🔄 Processing Level 5 data for Level 4 normalization...")
print("Creating pivot table for normalization processing...")

# Apply forward fill to complete missing data before normalization
level5_data_for_normalization = forward_fill_missing_data(level5_data_for_normalization)

# Create Level 4 normalization: 1:1 relationship with Level 5 data
print("\n🎯 Creating Level 4 indicators via per-decile normalization...")
print("Applying per-decile, per-year normalization for Level 4 indicators...")

# The normalization should be intra-decile, intra-indicator, AND intra-year
# This creates a 1:1 relationship between Level 5 and Level 4 records
scaled_min = 0.1
normalized_records = []

# Group by year, indicator, and decile for proper normalization
for (year, indicator, decile), group_data in level5_data_for_normalization.groupby(['year', 'primary_index', 'decile']):
    # Extract values for normalization
    values = group_data['value'].values
    
    # Check if all values are NaN (no data available for this year/indicator/decile)
    if np.all(np.isnan(values)):
        # Skip normalization entirely - don't create Level 4 records when no Level 5 data exists
        continue
    
    # Count valid (non-NaN) values
    valid_values = values[~np.isnan(values)]
    if len(valid_values) < 2:
        # If insufficient valid data for normalization, skip this group
        # Don't create Level 4 records when there's only 0 or 1 valid data points
        continue
    
    # Z-score normalization: (value - mean) / std, reversed
    # Reversed because higher values often indicate worse outcomes in social indicators
    # Use nanmean/nanstd to handle mixed NaN/data cases properly
    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    
    if std_val > 0 and not np.isnan(std_val):
        normalized_values = -1 * (values - mean_val) / std_val
    else:
        # If all non-NaN values are the same, use middle value for non-NaN, keep NaN as NaN
        normalized_values = np.where(np.isnan(values), np.nan, 0.5)
    
    # Handle infinite values but preserve NaN
    normalized_values = np.where(np.isinf(normalized_values), 0, normalized_values)
    
    # Scale between scaled_min and 1, but only scale non-NaN values
    if len(normalized_values) > 0:
        # Only consider non-NaN values for scaling
        finite_values = normalized_values[~np.isnan(normalized_values)]
        if len(finite_values) > 0:
            min_norm = np.min(finite_values)
            max_norm = np.max(finite_values)
            
            if max_norm > min_norm:
                # Scale only the non-NaN values
                scaled_values = scaled_min + (finite_values - min_norm) * (1 - scaled_min) / (max_norm - min_norm)
                # Put scaled values back, keeping NaN where they were
                normalized_values = np.where(np.isnan(normalized_values), np.nan, 
                                           scaled_min + (normalized_values - min_norm) * (1 - scaled_min) / (max_norm - min_norm))
            else:
                # All finite values are the same, set them to 0.5, keep NaN as NaN
                normalized_values = np.where(np.isnan(normalized_values), np.nan, 0.5)
    
    # Create normalized records
    for i, (_, row) in enumerate(group_data.iterrows()):
        normalized_row = row.copy()
        normalized_row['value'] = normalized_values[i]
        normalized_records.append(normalized_row)

# Convert to DataFrame
if normalized_records:
    level4_data = pd.DataFrame(normalized_records)
    print(f"✅ Created Level 4 normalized data: {len(level4_data):,} records (1:1 with provisional database)")
else:
    print("⚠️ No data available for Level 4 normalization")
    level4_data = pd.DataFrame()

# Verify the 1:1 relationship
if not level4_data.empty:
    print(f"Level 5 provisional records: {len(level5_data_for_normalization):,}")
    print(f"Level 4 normalized records: {len(level4_data):,}")
    if len(level4_data) == len(level5_data_for_normalization):
        print("✅ Confirmed: 1:1 relationship between Level 5 and Level 4")
    else:
        print("⚠️ Warning: Not a 1:1 relationship!")

level4_data


# ### Create Level 4 indicators and combine with Level 5 data in unified structure

print("\n🔗 Creating Level 4 indicators and stacking with Level 5 data...")

def create_level4_unified_structure_direct(level4_data, original_unified_df):
    """
    Convert Level 4 normalized data to unified structure and combine with original Level 5 data.
    
    Args:
        level4_data: Normalized Level 4 data in long format (1:1 with provisional database)
        original_unified_df: Original unified dataframe from 1_final_df.py (includes "All" values)
        
    Returns:
        Combined dataframe with both Level 4 (normalized) and Level 5 (original) data
    """
    print("Converting Level 4 normalized data to unified structure...")
    
    if not level4_data.empty:
        # Create Level 4 indicators with proper metadata
        level4_unified = level4_data.copy()
        level4_unified = level4_unified.rename(columns={
            'year': 'Year',
            'country': 'Country', 
            'decile': 'Decile',
            'quintile': 'Quintile',
            'primary_index': 'Primary and raw data',
            'value': 'Value'
        })
        
        # Add Level 4 metadata as specified
        level4_unified['Level'] = 4
        level4_unified['Type'] = 'Aggregation'
        level4_unified['Aggregation'] = 'Normalization level-1'
        level4_unified['EU priority'] = pd.NA  # To be determined in next phases
        level4_unified['Secondary'] = pd.NA   # To be determined in next phases
        
        # Filter out any 'All' values that shouldn't be in Level 4 data
        level4_unified = level4_unified[level4_unified['Decile'] != 'All'].copy()
        
        # Convert Decile to numeric first, handling any remaining string values
        level4_unified['Decile'] = pd.to_numeric(level4_unified['Decile'], errors='coerce')
        
        # Add quintile mapping from deciles if quintile is NaN
        def decile_to_quintile(decile_val):
            if pd.isna(decile_val):
                return pd.NA
            elif decile_val in [1.0, 2.0]:
                return 1.0
            elif decile_val in [3.0, 4.0]:
                return 2.0
            elif decile_val in [5.0, 6.0]:
                return 3.0
            elif decile_val in [7.0, 8.0]:
                return 4.0
            elif decile_val in [9.0, 10.0]:
                return 5.0
            else:
                return pd.NA
        
        # Only update quintile if it's NaN and decile is valid
        mask = (level4_unified['Quintile'].isna()) & (level4_unified['Decile'].notna())
        level4_unified.loc[mask, 'Quintile'] = level4_unified.loc[mask, 'Decile'].apply(decile_to_quintile)
        
        # Add datasource column (use original datasource info)
        if 'datasource' in level4_data.columns:
            level4_unified['datasource'] = level4_data['datasource']
        else:
            level4_unified['datasource'] = pd.NA
        
        print(f"Created {len(level4_unified):,} Level 4 indicator records")
        
        # Ensure column order matches original unified structure
        unified_columns = ['Year', 'Country', 'Decile', 'Quintile', 'Level', 'EU priority', 
                          'Secondary', 'Primary and raw data', 'Type', 'Aggregation', 'Value', 'datasource']
        level4_unified = level4_unified[unified_columns]
        
    else:
        print("No Level 4 data to process")
        level4_unified = pd.DataFrame()
    
    # Combine Level 4 with original Level 5 data
    print("Combining Level 4 and Level 5 data...")
    
    if not level4_unified.empty:
        combined_data = pd.concat([original_unified_df, level4_unified], ignore_index=True)
        print(f"✅ Combined data: {len(original_unified_df):,} Level 5 + {len(level4_unified):,} Level 4 = {len(combined_data):,} total")
    else:
        combined_data = original_unified_df
        print(f"Only Level 5 data: {len(combined_data):,} records")
    
    return combined_data, level4_unified

# Create combined Level 4 + Level 5 dataset
if 'level4_data' in locals() and not level4_data.empty:
    final_combined_data, level4_data_unified = create_level4_unified_structure_direct(level4_data, df_original)
else:
    print("No Level 4 data created - using only Level 5 data")
    final_combined_data = df_original
    level4_data_unified = pd.DataFrame()

print(f"\n📊 Final Combined Dataset Summary:")
print(f"  Total records: {len(final_combined_data):,}")
if not level4_data_unified.empty:
    print(f"  Level 4 (normalized): {len(level4_data_unified):,}")
    print(f"  Level 5 (raw): {len(df_original):,}")
else:
    print(f"  Level 5 (raw): {len(final_combined_data):,}")

# Build output paths
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, '..', 'output')

# Save the combined Level 4 + Level 5 unified data
final_unified_path = os.path.join(output_dir, 'level4_level5_unified_data.csv')
final_combined_data.to_csv(final_unified_path, index=False)
print(f"✅ Saved combined Level 4 + Level 5 unified data to: {final_unified_path}")

# Also save Level 4 normalized data in wide format for compatibility (skip this for direct approach)
if 'level4_data' in locals() and not level4_data.empty:
    level4_wide_path = os.path.join(output_dir, 'level4_normalized_wide.csv')  
    # Create wide format from long format for compatibility
    level4_wide = level4_data.pivot_table(values='value', index=['primary_index', 'decile', 'country'], columns='year')
    level4_wide.to_csv(level4_wide_path)
    print(f"✅ Saved Level 4 normalized data (wide format) to: {level4_wide_path}")

# Save Level 4 data separately for reference
if not level4_data_unified.empty:
    level4_only_path = os.path.join(output_dir, 'level4_indicators_only.csv')
    level4_data_unified.to_csv(level4_only_path, index=False)
    print(f"✅ Saved Level 4 indicators only to: {level4_only_path}")

print("\n🎉 Level 4 indicator processing completed successfully!")




# In[ ]:




