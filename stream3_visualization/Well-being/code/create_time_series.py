import pandas as pd

print("=== Creating Time Series Data ===")

# Load the enhanced data
df = pd.read_csv('../output/primary_data_preprocessed_with_decile_aggregates.csv')

print(f"Enhanced data shape: {df.shape}")

# Create time series data
time_series_data = []

for _, row in df.iterrows():
    country = row['country']
    primary_index = row['primary_index']
    decile = row['decile']
    
    # Get all year values
    years = [col for col in row.index if str(col).isdigit()]
    
    for year in years:
        value = row[year]
        if pd.notna(value):
            time_series_data.append({
                'country': country,
                'primary_index': primary_index,
                'decile': decile,
                'year': int(year),
                'value': value
            })

time_series_df = pd.DataFrame(time_series_data)
print(f"Time series data shape: {time_series_df.shape}")

# Save time series data
time_series_df.to_csv('../output/master_dataframe_time_series_with_decile_aggregates.csv', index=False)

print("✅ Time series data created successfully!")
print(f"Shape: {time_series_df.shape}")
print(f"Years: {sorted(time_series_df['year'].unique())}")
print(f"Countries: {len(time_series_df['country'].unique())}")
print(f"Deciles: {sorted(time_series_df['decile'].unique())}") 