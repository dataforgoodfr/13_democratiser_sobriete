import pandas as pd
import numpy as np
import os

# Check if we can access the raw EU-SILC data to understand the source of zeros
base_path = r"C:\Users\valentin.stuhlfauth\OneDrive - univ-lyon2.fr\Bureau\git-Data4Good\13_democratiser_sobriete\stream3_visualization\Well-being\output\0_raw_data_EUROSTAT\0_EU-SILC"

try:
    # Check the final merged household data
    final_merged_path = os.path.join(base_path, "3_final_merged_df", "EU_SILC_household_final_merged.csv")
    
    print("Checking raw data availability...")
    
    # Try to load larger chunks to find Germany data
    chunk_size = 100000
    
    # Read the first chunk to understand the structure
    df_chunk = pd.read_csv(final_merged_path, nrows=chunk_size)
    
    print("Columns in merged dataset:")
    print(df_chunk.columns.tolist())
    
    print(f"\nTotal records in sample: {len(df_chunk)}")
    print(f"Years available: {sorted(df_chunk['HB010'].unique())}")
    print(f"Countries available: {sorted(df_chunk['HB020'].unique())}")
    
    print("\nChecking HS021 (Behind on Utility Bills) values for early years...")
    
    # Check years 2004-2010 for all countries to see pattern
    early_years = df_chunk[df_chunk['HB010'].isin([2004, 2005, 2006, 2007, 2008, 2009, 2010])]
    
    if 'HS021' in df_chunk.columns and len(early_years) > 0:
        print(f"\nHS021 value distribution in early years ({len(early_years)} records):")
        print(early_years['HS021'].value_counts(dropna=False))
        
        print(f"\nHS021 values by year:")
        for year in sorted(early_years['HB010'].unique()):
            year_data = early_years[early_years['HB010'] == year]
            non_null = year_data['HS021'].notna().sum()
            total = len(year_data)
            print(f"  {year}: {non_null}/{total} non-null values")
            if non_null > 0:
                print(f"    Values: {year_data['HS021'].value_counts().to_dict()}")
    
    print("\nChecking overcrowded indicator for problem years...")
    
    # Check for missing overcrowded data
    problem_data = df_chunk[
        ((df_chunk['HB010'].isin([2015, 2016, 2017, 2018, 2019])) & (df_chunk['HB020'] == 'DE')) |
        ((df_chunk['HB010'].isin([2022, 2023])) & (df_chunk['HB020'] == 'DK'))
    ]
    
    if 'overcrowded' in df_chunk.columns and len(problem_data) > 0:
        print(f"\nOvercrowded values for problem cases ({len(problem_data)} records):")
        print(problem_data.groupby(['HB010', 'HB020'])['overcrowded'].value_counts(dropna=False))
    else:
        print(f"Found {len(problem_data)} records for problem cases")
    
except Exception as e:
    print(f"Error reading data: {e}")
    print("Trying to check if files exist...")
    
    files_to_check = [
        os.path.join(base_path, "3_final_merged_df", "EU_SILC_household_final_merged.csv"),
        os.path.join(base_path, "1_income_decile", "EU_SILC_household_data_with_decile.csv"),
        os.path.join(base_path, "2_overcrowding", "EU_SILC_household_data_with_overcrowding.csv")
    ]
    
    for file_path in files_to_check:
        exists = os.path.exists(file_path)
        if exists:
            size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"✓ {os.path.basename(file_path)}: {size:.1f} MB")
        else:
            print(f"✗ {os.path.basename(file_path)}: Not found")