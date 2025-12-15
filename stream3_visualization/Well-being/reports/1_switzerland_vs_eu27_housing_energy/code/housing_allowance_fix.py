import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Test the housing allowance data processing
def test_housing_allowance():
    # Load the dataset
    df = pd.read_excel('../external_data/ocde_Share of households receiving a housing allowance.xlsx')
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("First few rows:")
    print(df.head())
    
    # Fix the column mapping
    df['country'] = df['Unnamed: 0']
    
    # Check if Switzerland is in the data
    print(f"\nAll countries: {sorted(df['country'].tolist())}")
    
    if 'Switzerland' in df['country'].values:
        print("✓ Switzerland is included")
        swiss_data = df[df['country'] == 'Switzerland']
        print(f"Switzerland data: {swiss_data.to_dict('records')}")
    else:
        print("⚠ Switzerland is not in the data")
    
    return df

if __name__ == "__main__":
    test_housing_allowance()