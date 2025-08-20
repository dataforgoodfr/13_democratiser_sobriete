#!/usr/bin/env python3
"""
Test script for the CO2 decomposition preprocessor
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import CO2DecompositionPreprocessor

def test_preprocessing():
    """Test the preprocessing functionality"""
    
    print("=== Testing CO2 Decomposition Preprocessor ===\n")
    
    # Initialize preprocessor
    preprocessor = CO2DecompositionPreprocessor()
    
    # Check if data files exist
    print("Checking data files...")
    for zone, filename in preprocessor.zone_files.items():
        file_path = os.path.join(preprocessor.data_dir, filename)
        if os.path.exists(file_path):
            print(f"✓ {zone}: {filename}")
        else:
            print(f"✗ {zone}: {filename} - NOT FOUND")
    
    print("\n" + "="*50 + "\n")
    
    # Try to process data
    try:
        df_processed = preprocessor.save_processed_data()
        
        if df_processed is not None and len(df_processed) > 0:
            print("✓ Data processing successful!")
            print(f"Total records: {len(df_processed)}")
            print(f"Zones: {sorted(df_processed['Zone'].unique())}")
            print(f"Sectors: {sorted(df_processed['Sector'].unique())}")
            print(f"Scenarios: {sorted(df_processed['Scenario'].unique())}")
            print(f"Levers: {sorted(df_processed['Lever'].unique())}")
            
            # Show sample data
            print("\nSample data:")
            print(df_processed.head(10))
            
            # Show data structure
            print("\nData columns:")
            for col in df_processed.columns:
                print(f"  - {col}")
                
        else:
            print("✗ Data processing failed - no data returned")
            
    except Exception as e:
        print(f"✗ Error during data processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preprocessing() 