#!/usr/bin/env python3
"""
Sanity check for world CO2 decomposition data
Investigates extreme values and potential data quality issues
"""

import pandas as pd
import numpy as np
import os

# Get the absolute path to the data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))

def check_extreme_values():
    """Check for extreme values in the decomposition data"""
    print("🔍 Checking for extreme values in world decomposition data...")
    
    # Load the data
    df = pd.read_csv(os.path.join(DATA_DIR, 'world_unified_decomposition_data.csv'))
    
    # Filter out Total lever for analysis
    df_levers = df[df['Lever'] != 'Total'].copy()
    
    print(f"\n📊 Data Overview:")
    print(f"Total records: {len(df)}")
    print(f"Lever records (excluding Total): {len(df_levers)}")
    print(f"Zones: {len(df['Zone'].unique())}")
    print(f"Scenarios: {len(df['Scenario'].unique())}")
    
    # Check percentage contribution ranges
    print(f"\n📈 Percentage Contribution Ranges:")
    for period in ['Contrib_2015_2040_pct', 'Contrib_2040_2050_pct', 'Contrib_2015_2050_pct']:
        print(f"\n{period}:")
        print(f"  Min: {df_levers[period].min():.1f}%")
        print(f"  Max: {df_levers[period].max():.1f}%")
        print(f"  Mean: {df_levers[period].mean():.1f}%")
        print(f"  Std: {df_levers[period].std():.1f}%")
        
        # Check extreme values
        extreme_high = df_levers[df_levers[period] > 100]
        extreme_low = df_levers[df_levers[period] < -100]
        
        print(f"  Values > 100%: {len(extreme_high)}")
        print(f"  Values < -100%: {len(extreme_low)}")
    
    return df, df_levers

def investigate_extreme_cases(df_levers):
    """Investigate specific extreme cases"""
    print(f"\n🚨 Investigating Extreme Cases...")
    
    # Find the most extreme values
    for period in ['Contrib_2015_2040_pct', 'Contrib_2040_2050_pct', 'Contrib_2015_2050_pct']:
        print(f"\n📊 {period} - Top 5 Most Extreme Values:")
        
        # Most positive
        most_positive = df_levers.nlargest(5, period)[['Zone', 'Sector', 'Scenario', 'Lever', period]]
        print(f"\nMost Positive:")
        print(most_positive.to_string(index=False))
        
        # Most negative
        most_negative = df_levers.nsmallest(5, period)[['Zone', 'Sector', 'Scenario', 'Lever', period]]
        print(f"\nMost Negative:")
        print(most_negative.to_string(index=False))

def check_data_consistency():
    """Check for data consistency issues"""
    print(f"\n🔍 Checking Data Consistency...")
    
    # Load original world data to check baseline values
    world_data_path = os.path.join(DATA_DIR, 'world_data_european_format.csv')
    if os.path.exists(world_data_path):
        world_df = pd.read_csv(world_data_path)
        
        print(f"\n📊 Original World Data Overview:")
        print(f"Shape: {world_df.shape}")
        print(f"Years: {sorted(world_df['Year'].unique())}")
        
        # Check for very small CO2 values that could cause division issues
        print(f"\n🔍 Checking for very small CO2 baseline values:")
        co2_2015 = world_df[world_df['Year'] == 2015]['CO2 (Million tonn)']
        small_co2 = co2_2015[co2_2015 < 1.0]  # Less than 1 MtCO2
        
        if len(small_co2) > 0:
            print(f"Found {len(small_co2)} regions with CO2_2015 < 1 MtCO2:")
            small_co2_data = world_df[
                (world_df['Year'] == 2015) & 
                (world_df['CO2 (Million tonn)'] < 1.0)
            ][['Geography', 'Sector', 'Scenario', 'CO2 (Million tonn)']]
            print(small_co2_data.to_string(index=False))
        else:
            print("No regions with very small CO2 baseline values found.")
        
        # Check for zero CO2 values
        zero_co2 = co2_2015[co2_2015 == 0.0]
        if len(zero_co2) > 0:
            print(f"\n⚠️  Found {len(zero_co2)} regions with CO2_2015 = 0!")
            zero_co2_data = world_df[
                (world_df['Year'] == 2015) & 
                (world_df['CO2 (Million tonn)'] == 0.0)
            ][['Geography', 'Sector', 'Scenario', 'CO2 (Million tonn)']]
            print(zero_co2_data.to_string(index=False))

def check_lmdi_calculations():
    """Check if LMDI calculations are mathematically sound"""
    print(f"\n🧮 Checking LMDI Calculation Soundness...")
    
    # Load the data
    df = pd.read_csv(os.path.join(DATA_DIR, 'world_unified_decomposition_data.csv'))
    
    # Check if Total lever contributions sum to 100%
    print(f"\n📊 Checking Total Lever Consistency:")
    
    for zone in df['Zone'].unique():
        for sector in df[df['Zone'] == zone]['Sector'].unique():
            for scenario in df[(df['Zone'] == zone) & (df['Sector'] == sector)]['Scenario'].unique():
                
                scenario_data = df[
                    (df['Zone'] == zone) & 
                    (df['Sector'] == sector) & 
                    (df['Scenario'] == scenario)
                ]
                
                if len(scenario_data) > 0:
                    # Get Total lever data
                    total_data = scenario_data[scenario_data['Lever'] == 'Total']
                    if len(total_data) > 0:
                        total_pct = total_data.iloc[0]['Contrib_2015_2050_pct']
                        
                        # Sum individual lever contributions
                        lever_data = scenario_data[scenario_data['Lever'] != 'Total']
                        if len(lever_data) > 0:
                            lever_sum = lever_data['Contrib_2015_2050_pct'].sum()
                            
                            # Check if they match (allowing for small floating point errors)
                            if abs(total_pct - lever_sum) > 0.1:
                                print(f"⚠️  Mismatch in {zone} - {sector} - {scenario}:")
                                print(f"    Total: {total_pct:.1f}%, Sum of levers: {lever_sum:.1f}%")
                                print(f"    Difference: {abs(total_pct - lever_sum):.1f}%")

def main():
    """Main sanity check function"""
    print("🔍 World CO2 Decomposition Data Sanity Check")
    print("=" * 50)
    
    try:
        # Check extreme values
        df, df_levers = check_extreme_values()
        
        # Investigate extreme cases
        investigate_extreme_cases(df_levers)
        
        # Check data consistency
        check_data_consistency()
        
        # Check LMDI calculations
        check_lmdi_calculations()
        
        print(f"\n✅ Sanity check completed!")
        
    except Exception as e:
        print(f"❌ Error during sanity check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 