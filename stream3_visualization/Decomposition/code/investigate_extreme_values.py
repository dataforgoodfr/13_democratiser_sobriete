#!/usr/bin/env python3
"""
Investigate extreme values in world CO2 decomposition data
Deep dive into the root causes of values >1000% and <-1000%
"""

import pandas as pd
import numpy as np
import os

# Get the absolute path to the data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))

def investigate_extreme_cases():
    """Investigate specific extreme cases in detail"""
    print("🔍 Investigating Extreme Cases in Detail...")
    
    # Load the data
    df = pd.read_csv(os.path.join(DATA_DIR, 'world_unified_decomposition_data.csv'))
    
    # Find the most extreme case: Sub-Saharan Africa SSP1-1p5C
    extreme_case = df[
        (df['Zone'] == 'Sub-Saharan Africa') & 
        (df['Scenario'] == 'REMIND SSP1-1p5C') &
        (df['Lever'] == 'Energy Efficiency')
    ]
    
    if not extreme_case.empty:
        print(f"\n🚨 Most Extreme Case: Sub-Saharan Africa SSP1-1p5C Energy Efficiency")
        print(f"Contrib_2015_2040_pct: {extreme_case.iloc[0]['Contrib_2015_2040_pct']:.1f}%")
        print(f"Contrib_2015_2040_abs: {extreme_case.iloc[0]['Contrib_2015_2040_abs']:.6f}")
        
        # Get the Total lever data for this case
        total_case = df[
            (df['Zone'] == 'Sub-Saharan Africa') & 
            (df['Scenario'] == 'REMIND SSP1-1p5C') &
            (df['Lever'] == 'Total')
        ]
        
        if not total_case.empty:
            total_data = total_case.iloc[0]
            print(f"\n📊 Total Lever Data:")
            print(f"CO2_2015: {total_data['CO2_2015']:.6f}")
            print(f"CO2_2040: {total_data['CO2_2040']:.6f}")
            print(f"CO2_2050: {total_data['CO2_2050']:.6f}")
            print(f"Total Change 2015-2040: {total_data['Contrib_2015_2040_abs']:.6f}")
            print(f"Total Change 2015-2040 (%): {total_data['Contrib_2015_2040_pct']:.1f}%")
            
            # Calculate the percentage manually
            total_change = total_data['Contrib_2015_2040_abs']
            extreme_contrib = extreme_case.iloc[0]['Contrib_2015_2040_abs']
            manual_pct = (extreme_contrib / abs(total_change)) * 100 if total_change != 0 else 0
            
            print(f"\n🧮 Manual Calculation Check:")
            print(f"Extreme contribution: {extreme_contrib:.6f}")
            print(f"Total change: {total_change:.6f}")
            print(f"Manual percentage: {manual_pct:.1f}%")
            print(f"Stored percentage: {extreme_case.iloc[0]['Contrib_2015_2040_pct']:.1f}%")
            print(f"Match: {abs(manual_pct - extreme_case.iloc[0]['Contrib_2015_2040_pct']) < 0.1}")

def check_source_data_quality():
    """Check the quality of source data for extreme cases"""
    print(f"\n🔍 Checking Source Data Quality...")
    
    # Load original world data
    world_data_path = os.path.join(DATA_DIR, 'world_data_european_format.csv')
    if os.path.exists(world_data_path):
        world_df = pd.read_csv(world_data_path)
        
        # Check Sub-Saharan Africa SSP1-1p5C specifically
        ssa_data = world_df[
            (world_df['Geography'] == 'Sub-Saharan Africa') & 
            (world_df['Scenario'] == 'REMIND SSP1-1p5C')
        ]
        
        print(f"\n📊 Sub-Saharan Africa SSP1-1p5C Source Data:")
        print(f"Shape: {ssa_data.shape}")
        
        for year in [2015, 2040, 2050]:
            year_data = ssa_data[ssa_data['Year'] == year]
            if not year_data.empty:
                print(f"\nYear {year}:")
                print(f"  Population: {year_data['Population (Million)'].iloc[0]:.6f}")
                print(f"  Volume: {year_data['Volume'].iloc[0]:.6f}")
                print(f"  Energy: {year_data['Energy (Million toe)'].iloc[0]:.6f}")
                print(f"  CO2: {year_data['CO2 (Million tonn)'].iloc[0]:.6f}")
        
        # Check for very small or zero values
        print(f"\n🔍 Checking for Data Quality Issues:")
        co2_2015 = world_df[world_df['Year'] == 2015]['CO2 (Million tonn)']
        print(f"CO2 2015 - Min: {co2_2015.min():.6f}, Max: {co2_2015.max():.6f}")
        
        # Find regions with very small CO2 values
        small_co2_regions = world_df[
            (world_df['Year'] == 2015) & 
            (world_df['CO2 (Million tonn)'] < 0.1)
        ][['Geography', 'Sector', 'Scenario', 'CO2 (Million tonn)']]
        
        if len(small_co2_regions) > 0:
            print(f"\n⚠️  Regions with very small CO2_2015 (< 0.1):")
            print(small_co2_regions.to_string(index=False))

def check_lmdi_calculation_method():
    """Check the LMDI calculation method for potential issues"""
    print(f"\n🧮 Checking LMDI Calculation Method...")
    
    # Load intermediary data to see the raw calculations
    intermediary_path = os.path.join(DATA_DIR, 'world_intermediary_decomposition_data.csv')
    if os.path.exists(intermediary_path):
        intermediary_df = pd.read_csv(intermediary_path)
        
        # Check Sub-Saharan Africa SSP1-1p5C
        ssa_intermediary = intermediary_df[
            (intermediary_df['Zone'] == 'Sub-Saharan Africa') & 
            (intermediary_df['Scenario'] == 'REMIND SSP1-1p5C')
        ]
        
        print(f"\n📊 Sub-Saharan Africa SSP1-1p5C Intermediary Data:")
        print(f"Shape: {ssa_intermediary.shape}")
        
        if not ssa_intermediary.empty:
            print(f"\nSample data:")
            for year in [2015, 2040, 2050]:
                year_data = ssa_intermediary[ssa_intermediary['Year'] == year]
                if not year_data.empty:
                    data = year_data.iloc[0]
                    print(f"\nYear {year}:")
                    print(f"  Population: {data['Population_Intensity']:.6f}")
                    print(f"  Sufficiency: {data['Sufficiency_Intensity']:.6f}")
                    print(f"  Energy Efficiency: {data['Energy_Efficiency_Intensity']:.6f}")
                    print(f"  Carbon Intensity: {data['Carbon_Intensity']:.6f}")
                    print(f"  CO2: {data['CO2']:.6f}")

def main():
    """Main investigation function"""
    print("🔍 World CO2 Decomposition Data - Extreme Values Investigation")
    print("=" * 70)
    
    try:
        # Investigate extreme cases
        investigate_extreme_cases()
        
        # Check source data quality
        check_source_data_quality()
        
        # Check LMDI calculation method
        check_lmdi_calculation_method()
        
        print(f"\n✅ Investigation completed!")
        
    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 