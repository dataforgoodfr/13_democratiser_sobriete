#!/usr/bin/env python3
"""
Create Intermediary File for World Sufficiency Lab Scenarios
Generates a detailed audit file with raw data and calculated intensity factors
"""

import pandas as pd
import numpy as np
import os
from test_sufficiency_scenarios import generate_all_sufficiency_scenarios

def create_sufficiency_intermediary_file():
    """
    Create an intermediary dataset specifically for World Sufficiency Lab scenarios
    """
    print("="*80)
    print("CREATING INTERMEDIARY FILE FOR WORLD SUFFICIENCY LAB SCENARIOS")
    print("="*80)
    
    # Define paths
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
    unified_file_path = os.path.join(DATA_DIR, 'unified_decomposition_data.csv')
    intermediary_path = os.path.join(DATA_DIR, 'sufficiency_intermediary_data.csv')
    
    # Load existing unified data
    print(f"Loading data from: {unified_file_path}")
    existing_data = pd.read_csv(unified_file_path)
    print(f"Data shape: {existing_data.shape}")
    
    # Get only the original scenarios (exclude existing World Sufficiency Lab scenarios)
    original_scenarios = [
        'Base Scenario', 'EU Commission >85% Decrease by 2040', 
        'EU Commission >90% Decrease by 2040', 'EU Commission Fit-for-55', 
        'EU Commission LIFE Scenario', 'Scenario Zer0 A', 'Scenario Zer0 B', 'Scenario Zer0 C'
    ]
    
    original_data = existing_data[existing_data['Scenario'].isin(original_scenarios)]
    print(f"Original scenarios data shape: {original_data.shape}")
    
    # Generate new sufficiency scenarios
    print("\nGenerating World Sufficiency Lab scenarios...")
    new_scenarios = generate_all_sufficiency_scenarios(original_data)
    
    if not new_scenarios:
        print("ERROR: No new scenarios generated!")
        return False
    
    # Convert to DataFrame
    new_scenarios_df = pd.DataFrame(new_scenarios)
    print(f"New scenarios shape: {new_scenarios_df.shape}")
    print(f"New scenarios: {sorted(new_scenarios_df['Scenario'].unique())}")
    
    # Create intermediary dataset for auditing
    print("\nCreating intermediary dataset...")
    intermediary_data = []
    
    for _, row in new_scenarios_df.iterrows():
        zone = row["Zone"]
        sector = row["Sector"]
        scenario = row["Scenario"]
        lever = row["Lever"]
        
        # Get CO2 values from the original reference scenario (e.g., EU Commission Fit-for-55)
        # We need to find the original scenario that was used as reference
        if 'World Sufficiency Lab' in scenario:
            # For World Sufficiency Lab scenarios, get CO2 from the first original scenario
            if zone == 'EU':
                reference_scenario = 'EU Commission Fit-for-55'
            else:  # Switzerland
                reference_scenario = 'Base Scenario'
        else:
            reference_scenario = scenario
            
        # Get CO2 values from the reference scenario's Total row
        reference_total = original_data[
            (original_data['Zone'] == zone) & 
            (original_data['Sector'] == sector) & 
            (original_data['Scenario'] == reference_scenario) & 
            (original_data['Lever'] == 'Total')
        ]
        
        if not reference_total.empty:
            co2_2015 = reference_total.iloc[0]["CO2_2015"]
            co2_2040 = reference_total.iloc[0]["CO2_2040"]
            co2_2050 = reference_total.iloc[0]["CO2_2050"]
        else:
            co2_2015 = co2_2040 = co2_2050 = None
        
        # Calculate intensity factors based on lever
        if lever == 'Population':
            population_intensity = row["Contrib_2015_2050_abs"] / abs(row["Contrib_2015_2050_abs"]) if row["Contrib_2015_2050_abs"] != 0 else 0
            sufficiency_intensity = 0
            energy_efficiency_intensity = 0
            carbon_intensity = 0
        elif lever == 'Sufficiency':
            population_intensity = 0
            sufficiency_intensity = row["Contrib_2015_2050_abs"] / abs(row["Contrib_2015_2050_abs"]) if row["Contrib_2015_2050_abs"] != 0 else 0
            energy_efficiency_intensity = 0
            carbon_intensity = 0
        elif lever == 'Energy Efficiency':
            population_intensity = 0
            sufficiency_intensity = 0
            energy_efficiency_intensity = row["Contrib_2015_2050_abs"] / abs(row["Contrib_2015_2050_abs"]) if row["Contrib_2015_2050_abs"] != 0 else 0
            carbon_intensity = 0
        elif lever == 'Supply Side Decarbonation':
            population_intensity = 0
            sufficiency_intensity = 0
            energy_efficiency_intensity = 0
            carbon_intensity = row["Contrib_2015_2050_abs"] / abs(row["Contrib_2015_2050_abs"]) if row["Contrib_2015_2050_abs"] != 0 else 0
        else:  # Total
            population_intensity = sufficiency_intensity = energy_efficiency_intensity = carbon_intensity = 0
        
        # Create intermediary row
        intermediary_row = {
            "Zone": zone,
            "Sector": sector,
            "Scenario": scenario,
            "Lever": lever,
            "CO2_2015": co2_2015,
            "CO2_2040": co2_2040,
            "CO2_2050": co2_2050,
            "Contrib_2015_2040_abs": row["Contrib_2015_2040_abs"],
            "Contrib_2040_2050_abs": row["Contrib_2040_2050_abs"],
            "Contrib_2015_2050_abs": row["Contrib_2015_2050_abs"],
            "Contrib_2015_2040_pct": row["Contrib_2015_2040_pct"],
            "Contrib_2040_2050_pct": row["Contrib_2040_2050_pct"],
            "Contrib_2015_2050_pct": row["Contrib_2015_2050_pct"],
            "Population_Intensity": population_intensity,
            "Sufficiency_Intensity": sufficiency_intensity,
            "Energy_Efficiency_Intensity": energy_efficiency_intensity,
            "Carbon_Intensity": carbon_intensity
        }
        
        intermediary_data.append(intermediary_row)
    
    # Convert to DataFrame
    df_intermediary = pd.DataFrame(intermediary_data)
    
    # Save intermediary dataset
    print(f"\nSaving intermediary dataset to: {intermediary_path}")
    df_intermediary.to_csv(intermediary_path, index=False)
    
    print(f"\n" + "="*80)
    print("INTERMEDIARY FILE CREATION COMPLETE!")
    print("="*80)
    print(f"✅ Intermediary dataset saved to: {intermediary_path}")
    print(f"✅ Dataset shape: {df_intermediary.shape}")
    print(f"✅ Columns: {df_intermediary.columns.tolist()}")
    
    # Show sample data
    print(f"\nSample data:")
    print(df_intermediary.head(10))
    
    return True

if __name__ == "__main__":
    success = create_sufficiency_intermediary_file()
    if success:
        print("\n🎉 SUCCESS: World Sufficiency Lab intermediary file created!")
    else:
        print("\n❌ ERROR: Intermediary file creation failed!") 