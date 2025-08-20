#!/usr/bin/env python3
"""
Combine Original Scenarios with World Sufficiency Lab Scenarios
This script merges the original scenarios with the new World Sufficiency Lab scenarios
"""

import pandas as pd
import os

def combine_scenarios():
    """
    Combine original scenarios with World Sufficiency Lab scenarios
    """
    print("="*80)
    print("COMBINING ORIGINAL SCENARIOS WITH WORLD SUFFICIENCY LAB SCENARIOS")
    print("="*80)
    
    # Define paths
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
    
    # Load original scenarios
    original_file = os.path.join(DATA_DIR, 'unified_decomposition_data.csv')
    print(f"Loading original scenarios from: {original_file}")
    original_data = pd.read_csv(original_file)
    print(f"Original data shape: {original_data.shape}")
    
    # Load World Sufficiency Lab scenarios
    wsl_file = os.path.join(DATA_DIR, 'world_sufficiency_lab_scenarios.csv')
    print(f"Loading World Sufficiency Lab scenarios from: {wsl_file}")
    wsl_data = pd.read_csv(wsl_file)
    print(f"World Sufficiency Lab data shape: {wsl_data.shape}")
    
    # Combine the datasets
    print("\nCombining datasets...")
    combined_data = pd.concat([original_data, wsl_data], ignore_index=True)
    print(f"Combined data shape: {combined_data.shape}")
    
    # Sort the data to keep it organized
    combined_data = combined_data.sort_values(['Zone', 'Sector', 'Scenario', 'Lever'])
    
    # Verify the combination
    print("\nVerifying the combination...")
    print(f"Total rows: {len(combined_data)}")
    print(f"Original scenarios: {len(original_data)}")
    print(f"World Sufficiency Lab scenarios: {len(wsl_data)}")
    
    # Check for duplicates
    duplicate_check = combined_data.duplicated(subset=['Zone', 'Sector', 'Scenario', 'Lever']).sum()
    print(f"Duplicate rows found: {duplicate_check}")
    
    if duplicate_check > 0:
        print("WARNING: Duplicates found! Please check the data.")
        return None
    
    # Check scenario counts
    scenario_counts = combined_data['Scenario'].value_counts()
    print(f"\nScenario counts:")
    for scenario, count in scenario_counts.items():
        print(f"  {scenario}: {count} rows")
    
    return combined_data

def save_combined_data(data, filename='unified_decomposition_data_final.csv'):
    """Save the combined data to a new file"""
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
    output_path = os.path.join(DATA_DIR, filename)
    
    print(f"\nSaving combined data to: {output_path}")
    data.to_csv(output_path, index=False)
    print("Data saved successfully!")

if __name__ == "__main__":
    # Combine the scenarios
    combined_data = combine_scenarios()
    
    if combined_data is not None:
        # Save the combined data
        save_combined_data(combined_data)
        
        print("\n" + "="*80)
        print("COMBINATION COMPLETE!")
        print("="*80)
    else:
        print("\nCombination failed due to duplicates!") 