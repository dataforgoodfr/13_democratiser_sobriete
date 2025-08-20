#!/usr/bin/env python3
"""
Append World Sufficiency Lab Scenarios to Unified Data
Integrates the new sufficiency scenarios into the main dataset
"""

import pandas as pd
import numpy as np
import os
from test_sufficiency_scenarios import generate_all_sufficiency_scenarios

def append_sufficiency_scenarios_to_unified_data():
    """
    Load existing unified data, generate new sufficiency scenarios, and append them
    """
    print("="*80)
    print("APPENDING WORLD SUFFICIENCY LAB SCENARIOS TO UNIFIED DATA")
    print("="*80)
    
    # Define paths
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
    unified_file_path = os.path.join(DATA_DIR, 'unified_decomposition_data.csv')
    backup_file_path = os.path.join(DATA_DIR, 'unified_decomposition_data_backup.csv')
    
    # Load existing unified data
    print(f"Loading existing data from: {unified_file_path}")
    existing_data = pd.read_csv(unified_file_path)
    print(f"Existing data shape: {existing_data.shape}")
    print(f"Existing scenarios: {sorted(existing_data['Scenario'].unique())}")
    
    # Create backup
    print(f"Creating backup at: {backup_file_path}")
    existing_data.to_csv(backup_file_path, index=False)
    
    # Generate new sufficiency scenarios
    print("\nGenerating World Sufficiency Lab scenarios...")
    new_scenarios = generate_all_sufficiency_scenarios(existing_data)
    
    if not new_scenarios:
        print("ERROR: No new scenarios generated!")
        return False
    
    # Convert to DataFrame
    new_scenarios_df = pd.DataFrame(new_scenarios)
    print(f"New scenarios shape: {new_scenarios_df.shape}")
    print(f"New scenarios: {sorted(new_scenarios_df['Scenario'].unique())}")
    
    # Ensure column order matches existing data
    new_scenarios_df = new_scenarios_df[existing_data.columns]
    
    # Append new scenarios to existing data
    print("\nAppending new scenarios to existing data...")
    combined_data = pd.concat([existing_data, new_scenarios_df], ignore_index=True)
    print(f"Combined data shape: {combined_data.shape}")
    
    # Verify the combination
    print(f"\nVerification:")
    print(f"Original records: {len(existing_data)}")
    print(f"New records: {len(new_scenarios_df)}")
    print(f"Combined records: {len(combined_data)}")
    print(f"All scenarios: {sorted(combined_data['Scenario'].unique())}")
    
    # Save the combined data
    print(f"\nSaving combined data to: {unified_file_path}")
    combined_data.to_csv(unified_file_path, index=False)
    
    print("\n" + "="*80)
    print("INTEGRATION COMPLETE!")
    print("="*80)
    print(f"✅ Backup created: {backup_file_path}")
    print(f"✅ New unified file saved: {unified_file_path}")
    print(f"✅ Added {len(new_scenarios_df)} new records")
    print(f"✅ Total records: {len(combined_data)}")
    
    return True

if __name__ == "__main__":
    success = append_sufficiency_scenarios_to_unified_data()
    if success:
        print("\n🎉 SUCCESS: World Sufficiency Lab scenarios successfully integrated!")
    else:
        print("\n❌ ERROR: Integration failed!")