#!/usr/bin/env python3
"""
Test Script for World Sufficiency Lab Scenarios
Calculates new scenarios with different sufficiency assumptions
"""

import pandas as pd
import numpy as np
import os

# Load the existing unified data
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
data = pd.read_csv(os.path.join(DATA_DIR, 'unified_decomposition_data.csv'))

print("Data loaded successfully!")
print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

def calculate_sufficiency_scenario_1(base_data, sector, zone):
    """
    Scenario 1: No increase in Consumption or Production Intensity
    - Set Sufficiency impact to 0 Mt
    - Redistribute the difference equally between Energy Efficiency and Supply Side Decarbonation
    """
    print(f"\n--- Calculating Scenario 1 for {sector} ({zone}) ---")
    
    # Get base scenario data (let's use the first scenario as reference)
    base_scenario = base_data[
        (base_data['Zone'] == zone) & 
        (base_data['Sector'] == sector) & 
        (base_data['Lever'] != 'Total')
    ].iloc[0:4]  # First 4 levers (excluding Total)
    
    print("Base scenario data:")
    print(base_scenario[['Lever', 'Contrib_2015_2040_abs', 'Contrib_2040_2050_abs', 'Contrib_2015_2050_abs']])
    
    # Get current Sufficiency values
    sufficiency_2015_2040 = base_scenario[base_scenario['Lever'] == 'Sufficiency']['Contrib_2015_2040_abs'].iloc[0]
    sufficiency_2040_2050 = base_scenario[base_scenario['Lever'] == 'Sufficiency']['Contrib_2040_2050_abs'].iloc[0]
    sufficiency_2015_2050 = base_scenario[base_scenario['Lever'] == 'Sufficiency']['Contrib_2015_2050_abs'].iloc[0]
    
    print(f"\nCurrent Sufficiency values:")
    print(f"2015-2040: {sufficiency_2015_2040:.2f} Mt")
    print(f"2040-2050: {sufficiency_2040_2050:.2f} Mt")
    print(f"2015-2050: {sufficiency_2015_2050:.2f} Mt")
    
    # Calculate the difference to redistribute
    # Since Sufficiency is currently negative (emissions increase), we need to:
    # 1. Set it to 0 (no change)
    # 2. Distribute the "missing" reduction to other levers
    
    # The difference is the absolute value of current Sufficiency (because we're going from negative to 0)
    diff_2015_2040 = abs(sufficiency_2015_2040)
    diff_2040_2050 = abs(sufficiency_2040_2050)
    diff_2015_2050 = abs(sufficiency_2015_2050)
    
    print(f"\nDifference to redistribute:")
    print(f"2015-2040: {diff_2015_2040:.2f} Mt")
    print(f"2040-2050: {diff_2040_2050:.2f} Mt")
    print(f"2015-2050: {diff_2015_2050:.2f} Mt")
    
    # Redistribute equally between Energy Efficiency and Supply Side Decarbonation
    redistribution_per_lever_2015_2040 = diff_2015_2040 / 2
    redistribution_per_lever_2040_2050 = diff_2040_2050 / 2
    redistribution_per_lever_2015_2050 = diff_2015_2050 / 2
    
    print(f"\nRedistribution per lever:")
    print(f"2015-2040: {redistribution_per_lever_2015_2040:.2f} Mt each")
    print(f"2040-2050: {redistribution_per_lever_2040_2050:.2f} Mt each")
    print(f"2015-2050: {redistribution_per_lever_2015_2050:.2f} Mt each")
    
    # Create new scenario data
    new_scenario = base_scenario.copy()
    
    # Update Sufficiency to 0
    sufficiency_mask = new_scenario['Lever'] == 'Sufficiency'
    new_scenario.loc[sufficiency_mask, 'Contrib_2015_2040_abs'] = 0
    new_scenario.loc[sufficiency_mask, 'Contrib_2040_2050_abs'] = 0
    new_scenario.loc[sufficiency_mask, 'Contrib_2015_2050_abs'] = 0
    
    # Update Energy Efficiency
    ee_mask = new_scenario['Lever'] == 'Energy Efficiency'
    new_scenario.loc[ee_mask, 'Contrib_2015_2040_abs'] += redistribution_per_lever_2015_2040
    new_scenario.loc[ee_mask, 'Contrib_2040_2050_abs'] += redistribution_per_lever_2040_2050
    new_scenario.loc[ee_mask, 'Contrib_2015_2050_abs'] += redistribution_per_lever_2015_2050
    
    # Update Supply Side Decarbonation
    ssd_mask = new_scenario['Lever'] == 'Supply Side Decarbonation'
    new_scenario.loc[ssd_mask, 'Contrib_2015_2040_abs'] += redistribution_per_lever_2015_2040
    new_scenario.loc[ssd_mask, 'Contrib_2040_2050_abs'] += redistribution_per_lever_2040_2050
    new_scenario.loc[ssd_mask, 'Contrib_2015_2050_abs'] += redistribution_per_lever_2015_2050
    
    print(f"\nNew scenario data:")
    print(new_scenario[['Lever', 'Contrib_2015_2040_abs', 'Contrib_2040_2050_abs', 'Contrib_2015_2050_abs']])
    
    # Verify the total still adds up
    total_2015_2040 = new_scenario['Contrib_2015_2040_abs'].sum()
    total_2040_2050 = new_scenario['Contrib_2040_2050_abs'].sum()
    total_2015_2050 = new_scenario['Contrib_2015_2050_abs'].sum()
    
    print(f"\nVerification - Total contributions:")
    print(f"2015-2040: {total_2015_2040:.2f} Mt")
    print(f"2040-2050: {total_2040_2050:.2f} Mt")
    print(f"2015-2050: {total_2015_2050:.2f} Mt")
    
    return new_scenario

def calculate_sufficiency_scenario_2(base_data, sector, zone):
    """
    Scenario 2: 20% decrease in Consumption or Production Intensity
    - Assume Sufficiency creates a 20% decrease in the relevant metric
    - This would create a positive Sufficiency contribution
    - Adjust Energy Efficiency and Supply Side Decarbonation accordingly
    """
    print(f"\n--- Calculating Scenario 2 for {sector} ({zone}) ---")
    
    # Get base scenario data (let's use the first scenario as reference)
    base_scenario = base_data[
        (base_data['Zone'] == zone) & 
        (base_data['Sector'] == sector) & 
        (data['Lever'] != 'Total')
    ].iloc[0:4]  # First 4 levers (excluding Total)
    
    print("Base scenario data:")
    print(base_scenario[['Lever', 'Contrib_2015_2040_abs', 'Contrib_2040_2050_abs', 'Contrib_2015_2050_abs']])
    
    # Get current Sufficiency values
    sufficiency_2015_2040 = base_scenario[base_scenario['Lever'] == 'Sufficiency']['Contrib_2015_2040_abs'].iloc[0]
    sufficiency_2040_2050 = base_scenario[base_scenario['Lever'] == 'Sufficiency']['Contrib_2040_2050_abs'].iloc[0]
    sufficiency_2015_2050 = base_scenario[base_scenario['Lever'] == 'Sufficiency']['Contrib_2015_2050_abs'].iloc[0]
    
    print(f"\nCurrent Sufficiency values:")
    print(f"2015-2040: {sufficiency_2015_2040:.2f} Mt")
    print(f"2040-2050: {sufficiency_2040_2050:.2f} Mt")
    print(f"2015-2050: {sufficiency_2015_2050:.2f} Mt")
    
    # For Scenario 2, we want to create a 20% decrease in consumption/production intensity
    # This means Sufficiency should contribute to emissions REDUCTION (negative values in our current convention)
    # Since we're flipping the sign convention, we need to think about this carefully
    
    # Let's assume a 20% decrease in the relevant metric means:
    # - If current Sufficiency is +52.77 Mt (emissions increase), we want to go to -52.77 * 0.2 = -10.55 Mt (emissions decrease)
    # - This represents a 20% improvement from the current situation
    
    # Calculate the target Sufficiency values (20% of current absolute values, but negative for emissions reduction)
    target_sufficiency_2015_2040 = -abs(sufficiency_2015_2040) * 0.2
    target_sufficiency_2040_2050 = -abs(sufficiency_2040_2050) * 0.2
    target_sufficiency_2015_2050 = -abs(sufficiency_2015_2050) * 0.2
    
    print(f"\nTarget Sufficiency values (20% decrease in intensity):")
    print(f"2015-2040: {target_sufficiency_2015_2040:.2f} Mt")
    print(f"2040-2050: {target_sufficiency_2040_2050:.2f} Mt")
    print(f"2015-2050: {target_sufficiency_2015_2050:.2f} Mt")
    
    # Calculate the difference to redistribute
    # We're going from current Sufficiency to target Sufficiency
    diff_2015_2040 = sufficiency_2015_2040 - target_sufficiency_2015_2040
    diff_2040_2050 = sufficiency_2040_2050 - target_sufficiency_2040_2050
    diff_2015_2050 = sufficiency_2015_2050 - target_sufficiency_2015_2050
    
    print(f"\nDifference to redistribute:")
    print(f"2015-2040: {diff_2015_2040:.2f} Mt")
    print(f"2040-2050: {diff_2040_2050:.2f} Mt")
    print(f"2015-2050: {diff_2015_2050:.2f} Mt")
    
    # Redistribute equally between Energy Efficiency and Supply Side Decarbonation
    redistribution_per_lever_2015_2040 = diff_2015_2040 / 2
    redistribution_per_lever_2040_2050 = diff_2040_2050 / 2
    redistribution_per_lever_2015_2050 = diff_2015_2050 / 2
    
    print(f"\nRedistribution per lever:")
    print(f"2015-2040: {redistribution_per_lever_2015_2040:.2f} Mt each")
    print(f"2040-2050: {redistribution_per_lever_2040_2050:.2f} Mt each")
    print(f"2015-2050: {redistribution_per_lever_2015_2050:.2f} Mt each")
    
    # Create new scenario data
    new_scenario = base_scenario.copy()
    
    # Update Sufficiency to target values
    sufficiency_mask = new_scenario['Lever'] == 'Sufficiency'
    new_scenario.loc[sufficiency_mask, 'Contrib_2015_2040_abs'] = target_sufficiency_2015_2040
    new_scenario.loc[sufficiency_mask, 'Contrib_2040_2050_abs'] = target_sufficiency_2040_2050
    new_scenario.loc[sufficiency_mask, 'Contrib_2015_2050_abs'] = target_sufficiency_2015_2050
    
    # Update Energy Efficiency
    ee_mask = new_scenario['Lever'] == 'Energy Efficiency'
    new_scenario.loc[ee_mask, 'Contrib_2015_2040_abs'] += redistribution_per_lever_2015_2040
    new_scenario.loc[ee_mask, 'Contrib_2040_2050_abs'] += redistribution_per_lever_2040_2050
    new_scenario.loc[ee_mask, 'Contrib_2015_2050_abs'] += redistribution_per_lever_2015_2050
    
    # Update Supply Side Decarbonation
    ssd_mask = new_scenario['Lever'] == 'Supply Side Decarbonation'
    new_scenario.loc[ssd_mask, 'Contrib_2015_2040_abs'] += redistribution_per_lever_2015_2040
    new_scenario.loc[ssd_mask, 'Contrib_2040_2050_abs'] += redistribution_per_lever_2040_2050
    new_scenario.loc[ssd_mask, 'Contrib_2015_2050_abs'] += redistribution_per_lever_2015_2050
    
    print(f"\nNew scenario data:")
    print(new_scenario[['Lever', 'Contrib_2015_2040_abs', 'Contrib_2040_2050_abs', 'Contrib_2015_2050_abs']])
    
    # Verify the total still adds up
    total_2015_2040 = new_scenario['Contrib_2015_2040_abs'].sum()
    total_2040_2050 = new_scenario['Contrib_2040_2050_abs'].sum()
    total_2015_2050 = new_scenario['Contrib_2015_2050_abs'].sum()
    
    print(f"\nVerification - Total contributions:")
    print(f"2015-2040: {total_2015_2040:.2f} Mt")
    print(f"2040-2050: {total_2040_2050:.2f} Mt")
    print(f"2015-2050: {total_2015_2050:.2f} Mt")
    
    return new_scenario

def generate_all_sufficiency_scenarios(base_data):
    """
    Generate World Sufficiency Lab scenarios for all sectors and geographies
    Returns a list of new scenario data that can be appended to the main dataset
    """
    print("\n" + "="*80)
    print("GENERATING WORLD SUFFICIENCY LAB SCENARIOS FOR ALL SECTORS AND GEOGRAPHIES")
    print("="*80)
    
    # Get all unique combinations of Zone and Sector
    zones = base_data['Zone'].unique()
    sectors = base_data['Sector'].unique()
    
    print(f"Processing {len(zones)} zones: {list(zones)}")
    print(f"Processing {len(sectors)} sectors: {list(sectors)}")
    
    all_new_scenarios = []
    
    for zone in zones:
        for sector in sectors:
            print(f"\n--- Processing {sector} in {zone} ---")
            
            # Get base scenario data for this zone/sector combination
            base_scenario_data = base_data[
                (base_data['Zone'] == zone) & 
                (base_data['Sector'] == sector) & 
                (base_data['Lever'] != 'Total')
            ]
            
            if base_scenario_data.empty:
                print(f"No data found for {sector} in {zone}, skipping...")
                continue
            
            # Use the first scenario as reference (first 4 levers)
            reference_scenario = base_scenario_data.iloc[0:4].copy()
            
            # Generate Scenario 1: No increase in Consumption/Production Intensity
            scenario_1 = generate_single_scenario_1(reference_scenario, sector, zone)
            if scenario_1 is not None:
                all_new_scenarios.extend(scenario_1)
            
            # Generate Scenario 2: 20% decrease in Consumption/Production Intensity  
            scenario_2 = generate_single_scenario_2(reference_scenario, sector, zone)
            if scenario_2 is not None:
                all_new_scenarios.extend(scenario_2)
    
    print(f"\n" + "="*80)
    print(f"GENERATION COMPLETE: Created {len(all_new_scenarios)} new scenario records")
    print("="*80)
    
    return all_new_scenarios

def generate_single_scenario_1(base_scenario, sector, zone):
    """Generate Scenario 1 data for a single sector/zone combination"""
    try:
        # Get current Sufficiency values
        sufficiency_row = base_scenario[base_scenario['Lever'] == 'Sufficiency']
        if sufficiency_row.empty:
            print(f"No Sufficiency data for {sector} in {zone}")
            return None
            
        sufficiency_2015_2040 = sufficiency_row['Contrib_2015_2040_abs'].iloc[0]
        sufficiency_2040_2050 = sufficiency_row['Contrib_2040_2050_abs'].iloc[0]
        sufficiency_2015_2050 = sufficiency_row['Contrib_2015_2050_abs'].iloc[0]
        
        # Calculate redistribution amounts
        diff_2015_2040 = abs(sufficiency_2015_2040)
        diff_2040_2050 = abs(sufficiency_2040_2050)
        diff_2015_2050 = abs(sufficiency_2015_2050)
        
        redistribution_per_lever_2015_2040 = diff_2015_2040 / 2
        redistribution_per_lever_2040_2050 = diff_2040_2050 / 2
        redistribution_per_lever_2015_2050 = diff_2015_2050 / 2
        
        # Create new scenario data
        new_scenario = base_scenario.copy()
        new_scenario['Scenario'] = 'World Sufficiency Lab - No Increase in Consumption or Production Intensity'
        
        # Update Sufficiency to 0
        sufficiency_mask = new_scenario['Lever'] == 'Sufficiency'
        new_scenario.loc[sufficiency_mask, 'Contrib_2015_2040_abs'] = 0
        new_scenario.loc[sufficiency_mask, 'Contrib_2040_2050_abs'] = 0
        new_scenario.loc[sufficiency_mask, 'Contrib_2015_2050_abs'] = 0
        
        # Update Energy Efficiency
        ee_mask = new_scenario['Lever'] == 'Energy Efficiency'
        new_scenario.loc[ee_mask, 'Contrib_2015_2040_abs'] += redistribution_per_lever_2015_2040
        new_scenario.loc[ee_mask, 'Contrib_2040_2050_abs'] += redistribution_per_lever_2040_2050
        new_scenario.loc[ee_mask, 'Contrib_2015_2050_abs'] += redistribution_per_lever_2015_2050
        
        # Update Supply Side Decarbonation
        ssd_mask = new_scenario['Lever'] == 'Supply Side Decarbonation'
        new_scenario.loc[ssd_mask, 'Contrib_2015_2040_abs'] += redistribution_per_lever_2015_2040
        new_scenario.loc[ssd_mask, 'Contrib_2040_2050_abs'] += redistribution_per_lever_2040_2050
        new_scenario.loc[ssd_mask, 'Contrib_2015_2050_abs'] += redistribution_per_lever_2015_2050
        
        # Calculate percentage contributions (with sign flip)
        total_change_2015_2040 = new_scenario['Contrib_2015_2040_abs'].sum()
        total_change_2040_2050 = new_scenario['Contrib_2040_2050_abs'].sum()
        total_change_2015_2050 = new_scenario['Contrib_2015_2050_abs'].sum()
        
        for idx, row in new_scenario.iterrows():
            if row['Lever'] != 'Total':
                contrib_pct_1 = (row['Contrib_2015_2040_abs'] / abs(total_change_2015_2040)) * 100 if total_change_2015_2040 != 0 else 0
                contrib_pct_2 = (row['Contrib_2040_2050_abs'] / abs(total_change_2040_2050)) * 100 if total_change_2040_2050 != 0 else 0
                contrib_pct_total = (row['Contrib_2015_2050_abs'] / abs(total_change_2015_2050)) * 100 if total_change_2015_2050 != 0 else 0
                
                # Apply sign flip for display
                new_scenario.loc[idx, 'Contrib_2015_2040_pct'] = -contrib_pct_1
                new_scenario.loc[idx, 'Contrib_2040_2050_pct'] = -contrib_pct_2
                new_scenario.loc[idx, 'Contrib_2015_2050_pct'] = -contrib_pct_total
        
        # Add Total row
        total_row = new_scenario.iloc[0].copy()
        total_row['Lever'] = 'Total'
        total_row['CO2_2015'] = base_scenario.iloc[0]['CO2_2015'] if 'CO2_2015' in base_scenario.columns else None
        total_row['CO2_2040'] = base_scenario.iloc[0]['CO2_2040'] if 'CO2_2040' in base_scenario.columns else None
        total_row['CO2_2050'] = base_scenario.iloc[0]['CO2_2050'] if 'CO2_2050' in base_scenario.columns else None
        total_row['Contrib_2015_2040_abs'] = total_change_2015_2040
        total_row['Contrib_2040_2050_abs'] = total_change_2040_2050
        total_row['Contrib_2015_2050_abs'] = total_change_2015_2050
        total_row['Contrib_2015_2040_pct'] = 100.0
        total_row['Contrib_2040_2050_pct'] = 100.0
        total_row['Contrib_2015_2050_pct'] = 100.0
        
        # Convert to list of dictionaries
        result = []
        for _, row in new_scenario.iterrows():
            result.append(row.to_dict())
        result.append(total_row.to_dict())
        
        return result
        
    except Exception as e:
        print(f"Error generating Scenario 1 for {sector} in {zone}: {e}")
        return None

def generate_single_scenario_2(base_scenario, sector, zone):
    """Generate Scenario 2 data for a single sector/zone combination"""
    try:
        # Get current Sufficiency values
        sufficiency_row = base_scenario[base_scenario['Lever'] == 'Sufficiency']
        if sufficiency_row.empty:
            print(f"No Sufficiency data for {sector} in {zone}")
            return None
            
        sufficiency_2015_2040 = sufficiency_row['Contrib_2015_2040_abs'].iloc[0]
        sufficiency_2040_2050 = sufficiency_row['Contrib_2040_2050_abs'].iloc[0]
        sufficiency_2015_2050 = sufficiency_row['Contrib_2015_2050_abs'].iloc[0]
        
        # Calculate target Sufficiency values (20% decrease in intensity)
        target_sufficiency_2015_2040 = -abs(sufficiency_2015_2040) * 0.2
        target_sufficiency_2040_2050 = -abs(sufficiency_2040_2050) * 0.2
        target_sufficiency_2015_2050 = -abs(sufficiency_2015_2050) * 0.2
        
        # Calculate redistribution amounts
        diff_2015_2040 = sufficiency_2015_2040 - target_sufficiency_2015_2040
        diff_2040_2050 = sufficiency_2040_2050 - target_sufficiency_2040_2050
        diff_2015_2050 = sufficiency_2015_2050 - target_sufficiency_2015_2050
        
        redistribution_per_lever_2015_2040 = diff_2015_2040 / 2
        redistribution_per_lever_2040_2050 = diff_2040_2050 / 2
        redistribution_per_lever_2015_2050 = diff_2015_2050 / 2
        
        # Create new scenario data
        new_scenario = base_scenario.copy()
        new_scenario['Scenario'] = 'World Sufficiency Lab - Sufficiency'
        
        # Update Sufficiency to target values
        sufficiency_mask = new_scenario['Lever'] == 'Sufficiency'
        new_scenario.loc[sufficiency_mask, 'Contrib_2015_2040_abs'] = target_sufficiency_2015_2040
        new_scenario.loc[sufficiency_mask, 'Contrib_2040_2050_abs'] = target_sufficiency_2040_2050
        new_scenario.loc[sufficiency_mask, 'Contrib_2015_2050_abs'] = target_sufficiency_2015_2050
        
        # Update Energy Efficiency
        ee_mask = new_scenario['Lever'] == 'Energy Efficiency'
        new_scenario.loc[ee_mask, 'Contrib_2015_2040_abs'] += redistribution_per_lever_2015_2040
        new_scenario.loc[ee_mask, 'Contrib_2040_2050_abs'] += redistribution_per_lever_2040_2050
        new_scenario.loc[ee_mask, 'Contrib_2015_2050_abs'] += redistribution_per_lever_2015_2050
        
        # Update Supply Side Decarbonation
        ssd_mask = new_scenario['Lever'] == 'Supply Side Decarbonation'
        new_scenario.loc[ssd_mask, 'Contrib_2015_2040_abs'] += redistribution_per_lever_2015_2040
        new_scenario.loc[ssd_mask, 'Contrib_2040_2050_abs'] += redistribution_per_lever_2040_2050
        new_scenario.loc[ssd_mask, 'Contrib_2015_2050_abs'] += redistribution_per_lever_2015_2050
        
        # Calculate percentage contributions (with sign flip)
        total_change_2015_2040 = new_scenario['Contrib_2015_2040_abs'].sum()
        total_change_2040_2050 = new_scenario['Contrib_2040_2050_abs'].sum()
        total_change_2015_2050 = new_scenario['Contrib_2015_2050_abs'].sum()
        
        for idx, row in new_scenario.iterrows():
            if row['Lever'] != 'Total':
                contrib_pct_1 = (row['Contrib_2015_2040_abs'] / abs(total_change_2015_2040)) * 100 if total_change_2015_2040 != 0 else 0
                contrib_pct_2 = (row['Contrib_2040_2050_abs'] / abs(total_change_2040_2050)) * 100 if total_change_2040_2050 != 0 else 0
                contrib_pct_total = (row['Contrib_2015_2050_abs'] / abs(total_change_2015_2050)) * 100 if total_change_2015_2050 != 0 else 0
                
                # Apply sign flip for display
                new_scenario.loc[idx, 'Contrib_2015_2040_pct'] = -contrib_pct_1
                new_scenario.loc[idx, 'Contrib_2040_2050_pct'] = -contrib_pct_2
                new_scenario.loc[idx, 'Contrib_2015_2050_pct'] = -contrib_pct_total
        
        # Add Total row
        total_row = new_scenario.iloc[0].copy()
        total_row['Lever'] = 'Total'
        total_row['CO2_2015'] = base_scenario.iloc[0]['CO2_2015'] if 'CO2_2015' in base_scenario.columns else None
        total_row['CO2_2040'] = base_scenario.iloc[0]['CO2_2040'] if 'CO2_2040' in base_scenario.columns else None
        total_row['CO2_2050'] = base_scenario.iloc[0]['CO2_2050'] if 'CO2_2050' in base_scenario.columns else None
        total_row['Contrib_2015_2040_abs'] = total_change_2015_2040
        total_row['Contrib_2040_2050_abs'] = total_change_2040_2050
        total_row['Contrib_2015_2050_abs'] = total_change_2015_2050
        total_row['Contrib_2015_2040_pct'] = 100.0
        total_row['Contrib_2040_2050_pct'] = 100.0
        total_row['Contrib_2015_2050_pct'] = 100.0
        
        # Convert to list of dictionaries
        result = []
        for _, row in new_scenario.iterrows():
            result.append(row.to_dict())
        result.append(total_row.to_dict())
        
        return result
        
    except Exception as e:
        print(f"Error generating Scenario 2 for {sector} in {zone}: {e}")
        return None

# Test the comprehensive generation
print(f"Testing comprehensive generation for all sectors and geographies")
all_sufficiency_scenarios = generate_all_sufficiency_scenarios(data)

print(f"\nSample of generated scenarios:")
if all_sufficiency_scenarios:
    sample_df = pd.DataFrame(all_sufficiency_scenarios[:10])  # Show first 10 records
    print(sample_df[['Zone', 'Sector', 'Scenario', 'Lever', 'Contrib_2015_2050_abs', 'Contrib_2015_2050_pct']])
    
    # Show summary
    scenarios_df = pd.DataFrame(all_sufficiency_scenarios)
    print(f"\nSummary:")
    print(f"Total new records: {len(scenarios_df)}")
    print(f"Unique scenarios: {scenarios_df['Scenario'].unique()}")
    print(f"Records per scenario: {scenarios_df['Scenario'].value_counts()}")
else:
    print("No scenarios generated!")

print("\n" + "="*80)
print("COMPREHENSIVE GENERATION TEST COMPLETE")
print("="*80) 