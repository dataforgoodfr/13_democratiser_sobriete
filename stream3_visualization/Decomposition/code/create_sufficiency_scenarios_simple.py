#!/usr/bin/env python3
"""
Create World Sufficiency Lab Scenarios - Simple Approach
Loads CO2 values by sector/geography and creates scenarios directly
"""

import pandas as pd
import numpy as np
import os

def create_sufficiency_scenarios():
    """
    Create World Sufficiency Lab scenarios with direct CO2 value lookup
    """
    print("="*80)
    print("CREATING WORLD SUFFICIENCY LAB SCENARIOS - SIMPLE APPROACH")
    print("="*80)
    
    # Define paths
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Output'))
    unified_file_path = os.path.join(DATA_DIR, 'unified_decomposition_data.csv')
    
    # Load existing data
    print(f"Loading data from: {unified_file_path}")
    data = pd.read_csv(unified_file_path)
    print(f"Data shape: {data.shape}")
    
    # Get only original scenarios (exclude any existing World Sufficiency Lab scenarios)
    original_scenarios = [
        'Base Scenario', 'EU Commission >85% Decrease by 2040', 
        'EU Commission >90% Decrease by 2040', 'EU Commission Fit-for-55', 
        'EU Commission LIFE Scenario', 'Scenario Zer0 A', 'Scenario Zer0 B', 'Scenario Zer0 C'
    ]
    
    original_data = data[data['Scenario'].isin(original_scenarios)]
    print(f"Original scenarios data shape: {original_data.shape}")
    
    # Create CO2 lookup table by sector and geography
    print("\nCreating CO2 lookup table...")
    co2_lookup = {}
    
    for zone in ['EU', 'Switzerland']:
        co2_lookup[zone] = {}
        # Define sectors based on what actually exists in the data
        if zone == 'EU':
            sectors = [
                'Buildings - Residential', 'Buildings - Services', 
                'Transport - Passenger cars', 'Transport - Rail',
                'Industry - Steel industry', 'Industry - Non-ferrous metal industry',
                'Industry - Chemicals industry', 'Industry - Non-Metallic Minerals industry',
                'Industry - Pulp, Paper & Print industry'
            ]
        else:  # Switzerland
            sectors = [
                'Buildings - Residential', 'Buildings - Services',
                'Passenger Land Transport', 'Cement industry', 'Steel industry'
            ]
        
        for sector in sectors:
            # Get CO2 values from the first scenario's Total row for this zone/sector
            sector_data = original_data[
                (original_data['Zone'] == zone) & 
                (original_data['Sector'] == sector) & 
                (original_data['Lever'] == 'Total')
            ]
            
            if not sector_data.empty:
                first_scenario = sector_data.iloc[0]
                co2_lookup[zone][sector] = {
                    'CO2_2015': first_scenario['CO2_2015'],
                    'CO2_2040': first_scenario['CO2_2040'],
                    'CO2_2050': first_scenario['CO2_2050']
                }
                print(f"{zone} - {sector}: CO2_2015={first_scenario['CO2_2015']:.2f}, CO2_2040={first_scenario['CO2_2040']:.2f}, CO2_2050={first_scenario['CO2_2050']:.2f}")
            else:
                co2_lookup[zone][sector] = None
                print(f"{zone} - {sector}: No data found")
    
    # Generate new scenarios
    print("\nGenerating World Sufficiency Lab scenarios...")
    new_scenarios = []
    
    for zone in ['EU', 'Switzerland']:
        # Define sectors based on what actually exists in the data
        if zone == 'EU':
            sectors = [
                'Buildings - Residential', 'Buildings - Services', 
                'Transport - Passenger cars', 'Transport - Rail',
                'Industry - Steel industry', 'Industry - Non-ferrous metal industry',
                'Industry - Chemicals industry', 'Industry - Non-Metallic Minerals industry',
                'Industry - Pulp, Paper & Print industry'
            ]
        else:  # Switzerland
            sectors = [
                'Buildings - Residential', 'Buildings - Services',
                'Passenger Land Transport', 'Cement industry', 'Steel industry'
            ]
        
        for sector in sectors:
                
            print(f"\n--- Processing {sector} in {zone} ---")
            
            # Get base scenario data for this zone/sector combination
            base_scenario_data = original_data[
                (original_data['Zone'] == zone) & 
                (original_data['Sector'] == sector)
            ]
            
            if base_scenario_data.empty:
                print(f"No data found for {sector} in {zone}, skipping...")
                continue
            
            # Get the first scenario (all levers including Total)
            first_scenario_name = base_scenario_data.iloc[0]['Scenario']
            first_scenario_data = base_scenario_data[base_scenario_data['Scenario'] == first_scenario_name]
            
            # Get individual levers (excluding Total) for calculations
            reference_scenario = first_scenario_data[first_scenario_data['Lever'] != 'Total'].copy()
            
            # Get CO2 values from lookup table
            co2_values = co2_lookup[zone][sector]
            if co2_values is None:
                print(f"No CO2 values found for {sector} in {zone}, skipping...")
                continue
            
            # Generate Scenario 1: Consumption or Production per capita at 2015 Levels
            scenario_1 = generate_scenario_1(reference_scenario, sector, zone, co2_values)
            if scenario_1:
                new_scenarios.extend(scenario_1)
            
            # Generate Scenario 2: With Sufficiency Measures (20% decrease) - DISABLED
            # scenario_2 = generate_scenario_2(reference_scenario, sector, zone, co2_values)
            # if scenario_2:
            #     new_scenarios.extend(scenario_2)
    
    print(f"\n" + "="*80)
    print(f"GENERATION COMPLETE: Created {len(new_scenarios)} new scenario records")
    print("="*80)
    
    # Convert to DataFrame and save
    if new_scenarios:
        new_scenarios_df = pd.DataFrame(new_scenarios)
        print(f"New scenarios shape: {new_scenarios_df.shape}")
        print(f"New scenarios: {sorted(new_scenarios_df['Scenario'].unique())}")
        
        # Save to CSV
        output_path = os.path.join(DATA_DIR, 'world_sufficiency_lab_scenarios.csv')
        new_scenarios_df.to_csv(output_path, index=False)
        print(f"\n✅ New scenarios saved to: {output_path}")
        
        # Show sample
        print(f"\nSample data:")
        print(new_scenarios_df[['Zone', 'Sector', 'Scenario', 'Lever', 'CO2_2015', 'CO2_2040', 'CO2_2050']].head(10))
    
    return new_scenarios

def generate_scenario_1(base_scenario, sector, zone, co2_values):
    """Generate Scenario 1: Consumption or Production per capita at 2015 Levels"""
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
        new_scenario['Scenario'] = 'World Sufficiency Lab - Consumption or Production per capita at 2015 Levels'
        
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
        
        # Create Total row with CO2 values from lookup
        total_row = {
            'Zone': zone,
            'Sector': sector,
            'Scenario': 'World Sufficiency Lab - Consumption or Production per capita at 2015 Levels',
            'Lever': 'Total',
            'CO2_2015': co2_values['CO2_2015'],
            'CO2_2040': co2_values['CO2_2040'],
            'CO2_2050': co2_values['CO2_2050'],
            'Contrib_2015_2040_abs': total_change_2015_2040,
            'Contrib_2040_2050_abs': total_change_2040_2050,
            'Contrib_2015_2050_abs': total_change_2015_2050,
            'Contrib_2015_2040_pct': 100.0,
            'Contrib_2040_2050_pct': 100.0,
            'Contrib_2015_2050_pct': 100.0
        }
        
        # Convert to list of dictionaries
        result = []
        for _, row in new_scenario.iterrows():
            result.append(row.to_dict())
        result.append(total_row)
        
        return result
        
    except Exception as e:
        print(f"Error generating Scenario 1 for {sector} in {zone}: {e}")
        return None

def generate_scenario_2(base_scenario, sector, zone, co2_values):
    """Generate Scenario 2: With Sufficiency Measures (40% decrease)"""
    try:
        # Get current Sufficiency values
        sufficiency_row = base_scenario[base_scenario['Lever'] == 'Sufficiency']
        if sufficiency_row.empty:
            print(f"No Sufficiency data for {sector} in {zone}")
            return None
            
        sufficiency_2015_2040 = sufficiency_row['Contrib_2015_2040_abs'].iloc[0]
        sufficiency_2040_2050 = sufficiency_row['Contrib_2040_2050_abs'].iloc[0]
        sufficiency_2015_2050 = sufficiency_row['Contrib_2015_2050_abs'].iloc[0]
        
        # Calculate target Sufficiency values (40% decrease in intensity)
        target_sufficiency_2015_2040 = -abs(sufficiency_2015_2040) * 0.4
        target_sufficiency_2040_2050 = -abs(sufficiency_2040_2050) * 0.4
        target_sufficiency_2015_2050 = -abs(sufficiency_2015_2050) * 0.4
        
        # Calculate redistribution amounts
        diff_2015_2040 = sufficiency_2015_2040 - target_sufficiency_2015_2040
        diff_2040_2050 = sufficiency_2040_2050 - target_sufficiency_2040_2050
        diff_2015_2050 = sufficiency_2015_2050 - target_sufficiency_2015_2050
        
        redistribution_per_lever_2015_2040 = diff_2015_2040 / 2
        redistribution_per_lever_2040_2050 = diff_2040_2050 / 2
        redistribution_per_lever_2015_2050 = diff_2015_2050 / 2
        
        # Create new scenario data
        new_scenario = base_scenario.copy()
        new_scenario['Scenario'] = 'World Sufficiency Lab - With Sufficiency Measures'
        
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
        
        # Create Total row with CO2 values from lookup
        total_row = {
            'Zone': zone,
            'Sector': sector,
            'Scenario': 'World Sufficiency Lab - With Sufficiency Measures',
            'Lever': 'Total',
            'CO2_2015': co2_values['CO2_2015'],
            'CO2_2040': co2_values['CO2_2040'],
            'CO2_2050': co2_values['CO2_2050'],
            'Contrib_2015_2040_abs': total_change_2015_2040,
            'Contrib_2040_2050_abs': total_change_2040_2050,
            'Contrib_2015_2050_abs': total_change_2015_2050,
            'Contrib_2015_2040_pct': 100.0,
            'Contrib_2040_2050_pct': 100.0,
            'Contrib_2015_2050_pct': 100.0
        }
        
        # Convert to list of dictionaries
        result = []
        for _, row in new_scenario.iterrows():
            result.append(row.to_dict())
        result.append(total_row)
        
        return result
        
    except Exception as e:
        print(f"Error generating Scenario 2 for {sector} in {zone}: {e}")
        return None

if __name__ == "__main__":
    new_scenarios = create_sufficiency_scenarios()
    if new_scenarios:
        print(f"\n🎉 SUCCESS: Created {len(new_scenarios)} new scenario records!")
        print(f"📁 WSL scenarios saved to: world_sufficiency_lab_scenarios.csv")
        print(f"💡 Run data_preprocessing.py to combine with main data")
    else:
        print("\n❌ ERROR: No scenarios generated!") 