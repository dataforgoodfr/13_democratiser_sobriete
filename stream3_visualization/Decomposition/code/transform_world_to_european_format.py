#!/usr/bin/env python3
"""
Transform World Decomposition Data to European Format

This script transforms the world decomposition data from its current structure:
- Columns: Model, Scenario, Region, Sector, Variable_clean, Unit, 2015, 2040, 2050
- Rows: Each combination of Model+Scenario+Region+Sector+Variable

To the European format structure:
- Columns: Geography, Sector, Scenario, Year, Population, Volume, Volume Unit, Energy, CO2
- Rows: Each combination of Geography+Sector+Scenario+Year
"""

import pandas as pd
import numpy as np

def transform_world_to_european_format():
    """
    Transform world decomposition data to match European format
    """
    
    # Load the world data
    print("Loading world data...")
    world_data = pd.read_csv('Output/world_data_preprocessed.csv')
    
    print(f"Original data shape: {world_data.shape}")
    print(f"Columns: {list(world_data.columns)}")
    print(f"Variables: {world_data['Variable_clean'].unique()}")
    
    # Step 1: Concatenate Model + Scenario
    print("\nStep 1: Concatenating Model + Scenario...")
    world_data['Scenario'] = world_data['Model'] + ' - ' + world_data['Scenario']
    
    # Step 2: Rename columns to match European format
    print("Step 2: Renaming columns...")
    world_data = world_data.rename(columns={
        'Region': 'Geography',
        'Sector': 'Sector'
    })
    
    # Step 3: Apply unit conversions
    print("Step 3: Applying unit conversions...")
    
    # Energy Service: bn m² to million m² (×1000)
    energy_service_mask = world_data['Variable_clean'] == 'Energy Service'
    world_data.loc[energy_service_mask, '2015'] = world_data.loc[energy_service_mask, '2015'] * 1000
    world_data.loc[energy_service_mask, '2040'] = world_data.loc[energy_service_mask, '2040'] * 1000
    world_data.loc[energy_service_mask, '2050'] = world_data.loc[energy_service_mask, '2050'] * 1000
    
    # Final Energy: Exajoule to Mtoe (×23.88458966275)
    final_energy_mask = world_data['Variable_clean'] == 'Final Energy'
    world_data.loc[final_energy_mask, '2015'] = world_data.loc[final_energy_mask, '2015'] * 23.88458966275
    world_data.loc[final_energy_mask, '2040'] = world_data.loc[final_energy_mask, '2040'] * 23.88458966275
    world_data.loc[final_energy_mask, '2050'] = world_data.loc[final_energy_mask, '2050'] * 23.88458966275
    
    print("Unit conversions applied:")
    print("- Energy Service: bn m² → million m² (×1000)")
    print("- Final Energy: Exajoule → Mtoe (×23.88458966275)")
    
    # Step 4: Transform years from columns to rows
    print("\nStep 4: Transforming years from columns to rows...")
    
    # Melt the year columns
    world_data_melted = world_data.melt(
        id_vars=['Geography', 'Sector', 'Scenario', 'Variable_clean', 'Unit'],
        value_vars=['2015', '2040', '2050'],
        var_name='Year',
        value_name='Value'
    )
    
    # Convert Year to integer
    world_data_melted['Year'] = world_data_melted['Year'].astype(int)
    
    print(f"After melting: {world_data_melted.shape}")
    
    # Step 5: Transform variables from rows to columns
    print("\nStep 5: Transforming variables from rows to columns...")
    
    # Pivot the variables to columns
    world_data_pivoted = world_data_melted.pivot_table(
        index=['Geography', 'Sector', 'Scenario', 'Year'],
        columns='Variable_clean',
        values='Value',
        aggfunc='first'
    ).reset_index()
    
    print(f"After pivoting: {world_data_pivoted.shape}")
    
    # Step 6: Rename columns to match European format
    print("Step 6: Renaming columns to match European format...")
    
    # Create the final structure
    final_data = pd.DataFrame()
    final_data['Geography'] = world_data_pivoted['Geography']
    final_data['Sector'] = world_data_pivoted['Sector']
    final_data['Scenario'] = world_data_pivoted['Scenario']
    final_data['Year'] = world_data_pivoted['Year']
    
    # Add Population column
    if 'Population' in world_data_pivoted.columns:
        final_data['Population (Mill)'] = world_data_pivoted['Population']
    else:
        final_data['Population (Mill)'] = np.nan
    
    # Add Volume column (Energy Service)
    if 'Energy Service' in world_data_pivoted.columns:
        final_data['Volume'] = world_data_pivoted['Energy Service']
        final_data['Volume Unit'] = 'Floor area (Million m²)'
    else:
        final_data['Volume'] = np.nan
        final_data['Volume Unit'] = 'Floor area (Million m²)'
    
    # Add Energy column (Final Energy)
    if 'Final Energy' in world_data_pivoted.columns:
        final_data['Energy (Million toe)'] = world_data_pivoted['Final Energy']
    else:
        final_data['Energy (Million toe)'] = np.nan
    
    # Add CO2 column (Emissions)
    if 'Emissions' in world_data_pivoted.columns:
        final_data['CO2 (Million tonn)'] = world_data_pivoted['Emissions']
    else:
        final_data['CO2 (Million tonn)'] = np.nan
    
    # Step 7: Sort and clean up
    print("Step 7: Final cleanup and sorting...")
    
    final_data = final_data.sort_values(['Geography', 'Sector', 'Scenario', 'Year'])
    
    # Remove rows where all data columns are NaN
    data_columns = ['Population (Mill)', 'Volume', 'Energy (Million toe)', 'CO2 (Million tonn)']
    final_data = final_data.dropna(subset=data_columns, how='all')
    
    print(f"Final data shape: {final_data.shape}")
    print(f"Final columns: {list(final_data.columns)}")
    
    # Step 8: Save the transformed data
    print("\nStep 8: Saving transformed data...")
    output_file = 'Output/world_data_european_format.csv'
    final_data.to_csv(output_file, index=False)
    print(f"Transformed data saved to: {output_file}")
    
    # Display sample of the transformed data
    print("\nSample of transformed data:")
    print(final_data.head(10))
    
    # Display summary statistics
    print("\nSummary by Geography:")
    print(final_data.groupby('Geography').size())
    
    print("\nSummary by Sector:")
    print(final_data.groupby('Sector').size())
    
    print("\nSummary by Scenario:")
    print(final_data.groupby('Scenario').size())
    
    return final_data

if __name__ == "__main__":
    try:
        transformed_data = transform_world_to_european_format()
        print("\n✅ Transformation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during transformation: {e}")
        import traceback
        traceback.print_exc() 