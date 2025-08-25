#!/usr/bin/env python3
"""
World Data Preprocessing for CO2 Decomposition Analysis
Handles REMIND data structure and creates proper aggregates
"""

import pandas as pd
import numpy as np
import os

class WorldDataPreprocessor:
    """
    Preprocessor for world data from REMIND model
    Handles different data structure and creates proper aggregates
    """
    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.world_data_file = "2025-08-20_REMIND Shape_Data_Compiled.xlsx"
        
        # Gases to aggregate into Emissions
        self.gases = ['BC', 'CO', 'CO2', 'NH3', 'NO2', 'OC', 'SO2', 'VOC']
        
        # Target years for analysis
        self.target_years = [2015, 2040, 2050]
        
        # Region mapping to IPCC standard names
        self.region_mapping = {
            'CAZ': 'Canada, Australia, New Zealand',
            'CHA': 'China',
            'EUR': 'Europe',
            'IND': 'India', 
            'JPN': 'Japan',
            'LAM': 'Latin America and Caribbean',
            'MEA': 'Middle East and Africa',
            'NEU': 'Non-EU Europe',
            'OAS': 'Other Asia',
            'REF': 'Russia and Former Soviet Union',
            'SSA': 'Sub-Saharan Africa',
            'USA': 'United States',
            'China': 'China',
            'India': 'India',
            'United States of America': 'United States',
            'WLD': 'World'
        }
        
    def load_world_data(self):
        """Load world data from Excel file"""
        file_path = os.path.join(self.data_dir, self.world_data_file)
        print(f"Loading world data from: {file_path}")
        
        try:
            # Load the Combined_data sheet
            data = pd.read_excel(file_path, sheet_name="Combined_data")
            print(f"Data loaded successfully! Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            return data
        except Exception as e:
            print(f"Error loading world data: {e}")
            return None
    
    def clean_region_names(self, data):
        """Clean region names and map to IPCC standard names"""
        print("Cleaning and mapping region names...")
        
        # Remove the model prefix from region names
        data['Region'] = data['Region'].str.replace('REMIndia-MAgPIE 3.2-4.6|', '')
        data['Region'] = data['Region'].str.replace('REMIND-MAgPIE 3.2-4.6|', '')
        
        # Map regions to IPCC standard names
        data['Region'] = data['Region'].map(self.region_mapping).fillna(data['Region'])
        
        # Get unique regions
        unique_regions = data['Region'].unique()
        print(f"Unique regions after mapping: {unique_regions}")
        
        return data
    
    def add_sector_column(self, data):
        """Add sector column with fixed value for all rows"""
        print("Adding sector column...")
        
        # All data is for Buildings - Residential and Commercial sector
        data['Sector'] = 'Buildings - Residential and Commercial'
        
        print("Sector column added with value: 'Buildings - Residential and Commercial'")
        
        return data
    
    def identify_emission_variables(self, data):
        """Identify which variables are emissions that need to be aggregated"""
        print("Identifying emission variables...")
        
        # Check if the variable contains any of the gases
        data['is_emission'] = data['Variable'].str.contains('|'.join(self.gases), case=False, na=False)
        
        # Also check if it's explicitly an emissions variable
        data['is_emission'] = data['is_emission'] | data['Variable'].str.contains('Emissions', case=False, na=False)
        
        # Explicitly exclude non-emission variables that might contain "Energy" in their name
        data['is_emission'] = data['is_emission'] & ~data['Variable'].str.contains('Final Energy', case=False, na=False)
        data['is_emission'] = data['is_emission'] & ~data['Variable'].str.contains('Energy Service', case=False, na=False)
        data['is_emission'] = data['is_emission'] & ~data['Variable'].str.contains('Population', case=False, na=False)
        
        emission_count = data['is_emission'].sum()
        non_emission_count = (~data['is_emission']).sum()
        
        print(f"Emission variables: {emission_count}")
        print(f"Non-emission variables: {non_emission_count}")
        
        # Show what variables are being classified as what
        print("\nEmission variables found:")
        emission_vars = data[data['is_emission']]['Variable'].unique()
        print(emission_vars)
        
        print("\nNon-emission variables found:")
        non_emission_vars = data[~data['is_emission']]['Variable'].unique()
        print(non_emission_vars)
        
        return data
    
    def aggregate_emissions(self, data):
        """Aggregate emission variables into a single Emissions variable"""
        print("Aggregating emission variables...")
        
        # Get emission data
        emission_data = data[data['is_emission']].copy()
        
        # Get year columns
        year_cols = [col for col in data.columns if str(col).isdigit()]
        
        # Convert comma decimal separator to dot if present
        for year_col in year_cols:
            emission_data[year_col] = pd.to_numeric(
                emission_data[year_col].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
        
        # Group by Model, Scenario, Region, Sector and sum emissions
        group_cols = ['Model', 'Scenario', 'Region', 'Sector']
        
        aggregated_emissions = emission_data.groupby(group_cols)[year_cols].sum().reset_index()
        
        # Create the new Emissions variable
        aggregated_emissions['Variable_clean'] = 'Emissions'
        aggregated_emissions['Unit'] = 'Mt CO2e/yr'
        
        print(f"Aggregated emissions shape: {aggregated_emissions.shape}")
        
        return aggregated_emissions
    
    def prepare_non_emission_data(self, data):
        """Prepare non-emission data for final aggregation"""
        print("Preparing non-emission data...")
        
        # Get non-emission data
        non_emission_data = data[~data['is_emission']].copy()
        
        # Clean up variable names using the new logic:
        # 1. Take everything left of the second "|" (or keep whole if fewer than 2 "|")
        # 2. Remove "|Residential and Commercial" if present
        def clean_variable_name(var_name):
            if pd.isna(var_name):
                return var_name
            
            parts = var_name.split('|')
            if len(parts) >= 2:
                # Take everything left of the second "|"
                result = '|'.join(parts[:2])
            else:
                # Keep the whole string if fewer than 2 "|"
                result = var_name
            
            # Remove "|Residential and Commercial" if present
            result = result.replace('|Residential and Commercial', '')
            
            return result
        
        non_emission_data['Variable_clean'] = non_emission_data['Variable'].apply(clean_variable_name)
        
        # Select only the columns we need
        year_cols = [col for col in data.columns if str(col).isdigit()]
        essential_cols = ['Model', 'Scenario', 'Region', 'Sector', 'Variable_clean', 'Unit'] + year_cols
        
        # Ensure all columns exist
        for col in essential_cols:
            if col not in non_emission_data.columns:
                non_emission_data[col] = ''
        
        non_emission_data = non_emission_data[essential_cols]
        
        print(f"Non-emission data shape: {non_emission_data.shape}")
        
        # Show what variables we have
        print("\nNon-emission variables found:")
        unique_vars = non_emission_data['Variable_clean'].unique()
        for var in unique_vars:
            count = len(non_emission_data[non_emission_data['Variable_clean'] == var])
            print(f"  {var}: {count} rows")
        
        return non_emission_data
    
    def create_world_aggregate(self, data):
        """Create world aggregate by summing all regions"""
        print("Creating world aggregate...")
        
        # Get year columns
        year_cols = [col for col in data.columns if str(col).isdigit()]
        
        # Group by Model, Scenario, Sector, Variable_clean and sum across all regions
        group_cols = ['Model', 'Scenario', 'Sector', 'Variable_clean', 'Unit']
        
        world_aggregate = data.groupby(group_cols)[year_cols].sum().reset_index()
        
        # Add 'WLD' as region
        world_aggregate['Region'] = 'WLD'
        
        print(f"World aggregate shape: {world_aggregate.shape}")
        
        return world_aggregate
    
    def filter_target_years(self, data):
        """Filter data to only include target years and essential columns"""
        print("Filtering to target years...")
        
        # Get year columns that match our target years
        available_years = [col for col in data.columns if str(col).isdigit()]
        target_year_cols = [col for col in available_years if int(col) in self.target_years]
        
        # Essential non-year columns
        essential_cols = ['Model', 'Scenario', 'Region', 'Sector', 'Variable_clean', 'Unit']
        
        # Combine essential and target year columns
        final_cols = essential_cols + target_year_cols
        
        # Filter data
        filtered_data = data[final_cols].copy()
        
        print(f"Filtered data shape: {filtered_data.shape}")
        print(f"Target years available: {target_year_cols}")
        
        return filtered_data
    
    def process_world_data(self):
        """Main processing pipeline for world data"""
        print("="*80)
        print("WORLD DATA PREPROCESSING - COMPLETE PIPELINE")
        print("="*80)
        
        # Load data
        data = self.load_world_data()
        if data is None:
            return None
        
        # Step 1.1: Clean region names and map to IPCC standards
        data = self.clean_region_names(data)
        
        # Step 1.2: Add sector column with fixed value
        data = self.add_sector_column(data)
        
        # Step 1.3: Identify emission variables
        data = self.identify_emission_variables(data)
        
        # Step 1.4: Aggregate emissions
        aggregated_emissions = self.aggregate_emissions(data)
        
        # Step 1.5: Prepare non-emission data
        non_emission_data = self.prepare_non_emission_data(data)
        
        # Step 1.6: Combine aggregated emissions with non-emission data
        combined_data = pd.concat([aggregated_emissions, non_emission_data], ignore_index=True)
        
        # Step 1.7: Create world aggregate
        world_aggregate = self.create_world_aggregate(combined_data)
        
        # Step 1.8: Combine regional and world data
        final_combined_data = pd.concat([combined_data, world_aggregate], ignore_index=True)
        
        # Step 1.9: Filter to target years
        final_data = self.filter_target_years(final_combined_data)
        
        # Save only the final comprehensive output
        output_dir = os.path.join(self.data_dir, '..', 'Output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final comprehensive data
        final_output_path = os.path.join(output_dir, 'world_data_preprocessed.csv')
        final_data.to_csv(final_output_path, index=False)
        print(f"Final comprehensive world data saved to: {final_output_path}")
        
        print("\n" + "="*80)
        print("WORLD DATA PREPROCESSING COMPLETED")
        print("="*80)
        
        return final_data

def main():
    """Main function to run world data preprocessing"""
    preprocessor = WorldDataPreprocessor()
    result = preprocessor.process_world_data()
    
    if result is not None:
        print("\nWorld data preprocessing completed successfully!")
        print(f"Final data shape: {result.shape}")
        print(f"Final columns: {list(result.columns)}")
    else:
        print("\nWorld data preprocessing failed!")

if __name__ == "__main__":
    main() 