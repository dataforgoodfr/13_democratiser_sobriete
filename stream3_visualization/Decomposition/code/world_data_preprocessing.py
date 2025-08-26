#!/usr/bin/env python3
"""
World Data Preprocessing for CO2 Decomposition Analysis
Handles REMIND data structure and creates proper aggregates
"""

import pandas as pd
import numpy as np
import os

class WorldCO2DecompositionPreprocessor:
    """
    Data preprocessor for world CO2 emission decomposition analysis
    Handles world data from REMIND-MAgPIE scenarios with proper LMDI decomposition
    Mirrors the structure of the EU/Switzerland preprocessor
    """
    
    def __init__(self, data_file="../Output/world_data_european_format.csv"):
        self.data_file = data_file
        
        # World regions (geographies) from the data
        self.world_regions = [
            "Canada, Australia, New Zealand", "China", "Europe", "India", "Japan",
            "Latin America and Caribbean", "Middle East and Africa", "Non-EU Europe",
            "Other Asia", "Russia and Former Soviet Union", "Sub-Saharan Africa",
            "United States", "WLD"
        ]
        
        # Sector configuration (only one sector available in world data)
        self.sector_configs = {
            "Buildings - Residential and Commercial": [
                "REMIND-MAgPIE 3.2-4.6 - SDP_EI-1p5C-CCimp",
                "REMIND-MAgPIE 3.2-4.6 - SDP_MC-1p5C-CCimp", 
                "REMIND-MAgPIE 3.2-4.6 - SDP_RC-1p5C-CCimp",
                "REMIND-MAgPIE 3.2-4.6 - SSP1-1p5C",
                "REMIND-MAgPIE 3.2-4.6 - SSP1-NPi",
                "REMIND-MAgPIE 3.2-4.6 - SSP2-1p5C-CCimp",
                "REMIndia-MAgPIE 3.2-4.6 - SSP2-NPi-CCimp"
            ]
        }
        
        # Column mapping (same as European format)
        self.sector_columns = ["Year", "Population (Mill)", "Volume", "Energy (Million toe)", "CO2 (Million tonn)"]
        
        # Universal lever names (same as EU/Switzerland)
        self.universal_levers = ["Population", "Sufficiency", "Energy Efficiency", "Supply Side Decarbonation"]
        
        # Scenario name mappings for display (internal name -> display name)
        self.scenario_display_names = {
            "REMIND-MAgPIE 3.2-4.6 - SDP_EI-1p5C-CCimp": "REMIND SDP_EI-1p5C-CCimp",
            "REMIND-MAgPIE 3.2-4.6 - SDP_MC-1p5C-CCimp": "REMIND SDP_MC-1p5C-CCimp",
            "REMIND-MAgPIE 3.2-4.6 - SDP_RC-1p5C-CCimp": "REMIND SDP_RC-1p5C-CCimp",
            "REMIND-MAgPIE 3.2-4.6 - SSP1-1p5C": "REMIND SSP1-1p5C",
            "REMIND-MAgPIE 3.2-4.6 - SSP1-NPi": "REMIND SSP1-NPi",
            "REMIND-MAgPIE 3.2-4.6 - SSP2-1p5C-CCimp": "REMIND SSP2-1p5C-CCimp",
            "REMIndia-MAgPIE 3.2-4.6 - SSP2-NPi-CCimp": "REMIndia SSP2-NPi-CCimp"
        }
        
        # Sector name mappings for display (internal name -> display name)
        self.sector_display_names = {
            "Buildings - Residential and Commercial": "Buildings - Residential and Commercial"
        }
    
    def calculate_intensity_factors(self, data_rows, sector):
        """Calculate intensity factors for LMDI decomposition"""
        intensity_factors = {}
        
        # All sectors use the same unified column structure
        intensity_factors["Population"] = data_rows["Population (Mill)"]
        intensity_factors["Sufficiency"] = data_rows["Volume"] / data_rows["Population (Mill)"]
        intensity_factors["Energy Efficiency"] = data_rows["Energy (Million toe)"] / data_rows["Volume"]
        intensity_factors["Supply Side Decarbonation"] = data_rows["CO2 (Million tonn)"] / data_rows["Energy (Million toe)"]
        
        return intensity_factors
    
    def safe_lmdi_contribution(self, co2_0, co2_t, x_0, x_t):
        """Calculate LMDI contribution with safety checks"""
        if co2_0 == co2_t or x_0 == 0 or x_t == 0:
            return 0
        try:
            weight = (co2_t - co2_0) / (np.log(np.abs(co2_t)) - np.log(np.abs(co2_0)))
            return weight * np.log(np.abs(x_t) / np.abs(x_0))
        except:
            return 0
    
    def process_world_data(self):
        """Process world data from the CSV file"""
        if not os.path.exists(self.data_file):
            print(f"Warning: World data file not found: {self.data_file}")
            return []
            
        all_scenario_data = []
        
        try:
            # Load world data
            df_world = pd.read_csv(self.data_file)
            print(f"World data loaded, shape: {df_world.shape}")
            
            # Process each geography and scenario combination
            for geography in self.world_regions:
                for sector, scenarios in self.sector_configs.items():
                    for scenario in scenarios:
                        try:
                            # Filter data for this geography, sector and scenario
                            sector_data = df_world[
                                (df_world["Geography"] == geography) & 
                                (df_world["Sector"] == sector) & 
                                (df_world["Scenario"] == scenario)
                            ].copy()
                            
                            if sector_data.empty:
                                print(f"Warning: No data found for {geography} - {sector} - {scenario}")
                                continue
                            
                            # Filter to only years 2015, 2040, 2050
                            year_data = sector_data[sector_data["Year"].isin([2015, 2040, 2050])].copy()
                            
                            if len(year_data) != 3:
                                print(f"Warning: Missing years for {geography} - {sector} - {scenario}. Found: {year_data['Year'].tolist()}")
                                continue
                            
                            # Set year as index for easier access
                            year_data.set_index("Year", inplace=True)
                            
                            # Convert only the numerical columns to float
                            numerical_columns = ["Population (Mill)", "Volume", "Energy (Million toe)", "CO2 (Million tonn)"]
                            for col in numerical_columns:
                                if col in year_data.columns:
                                    year_data[col] = pd.to_numeric(year_data[col], errors='coerce')
                            
                            # Extract CO2 values
                            co2_2015 = year_data.loc[2015, "CO2 (Million tonn)"]
                            co2_2040 = year_data.loc[2040, "CO2 (Million tonn)"]
                            co2_2050 = year_data.loc[2050, "CO2 (Million tonn)"]
                            
                            # Calculate intensity factors
                            intensity_factors = self.calculate_intensity_factors(year_data, sector)
                            
                            # Calculate LMDI contributions for each period
                            contrib_2015_2040 = {}
                            contrib_2040_2050 = {}
                            
                            for lever in self.universal_levers:
                                # 2015-2040 period
                                x0, xt = intensity_factors[lever].loc[2015], intensity_factors[lever].loc[2040]
                                contrib_2015_2040[lever] = self.safe_lmdi_contribution(co2_2015, co2_2040, x0, xt)
                                
                                # 2040-2050 period
                                x0b, xtb = intensity_factors[lever].loc[2040], intensity_factors[lever].loc[2050]
                                contrib_2040_2050[lever] = self.safe_lmdi_contribution(co2_2040, co2_2050, x0b, xtb)
                            
                            # Store data
                            all_scenario_data.append({
                                "Zone": geography,
                                "Sector": sector,
                                "Scenario": scenario,
                                "CO2_2015": co2_2015,
                                "CO2_2040": co2_2040,
                                "CO2_2050": co2_2050,
                                "Contrib_2015_2040": contrib_2015_2040,
                                "Contrib_2040_2050": contrib_2040_2050,
                                "Intensity_Factors": intensity_factors
                            })
                            
                        except Exception as e:
                            print(f"Error processing {geography} scenario '{scenario}' in {sector}: {e}")
                            continue
                            
        except Exception as e:
            print(f"Error processing world data: {e}")
            return []
        
        return all_scenario_data
    
    def create_unified_dataset(self):
        """Create a unified dataset for all world regions, sectors, and scenarios"""
        all_data = []
        
        print("Processing world data...")
        all_data = self.process_world_data()
        
        # Process all scenarios into lever-level data
        unified_data = []
        
        for scenario_data in all_data:
            zone = scenario_data["Zone"]
            sector = scenario_data["Sector"]
            scenario = scenario_data["Scenario"]
            
            # Apply display name mappings
            display_sector = self.sector_display_names.get(sector, sector)
            display_scenario = self.scenario_display_names.get(scenario, scenario)
            
            co2_2015 = scenario_data["CO2_2015"]
            co2_2040 = scenario_data["CO2_2040"]
            co2_2050 = scenario_data["CO2_2050"]
            
            contrib_2015_2040 = scenario_data["Contrib_2015_2040"]
            contrib_2040_2050 = scenario_data["Contrib_2040_2050"]
            
            # Calculate total changes
            total_change_2015_2040 = co2_2040 - co2_2015
            total_change_2040_2050 = co2_2050 - co2_2040
            total_change_2015_2050 = co2_2050 - co2_2015
            
            # Add "Total" lever first (shows actual CO2 emissions)
            unified_data.append({
                "Zone": zone,
                "Sector": display_sector,
                "Scenario": display_scenario,
                "Lever": "Total",
                "CO2_2015": co2_2015,
                "CO2_2040": co2_2040,
                "CO2_2050": co2_2050,
                "Contrib_2015_2040_abs": total_change_2015_2040,
                "Contrib_2040_2050_abs": total_change_2040_2050,
                "Contrib_2015_2050_abs": total_change_2015_2050,
                "Contrib_2015_2040_pct": 100.0,  # Total represents 100% of change
                "Contrib_2040_2050_pct": 100.0,
                "Contrib_2015_2050_pct": 100.0
            })
            
            # Add data for each individual lever
            for lever in self.universal_levers:
                # 2015-2040 period
                contrib_abs_1 = contrib_2015_2040[lever]
                contrib_pct_1 = (contrib_abs_1 / abs(total_change_2015_2040)) * 100 if total_change_2015_2040 != 0 else 0
                
                # 2040-2050 period  
                contrib_abs_2 = contrib_2040_2050[lever]
                contrib_pct_2 = (contrib_abs_2 / abs(total_change_2040_2050)) * 100 if total_change_2040_2050 != 0 else 0
                
                # Total period
                contrib_abs_total = contrib_abs_1 + contrib_abs_2
                contrib_pct_total = (contrib_abs_total / abs(total_change_2015_2050)) * 100 if total_change_2015_2050 != 0 else 0
                
                # Flip the sign convention: emissions reductions = positive, emissions increases = negative
                contrib_pct_1 = -contrib_pct_1
                contrib_pct_2 = -contrib_pct_2
                contrib_pct_total = -contrib_pct_total
                
                unified_data.append({
                    "Zone": zone,
                    "Sector": display_sector,
                    "Scenario": display_scenario,
                    "Lever": lever,
                    "CO2_2015": None,  # N/A for individual levers
                    "CO2_2040": None,   # N/A for individual levers
                    "CO2_2050": None,   # N/A for individual levers
                    "Contrib_2015_2040_abs": contrib_abs_1,
                    "Contrib_2040_2050_abs": contrib_abs_2,
                    "Contrib_2015_2050_abs": contrib_abs_total,
                    "Contrib_2015_2040_pct": contrib_pct_1,
                    "Contrib_2040_2050_pct": contrib_pct_2,
                    "Contrib_2015_2050_pct": contrib_pct_total
                })
        
        return pd.DataFrame(unified_data)
    
    def create_intermediary_dataset(self, all_data):
        """Create an intermediary dataset with raw data and calculated intensity factors"""
        intermediary_data = []
        
        for scenario_data in all_data:
            zone = scenario_data["Zone"]
            sector = scenario_data["Sector"]
            scenario = scenario_data["Scenario"]
            
            # Apply display name mappings
            display_sector = self.sector_display_names.get(sector, sector)
            display_scenario = self.scenario_display_names.get(scenario, scenario)
            
            co2_2015 = scenario_data["CO2_2015"]
            co2_2040 = scenario_data["CO2_2040"]
            co2_2050 = scenario_data["CO2_2050"]
            
            intensity_factors = scenario_data["Intensity_Factors"]
            
            # Get raw data for each year
            for year in [2015, 2040, 2050]:
                row_data = {
                    "Zone": zone,
                    "Sector": display_sector,
                    "Scenario": display_scenario,
                    "Year": year,
                    "CO2_2015": co2_2015,
                    "CO2_2040": co2_2040,
                    "CO2_2050": co2_2050,
                    "Population": intensity_factors["Population"].loc[year],
                    "Volume": intensity_factors["Sufficiency"].loc[year] * intensity_factors["Population"].loc[year],  # Volume = Sufficiency * Population
                    "Energy": intensity_factors["Energy Efficiency"].loc[year] * intensity_factors["Sufficiency"].loc[year] * intensity_factors["Population"].loc[year],  # Energy = Energy Efficiency * Volume
                    "CO2": intensity_factors["Supply Side Decarbonation"].loc[year] * intensity_factors["Energy Efficiency"].loc[year] * intensity_factors["Sufficiency"].loc[year] * intensity_factors["Population"].loc[year],  # CO2 = Carbon Intensity * Energy
                    "Population_Intensity": intensity_factors["Population"].loc[year],
                    "Sufficiency_Intensity": intensity_factors["Sufficiency"].loc[year],
                    "Energy_Efficiency_Intensity": intensity_factors["Energy Efficiency"].loc[year],
                    "Carbon_Intensity": intensity_factors["Supply Side Decarbonation"].loc[year]
                }
                intermediary_data.append(row_data)
        
        return pd.DataFrame(intermediary_data)
    
    def save_processed_data(self, output_dir="../Output"):
        """Process and save the unified world dataset"""
        print("Starting world data processing...")
        
        # Create unified dataset
        df_unified = self.create_unified_dataset()
        
        if df_unified.empty:
            print("Warning: No world data was processed!")
            return None
        
        # Create intermediary dataset for auditing
        print("Creating intermediary dataset for auditing...")
        all_data = self.process_world_data()
        df_intermediary = self.create_intermediary_dataset(all_data)
        
        # Save both datasets
        unified_path = os.path.join(output_dir, "world_unified_decomposition_data.csv")
        intermediary_path = os.path.join(output_dir, "world_intermediary_decomposition_data.csv")
        
        df_unified.to_csv(unified_path, index=False)
        df_intermediary.to_csv(intermediary_path, index=False)
        
        print(f"World unified dataset saved to: {unified_path}")
        print(f"World intermediary dataset saved to: {intermediary_path}")
        
        # Summary statistics for validation (not saved to file)
        total_scenarios = len(df_unified.groupby(["Zone", "Sector", "Scenario"]).size())
        
        # Data validation summary
        print("\nWorld Data Validation Summary:")
        print(f"Total scenarios processed: {total_scenarios}")
        print(f"Zones: {sorted(df_unified['Zone'].unique())}")
        print(f"Sectors: {sorted(df_unified['Sector'].unique())}")
        print(f"Scenarios: {sorted(df_unified['Scenario'].unique())}")
        print(f"Levers: {sorted(df_unified['Lever'].unique())}")
        print(f"\nIntermediary dataset shape: {df_intermediary.shape}")
        print(f"Intermediary dataset columns: {df_intermediary.columns.tolist()}")
        
        return df_unified

if __name__ == "__main__":
    # Initialize world preprocessor
    preprocessor = WorldCO2DecompositionPreprocessor()
    
    # Process and save world data
    df_processed = preprocessor.save_processed_data()
    
    if df_processed is not None:
        print("\nWorld data processing complete!")
        print(f"Total records: {len(df_processed)}")
        
        # Show sample data
        print("\nSample world data:")
        print(df_processed.head(10))
        
    else:
        print("\nWorld data processing failed!") 