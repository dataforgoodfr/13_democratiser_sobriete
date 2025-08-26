import pandas as pd
import numpy as np
import os

class CO2DecompositionPreprocessor:
    """
    Comprehensive data preprocessor for CO2 emission decomposition analysis
    Handles EU and Switzerland data sources with proper LMDI decomposition
    """
    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.zones = ["EU", "Switzerland"]  # World not available yet
        
        # File mappings for each zone
        self.zone_files = {
            "EU": "2025-04-28_EC scenarios data_Decomposition_compiled.xlsx",
            "Switzerland": "2025-08-13_CH scenarios data_Decomposition_Compiled.xlsx"
        }
        
        # Sector configurations for each zone
        self.sector_configs = {
            "EU": {
                "Buildings-Residential": ["Scenario 1", "Scenario 2", "Scenario 3", "Life Scenario"],
                "Buildings -Services": ["Scenario 1", "Scenario 2", "Scenario 3", "Life Scenario"],
                "Industry": ["Scenario 1", "Scenario 2", "Scenario 3", "Life Scenario"],
                "PassLandTransport": ["Scenario 1", "Scenario 2", "Scenario 3", "Life Scenario"]
            },
            "Switzerland": {
                "Buildings-Residential": ["Scenario Basis", "Scenario Zer0 A", "Scenario Zer0 B", "Scenario Zer0 C"],
                "Buildings -Services": ["Scenario Basis", "Scenario Zer0 A", "Scenario Zer0 B", "Scenario Zer0 C"],
                "PassLandTransport": ["Scenario Basis", "Scenario Zer0 A", "Scenario Zer0 B", "Scenario Zer0 C"]
            }
        }
        
        # Column mapping for each sector (actual column names in Excel)
        self.sector_columns = {
            "Buildings-Residential": ["Year", "Population (Million)", "Volume", "Energy (Million toe)", "CO2 (Million tonnes)"],
            "Buildings -Services": ["Year", "Population (Million)", "Volume", "Energy (Million toe)", "CO2 (Million tonnes)"],
            "Industry": ["Year", "Population (Million)", "Volume", "Energy (Million toe)", "CO2 (Million tonnes)"],
            "PassLandTransport": ["Year", "Population (Million)", "Volume", "Energy (Million toe)", "CO2 (Million tonnes)"]
        }
        
        # Switzerland specific column mapping for the new compiled structure
        self.switzerland_columns = ["Geography", "Sector", "Scenario", "Year", "Population (Million)", "Volume", "Volume Unit", "Energy (Million toe)", "CO2 (Million tonnes)"]
        
        # EU specific column mapping for the new compiled structure  
        self.eu_columns = ["Geography", "Sector", "Scenario", "Year", "Population (Million)", "Volume", "Volume Unit", "Energy (Million toe)", "CO2 (Million tonnes)"]
        
        # Universal lever names
        self.universal_levers = ["Population", "Sufficiency", "Energy Efficiency", "Supply Side Decarbonation"]
        
        # Scenario name mappings for display (internal name -> display name)
        self.scenario_display_names = {
            "EU": {
                "Scenario 1": "EU Commission Fit-for-55",
                "Scenario 2": "EU Commission >85% Decrease by 2040", 
                "Scenario 3": "EU Commission >90% Decrease by 2040",
                "Life Scenario": "EU Commission LIFE Scenario"
            },
            "Switzerland": {
                "Scenario Basis": "Base Scenario",
                "Scenario Zer0 A": "Scenario Zer0 A",
                "Scenario Zer0 B": "Scenario Zer0 B", 
                "Scenario Zer0 C": "Scenario Zer0 C"
            }
        }
        
        # Sector name mappings for display (internal name -> display name)
        self.sector_display_names = {
            "Buildings-Residential": "Buildings - Residential",
            "Buildings -Services": "Buildings - Services", 
            "PassLandTransport": "Passenger Land Transportation",
            "Industry": "Industry"
        }
    
    def calculate_intensity_factors(self, data_rows, sector):
        """Calculate intensity factors for LMDI decomposition"""
        intensity_factors = {}
        
        # All sectors now use the same unified column structure
        intensity_factors["Population"] = data_rows["Population (Million)"]
        intensity_factors["Sufficiency"] = data_rows["Volume"] / data_rows["Population (Million)"]
        intensity_factors["Energy Efficiency"] = data_rows["Energy (Million toe)"] / data_rows["Volume"]
        intensity_factors["Supply Side Decarbonation"] = data_rows["CO2 (Million tonnes)"] / data_rows["Energy (Million toe)"]
        
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
    
    def process_zone_data(self, zone):
        """Process data for a specific zone"""
        file_path = os.path.join(self.data_dir, self.zone_files[zone])
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found for {zone}: {file_path}")
            return []
            
        all_scenario_data = []
        
        try:
            # Both zones now use the same "All Sectors" sheet structure
            df_sheet = pd.read_excel(file_path, sheet_name="All Sectors")
            print(f"{zone} data loaded from 'All Sectors' sheet, shape: {df_sheet.shape}")
            
            # Process each sector and scenario combination
            for sector, scenarios in self.sector_configs[zone].items():
                for scenario in scenarios:
                    try:
                        # Filter data for this sector and scenario
                        sector_data = df_sheet[
                            (df_sheet["Sector"] == sector) & 
                            (df_sheet["Scenario"] == scenario)
                        ].copy()
                        
                        if sector_data.empty:
                            print(f"Warning: No data found for {sector} - {scenario}")
                            continue
                        
                        # Filter to only years 2015, 2040, 2050
                        year_data = sector_data[sector_data["Year"].isin([2015, 2040, 2050])].copy()
                        
                        if len(year_data) != 3:
                            print(f"Warning: Missing years for {sector} - {scenario}. Found: {year_data['Year'].tolist()}")
                            continue
                        
                        # Set year as index for easier access
                        year_data.set_index("Year", inplace=True)
                        
                        # Convert only the numerical columns to float, excluding Geography, Sector, and Scenario
                        numerical_columns = ["Population (Million)", "Volume", "Energy (Million toe)", "CO2 (Million tonnes)"]
                        for col in numerical_columns:
                            if col in year_data.columns:
                                year_data[col] = pd.to_numeric(year_data[col], errors='coerce')
                        
                        # Extract CO2 values (both zones use the same column names now)
                        co2_2015 = year_data.loc[2015, "CO2 (Million tonnes)"]
                        co2_2040 = year_data.loc[2040, "CO2 (Million tonnes)"]
                        co2_2050 = year_data.loc[2050, "CO2 (Million tonnes)"]
                        
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
                            "Zone": zone,
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
                        print(f"Error processing {zone} scenario '{scenario}' in {sector}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error processing {zone} data: {e}")
            return []
        
        return all_scenario_data
    
    def create_unified_dataset(self):
        """Create a unified dataset for all zones, sectors, and scenarios"""
        all_data = []
        
        for zone in self.zones:
            print(f"Processing {zone}...")
            zone_data = self.process_zone_data(zone)
            all_data.extend(zone_data)
        
        # Process all scenarios into lever-level data
        unified_data = []
        
        for scenario_data in all_data:
            zone = scenario_data["Zone"]
            sector = scenario_data["Sector"]
            scenario = scenario_data["Scenario"]
            
            # Apply display name mappings
            display_sector = self.sector_display_names.get(sector, sector)
            display_scenario = self.scenario_display_names[zone].get(scenario, scenario)
            
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
                "Contrib_2015_2040_pct": (total_change_2015_2040 / abs(total_change_2015_2040)) * 100 if total_change_2015_2040 != 0 else 0,
                "Contrib_2040_2050_pct": (total_change_2040_2050 / abs(total_change_2040_2050)) * 100 if total_change_2040_2050 != 0 else 0,
                "Contrib_2015_2050_pct": (total_change_2015_2050 / abs(total_change_2015_2050)) * 100 if total_change_2015_2050 != 0 else 0
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
                
                # Align sign convention with dashboards: negative = reductions, positive = increases
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
            display_scenario = self.scenario_display_names[zone].get(scenario, scenario)
            
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
        """Process and save the unified dataset"""
        print("Starting data processing...")
        
        # Create unified dataset
        df_unified = self.create_unified_dataset()
        
        if df_unified.empty:
            print("Warning: No data was processed!")
            return None
        
        # Create intermediary dataset for auditing
        print("Creating intermediary dataset for auditing...")
        all_data = []
        for zone in self.zones:
            zone_data = self.process_zone_data(zone)
            all_data.extend(zone_data)
        
        df_intermediary = self.create_intermediary_dataset(all_data)
        
        # Save both datasets
        unified_path = os.path.join(output_dir, "unified_decomposition_data.csv")
        intermediary_path = os.path.join(output_dir, "intermediary_decomposition_data.csv")
        
        df_unified.to_csv(unified_path, index=False)
        df_intermediary.to_csv(intermediary_path, index=False)
        
        print(f"Unified dataset saved to: {unified_path}")
        print(f"Intermediary dataset saved to: {intermediary_path}")
        
        # Summary statistics for validation (not saved to file)
        total_scenarios = len(df_unified.groupby(["Zone", "Sector", "Scenario"]).size())
        
        # Data validation summary
        print("\nData Validation Summary:")
        print(f"Total scenarios processed: {total_scenarios}")
        print(f"Zones: {sorted(df_unified['Zone'].unique())}")
        print(f"Sectors: {sorted(df_unified['Sector'].unique())}")
        print(f"Scenarios: {sorted(df_unified['Scenario'].unique())}")
        print(f"Levers: {sorted(df_unified['Lever'].unique())}")
        print(f"\nIntermediary dataset shape: {df_intermediary.shape}")
        print(f"Intermediary dataset columns: {df_intermediary.columns.tolist()}")
        
        return df_unified

    def combine_with_wsl_scenarios(self, output_dir="../Output"):
        """Combine main data with World Sufficiency Lab scenarios"""
        print("\n🔄 Combining with World Sufficiency Lab scenarios...")
        
        try:
            # Import the sufficiency scenarios creation function
            import sys
            sys.path.append(os.path.dirname(__file__))
            from create_sufficiency_scenarios_simple import create_sufficiency_scenarios
            
            # Generate WSL scenarios
            wsl_scenarios = create_sufficiency_scenarios()
            
            if wsl_scenarios:
                # Convert to DataFrame
                df_wsl = pd.DataFrame(wsl_scenarios)
                print(f"✅ WSL scenarios generated: {len(df_wsl)} records")
                
                # Load the main unified data
                main_data_path = os.path.join(output_dir, "unified_decomposition_data.csv")
                df_main = pd.read_csv(main_data_path)
                print(f"📊 Main data loaded: {len(df_main)} records")
                
                # Combine both datasets
                df_combined = pd.concat([df_main, df_wsl], ignore_index=True)
                print(f"🔗 Combined data: {len(df_combined)} records")
                
                # Save the combined data back to unified_decomposition_data.csv
                df_combined.to_csv(main_data_path, index=False)
                print(f"💾 Combined data saved to: {main_data_path}")
                print(f"📈 Final dataset: {len(df_combined)} total records")
                
                return df_combined
            else:
                print("⚠️  No WSL scenarios generated")
                return None
                
        except Exception as e:
            print(f"⚠️  Warning: Could not combine with WSL scenarios: {e}")
            print("   Main data saved without WSL scenarios")
            return None

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = CO2DecompositionPreprocessor()
    
    # Process and save main data
    df_processed = preprocessor.save_processed_data()
    
    if df_processed is not None:
        print("\nData processing complete!")
        print(f"Total records: {len(df_processed)}")
        
        # Combine with WSL scenarios
        df_final = preprocessor.combine_with_wsl_scenarios()
        
        if df_final is not None:
            print(f"\n🎉 FINAL RESULT: Complete dataset with {len(df_final)} records!")
            print("📁 All scenarios now available in unified_decomposition_data.csv")
        else:
            print("\n📁 Main data saved (WSL scenarios not combined)")
        
        # Show sample data
        print("\nSample data:")
        print(df_processed.head(10))
        
    else:
        print("\nData processing failed!") 