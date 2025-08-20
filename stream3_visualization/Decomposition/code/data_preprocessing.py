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
            "EU": "2025-04-28_EC scenarios data_Decomposition.xlsx",
            "Switzerland": "2025-08-13_CH scenarios data_Decomposition.xlsx"
        }
        
        # Sector configurations for each zone
        self.sector_configs = {
            "EU": {
                "Buildings-Residential": ["scenario 1", "scenario 2", "scenario 3", "life scenario"],
                "Buildings -Services": ["scenario 1", "scenario 2", "scenario 3", "life scenario"],
                "Industry": ["scenario 1 -standard", "scenario 2-standard", "scenario 3-standard", "life scenario-circular economy"],
                "PassLandTransport": ["scenario 1", "scenario 2", "scenario 3", "life scenario"]
            },
            "Switzerland": {
                "Buildings-Residential": ["scenario 1", "scenario 2", "scenario 3", "life scenario"],
                "Buildings -Services": ["scenario 1", "scenario 2", "scenario 3", "life scenario"],
                "PassLandTransport": ["scenario 1", "scenario 2", "scenario 3", "life scenario"]
            }
        }
        
        # Column mapping for each sector (actual column names in Excel)
        self.sector_columns = {
            "Buildings-Residential": ["Year", "Population (Million)", "Floor area (Million m²)", "Final Energy (Million toe)", "CO2 (Million tonnes)"],
            "Buildings -Services": ["Year", "Population (Million)", "Floor area (Million m²)", "Final Energy (Million toe)", "CO2 (Million tonnes)"],
            "Industry": ["Year", "Population (Millions)", "Production (Million of Tonnes)", "Energy (Mtoe)", "CO2 (Mt)"],
            "PassLandTransport": ["Year", "Population (Millions)", "Passenger Transport (Tpkm)", "Energy (Mtoe)", "CO2 (Mtonnes)"]
        }
        
        # Universal lever names
        self.universal_levers = ["Population", "Sufficiency", "Energy Efficiency", "Supply Side Decarbonation"]
        
    def calculate_intensity_factors(self, data_rows, sector):
        """Calculate intensity factors for LMDI decomposition"""
        intensity_factors = {}
        
        if sector in ["Buildings-Residential", "Buildings -Services"]:
            # Buildings: CO2 = Population × (m²/population) × (energy/m²) × (CO2/energy)
            intensity_factors["Population"] = data_rows["Population (Million)"]
            intensity_factors["Sufficiency"] = data_rows["Floor area (Million m²)"] / data_rows["Population (Million)"]
            intensity_factors["Energy Efficiency"] = data_rows["Final Energy (Million toe)"] / data_rows["Floor area (Million m²)"]
            intensity_factors["Supply Side Decarbonation"] = data_rows["CO2 (Million tonnes)"] / data_rows["Final Energy (Million toe)"]
            
        elif sector == "Industry":
            # Industry: CO2 = Population × (Production/population) × (Energy/Production) × (CO2/Energy)
            intensity_factors["Population"] = data_rows["Population (Millions)"]
            intensity_factors["Sufficiency"] = data_rows["Production (Million of Tonnes)"] / data_rows["Population (Millions)"]
            intensity_factors["Energy Efficiency"] = data_rows["Energy (Mtoe)"] / data_rows["Production (Million of Tonnes)"]
            intensity_factors["Supply Side Decarbonation"] = data_rows["CO2 (Mt)"] / data_rows["Energy (Mtoe)"]
            
        elif sector == "PassLandTransport":
            # Transport: CO2 = Population × (Passenger Transport/population) × (Energy/Passenger Transport) × (CO2/Energy)
            intensity_factors["Population"] = data_rows["Population (Millions)"]
            intensity_factors["Sufficiency"] = data_rows["Passenger Transport (Tpkm)"] / data_rows["Population (Millions)"]
            intensity_factors["Energy Efficiency"] = data_rows["Energy (Mtoe)"] / data_rows["Passenger Transport (Tpkm)"]
            intensity_factors["Supply Side Decarbonation"] = data_rows["CO2 (Mtonnes)"] / data_rows["Energy (Mtoe)"]
        
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
        
        for sector, scenarios in self.sector_configs[zone].items():
            try:
                # Read the sheet
                df_sheet = pd.read_excel(file_path, sheet_name=sector, header=None)
                
                for scenario in scenarios:
                    try:
                        # Find scenario start row
                        scenario_rows = df_sheet[df_sheet[0].astype(str).str.strip().str.lower() == scenario.lower()]
                        if len(scenario_rows) == 0:
                            print(f"Warning: Scenario '{scenario}' not found in {zone} - {sector}")
                            continue
                            
                        idx = scenario_rows.index[0]
                        
                        # Extract data (assuming 3 rows: 2015, 2040, 2050)
                        data_rows = df_sheet.iloc[idx + 2 : idx + 5, [0, 1, 2, 3, 4]].copy()
                        
                        # Use correct column names for this sector
                        sector_cols = self.sector_columns[sector]
                        data_rows.columns = sector_cols
                        data_rows.set_index("Year", inplace=True)
                        data_rows.index = data_rows.index.astype(int)
                        data_rows = data_rows.astype(float)
                        
                        # Get actual CO2 emissions from raw data (for starting points)
                        if sector in ["Buildings-Residential", "Buildings -Services"]:
                            co2_2015 = data_rows.loc[2015, "CO2 (Million tonnes)"]
                            co2_2040 = data_rows.loc[2040, "CO2 (Million tonnes)"]
                            co2_2050 = data_rows.loc[2050, "CO2 (Million tonnes)"]
                        elif sector == "Industry":
                            co2_2015 = data_rows.loc[2015, "CO2 (Mt)"]
                            co2_2040 = data_rows.loc[2040, "CO2 (Mt)"]
                            co2_2050 = data_rows.loc[2050, "CO2 (Mt)"]
                        elif sector == "PassLandTransport":
                            co2_2015 = data_rows.loc[2015, "CO2 (Mtonnes)"]
                            co2_2040 = data_rows.loc[2040, "CO2 (Mtonnes)"]
                            co2_2050 = data_rows.loc[2050, "CO2 (Mtonnes)"]
                        
                        # Calculate intensity factors
                        intensity_factors = self.calculate_intensity_factors(data_rows, sector)
                        
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
                        
                        # Clean scenario name
                        scenario_clean = scenario.replace("-standard", "").replace("-circular economy", "").title()
                        
                        # Store data directly in the list
                        all_scenario_data.append({
                            "Zone": zone,
                            "Sector": sector,
                            "Scenario": scenario_clean,
                            "CO2_2015": co2_2015,
                            "CO2_2040": co2_2040,
                            "CO2_2050": co2_2050,
                            "Contrib_2015_2040": contrib_2015_2040,
                            "Contrib_2040_2050": contrib_2040_2050,
                            "Intensity_Factors": intensity_factors
                        })
                        
                    except Exception as e:
                        print(f"Error processing scenario '{scenario}' in {zone} - {sector}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing sector '{sector}' in {zone}: {e}")
                continue
        
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
                "Sector": sector,
                "Scenario": scenario,
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
                
                unified_data.append({
                    "Zone": zone,
                    "Sector": sector,
                    "Scenario": scenario,
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
    
    def save_processed_data(self, output_dir="../Output"):
        """Process and save the unified dataset"""
        print("Starting data processing...")
        
        # Create unified dataset
        df_unified = self.create_unified_dataset()
        
        if df_unified.empty:
            print("Warning: No data was processed!")
            return None
        
        # Save to CSV
        output_path = os.path.join(output_dir, "unified_decomposition_data.csv")
        df_unified.to_csv(output_path, index=False)
        print(f"Unified dataset saved to: {output_path}")
        
        # Create summary statistics
        summary = df_unified.groupby(["Zone", "Sector", "Scenario"]).agg({
            "CO2_2015": "first",
            "CO2_2040": "first",
            "CO2_2050": "first"
        }).reset_index()
        
        # Calculate total changes for summary
        summary["Total_Change_2015_2040"] = summary["CO2_2040"] - summary["CO2_2015"]
        summary["Total_Change_2040_2050"] = summary["CO2_2050"] - summary["CO2_2040"]
        summary["Total_Change_2015_2050"] = summary["CO2_2050"] - summary["CO2_2015"]
        
        summary_path = os.path.join(output_dir, "decomposition_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")
        
        # Data validation summary
        print("\nData Validation Summary:")
        print(f"Total scenarios processed: {len(summary)}")
        print(f"Zones: {sorted(df_unified['Zone'].unique())}")
        print(f"Sectors: {sorted(df_unified['Sector'].unique())}")
        print(f"Scenarios: {sorted(df_unified['Scenario'].unique())}")
        print(f"Levers: {sorted(df_unified['Lever'].unique())}")
        
        return df_unified

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = CO2DecompositionPreprocessor()
    
    # Process and save data
    df_processed = preprocessor.save_processed_data()
    
    if df_processed is not None:
        print("\nData processing complete!")
        print(f"Total records: {len(df_processed)}")
        
        # Show sample data
        print("\nSample data:")
        print(df_processed.head(10))
        
    else:
        print("\nData processing failed!") 