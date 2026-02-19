#!/usr/bin/env python3
"""
Switzerland GHG Emissions Analysis Script

Analyzes and visualizes Switzerland's greenhouse gas emissions over time using multiple data sources:
1. Swiss Federal Office territorial emissions (CO2 and total GHG)
2. Eurostat carbon footprint (consumption-based)
3. Eurostat GHG footprint (consumption-based)

Creates visualization showing territorial vs consumption-based emissions for 2010-2023.

Data sources:
- swissfederaloffice_emissions.xlsx: Swiss territorial emissions by gas type
- eurostat_carbon_footprint.csv: Switzerland's carbon footprint (consumption-based)
- eurostat_ghg_footprint.csv: Switzerland's GHG footprint (consumption-based)

Output: PNG file in outputs/graphs/emissions/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).parent.parent  # Go up to the report directory
DATA_DIR = BASE_DIR / 'external_data'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'graphs' / 'emissions'

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette - consistent with eurostat_analysis_swiss.py
COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3', 
          '#80b1d3', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']

class SwissEmissionsAnalyzer:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.territorial_emissions = None
        self.carbon_footprint = None
        self.ghg_footprint = None
        self.population_data = None
        
    def load_data(self):
        """Load all emission datasets"""
        print("Loading Swiss emissions datasets...")
        
        try:
            # Load Swiss Federal Office territorial emissions
            print("  Loading Swiss territorial emissions...")
            self.territorial_emissions = self.load_swiss_territorial_emissions()
            
            # Load Eurostat footprint data
            print("  Loading Eurostat carbon footprint...")
            carbon_fp_raw = pd.read_csv(self.data_dir / 'eurostat_carbon_footprint.csv')
            self.carbon_footprint = self.process_eurostat_footprint(carbon_fp_raw, 'carbon')
            
            print("  Loading Eurostat GHG footprint...")
            ghg_fp_raw = pd.read_csv(self.data_dir / 'eurostat_ghg_footprint.csv')
            self.ghg_footprint = self.process_eurostat_footprint(ghg_fp_raw, 'ghg')
            
            print("  Loading Switzerland population data...")
            population_raw = pd.read_csv(self.data_dir / 'eurostat_population.csv')
            self.population_data = self.process_population_data(population_raw)
            
            print("[OK] All datasets loaded successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading datasets: {str(e)}")
            return False
    
    def load_swiss_territorial_emissions(self):
        """Load and process Swiss Federal Office territorial emissions"""
        print("    Loading simplified Swiss emissions data (year, ghg, co2)...")
        
        # Read the simplified Excel file with 3 columns: year, ghg, co2
        swiss_raw = pd.read_excel(self.data_dir / 'swissfederaloffice_emissions.xlsx', 
                                 engine='openpyxl')
        
        print(f"    Raw data shape: {swiss_raw.shape}")
        print(f"    Columns: {list(swiss_raw.columns)}")
        
        # The file should have columns: year, ghg, co2
        # Rename columns to standardized names if needed
        expected_cols = ['year', 'ghg', 'co2']
        if len(swiss_raw.columns) >= 3:
            swiss_raw.columns = expected_cols[:len(swiss_raw.columns)]
        
        # Filter for 2010-2023
        swiss_filtered = swiss_raw[
            (swiss_raw['year'] >= 2010) & 
            (swiss_raw['year'] <= 2023)
        ].copy()
        
        # Convert from Megatonnes to thousand tonnes (multiply by 1000)
        if 'ghg' in swiss_filtered.columns:
            swiss_filtered['Total_GHG'] = swiss_filtered['ghg'] * 1000
            print(f"    Converted GHG from Megatonnes to thousand tonnes")
            
        if 'co2' in swiss_filtered.columns:
            swiss_filtered['CO2'] = swiss_filtered['co2'] * 1000
            print(f"    Converted CO2 from Megatonnes to thousand tonnes")
        
        # Keep only the standardized columns
        result_cols = ['year']
        if 'Total_GHG' in swiss_filtered.columns:
            result_cols.append('Total_GHG')
        if 'CO2' in swiss_filtered.columns:
            result_cols.append('CO2')
            
        df_result = swiss_filtered[result_cols].copy()
        
        print(f"    Territorial emissions loaded: {len(df_result)} years, columns: {list(df_result.columns)}")
        
        # Show sample values to verify conversion
        for col in df_result.columns:
            if col != 'year':
                sample_val = df_result[col].dropna().iloc[0] if not df_result[col].dropna().empty else 'N/A'
                print(f"    Sample {col}: {sample_val} thousand tonnes")
                
        return df_result
    
    def process_eurostat_footprint(self, df, footprint_type):
        """Process Eurostat footprint data for Switzerland"""
        # Filter for 2010-2023 and Switzerland
        df_filtered = df[
            (df['TIME_PERIOD'] >= 2010) & 
            (df['TIME_PERIOD'] <= 2023) &
            (df['c_dest'] == 'Switzerland')
        ].copy()
        
        df_filtered = df_filtered.rename(columns={'TIME_PERIOD': 'year'})
        df_filtered = df_filtered[['year', 'OBS_VALUE']].sort_values('year')
        
        column_name = f'{footprint_type}_footprint'
        df_filtered = df_filtered.rename(columns={'OBS_VALUE': column_name})
        
        print(f"    {footprint_type.upper()} footprint loaded: {len(df_filtered)} years")
        return df_filtered
    
    def process_population_data(self, df):
        """Process Eurostat population data for Switzerland"""
        # Filter for 2010-2023 and Switzerland
        df_filtered = df[
            (df['TIME_PERIOD'] >= 2010) & 
            (df['TIME_PERIOD'] <= 2023) &
            (df['geo'] == 'CH')  # Switzerland code in Eurostat
        ].copy()
        
        if df_filtered.empty:
            # Try alternative country codes
            df_filtered = df[
                (df['TIME_PERIOD'] >= 2010) & 
                (df['TIME_PERIOD'] <= 2023) &
                (df['geo'].str.contains('Switzerland', case=False, na=False))
            ].copy()
        
        df_filtered = df_filtered.rename(columns={'TIME_PERIOD': 'year'})
        df_filtered = df_filtered[['year', 'OBS_VALUE']].sort_values('year')
        
        # Population is in number of people
        df_filtered = df_filtered.rename(columns={'OBS_VALUE': 'population'})
        
        print(f"    Population data loaded: {len(df_filtered)} years")
        return df_filtered
    
    def combine_emissions_data(self):
        """Combine all emissions data sources"""
        print("\nCombining emissions data...")
        
        # Start with territorial emissions
        combined = self.territorial_emissions.copy()
        
        # Merge carbon footprint
        if self.carbon_footprint is not None:
            combined = combined.merge(self.carbon_footprint, on='year', how='outer')
            
        # Merge GHG footprint  
        if self.ghg_footprint is not None:
            combined = combined.merge(self.ghg_footprint, on='year', how='outer')
        
        # Merge population data
        if self.population_data is not None:
            combined = combined.merge(self.population_data, on='year', how='outer')
            
            # Calculate per capita emissions (convert thousand tonnes to tonnes, then divide by population)
            for col in ['Total_GHG', 'CO2', 'carbon_footprint', 'ghg_footprint']:
                if col in combined.columns:
                    # Convert from thousand tonnes to tonnes (multiply by 1000), then divide by population
                    combined[f'{col}_per_capita'] = (combined[col] * 1000) / combined['population']
                    print(f"    Calculated {col} per capita emissions")
        
        # Sort by year and filter 2010-2023
        combined = combined[(combined['year'] >= 2010) & (combined['year'] <= 2023)].sort_values('year')
        
        print(f"Combined emissions data: {len(combined)} years")
        print("Available emission types:", [col for col in combined.columns if col != 'year'])
        
        return combined
    
    def plot_emissions_trends(self, emissions_data):
        """Create emissions trends visualization"""
        print("\nCreating emissions trends visualization...")
        
        plt.figure(figsize=(14, 10))
        
        # Define series to plot with labels and colors
        series_config = []
        
        if 'Total_GHG' in emissions_data.columns:
            series_config.append(('Total_GHG', 'Swiss Territorial GHG Emissions', COLORS[0], '-'))
            
        if 'CO2' in emissions_data.columns:
            series_config.append(('CO2', 'Swiss Territorial CO2 Emissions', COLORS[1], '-'))
            
        if 'carbon_footprint' in emissions_data.columns:
            series_config.append(('carbon_footprint', 'Swiss Carbon Footprint (Consumption)', COLORS[2], '-'))
            
        if 'ghg_footprint' in emissions_data.columns:
            series_config.append(('ghg_footprint', 'Swiss GHG Footprint (Consumption)', COLORS[3], '-'))
        
        # Plot each series
        for column, label, color, linestyle in series_config:
            data = emissions_data[emissions_data[column].notna()]
            if len(data) > 0:
                plt.plot(data['year'], data[column], 
                        marker='o', linewidth=2.5, markersize=6,
                        label=label, color=color, linestyle=linestyle)
                print(f"  Plotted {label}: {len(data)} points")
        
        # Formatting
        plt.title('Switzerland GHG and CO2 Emissions (2010-2023)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Emissions (Thousand Tonnes)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis to show all years
        plt.xlim(2009.5, 2023.5)
        plt.xticks(range(2010, 2024, 2))
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add explanatory note
        note_text = ("Territorial emissions: Direct emissions within Switzerland\\n"
                    "Consumption footprint: Emissions from Swiss consumption worldwide")
        plt.figtext(0.02, 0.02, note_text, fontsize=9, style='italic', 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / 'switzerland_emissions_trends.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Emissions trends chart saved: {output_path}")
        plt.close()
        
        return output_path
    
    def plot_per_capita_emissions(self, emissions_data):
        """Create per capita emissions trends visualization"""
        print("\nCreating per capita emissions trends visualization...")
        
        plt.figure(figsize=(14, 10))
        
        # Define per capita series to plot
        per_capita_series = []
        
        if 'Total_GHG_per_capita' in emissions_data.columns:
            per_capita_series.append(('Total_GHG_per_capita', 'Swiss Territorial GHG Emissions per Capita', COLORS[0], '-'))
            
        if 'CO2_per_capita' in emissions_data.columns:
            per_capita_series.append(('CO2_per_capita', 'Swiss Territorial CO2 Emissions per Capita', COLORS[1], '-'))
            
        if 'carbon_footprint_per_capita' in emissions_data.columns:
            per_capita_series.append(('carbon_footprint_per_capita', 'Swiss Carbon Footprint per Capita (Consumption)', COLORS[2], '-'))
            
        if 'ghg_footprint_per_capita' in emissions_data.columns:
            per_capita_series.append(('ghg_footprint_per_capita', 'Swiss GHG Footprint per Capita (Consumption)', COLORS[3], '-'))
        
        # Plot each per capita series
        for column, label, color, linestyle in per_capita_series:
            data = emissions_data[emissions_data[column].notna()]
            if len(data) > 0:
                plt.plot(data['year'], data[column], 
                        marker='o', linewidth=2.5, markersize=6,
                        label=label, color=color, linestyle=linestyle)
                print(f"  Plotted {label}: {len(data)} points")
        
        # Formatting
        plt.title('Switzerland GHG and CO2 Emissions per Capita (2010-2023)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Emissions (Tonnes per Inhabitant)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Set x-axis to show all years
        plt.xlim(2009.5, 2023.5)
        plt.xticks(range(2010, 2024, 2))
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add explanatory note
        note_text = ("Territorial emissions: Direct emissions within Switzerland per capita\\n"
                    "Consumption footprint: Emissions from Swiss consumption per capita worldwide")
        plt.figtext(0.02, 0.02, note_text, fontsize=9, style='italic', 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / 'switzerland_emissions_per_capita_trends.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Per capita emissions trends chart saved: {output_path}")
        plt.close()
        
        # Export data to Excel with requested format
        excel_output_path = self.output_dir / 'switzerland_emissions_per_capita_trends.xlsx'
        self.export_per_capita_data_to_excel(emissions_data, per_capita_series, excel_output_path)
        
        return output_path
    
    def export_per_capita_data_to_excel(self, emissions_data, per_capita_series, excel_output_path):
        """Export per capita emissions data to Excel with requested format"""
        print("  Exporting per capita emissions data to Excel...")
        
        # Create list to store all data rows
        excel_data = []
        
        # Graph title for visual_name
        visual_name = "Switzerland GHG and CO2 Emissions per Capita (2010-2023)"
        
        # Process each emission type
        for column, label, color, linestyle in per_capita_series:
            data = emissions_data[emissions_data[column].notna()]
            
            # Determine unit based on column type
            unit = "Tonnes per Inhabitant"
            
            # Determine filter value based on column type
            if 'Total_GHG_per_capita' in column:
                filter_value = "Territorial GHG"
            elif 'CO2_per_capita' in column:
                filter_value = "Territorial CO2"
            elif 'carbon_footprint_per_capita' in column:
                filter_value = "Footprint CO2"
            elif 'ghg_footprint_per_capita' in column:
                filter_value = "Footprint GHG"
            else:
                filter_value = "Unknown"
            
            # Add data for each year
            for _, row in data.iterrows():
                excel_data.append({
                    'visual_number': np.nan,
                    'visual_name': visual_name,
                    'year': int(row['year']),
                    'filter': filter_value,
                    'decile': np.nan,
                    'value': row[column],
                    'unit': unit
                })
        
        # Create DataFrame
        df_excel = pd.DataFrame(excel_data)
        
        # Sort by year
        df_excel = df_excel.sort_values('year')
        
        # Save to Excel
        df_excel.to_excel(excel_output_path, index=False)
        print(f"[OK] Per capita emissions data exported to Excel: {excel_output_path}")
    
    def print_data_summary(self, emissions_data):
        """Print summary of emissions data"""
        print("\n" + "="*60)
        print("EMISSIONS DATA SUMMARY")
        print("="*60)
        
        for column in emissions_data.columns:
            if column == 'year':
                continue
                
            data = emissions_data[emissions_data[column].notna()]
            if len(data) > 0:
                print(f"\n{column.upper()}:")
                print(f"  Years available: {data['year'].min():.0f}-{data['year'].max():.0f}")
                print(f"  Range: {data[column].min():.0f} - {data[column].max():.0f} thousand tonnes")
                print(f"  2010 value: {data[data['year'] == 2010][column].iloc[0]:.0f}" 
                      if 2010 in data['year'].values else "  2010: Not available")
                print(f"  2023 value: {data[data['year'] == 2023][column].iloc[0]:.0f}" 
                      if 2023 in data['year'].values else "  2023: Not available")

def main():
    """Main execution function"""
    print("="*60)
    print("SWITZERLAND GHG EMISSIONS ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = SwissEmissionsAnalyzer()
    
    # Load and process data
    if analyzer.load_data():
        # Combine all emissions data
        emissions_data = analyzer.combine_emissions_data()
        
        # Create visualization
        if len(emissions_data) > 0:
            analyzer.plot_emissions_trends(emissions_data)
            analyzer.plot_per_capita_emissions(emissions_data)
            analyzer.print_data_summary(emissions_data)
        else:
            print("[ERROR] No emissions data to plot")
    else:
        print("[ERROR] Failed to load emissions data")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("Charts saved in:", OUTPUT_DIR)
    print("="*60)

if __name__ == "__main__":
    main()
