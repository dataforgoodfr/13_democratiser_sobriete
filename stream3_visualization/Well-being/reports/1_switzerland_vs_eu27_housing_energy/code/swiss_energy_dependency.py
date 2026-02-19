#!/usr/bin/env python3
"""
Energy Dependency Analysis Script - France & Switzerland

Analyzes energy dependency for gas and oil using Eurostat data for both countries.
Creates visualizations showing:
1. Dependency ratio over time for France & Switzerland (gas and oil) as percentages 
2. Area charts decomposed by top 7 suppliers + others (net imports)

Data sources:
FRANCE:
- eurostat_oil_imports.csv / eurostat_oil_exports.csv (unit: Thousand tonnes)
- eurostat_gas_imports.csv / eurostat_gas_exports.csv (unit: Terajoule GCV)
- eurostat_gross_available_energy.csv (unit: KTOE - Thousand tonnes of oil equivalent)

SWITZERLAND:
- eurostat_trade_oil-gas_swiss.csv (units: kg, flow codes: 1=import/2=export, products: 33=oil/343=gas)
- eurostat_gross_available_energy.csv (unit: KTOE - Thousand tonnes of oil equivalent)

Conversion factors:
- 1 TJ = 23.88458966275 TOE
- 1 kg natural gas = 0.001194 TOE
- 1 kg oil ≈ 0.001 TOE

Output: PNG files in outputs/graphs/EUROSTAT_trade/
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
OUTPUT_DIR = BASE_DIR / 'outputs' / 'graphs' / 'EUROSTAT_trade'

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
TJ_TO_TOE = 23.88458966275  # 1 Terajoule = 23.88458966275 Toe
KG_GAS_TO_TOE = 0.001194  # 1 kilogram of natural gas = 0.001194 TOE
KG_OIL_TO_TOE = 1 / 1000  # Approximate: 1 kg oil ≈ 0.001 TOE (1000 kg = 1 TOE)
TARGET_COUNTRY = 'France'

# Color palette for visualizations
COLORS = ['#fb8072', '#ffd558', '#b3de69', '#80b1d3', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']

class FrenchEnergyDependencyAnalyzer:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.datasets = {}
        
    def load_data(self):
        """Load all required datasets"""
        print("Loading Eurostat datasets...")
        
        try:
            # Load oil trade data
            self.datasets['oil_imports'] = pd.read_csv(self.data_dir / 'eurostat_oil_imports.csv')
            self.datasets['oil_exports'] = pd.read_csv(self.data_dir / 'eurostat_oil_exports.csv')
            
            # Load gas trade data  
            self.datasets['gas_imports'] = pd.read_csv(self.data_dir / 'eurostat_gas_imports.csv')
            self.datasets['gas_exports'] = pd.read_csv(self.data_dir / 'eurostat_gas_exports.csv')
            
            # Load gross available energy
            self.datasets['gross_energy'] = pd.read_csv(self.data_dir / 'eurostat_gross_available_energy.csv')
            
            print("[OK] All datasets loaded successfully")
            
            # Print basic info about datasets
            for name, df in self.datasets.items():
                print(f"  - {name}: {len(df)} rows")
                
            return True
                
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            return False
    
    def explore_data(self):
        """Explore data structure and available countries"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        # Check available countries
        for dataset_name, df in self.datasets.items():
            if 'geo' in df.columns:
                countries = df['geo'].unique()
                print(f"\n{dataset_name.upper()}:")
                print(f"  Countries: {len(countries)}")
                # Check for France variants
                france_variants = [c for c in countries if 'france' in c.lower() or c == 'FR']
                if france_variants:
                    print(f"  France found as: {france_variants}")
                else:
                    print(f"  France variants: {[c for c in countries if 'FR' in c or 'france' in c.lower()]}")
                
                # Show first few countries for reference
                print(f"  Sample countries: {sorted(countries)[:10]}")
                
                if 'partner' in df.columns:
                    partners = df['partner'].unique()
                    print(f"  Partners: {len(partners)}")
                    print(f"  Sample partners: {sorted(partners)[:15]}")
                    
                if 'siec' in df.columns:
                    siec_values = df['siec'].unique()
                    print(f"  SIEC values: {siec_values}")
    
    def get_france_data(self, df, france_identifier='France'):
        """Filter data for France"""
        # Try different possible identifiers for France
        possible_names = ['France', 'FR', 'French']
        
        france_data = pd.DataFrame()
        for name in possible_names:
            if name in df['geo'].unique():
                france_data = df[df['geo'] == name].copy()
                print(f"  Found France data as '{name}': {len(france_data)} rows")
                break
        
        if france_data.empty:
            print(f"  WARNING: No France data found in dataset")
            print(f"  Available countries: {sorted(df['geo'].unique())}")
        
        return france_data
    
    def calculate_dependency_ratios(self):
        """Calculate energy dependency ratios for Switzerland"""
        print("\n" + "="*50) 
        print("CALCULATING DEPENDENCY RATIOS")
        print("="*50)
        
        dependencies = {}
        
        # Process Oil Data
        print("\n1. Processing Oil Data...")
        oil_deps = self.process_energy_source('oil')
        if not oil_deps.empty:
            dependencies['oil'] = oil_deps
            print(f"   [OK] Oil dependency calculated: {len(oil_deps)} years")
        else:
            print("   ✗ No oil dependency data calculated")
        
        # Process Gas Data
        print("\n2. Processing Gas Data...")
        gas_deps = self.process_energy_source('gas')
        if not gas_deps.empty:
            dependencies['gas'] = gas_deps
            print(f"   [OK] Gas dependency calculated: {len(gas_deps)} years")
        else:
            print("   ✗ No gas dependency data calculated")
        
        return dependencies
    
    def process_energy_source(self, source):
        """Process energy dependency for a specific source (oil or gas)"""
        try:
            # Get trade data
            imports_df = self.datasets[f'{source}_imports']
            exports_df = self.datasets[f'{source}_exports']
            
            # Get France trade data
            france_imports = self.get_france_data(imports_df)
            france_exports = self.get_france_data(exports_df)
            
            if france_imports.empty or france_exports.empty:
                print(f"    No France trade data for {source}")
                return pd.DataFrame()
            
            # Prepare trade data - rename columns to standard format
            france_imports = france_imports.rename(columns={'TIME_PERIOD': 'year', 'OBS_VALUE': 'value'})
            france_exports = france_exports.rename(columns={'TIME_PERIOD': 'year', 'OBS_VALUE': 'value'})
            

            
            # Calculate total imports and exports per year
            # For dependency calculation, prioritize "Total" partner if available, otherwise sum all
            if 'Total' in france_imports['partner'].values:
                imports_total = france_imports[france_imports['partner'] == 'Total'].groupby('year')['value'].sum().reset_index()
                print(f"    Using 'Total' partner for {source} imports: {len(imports_total)} years")
            else:
                imports_total = france_imports.groupby('year')['value'].sum().reset_index()
                print(f"    Summing all partners for {source} imports: {len(imports_total)} years")
            imports_total = imports_total.rename(columns={'value': 'total_imports'})
            
            if 'Total' in france_exports['partner'].values:
                exports_total = france_exports[france_exports['partner'] == 'Total'].groupby('year')['value'].sum().reset_index()
                print(f"    Using 'Total' partner for {source} exports: {len(exports_total)} years")
            else:
                exports_total = france_exports.groupby('year')['value'].sum().reset_index()
                print(f"    Summing all partners for {source} exports: {len(exports_total)} years")
            exports_total = exports_total.rename(columns={'value': 'total_exports'})
            

            
            # Merge imports and exports
            trade_data = imports_total.merge(exports_total, on='year', how='outer').fillna(0)
            trade_data['net_imports'] = trade_data['total_imports'] - trade_data['total_exports']
            
            # Convert gas from TJ to TOE if needed for partner analysis
            if source == 'gas':
                # Keep gas in TJ for dependency calculation, but convert to TOE for partner analysis
                trade_data['net_imports_toe'] = trade_data['net_imports'] * TJ_TO_TOE
                trade_data['total_imports_toe'] = trade_data['total_imports'] * TJ_TO_TOE
                trade_data['total_exports_toe'] = trade_data['total_exports'] * TJ_TO_TOE
            else:
                # Oil is already in thousand tonnes
                trade_data['net_imports_toe'] = trade_data['net_imports']
                trade_data['total_imports_toe'] = trade_data['total_imports']
                trade_data['total_exports_toe'] = trade_data['total_exports']
            
            # Get gross available energy data
            gross_energy = self.datasets['gross_energy']
            france_gross = self.get_france_data(gross_energy)
            
            if france_gross.empty:
                print(f"    No France gross energy data")
                return pd.DataFrame()
            
            # Filter for relevant energy source in gross available energy
            print(f"    Looking for France gross energy data for {source}")
            print(f"    France data shape: {france_gross.shape}")
            print(f"    Unique SIEC values in France data: {france_gross['siec'].unique()}")
            
            if source == 'gas':
                # Look for natural gas data with exact SIEC name
                target_siec = 'Natural gas'
                france_gross_filtered = france_gross[france_gross['siec'] == target_siec]
                print(f"    Looking for exact match: '{target_siec}'")
                print(f"    Found {len(france_gross_filtered)} rows with Natural gas")
            else:
                # Look for oil data with exact SIEC name
                target_siec = 'Oil and petroleum products (excluding biofuel portion)'
                france_gross_filtered = france_gross[france_gross['siec'] == target_siec]
                print(f"    Looking for exact match: '{target_siec}'")
                print(f"    Found {len(france_gross_filtered)} rows with Oil data")
            
            if france_gross_filtered.empty:
                print(f"    No '{target_siec}' data found")
                print(f"    Available SIEC values: {france_gross['siec'].unique()}")
                return pd.DataFrame()
            
            # Prepare gross energy data
            france_gross_filtered = france_gross_filtered.rename(columns={'TIME_PERIOD': 'year', 'OBS_VALUE': 'gross_available'})
            gross_by_year = france_gross_filtered.groupby('year')['gross_available'].sum().reset_index()
            
            # Merge with trade data
            dependency_data = trade_data.merge(gross_by_year, on='year', how='inner')
            
            # DEBUG: Print actual values for a few years
            print(f"    === DEBUGGING {source.upper()} CALCULATION ===")
            if len(dependency_data) > 0:
                sample_year = dependency_data['year'].iloc[0]
                sample_row = dependency_data.iloc[0]
                
                print(f"    Sample year: {sample_year}")
                print(f"    Raw imports: {sample_row['total_imports']:,.1f}")
                print(f"    Raw exports: {sample_row['total_exports']:,.1f}")
                print(f"    Net imports (raw): {sample_row['net_imports']:,.1f}")
                print(f"    Net imports (TOE): {sample_row['net_imports_toe']:,.1f}")
                print(f"    Gross available energy (KTOE): {sample_row['gross_available']:,.1f}")
                
                # Calculate dependency manually
                if source == 'gas':
                    # Gas: TJ -> TOE -> KTOE, then compare with gross in KTOE
                    net_imports_ktoe = sample_row['net_imports'] * TJ_TO_TOE / 1000  # Convert TJ to KTOE
                    print(f"    Net imports (KTOE): {net_imports_ktoe:,.1f}")
                    manual_dependency = (net_imports_ktoe / sample_row['gross_available']) * 100
                else:
                    # Oil: already in thousand tonnes, gross is KTOE, assume 1:1 conversion
                    manual_dependency = (sample_row['net_imports_toe'] / sample_row['gross_available']) * 100
                
                print(f"    Manual dependency calculation: {manual_dependency:.1f}%")
            
            # Calculate dependency ratio as percentage - using TJ units directly for gas
            if source == 'gas':
                # Gas: Convert gross available energy from KTOE to TJ for direct comparison
                dependency_data['gross_available_tj'] = dependency_data['gross_available'] * 1000 / TJ_TO_TOE
                dependency_data['dependency_ratio'] = np.where(
                    dependency_data['gross_available_tj'] != 0,
                    (dependency_data['net_imports'] / dependency_data['gross_available_tj']) * 100,
                    0
                )
            else:
                # Oil: Assume thousand tonnes ≈ KTOE (1:1 conversion for approximation)
                dependency_data['dependency_ratio'] = np.where(
                    dependency_data['gross_available'] != 0,
                    (dependency_data['net_imports_toe'] / dependency_data['gross_available']) * 100,
                    0
                )
            
            print(f"    Final dependency ratios: {dependency_data['dependency_ratio'].tolist()}")
            print(f"    === END DEBUG ===")
            
            return dependency_data
            
        except Exception as e:
            print(f"    Error processing {source}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_partner_decomposition(self):
        """Calculate net imports by partner for area chart decomposition"""
        print("\n" + "="*50)
        print("CALCULATING PARTNER DECOMPOSITION") 
        print("="*50)
        
        decompositions = {}
        
        # Process Oil Partners
        print("\n1. Processing Oil Partners...")
        oil_partners = self.process_partners('oil')
        if not oil_partners.empty:
            decompositions['oil'] = oil_partners
            print(f"   [OK] Oil partners processed: {len(oil_partners)} rows")
        
        # Process Gas Partners
        print("\n2. Processing Gas Partners...")
        gas_partners = self.process_partners('gas')
        if not gas_partners.empty:
            decompositions['gas'] = gas_partners
            print(f"   [OK] Gas partners processed: {len(gas_partners)} rows")
            
        return decompositions
    
    def process_partners(self, source):
        """Process partner data for a specific energy source"""
        try:
            imports_df = self.datasets[f'{source}_imports']
            exports_df = self.datasets[f'{source}_exports']
            
            # Get France data
            france_imports = self.get_france_data(imports_df)
            france_exports = self.get_france_data(exports_df)
            
            if france_imports.empty:
                return pd.DataFrame()
            
            # Rename columns
            france_imports = france_imports.rename(columns={'TIME_PERIOD': 'year', 'OBS_VALUE': 'value'})
            france_exports = france_exports.rename(columns={'TIME_PERIOD': 'year', 'OBS_VALUE': 'value'}) if not france_exports.empty else pd.DataFrame(columns=['year', 'partner', 'value'])
            
            # Filter out TOTAL and aggregate records
            france_imports_clean = france_imports[
                (france_imports['partner'] != 'Total') & 
                (~france_imports['partner'].isna())
            ].copy()
            
            if not france_exports.empty:
                france_exports_clean = france_exports[
                    (france_exports['partner'] != 'Total') & 
                    (~france_exports['partner'].isna()) 
                ].copy()
            else:
                france_exports_clean = pd.DataFrame(columns=['year', 'partner', 'value'])
            
            # Calculate net imports by partner and year
            imports_by_partner = france_imports_clean.groupby(['year', 'partner'])['value'].sum().reset_index()
            imports_by_partner = imports_by_partner.rename(columns={'value': 'imports'})
            
            if not france_exports_clean.empty:
                exports_by_partner = france_exports_clean.groupby(['year', 'partner'])['value'].sum().reset_index()
                exports_by_partner = exports_by_partner.rename(columns={'value': 'exports'})
            else:
                # Create empty exports dataframe
                exports_by_partner = pd.DataFrame(columns=['year', 'partner', 'exports'])
            
            # Merge imports and exports
            net_imports = imports_by_partner.merge(exports_by_partner, on=['year', 'partner'], how='left')
            net_imports['exports'] = net_imports['exports'].fillna(0)
            net_imports['net_imports'] = net_imports['imports'] - net_imports['exports']
            
            # Keep original units for display
            if source == 'gas':
                # Keep gas in TJ (original units)
                net_imports['net_imports_toe'] = net_imports['net_imports']
            else:
                # Oil is already in thousand tonnes
                net_imports['net_imports_toe'] = net_imports['net_imports']
            
            return net_imports
            
        except Exception as e:
            print(f"    Error processing {source} partners: {str(e)}")
            return pd.DataFrame()
    
    def plot_dependency_trends(self, dependencies):
        """Plot dependency ratio trends over time"""
        print("\n" + "="*50)
        print("CREATING DEPENDENCY TREND CHARTS")
        print("="*50)
        
        if not dependencies:
            print("No dependency data available for plotting")
            return
        
        plt.figure(figsize=(12, 8))
        
        for i, (source, data) in enumerate(dependencies.items()):
            if len(data) > 0:
                color = COLORS[i % len(COLORS)]
                label = 'Natural Gas' if source == 'gas' else 'Oil & Petroleum Products'
                
                plt.plot(data['year'], data['dependency_ratio'], 
                        marker='o', linewidth=3, markersize=8, 
                        label=label, color=color, alpha=0.8)
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('France Energy Dependency Trends\n(Net Imports / Gross Available Energy)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Dependency Ratio (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add text annotations
        plt.text(0.02, 0.98, 'Higher values = More dependent on imports\nNegative values = Net exporter', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        output_path = self.output_dir / 'france_dependency_trends.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Dependency trends chart saved: {output_path}")
        plt.close()
    
    def plot_partner_decomposition(self, decompositions, top_n=7):
        """Plot area charts showing top suppliers decomposition"""
        print("\n" + "="*50)
        print("CREATING PARTNER DECOMPOSITION CHARTS")
        print("="*50)
        
        for source, data in decompositions.items():
            if data.empty:
                print(f"No partner data for {source}")
                continue
                
            print(f"\nProcessing {source} partners...")
            
            # Find top N partners by total net imports
            partner_totals = data.groupby('partner')['net_imports_toe'].sum()
            partner_totals = partner_totals[partner_totals > 0]  # Only positive net imports
            top_partners = partner_totals.nlargest(top_n).index.tolist()
            
            print(f"   Top {top_n} partners: {top_partners}")
            
            if not top_partners:
                print(f"   No positive net import partners found for {source}")
                continue
            
            # Create pivot table for top partners
            top_data = data[data['partner'].isin(top_partners)].copy()
            pivot_data = top_data.pivot_table(index='year', columns='partner', values='net_imports_toe', fill_value=0)
            
            # Calculate "Others" category
            all_partners_by_year = data.groupby('year')['net_imports_toe'].sum()
            top_partners_by_year = pivot_data.sum(axis=1)
            others_by_year = all_partners_by_year - top_partners_by_year
            others_by_year = others_by_year.clip(lower=0)  # Ensure non-negative
            
            # Create the area chart
            plt.figure(figsize=(14, 8))
            
            # Prepare data for stacking (including Others)
            stack_data = []
            stack_labels = list(pivot_data.columns) 
            colors_to_use = COLORS[:len(pivot_data.columns)]
            
            # Add data for top partners
            for col in pivot_data.columns:
                stack_data.append(pivot_data[col])
            
            # Add "Others" if significant
            if not others_by_year.empty and others_by_year.max() > 0:
                # Align others_by_year with pivot_data index
                others_aligned = others_by_year.reindex(pivot_data.index, fill_value=0)
                stack_data.append(others_aligned.values)
                stack_labels.append('Others')
                colors_to_use.append(COLORS[len(pivot_data.columns) % len(COLORS)])
            
            # Plot all as one stacked area
            plt.stackplot(pivot_data.index, *stack_data,
                         labels=stack_labels,
                         colors=colors_to_use,
                         alpha=0.8)
            
            # Format chart
            source_label = 'Natural Gas' if source == 'gas' else 'Oil & Petroleum Products'
            unit = 'Terajoule (GCV)' if source == 'gas' else 'Thousand tonnes'
            
            plt.title(f'France {source_label} - Net Imports by Partner\nTop {top_n} Suppliers + Others ({unit})', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Year', fontsize=12)
            plt.ylabel(f'Net Imports ({unit})', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = self.output_dir / f'france_{source}_partners_decomposition.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[OK] {source_label} decomposition chart saved: {output_path}")
            plt.close()


class SwissEnergyDependencyAnalyzer:
    """Swiss Energy Dependency Analysis using eurostat_trade_oil-gas_swiss dataset"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.swiss_trade_data = None
        self.gross_energy_data = None
        
    def load_data(self):
        """Load Swiss trade data and gross available energy"""
        print("Loading Swiss datasets...")
        
        try:
            # Load Swiss trade data
            swiss_file = self.data_dir / 'eurostat_trade_oil-gas_swiss.csv'
            if not swiss_file.exists():
                print(f"[ERROR] Swiss trade file not found: {swiss_file}")
                return False
            
            self.swiss_trade_data = pd.read_csv(swiss_file)
            
            # Load Swiss Federal Office gross available energy data
            swiss_energy_file = self.data_dir / 'swissfederaloffice_energy.csv'
            if not swiss_energy_file.exists():
                print(f"[ERROR] Swiss energy file not found: {swiss_energy_file}")
                return False
                
            # Load with semicolon separator for German data
            self.swiss_energy = pd.read_csv(swiss_energy_file, sep=';')
            
            # Also keep the old reference for compatibility
            self.gross_energy_data = self.swiss_energy
            
            print("[OK] All Swiss datasets loaded successfully")
            print(f"  - swiss_trade: {len(self.swiss_trade_data)} rows")
            print(f"  - swiss_energy: {len(self.swiss_energy)} rows")
            
            # Show Swiss energy data structure
            print(f"  - swiss_energy columns: {list(self.swiss_energy.columns)}")
            if 'energietraeger' in self.swiss_energy.columns:
                sources = sorted(self.swiss_energy['energietraeger'].unique())
                print(f"  - Available energy sources: {sources[:10]}...")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading datasets: {str(e)}")
            return False
    
    def explore_data(self):
        """Explore Swiss trade data structure"""
        print("\n" + "="*50)
        print("SWISS DATA EXPLORATION") 
        print("="*50)
        
        if self.swiss_trade_data is not None:
            print(f"\nSWISS_TRADE:")
            print(f"  Shape: {self.swiss_trade_data.shape}")
            print(f"  Columns: {list(self.swiss_trade_data.columns)}")
            
            if 'flow' in self.swiss_trade_data.columns:
                print(f"  Flow values: {sorted(self.swiss_trade_data['flow'].unique())}")
            if 'product' in self.swiss_trade_data.columns:
                print(f"  Product values: {sorted(self.swiss_trade_data['product'].unique())}")
            if 'partner' in self.swiss_trade_data.columns:
                print(f"  Partners: {len(self.swiss_trade_data['partner'].unique())} unique")
                print(f"  Sample partners: {sorted(self.swiss_trade_data['partner'].unique())[:10]}")
        
        if self.gross_energy_data is not None:
            print(f"\nSWISS_ENERGY (Federal Office):")
            print(f"  Shape: {self.gross_energy_data.shape}")
            print(f"  Columns: {list(self.gross_energy_data.columns)}")
            
            # Show energy sources (Energietraeger) 
            if 'Energietraeger' in self.gross_energy_data.columns:
                energy_sources = sorted(self.gross_energy_data['Energietraeger'].unique())
                print(f"  Energy sources: {energy_sources}")
                
            # Show categories (Rubrik)
            if 'Rubrik' in self.gross_energy_data.columns:
                categories = sorted(self.gross_energy_data['Rubrik'].unique())
                print(f"  Categories: {categories}")
                
            # Show year range
            if 'Jahr' in self.gross_energy_data.columns:
                years = self.gross_energy_data['Jahr'].min(), self.gross_energy_data['Jahr'].max()
                print(f"  Year range: {years[0]} - {years[1]}")
                
        # Show top 5 energy sources by average consumption
        if self.gross_energy_data is not None:
            brutto_data = self.gross_energy_data[
                self.gross_energy_data['Rubrik'] == 'Bruttoverbrauch'
            ].copy()
            
            if not brutto_data.empty:
                # Convert TJ to numeric (handle 'NA' values)
                brutto_data['TJ_numeric'] = pd.to_numeric(brutto_data['TJ'], errors='coerce')
                
                # Calculate average consumption by energy source
                avg_consumption = brutto_data.groupby('Energietraeger')['TJ_numeric'].mean().sort_values(ascending=False)
                
                print(f"\n  TOP 5 ENERGY SOURCES (Average TJ):")
                for i, (source, value) in enumerate(avg_consumption.head(5).items(), 1):
                    print(f"    {i}. {source}: {value:,.0f} TJ")
                
    def process_swiss_trade_data(self):
        """Process Swiss trade data into imports/exports by product and partner"""
        if self.swiss_trade_data is None:
            return {}, {}
            
        # Filter for imports and exports (using string values)
        imports = self.swiss_trade_data[self.swiss_trade_data['flow'] == 'IMPORT'].copy()
        exports = self.swiss_trade_data[self.swiss_trade_data['flow'] == 'EXPORT'].copy()
        
        # Separate by product using full string names
        oil_imports = imports[imports['product'] == 'Petroleum, petroleum products and related materials'].copy()
        oil_exports = exports[exports['product'] == 'Petroleum, petroleum products and related materials'].copy()
        gas_imports = imports[imports['product'] == 'Natural gas, whether or not liquefied'].copy() 
        gas_exports = exports[exports['product'] == 'Natural gas, whether or not liquefied'].copy()
        
        # Convert from kg to TOE
        oil_imports['value_toe'] = oil_imports['OBS_VALUE'] * KG_OIL_TO_TOE
        oil_exports['value_toe'] = oil_exports['OBS_VALUE'] * KG_OIL_TO_TOE
        gas_imports['value_toe'] = gas_imports['OBS_VALUE'] * KG_GAS_TO_TOE 
        gas_exports['value_toe'] = gas_exports['OBS_VALUE'] * KG_GAS_TO_TOE
        
        return {
            'oil': {'imports': oil_imports, 'exports': oil_exports},
            'gas': {'imports': gas_imports, 'exports': gas_exports}
        }
    
    def get_switzerland_gross_energy(self, source):
        """Get Switzerland gross available energy data from Swiss Federal Office"""
        if self.gross_energy_data is None:
            return pd.DataFrame()
            
        # Filter for gross consumption (Bruttoverbrauch) 
        swiss_data = self.gross_energy_data[
            self.gross_energy_data['Rubrik'] == 'Bruttoverbrauch'
        ].copy()
        
        if swiss_data.empty:
            print(f"    No Swiss gross energy data found")
            return pd.DataFrame()
            
        # Map energy sources to German names
        if source == 'gas':
            target_sources = ['Gas']  # Natural gas
        else:  # oil
            target_sources = ['Erdölprodukte', 'Rohöl']  # Oil products + crude oil
            
        # Filter by energy sources
        swiss_energy = swiss_data[
            swiss_data['Energietraeger'].isin(target_sources)
        ].copy()
        
        if swiss_energy.empty:
            print(f"    No Swiss {source} energy data found")
            return pd.DataFrame()
            
        # Convert TJ to numeric (handle 'NA' values)
        swiss_energy['TJ_numeric'] = pd.to_numeric(swiss_energy['TJ'], errors='coerce')
        
        # Sum by year (combine oil products + crude oil for oil source)
        yearly_totals = swiss_energy.groupby('Jahr')['TJ_numeric'].sum().reset_index()
        yearly_totals.columns = ['year', 'gross_available_tj']
        
        print(f"    Found Swiss {source} energy data: {len(yearly_totals)} years")
        print(f"    Sample values: {yearly_totals['gross_available_tj'].iloc[:3].tolist()}")
        
        return yearly_totals
    
    def calculate_swiss_dependency_ratios(self):
        """Calculate Swiss energy dependency ratios for Gas, Petroleum Products, and Electricity"""
        print("\n" + "="*50)
        print("CALCULATING SWISS DEPENDENCY RATIOS - 3 SPECIFIED SOURCES")
        print("="*50)
        
        if self.swiss_energy is None:
            print("[ERROR] No Swiss energy data loaded")
            return {}
            
        # Target energy sources (German names as they appear in data)
        target_sources = {
            'Gas': 'Gas',
            'Petroleum Products': 'Erdölprodukte', 
            'Electricity': 'Elektrizität'
        }
        
        # Check available energy sources
        available_sources = sorted(self.swiss_energy['Energietraeger'].unique())
        print(f"[INFO] Available energy sources: {available_sources}")
        
        # Verify our target sources exist
        matched_sources = {}
        for eng_name, ger_name in target_sources.items():
            if ger_name in available_sources:
                matched_sources[eng_name] = ger_name
                print(f"   [OK] {eng_name} -> {ger_name}")
            else:
                print(f"   [ERROR] {eng_name} ({ger_name}) not found")
        
        if not matched_sources:
            print("[ERROR] No target energy sources found")
            return {}
            
        print(f"\n[OK] Processing {len(matched_sources)} energy sources")
        dependency_ratios = {}
        
        for eng_name, ger_name in matched_sources.items():
            print(f"\n[PROCESSING] Processing {eng_name} ({ger_name})...")
            
            # Filter data by energy source
            source_data = self.swiss_energy[self.swiss_energy['Energietraeger'] == ger_name].copy()
            
            if source_data.empty:
                print(f"   [ERROR] No data for {ger_name}")
                continue
            
            # Check available Rubrik values for this energy source
            available_rubriks = sorted(source_data['Rubrik'].unique())
            print(f"   [INFO] Available Rubrik values: {available_rubriks}")
            
            # Check if we have the required categories
            required_rubriks = ['Bruttoverbrauch', 'Import', 'Export']
            missing_rubriks = [r for r in required_rubriks if r not in available_rubriks]
            
            if missing_rubriks:
                print(f"   [WARNING] Missing Rubrik categories: {missing_rubriks}")
                print(f"   Available: {available_rubriks}")
                # Continue with available data if we have at least gross consumption
                if 'Bruttoverbrauch' not in available_rubriks:
                    print(f"   [ERROR] Cannot calculate dependency without Bruttoverbrauch")
                    continue
            
            # Convert TJ values to numeric and handle NA values (turn to 0)
            source_data['TJ_numeric'] = pd.to_numeric(source_data['TJ'], errors='coerce').fillna(0)
            
            # Pivot the data to get separate columns for each Rubrik
            pivot_data = source_data.pivot_table(
                index='Jahr', 
                columns='Rubrik', 
                values='TJ_numeric', 
                fill_value=0
            )
            
            print(f"   [CALC] Pivoted data shape: {pivot_data.shape}")
            print(f"   [CALC] Pivoted columns: {list(pivot_data.columns)}")
            
            # Handle missing categories by filling with 0
            if 'Import' not in pivot_data.columns:
                pivot_data['Import'] = 0
                print(f"   [WARNING] No Import data - assuming 0")
            if 'Export' not in pivot_data.columns:
                pivot_data['Export'] = 0  
                print(f"   [WARNING] No Export data - assuming 0")
            
            # Filter out rows where gross consumption is 0 or NaN
            valid_data = pivot_data[
                pivot_data['Bruttoverbrauch'].notna() & 
                (pivot_data['Bruttoverbrauch'] > 0)
            ].copy()
            
            if valid_data.empty:
                print(f"   [ERROR] No valid consumption data for {ger_name}")
                continue
            
            # Apply user's formula: Dependency = (Import + Export) / Bruttoverbrauch * 100
            # Note: Export values should be negative for net imports calculation
            # But we'll use them as-is since they represent the actual flow values
            valid_data['dependency_ratio'] = (
                (valid_data['Import'] + valid_data['Export']) / valid_data['Bruttoverbrauch'] * 100
            )
            
            # Handle any infinite or invalid values
            valid_data['dependency_ratio'] = valid_data['dependency_ratio'].replace([float('inf'), -float('inf')], 0)
            
            # Show example calculation for verification
            if len(valid_data) > 0:
                # Find a year with data for example (prefer 1980 if available, or use first available)
                example_year = 1980 if 1980 in valid_data.index else valid_data.index[0]
                example_row = valid_data.loc[example_year]
                
                print(f"   [CALC] Example calculation ({example_year}):")
                print(f"      - Bruttoverbrauch: {example_row['Bruttoverbrauch']:,.0f} TJ")
                print(f"      - Import: {example_row['Import']:,.0f} TJ")
                print(f"      - Export: {example_row['Export']:,.0f} TJ")
                print(f"      - Net imports: {example_row['Import'] + example_row['Export']:,.0f} TJ")
                print(f"      - Dependency ratio: {example_row['dependency_ratio']:.1f}%")
            
            print(f"   [OK] Calculated dependency ratios for {len(valid_data)} years")
            print(f"   [RANGE] Range: {valid_data['dependency_ratio'].min():.1f}% to {valid_data['dependency_ratio'].max():.1f}%")
            
            # Reset index to get year as a column
            result_data = valid_data.reset_index()[['Jahr', 'dependency_ratio']].copy()
            result_data = result_data.rename(columns={'Jahr': 'year'})
            
            dependency_ratios[eng_name] = result_data
                
        return dependency_ratios
    
    def get_single_source_gross_energy(self, energy_source):
        """Get gross energy data for a single energy source"""
        if self.gross_energy_data is None:
            return pd.DataFrame()
            
        # Filter for gross consumption and specific energy source
        energy_data = self.gross_energy_data[
            (self.gross_energy_data['Rubrik'] == 'Bruttoverbrauch') &
            (self.gross_energy_data['Energietraeger'] == energy_source)
        ].copy()
        
        if energy_data.empty:
            return pd.DataFrame()
            
        # Convert TJ to numeric
        energy_data['TJ_numeric'] = pd.to_numeric(energy_data['TJ'], errors='coerce')
        
        # Prepare data
        result = energy_data[['Jahr', 'TJ_numeric']].copy()
        result.columns = ['year', 'gross_available_tj']
        
        return result
        
    def calculate_swiss_partner_decomposition(self):
        """Calculate Swiss partner decomposition - countries only"""
        print("\n" + "="*50)
        print("CALCULATING SWISS PARTNER DECOMPOSITION - COUNTRIES ONLY")
        print("="*50)
        
        trade_data = self.process_swiss_trade_data()
        decompositions = {}
        
        # Define non-country partners to exclude (similar to EU trade data handling)
        non_country_partners = [
            'Intra-EU', 'Intra-EU27', 'Extra-EU', 'Extra-EU27', 
            'EU-28', 'EU-27', 'Total', 'World', 'Not specified',
            'Other African countries (aggregate changing according to the context)',
            'Other American countries (aggregate changing according to the context)', 
            'Other Asian countries (aggregate changing according to the context)',
            'Other European countries (aggregate changing according to the context)',
            'Other Near and Middle East Asian countries',
            'Other countries of former Soviet Union (before 1991)'
        ]
        
        for source in ['oil', 'gas']:
            print(f"\n{source.upper()} Partners (Countries Only):")
            
            if source not in trade_data:
                continue
                
            imports_df = trade_data[source]['imports']
            exports_df = trade_data[source]['exports']
            
            if imports_df.empty:
                continue
                
            # Filter to countries only (exclude aggregates)
            imports_countries = imports_df[
                ~imports_df['partner'].isin(non_country_partners)
            ].copy()
            
            if exports_df.empty:
                exports_countries = pd.DataFrame(columns=['TIME_PERIOD', 'partner', 'value_toe'])
            else:
                exports_countries = exports_df[
                    ~exports_df['partner'].isin(non_country_partners)
                ].copy()
                
            print(f"    Countries found: {len(imports_countries['partner'].unique())} import partners")
            print(f"    Sample countries: {sorted(imports_countries['partner'].unique())[:10]}")
                
            # Group by year and partner (countries only)
            imports_by_partner = imports_countries.groupby(['TIME_PERIOD', 'partner'])['value_toe'].sum().reset_index()
            imports_by_partner.columns = ['year', 'partner', 'imports']
            
            if not exports_countries.empty:
                exports_by_partner = exports_countries.groupby(['TIME_PERIOD', 'partner'])['value_toe'].sum().reset_index()
                exports_by_partner.columns = ['year', 'partner', 'exports']
            else:
                exports_by_partner = pd.DataFrame(columns=['year', 'partner', 'exports'])
                
            # Calculate net imports by partner
            partner_data = imports_by_partner.merge(exports_by_partner, on=['year', 'partner'], how='left')
            partner_data['exports'] = partner_data['exports'].fillna(0)
            partner_data['net_imports'] = partner_data['imports'] - partner_data['exports']
            partner_data['net_imports_toe'] = partner_data['net_imports']  # Already in TOE
            
            decompositions[source] = partner_data
            print(f"    [OK] {source.capitalize()} country partners processed: {len(partner_data)} rows")
            
        return decompositions
    
    def plot_swiss_dependency_trends(self, dependency_ratios):
        """Plot Swiss dependency trends for 3 specified sources with percentages"""
        print("\n" + "="*50)
        print("CREATING SWISS DEPENDENCY TRENDS CHART - 3 SPECIFIED SOURCES")
        print("="*50)
        
        if not dependency_ratios:
            print("[ERROR] No dependency ratios to plot")
            return
            
        plt.figure(figsize=(14, 10))
        
        # Use consistent colors from global palette
        colors_to_use = COLORS[:len(dependency_ratios)]
        
        for i, (source, data) in enumerate(dependency_ratios.items()):
            if data.empty:
                continue
                
            plt.plot(data['year'], data['dependency_ratio'], 
                    marker='o', linewidth=2.5, markersize=6,
                    label=source, color=colors_to_use[i])
            
            print(f"   [OK] Plotted {source}: {len(data)} data points")
            print(f"     Range: {data['dependency_ratio'].min():.1f}% to {data['dependency_ratio'].max():.1f}%")
        
        plt.title('Switzerland Energy Dependency Trends\nGas, Petroleum Products & Electricity (2000-2022)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Dependency Ratio (%)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Set y-axis to percentage format
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        
        # Add horizontal line at 100% for reference
        plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% Dependency')
        
        plt.tight_layout()
        output_path = self.output_dir / 'switzerland_dependency_trends.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Swiss dependency trends chart saved: {output_path}")
        plt.close()
    
    def plot_swiss_partner_decomposition(self, decompositions, top_n=7):
        """Plot Swiss partner decomposition charts - countries only"""
        print("\n" + "="*50)
        print("CREATING SWISS PARTNER DECOMPOSITION CHARTS - COUNTRIES ONLY")
        print("="*50)
        
        for source, data in decompositions.items():
            if data.empty:
                continue
                
            print(f"\nProcessing Swiss {source} country partners...")
            
            # Get top countries by average net imports
            partner_totals = data.groupby('partner')['net_imports_toe'].mean().sort_values(ascending=False)
            
            # Filter out negative values (net exporters) for cleaner visualization
            partner_totals = partner_totals[partner_totals > 0]
            top_partners = partner_totals.head(top_n).index.tolist()
            
            print(f"   Top {top_n} country partners: {top_partners}")
            
            # Filter for top partners
            top_data = data[data['partner'].isin(top_partners)].copy()
            others_data = data[
                (~data['partner'].isin(top_partners)) & 
                (data['net_imports_toe'] > 0)  # Only positive net imports for "Others"
            ].copy()
            
            # Create pivot for stacking
            pivot_data = top_data.pivot_table(index='year', columns='partner', values='net_imports_toe', fill_value=0)
            
            # Calculate "Others" (sum of remaining countries)
            others_by_year = others_data.groupby('year')['net_imports_toe'].sum()
            
            plt.figure(figsize=(12, 8))
            
            # Prepare stacking data with consistent colors
            stack_data = []
            stack_labels = list(pivot_data.columns)
            colors_to_use = COLORS[:len(pivot_data.columns)]
            
            for col in pivot_data.columns:
                stack_data.append(pivot_data[col])
                
            # Add "Others" on top with next color in sequence
            if not others_by_year.empty and others_by_year.max() > 0:
                others_aligned = others_by_year.reindex(pivot_data.index, fill_value=0)
                stack_data.append(others_aligned.values)
                stack_labels.append('Others')
                colors_to_use.append(COLORS[len(pivot_data.columns) % len(COLORS)])
            
            plt.stackplot(pivot_data.index, *stack_data,
                         labels=stack_labels,
                         colors=colors_to_use,
                         alpha=0.8)
            
            source_label = 'Natural Gas' if source == 'gas' else 'Oil & Petroleum Products'
            plt.title(f'Switzerland {source_label} - Net Imports by Country\nTop {top_n} Suppliers + Others (TOE)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Net Imports (TOE)', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = self.output_dir / f'switzerland_{source}_partners_decomposition.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[OK] {source_label} country decomposition chart saved: {output_path}")
            plt.close()
    
    def plot_top5_energy_sources(self):
        """Plot top 5 Swiss energy sources consumption over time"""
        print("\n" + "="*50)
        print("CREATING TOP 5 SWISS ENERGY SOURCES CHART")
        print("="*50)
        
        if self.gross_energy_data is None:
            print("No Swiss energy data available")
            return
            
        # Filter for gross consumption
        brutto_data = self.gross_energy_data[
            self.gross_energy_data['Rubrik'] == 'Bruttoverbrauch'
        ].copy()
        
        if brutto_data.empty:
            print("No gross consumption data found")
            return
            
        # Convert TJ to numeric
        brutto_data['TJ_numeric'] = pd.to_numeric(brutto_data['TJ'], errors='coerce')
        
        # Get top 5 energy sources by average consumption
        avg_consumption = brutto_data.groupby('Energietraeger')['TJ_numeric'].mean().sort_values(ascending=False)
        top5_sources = avg_consumption.head(5).index.tolist()
        
        print(f"Top 5 energy sources: {top5_sources}")
        
        # Filter data for top 5 sources
        top5_data = brutto_data[brutto_data['Energietraeger'].isin(top5_sources)].copy()
        
        # Create pivot table for plotting
        pivot_data = top5_data.pivot_table(
            index='Jahr', 
            columns='Energietraeger', 
            values='TJ_numeric', 
            fill_value=0
        )
        
        # Reorder columns by consumption level
        pivot_data = pivot_data.reindex(columns=top5_sources)
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Plot each energy source
        colors = COLORS[:len(top5_sources)]
        for i, source in enumerate(top5_sources):
            plt.plot(pivot_data.index, pivot_data[source], 
                    marker='o', linewidth=2.5, markersize=6,
                    color=colors[i], label=source)
        
        plt.title('Switzerland Top 5 Energy Sources - Gross Consumption Over Time', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Gross Consumption (TJ)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add average consumption annotation
        plt.text(0.02, 0.98, 
                f'Top 5 energy sources by average consumption:\n' +
                '\n'.join([f'{i+1}. {source}: {val:,.0f} TJ' 
                          for i, (source, val) in enumerate(avg_consumption.head(5).items())]),
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / 'switzerland_top5_energy_sources.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Top 5 energy sources chart saved: {output_path}")
        plt.close()


def main():
    """Main execution function"""
    print("="*60)
    print("ENERGY DEPENDENCY ANALYSIS - FRANCE & SWITZERLAND")
    print("="*60)
    
    # ===== FRANCE ANALYSIS =====
    print("\n" + "[FR] " * 20)
    print("FRENCH ENERGY DEPENDENCY ANALYSIS")
    print("[FR] " * 20)
    
    # Initialize French analyzer
    french_analyzer = FrenchEnergyDependencyAnalyzer()
    
    # Load and explore data
    if french_analyzer.load_data():
        french_analyzer.explore_data()
        
        # Calculate dependency ratios
        french_dependencies = french_analyzer.calculate_dependency_ratios()
        
        # Calculate partner decompositions
        french_decompositions = french_analyzer.calculate_partner_decomposition()
        
        # Create visualizations
        french_analyzer.plot_dependency_trends(french_dependencies)
        french_analyzer.plot_partner_decomposition(french_decompositions)
    
    # ===== SWITZERLAND ANALYSIS =====
    print("\n" + "[CH] " * 20)
    print("SWISS ENERGY DEPENDENCY ANALYSIS")  
    print("[CH] " * 20)
    
    # Initialize Swiss analyzer
    swiss_analyzer = SwissEnergyDependencyAnalyzer()
    
    # Load and explore data
    if swiss_analyzer.load_data():
        swiss_analyzer.explore_data()
        
        # Calculate Swiss dependency ratios for 4 specified sources
        swiss_dependencies = swiss_analyzer.calculate_swiss_dependency_ratios()
        
        # Create Swiss dependency trend chart
        if swiss_dependencies:
            swiss_analyzer.plot_swiss_dependency_trends(swiss_dependencies)
        else:
            print("[WARNING] No Swiss dependency data to plot")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("Charts saved in:", OUTPUT_DIR)
    print("="*60)

if __name__ == "__main__":
    main()