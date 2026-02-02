"""
EUROSTAT Analysis - Generate graphs from Eurostat housing data
This script creates visualizations for EU27 and individual countries
Outputs saved as PNG files in outputs/graphs/EUROSTAT folder
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Base paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, 'external_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'graphs', 'EUROSTAT')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EU27 and EFTA countries
EU27_COUNTRIES = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
    'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
    'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
    'Spain', 'Sweden'
]
EFTA_COUNTRIES = ['Iceland', 'Norway', 'Switzerland']

# Map Eurostat country names to our list
EUROSTAT_TO_STANDARD = {
    'Austria': 'Austria', 'Belgium': 'Belgium', 'Bulgaria': 'Bulgaria',
    'Croatia': 'Croatia', 'Cyprus': 'Cyprus', 'Czechia': 'Czech Republic',
    'Denmark': 'Denmark', 'Estonia': 'Estonia', 'Finland': 'Finland',
    'France': 'France', 'Germany': 'Germany', 'Greece': 'Greece',
    'Hungary': 'Hungary', 'Iceland': 'Iceland', 'Ireland': 'Ireland',
    'Italy': 'Italy', 'Latvia': 'Latvia', 'Lithuania': 'Lithuania',
    'Luxembourg': 'Luxembourg', 'Malta': 'Malta', 'Netherlands': 'Netherlands',
    'Norway': 'Norway', 'Poland': 'Poland', 'Portugal': 'Portugal',
    'Romania': 'Romania', 'Slovakia': 'Slovakia', 'Slovenia': 'Slovenia',
    'Spain': 'Spain', 'Sweden': 'Sweden', 'Switzerland': 'Switzerland'
}

TARGET_COUNTRIES = EU27_COUNTRIES + EFTA_COUNTRIES

# Color palette (matching hbs_test_lu_2020.py)
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_AGGREGATE_COLOR = '#80b1d3'
COMPONENT_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3', 
                    '#80b1d3', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']

# Create comprehensive color map for all countries (31 distinct colors)
_all_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
                  'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
                  'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
                  'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
                  'Spain', 'Sweden', 'Iceland', 'Liechtenstein', 'Norway', 'Switzerland']
_color_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf',
                  '#999999', '#66c2a5', '#e67e22', '#8da0cb', '#d946ef', '#a6d854', '#ffd92f',
                  '#e5c494', '#b3b3b3', '#8dd3c7', '#ffffb3', '#bebada', '#1f77b4', '#80b1d3',
                  '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f',
                  '#a6cee3', '#1f78b4', '#ff1493']
COUNTRY_COLOR_MAP = {country: _color_palette[i] for i, country in enumerate(_all_countries)}
DEFAULT_COUNTRY_COLOR = '#8dd3c7'

def get_country_color(country):
    """Return color for country"""
    if country in COUNTRY_COLOR_MAP:
        return COUNTRY_COLOR_MAP[country]
    return DEFAULT_COUNTRY_COLOR

def standardize_country_name(country):
    """Convert Eurostat country name to standard name"""
    return EUROSTAT_TO_STANDARD.get(country, None)

def load_real_estate_data():
    """Load and process real estate data"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_real estate other than main.csv')
    
    if not os.path.exists(file_path):
        print("  Missing data file: eurostat_real estate other than main.csv")
        return None
    
    df = pd.read_csv(file_path)
    
    # Keep only latest year
    df = df[df['TIME_PERIOD'] == df['TIME_PERIOD'].max()]
    
    # Filter to EU27 aggregate and individual countries
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label] + list(EUROSTAT_TO_STANDARD.keys()))].copy()
    
    # Standardize country names
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    
    # Convert value to numeric
    df_filtered['value'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['value'])
    
    return df_filtered

def load_rooms_data():
    """Load and process average rooms data"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_Average number of rooms.csv')
    
    if not os.path.exists(file_path):
        print("  Missing data file: eurostat_Average number of rooms.csv")
        return None
    
    df = pd.read_csv(file_path)
    
    # Keep only total dwelling type
    df = df[df['building'] == 'Total']
    
    # Keep only latest year
    df = df[df['TIME_PERIOD'] == df['TIME_PERIOD'].max()]
    
    # Filter to EU27 aggregate and individual countries
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label] + list(EUROSTAT_TO_STANDARD.keys()))].copy()
    
    # Standardize country names
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    
    # Convert value to numeric
    df_filtered['value'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['value'])
    
    return df_filtered

def create_real_estate_graphs():
    """Create real estate visualization graphs"""
    print("\n[1] Creating real estate by income quintile graphs...")
    
    df = load_real_estate_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    # Graph 1: EU27 aggregate with all quintiles
    df_eu27 = df[df['country_name'] == 'EU27'].copy()
    if not df_eu27.empty:
        # Order quintiles
        quintile_order = ['First quintile', 'Second quintile', 'Third quintile', 
                         'Fourth quintile', 'Fifth quintile', 'Total']
        df_eu27 = df_eu27[df_eu27['quant_inc'].isin(quintile_order)]
        df_eu27['quant_inc'] = pd.Categorical(df_eu27['quant_inc'], categories=quintile_order, ordered=True)
        df_eu27 = df_eu27.sort_values('quant_inc')
        
        fig, ax = plt.subplots(figsize=(11, 7))
        
        colors = COMPONENT_COLORS[:len(df_eu27)]
        bars = ax.barh(range(len(df_eu27)), df_eu27['value'], color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_eu27)))
        ax.set_yticklabels(df_eu27['quant_inc'], fontsize=11)
        ax.set_xlabel('Persons owning real estate (%)', fontsize=12, fontweight='bold')
        latest_year = int(df['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df.columns else 'Latest'
        ax.set_title(f'EU27: Persons Owning Real Estate Other Than Main Residence ({latest_year})\nby Income Quintile', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_eu27.iterrows()):
            ax.text(row['value'] + 1, i, f"{row['value']:.1f}%", va='center', fontsize=10)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        ax.set_xlim(0, max(df_eu27['value']) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '1_real_estate_eu27_quintiles.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 1_real_estate_eu27_quintiles.png")
    
    # Graph 2: Countries - Total values only
    df_countries = df[(df['country_name'] != 'EU27') & (df['quant_inc'] == 'Total')].copy()
    if not df_countries.empty:
        df_countries = df_countries.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(11, 12))
        
        colors_list = [get_country_color(c) for c in df_countries['country_name']]
        bars = ax.barh(range(len(df_countries)), df_countries['value'], color=colors_list, 
                      edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_countries)))
        ax.set_yticklabels(df_countries['country_name'], fontsize=10)
        ax.set_xlabel('Persons owning real estate (%)', fontsize=12, fontweight='bold')
        latest_year = int(df_countries['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df_countries.columns else 'Latest'
        ax.set_title(f'Real Estate Ownership Other Than Main Residence (Total) ({latest_year})\nby Country', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_countries.iterrows()):
            ax.text(row['value'] + 1, i, f"{row['value']:.1f}%", va='center', fontsize=9)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        ax.set_xlim(0, max(df_countries['value']) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '2_real_estate_countries_total.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 2_real_estate_countries_total.png")

def create_rooms_graphs():
    """Create average rooms visualization graphs"""
    print("\n[2] Creating average rooms graphs...")
    
    df = load_rooms_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    # Graph 1: EU27 aggregate with tenure types
    df_eu27 = df[df['country_name'] == 'EU27'].copy()
    if not df_eu27.empty:
        tenure_order = ['Tenant', 'Owner', 'Total']
        df_eu27 = df_eu27[df_eu27['tenure'].isin(tenure_order)]
        df_eu27['tenure'] = pd.Categorical(df_eu27['tenure'], categories=tenure_order, ordered=True)
        df_eu27 = df_eu27.sort_values('tenure')
        
        fig, ax = plt.subplots(figsize=(11, 7))
        
        colors = COMPONENT_COLORS[:len(df_eu27)]
        bars = ax.barh(range(len(df_eu27)), df_eu27['value'], color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_eu27)))
        ax.set_yticklabels(df_eu27['tenure'], fontsize=11)
        ax.set_xlabel('Average number of rooms per person', fontsize=12, fontweight='bold')
        latest_year = int(df['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df.columns else 'Latest'
        ax.set_title(f'EU27: Average Number of Rooms Per Person ({latest_year})\nby Tenure Status', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_eu27.iterrows()):
            ax.text(row['value'] + 0.05, i, f"{row['value']:.2f}", va='center', fontsize=10)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        ax.set_xlim(0, max(df_eu27['value']) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '3_rooms_eu27_tenure.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 3_rooms_eu27_tenure.png")
    
    # Graph 2: Countries - Total tenure only
    df_countries = df[(df['country_name'] != 'EU27') & (df['tenure'] == 'Total')].copy()
    if not df_countries.empty:
        df_countries = df_countries.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(11, 12))
        
        colors_list = [get_country_color(c) for c in df_countries['country_name']]
        bars = ax.barh(range(len(df_countries)), df_countries['value'], color=colors_list, 
                      edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_countries)))
        ax.set_yticklabels(df_countries['country_name'], fontsize=10)
        ax.set_xlabel('Average number of rooms per person', fontsize=12, fontweight='bold')
        latest_year = int(df_countries['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df_countries.columns else 'Latest'
        ax.set_title(f'Average Number of Rooms Per Person (Total) ({latest_year})\nby Country', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_countries.iterrows()):
            ax.text(row['value'] + 0.05, i, f"{row['value']:.2f}", va='center', fontsize=9)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        ax.set_xlim(0, max(df_countries['value']) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '4_rooms_countries_total.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 4_rooms_countries_total.png")

def load_energy_efficiency_data():
    """Load and process energy efficiency data"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_energy efficiency.csv')
    
    if not os.path.exists(file_path):
        print("  Missing data file: eurostat_energy efficiency.csv")
        return None
    
    df = pd.read_csv(file_path)
    
    # Filter to EU27 aggregate and individual countries
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label] + list(EUROSTAT_TO_STANDARD.keys()))].copy()
    
    # Standardize country names
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    
    # Convert value to numeric
    df_filtered['value'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['value'])
    
    return df_filtered

def load_berd_data():
    """Load and process Business Enterprise R&D Expenditure data"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_BERD by NACE.csv')
    
    if not os.path.exists(file_path):
        print("  Missing data file: eurostat_BERD by NACE.csv")
        return None
    
    df = pd.read_csv(file_path)
    
    # Keep only latest year
    df = df[df['TIME_PERIOD'] == df['TIME_PERIOD'].max()]
    
    # Filter to EU27 aggregate and individual countries
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label] + list(EUROSTAT_TO_STANDARD.keys()))].copy()
    
    # Standardize country names
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    
    # Convert value to numeric
    df_filtered['value'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['value'])
    
    return df_filtered

def load_hrst_data():
    """Load and process Employed Persons in Science and Technology (HRST) data"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_Employed persons in science and technology (HRST).csv')
    
    if not os.path.exists(file_path):
        print("  Missing data file: eurostat_Employed persons in science and technology (HRST).csv")
        return None
    
    df = pd.read_csv(file_path)
    
    # Keep only latest year
    df = df[df['TIME_PERIOD'] == df['TIME_PERIOD'].max()]
    
    # Filter to single category to avoid double-counting
    df = df[df['category'] == 'Persons employed in science and technology']
    
    # Filter to EU27 aggregate and individual countries
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label] + list(EUROSTAT_TO_STANDARD.keys()))].copy()
    
    # Standardize country names
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    
    # Convert value to numeric
    df_filtered['value'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['value'])
    
    # Sum across age groups
    df_filtered = df_filtered.groupby(['country_name', 'nace_r2']).agg({'value': 'mean'}).reset_index()
    
    return df_filtered

def create_energy_efficiency_graphs():
    """Create energy efficiency visualization graphs"""
    print("\n[3] Creating energy efficiency graphs...")
    
    df = load_energy_efficiency_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    # Age group order
    age_order = ['16 years or over', 'From 16 to 29 years', 'From 25 to 34 years',
                 'From 35 to 44 years', 'From 45 to 64 years', '65 years or over',
                 'From 16 to 44 years', 'From 16 to 64 years']
    
    # Graph 1: EU27 aggregate with all age groups
    df_eu27 = df[df['country_name'] == 'EU27'].copy()
    if not df_eu27.empty:
        df_eu27 = df_eu27[df_eu27['age'].isin(age_order)]
        df_eu27['age'] = pd.Categorical(df_eu27['age'], categories=age_order, ordered=True)
        df_eu27 = df_eu27.sort_values('age')
        
        fig, ax = plt.subplots(figsize=(11, 8))
        
        colors = COMPONENT_COLORS[:len(df_eu27)]
        bars = ax.barh(range(len(df_eu27)), df_eu27['value'], color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_eu27)))
        ax.set_yticklabels(df_eu27['age'], fontsize=10)
        ax.set_xlabel('Persons living in improved dwellings (%)', fontsize=12, fontweight='bold')
        latest_year = int(df['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df.columns else '2023'
        ax.set_title(f'EU27: Persons Living in Dwellings Whose Energy Efficiency ({latest_year})\nHas Been Improved in the Last 5 Years by Age', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_eu27.iterrows()):
            ax.text(row['value'] + 1, i, f"{row['value']:.1f}%", va='center', fontsize=9)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        ax.set_xlim(0, max(df_eu27['value']) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '5_energy_efficiency_eu27_age.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 5_energy_efficiency_eu27_age.png")
    
    # Graph 2: Countries - "16 years or over" only
    df_countries = df[(df['country_name'] != 'EU27') & (df['age'] == '16 years or over')].copy()
    if not df_countries.empty:
        df_countries = df_countries.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(11, 12))
        
        colors_list = [get_country_color(c) for c in df_countries['country_name']]
        bars = ax.barh(range(len(df_countries)), df_countries['value'], color=colors_list, 
                      edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_countries)))
        ax.set_yticklabels(df_countries['country_name'], fontsize=10)
        ax.set_xlabel('Persons living in improved dwellings (%)', fontsize=12, fontweight='bold')
        latest_year = int(df_countries['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df_countries.columns else '2023'
        ax.set_title(f'Energy Efficiency: Persons in Improved Dwellings (16 years+) ({latest_year})\nby Country', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_countries.iterrows()):
            ax.text(row['value'] + 1, i, f"{row['value']:.1f}%", va='center', fontsize=9)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        ax.set_xlim(0, max(df_countries['value']) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '6_energy_efficiency_countries_16plus.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 6_energy_efficiency_countries_16plus.png")

def create_berd_graphs():
    """Create Business Enterprise R&D Expenditure graphs"""
    print("\n[4] Creating BERD graphs...")
    
    df = load_berd_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    # Get the latest year for title
    latest_year = int(df['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df.columns else 'Latest'
    
    # NACE order
    nace_order = ['Total - all NACE activities', 'Manufacturing', 'Services', 'Construction',
                  'Agriculture, forestry and fishing', 'Mining and quarrying',
                  'Electricity, gas, steam and air conditioning supply; water supply; sewerage, waste management and remediation activities']
    
    # Create separate graphs for each unit
    for unit_val in df['unit'].unique():
        df_unit = df[df['unit'] == unit_val].copy()
        
        # Short label for unit
        unit_label = 'PPS per inhabitant' if 'PPS' in unit_val else 'Percentage of GDP'
        
        # Graph 1: EU27 aggregate with NACE breakdown
        df_eu27 = df_unit[df_unit['country_name'] == 'EU27'].copy()
        if not df_eu27.empty:
            df_eu27 = df_eu27[df_eu27['nace_r2'].isin(nace_order)]
            df_eu27['nace_r2'] = pd.Categorical(df_eu27['nace_r2'], categories=nace_order, ordered=True)
            df_eu27 = df_eu27.sort_values('nace_r2')
            
            fig, ax = plt.subplots(figsize=(11, 8))
            
            colors = COMPONENT_COLORS[:len(df_eu27)]
            bars = ax.barh(range(len(df_eu27)), df_eu27['value'], color=colors, edgecolor='white', linewidth=1.5)
            
            ax.set_yticks(range(len(df_eu27)))
            ax.set_yticklabels(df_eu27['nace_r2'], fontsize=9)
            
            if 'PPS' in unit_val:
                ax.set_xlabel(f'BERD ({unit_label})', fontsize=12, fontweight='bold')
                value_format = f"{{:.0f}}"
            else:
                ax.set_xlabel(f'BERD ({unit_label})', fontsize=12, fontweight='bold')
                value_format = f"{{:.2f}}"
            
            ax.set_title(f'EU27: Business Enterprise R&D Expenditure by NACE ({latest_year})\n({unit_label})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            # Add value labels
            for i, (idx, row) in enumerate(df_eu27.iterrows()):
                val_str = value_format.format(row['value'])
                if 'PPS' in unit_val:
                    ax.text(row['value'] + max(df_eu27['value'])*0.02, i, val_str, va='center', fontsize=9)
                else:
                    ax.text(row['value'] + max(df_eu27['value'])*0.03, i, f"{val_str}%", va='center', fontsize=9)
            
            ax.grid(axis='x', alpha=0.2)
            ax.set_facecolor('white')
            ax.set_xlim(0, max(df_eu27['value']) * 1.15)
            plt.tight_layout()
            
            file_suffix = 'pps' if 'PPS' in unit_val else 'pct_gdp'
            plt.savefig(os.path.join(OUTPUT_DIR, f'7_berd_eu27_nace_{file_suffix}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 7_berd_eu27_nace_{file_suffix}.png")
        
        # Graph 2: Countries - Stacked bar chart by NACE sectors
        # Note: 2024 only has Total NACE data, so we use 2023 for sector breakdown
        # For missing 2023 data per country-NACE, use last available year
        berd_file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_BERD by NACE.csv')
        df_all_years = pd.read_csv(berd_file_path)
        df_all_years['value'] = pd.to_numeric(df_all_years['OBS_VALUE'], errors='coerce')
        df_all_years = df_all_years.dropna(subset=['value'])
        
        # Exclude EU27
        eu27_label = 'European Union - 27 countries (from 2020)'
        df_all_years = df_all_years[df_all_years['geo'] != eu27_label].copy()
        df_all_years['country_name'] = df_all_years['geo'].apply(standardize_country_name)
        df_all_years = df_all_years.dropna(subset=['country_name'])
        df_all_years = df_all_years[df_all_years['unit'] == unit_val].copy()
        
        # Filter to major NACE sectors, excluding Total
        nace_sectors = ['Manufacturing', 'Services', 'Construction',
                       'Agriculture, forestry and fishing', 'Mining and quarrying',
                       'Electricity, gas, steam and air conditioning supply; water supply; sewerage, waste management and remediation activities']
        df_all_years = df_all_years[df_all_years['nace_r2'].isin(nace_sectors)]
        
        if not df_all_years.empty:
            # For each country-NACE pair, get 2023 data if available, otherwise last available year
            df_pivot_data = []
            for (country, nace), group in df_all_years.groupby(['country_name', 'nace_r2']):
                # Check if 2023 data exists for this country-NACE
                data_2023 = group[group['TIME_PERIOD'] == 2023]
                if not data_2023.empty:
                    df_pivot_data.append({'country_name': country, 'nace_r2': nace, 'value': data_2023['value'].iloc[0]})
                else:
                    # Use last available year
                    last_year = group.sort_values('TIME_PERIOD', ascending=False).iloc[0]
                    df_pivot_data.append({'country_name': country, 'nace_r2': nace, 'value': last_year['value']})
            
            df_countries = pd.DataFrame(df_pivot_data)
            
            # Pivot to get NACE sectors as columns
            df_pivot = df_countries.pivot(index='country_name', columns='nace_r2', values='value').fillna(0)
            
            # Reorder columns
            available_nace = [n for n in nace_sectors if n in df_pivot.columns]
            df_pivot = df_pivot[available_nace]
            
            # Sort by Total value
            df_pivot['total'] = df_pivot.sum(axis=1)
            df_pivot = df_pivot.sort_values('total', ascending=True)
            df_pivot = df_pivot.drop('total', axis=1)
            
            fig, ax = plt.subplots(figsize=(14, 12))
            
            # Create stacked bar chart
            colors_nace = COMPONENT_COLORS[:len(available_nace)]
            
            df_pivot.plot(kind='barh', stacked=True, ax=ax, color=colors_nace, 
                         edgecolor='white', linewidth=1.5, width=0.8)
            
            ax.set_xlabel(f'BERD ({unit_label})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Country', fontsize=12, fontweight='bold')
            ax.set_title(f'Business Enterprise R&D Expenditure by Country (2023)\n({unit_label}) - Decomposed by NACE Sector', 
                        fontsize=13, fontweight='bold', pad=15)
            
            # Simplify legend labels
            handles, labels = ax.get_legend_handles_labels()
            short_labels = [l.replace('Electricity, gas, steam and air conditioning supply; water supply; sewerage, waste management and remediation activities', 
                                     'Electricity, gas, water & utilities')
                           .replace('Agriculture, forestry and fishing', 'Agriculture & forestry')
                           for l in labels]
            ax.legend(handles, short_labels, fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
            
            ax.grid(axis='x', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            file_suffix = 'pps' if 'PPS' in unit_val else 'pct_gdp'
            plt.savefig(os.path.join(OUTPUT_DIR, f'8_berd_countries_total_{file_suffix}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 8_berd_countries_total_{file_suffix}.png")
    
    # Construction focused graphs
    print("\n  Creating Construction-focused BERD graphs...")
    # Load full BERD data to ensure Construction is present
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_BERD by NACE.csv')
    if os.path.exists(file_path):
        df_all = pd.read_csv(file_path)
        
        eu27_label = 'European Union - 27 countries (from 2020)'
        df_all = df_all[df_all['geo'].isin([eu27_label] + list(EUROSTAT_TO_STANDARD.keys()))].copy()
        df_all['country_name'] = df_all['geo'].apply(
            lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
        )
        df_all = df_all.dropna(subset=['country_name'])
        df_all['value'] = pd.to_numeric(df_all['OBS_VALUE'], errors='coerce')
        df_all = df_all.dropna(subset=['value'])
        
        # Find latest year with Construction data (avoiding 2024 which has only 6 countries)
        df_construction_all = df_all[df_all['nace_r2'] == 'Construction'].copy()
        if not df_construction_all.empty:
            # Get all years with Construction and at least 40 countries
            years_with_const = df_construction_all.groupby('TIME_PERIOD').size()
            years_with_const = years_with_const[years_with_const >= 40]
            if not years_with_const.empty:
                latest_construction_year = int(years_with_const.index.max())
            else:
                latest_construction_year = int(df_construction_all['TIME_PERIOD'].max())
            print(f"  Latest year with good Construction data: {latest_construction_year} ({len(years_with_const)} years available)")
        else:
            latest_construction_year = int(df_all['TIME_PERIOD'].max())
            print(f"  No Construction data found, using latest year: {latest_construction_year}")
        
        df_const_data = df_all[df_all['TIME_PERIOD'] == latest_construction_year]
        
        for unit_val in df_const_data['unit'].unique():
            df_unit = df_const_data[df_const_data['unit'] == unit_val].copy()
            unit_label = 'PPS per inhabitant' if 'PPS' in unit_val else 'Percentage of GDP'
            
            # Countries - Construction vs Total comparison
            df_comparison = df_unit[df_unit['nace_r2'].isin(['Construction', 'Total - all NACE activities'])].copy()
            df_comparison = df_comparison[df_comparison['country_name'] != 'EU27']
            
            if not df_comparison.empty and len(df_comparison[df_comparison['nace_r2'] == 'Construction']) > 0:
                # Pivot for comparison
                df_pivot = df_comparison.pivot(index='country_name', columns='nace_r2', values='value').fillna(0)
                if 'Construction' in df_pivot.columns and 'Total - all NACE activities' in df_pivot.columns:
                    # Sort by Construction value descending (highest first)
                    df_pivot = df_pivot[['Total - all NACE activities', 'Construction']].sort_values('Construction', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(11, 12))
                    
                    x = np.arange(len(df_pivot))
                    width = 0.35
                    
                    colors_total = ['#80b1d3'] * len(df_pivot)
                    colors_const = ['#fb8072'] * len(df_pivot)
                    
                    bars1 = ax.barh(x - width/2, df_pivot['Total - all NACE activities'], width, 
                                   label='Total NACE', color=colors_total, edgecolor='white', linewidth=1.5)
                    bars2 = ax.barh(x + width/2, df_pivot['Construction'], width, 
                                   label='Construction', color=colors_const, edgecolor='white', linewidth=1.5)
                    
                    ax.set_yticks(x)
                    ax.set_yticklabels(df_pivot.index, fontsize=9)
                    
                    ax.set_xlabel(f'BERD ({unit_label})', fontsize=12, fontweight='bold')
                    ax.set_title(f'Construction BERD vs Total NACE by Country ({latest_construction_year})\n({unit_label})', 
                                fontsize=13, fontweight='bold', pad=15)
                    ax.legend(fontsize=10, loc='best')
                    ax.grid(axis='x', alpha=0.2)
                    ax.set_facecolor('white')
                    
                    plt.tight_layout()
                    file_suffix = 'pps' if 'PPS' in unit_val else 'pct_gdp'
                    plt.savefig(os.path.join(OUTPUT_DIR, f'11_berd_construction_vs_total_{file_suffix}.png'), 
                               dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
                    print(f"  [SAVED] 11_berd_construction_vs_total_{file_suffix}.png")

def create_hrst_graphs():
    """Create Employed Persons in Science and Technology (HRST) graphs"""
    print("\n[5] Creating HRST graphs...")
    
    df = load_hrst_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    # Get latest year for title
    latest_year = int(df['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df.columns else 'Latest'
    
    # NACE order
    nace_order = ['Total - all NACE activities', 'Manufacturing', 'Services',
                  'Agriculture, forestry and fishing; mining and quarrying',
                  'Electricity, gas, steam and air conditioning supply; water supply and construction']
    
    # Graph 1: EU27 aggregate with NACE breakdown
    df_eu27 = df[df['country_name'] == 'EU27'].copy()
    if not df_eu27.empty:
        df_eu27 = df_eu27[df_eu27['nace_r2'].isin(nace_order)]
        df_eu27['nace_r2'] = pd.Categorical(df_eu27['nace_r2'], categories=nace_order, ordered=True)
        df_eu27 = df_eu27.sort_values('nace_r2')
        
        fig, ax = plt.subplots(figsize=(11, 8))
        
        colors = COMPONENT_COLORS[:len(df_eu27)]
        bars = ax.barh(range(len(df_eu27)), df_eu27['value'], color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_eu27)))
        ax.set_yticklabels(df_eu27['nace_r2'], fontsize=10)
        ax.set_xlabel('Percentage of total employment (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'EU27: Employed Persons in Science and Technology (HRST) ({latest_year})\nby NACE Rev. 2 Activity', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_eu27.iterrows()):
            ax.text(row['value'] + max(df_eu27['value'])*0.02, i, f"{row['value']:.1f}%", va='center', fontsize=9)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        ax.set_xlim(0, max(df_eu27['value']) * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '9_hrst_eu27_nace.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 9_hrst_eu27_nace.png")
    
    # Graph 2: Countries - Stacked bar chart decomposed by NACE (EXCLUDING Total)
    df_countries = df[df['country_name'] != 'EU27'].copy()
    if not df_countries.empty:
        # Filter to countries that have data, EXCLUDING Total NACE
        nace_sectors_only = [n for n in nace_order if n != 'Total - all NACE activities']
        df_countries = df_countries[df_countries['nace_r2'].isin(nace_sectors_only)]
        
        # Pivot to get NACE sectors as columns
        df_pivot = df_countries.pivot(index='country_name', columns='nace_r2', values='value').fillna(0)
        
        # Reorder columns
        available_nace = [n for n in nace_sectors_only if n in df_pivot.columns]
        df_pivot = df_pivot[available_nace]
        
        # Calculate percentages: normalize each row to sum to 100%
        df_pivot = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100
        
        # Sort by first sector value for better visualization
        df_pivot = df_pivot.sort_values(available_nace[0], ascending=True)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create stacked bar chart
        colors_nace = COMPONENT_COLORS[:len(available_nace)]
        
        df_pivot.plot(kind='barh', stacked=True, ax=ax, color=colors_nace, 
                     edgecolor='white', linewidth=1.5, width=0.8)
        
        ax.set_xlabel('Percentage of HRST Employment (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Country', fontsize=12, fontweight='bold')
        ax.set_title(f'Employed Persons in Science and Technology (HRST) by Country ({latest_year})\nDecomposed by NACE Rev. 2 Sector (% of HRST)', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Simplify legend labels
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [l.replace('Electricity, gas, steam and air conditioning supply; water supply and construction', 
                                       'Electricity, gas, water & construction').replace('Agriculture, forestry and fishing; mining and quarrying', 'Agriculture, forestry & mining')
                           for l in labels], 
                 fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '10_hrst_countries_by_nace_stacked.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 10_hrst_countries_by_nace_stacked.png")

def create_berd_timeseries():
    """Create BERD time series graphs"""
    print("\n[6] Creating BERD time series graphs...")
    
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_BERD by NACE.csv')
    if not os.path.exists(file_path):
        print("  Missing data file")
        return
    
    df = pd.read_csv(file_path)
    
    # Filter to EU27 aggregate
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_eu27 = df[df['geo'] == eu27_label].copy()
    
    if df_eu27.empty:
        print("  No EU27 data available")
        return
    
    # For each unit
    for unit_val in df_eu27['unit'].unique():
        df_unit = df_eu27[df_eu27['unit'] == unit_val].copy()
        unit_label = 'PPS per inhabitant' if 'PPS' in unit_val else 'Percentage of GDP'
        
        # Get ALL NACE sectors (excluding Total for clarity)
        df_time = df_unit[df_unit['nace_r2'] != 'Total - all NACE activities'].copy()
        df_time['value'] = pd.to_numeric(df_time['OBS_VALUE'], errors='coerce')
        df_time['TIME_PERIOD'] = pd.to_numeric(df_time['TIME_PERIOD'], errors='coerce')
        df_time = df_time.dropna(subset=['value', 'TIME_PERIOD'])
        
        if df_time.empty:
            print(f"  No data for {unit_label}")
            continue
        
        # Get unique NACE sectors
        sectors = df_time['nace_r2'].unique()
        print(f"  Plotting {len(sectors)} NACE sectors for EU27 ({unit_label})")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define color palette for sectors
        colors_nace = {
            'Manufacturing': '#fb8072',
            'Construction': '#ffd558',
            'Services': '#8dd3c7',
            'Agriculture, forestry and fishing': '#a6d854',
            'Mining and quarrying': '#998ec3',
            'Electricity, gas, steam and air conditioning supply; water supply; sewerage, waste management and remediation activities': '#80b1d3'
        }
        
        # Plot each NACE sector as a line
        for sector in sorted(sectors):
            df_sector = df_time[df_time['nace_r2'] == sector].sort_values('TIME_PERIOD')
            if not df_sector.empty:
                color = colors_nace.get(sector, '#999999')
                ax.plot(df_sector['TIME_PERIOD'].astype(int), df_sector['value'], 
                       marker='o', linewidth=2.5, markersize=6, label=sector, 
                       color=color, alpha=0.8)
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'BERD ({unit_label})', fontsize=12, fontweight='bold')
        ax.set_title(f'EU27: Business Enterprise R&D Expenditure by NACE Sector\n({unit_label}) - All Available Data', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Simplify legend labels
        handles, labels = ax.get_legend_handles_labels()
        short_labels = [l.replace('Electricity, gas, steam and air conditioning supply; water supply; sewerage, waste management and remediation activities', 
                                 'Electricity, gas, water & utilities')
                       .replace('Agriculture, forestry and fishing', 'Agriculture & forestry')
                       for l in labels]
        ax.legend(handles, short_labels, fontsize=10, loc='best', framealpha=0.9)
        
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('white')
        
        # Format x-axis to show every other year
        all_years = sorted(df_time['TIME_PERIOD'].unique())
        ax.set_xticks(all_years[::2])
        ax.set_xticklabels([int(y) for y in all_years[::2]])
        
        plt.tight_layout()
        file_suffix = 'pps' if 'PPS' in unit_val else 'pct_gdp'
        plt.savefig(os.path.join(OUTPUT_DIR, f'12_berd_timeseries_{file_suffix}.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] 12_berd_timeseries_{file_suffix}.png")

def create_france_berd_timeseries():
    """Create France-specific BERD time series graphs with ALL NACE sectors"""
    print("\n[7] Creating France BERD time series graphs...")
    
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_BERD by NACE.csv')
    if not os.path.exists(file_path):
        print("  Missing data file")
        return
    
    df = pd.read_csv(file_path)
    
    # Filter to France
    df_france = df[df['geo'] == 'France'].copy()
    
    if df_france.empty:
        print("  No France data available")
        return
    
    # For each unit
    for unit_val in df_france['unit'].unique():
        df_unit = df_france[df_france['unit'] == unit_val].copy()
        unit_label = 'PPS per inhabitant' if 'PPS' in unit_val else 'Percentage of GDP'
        
        # Get ALL NACE sectors (excluding Total for clarity)
        df_time = df_unit[df_unit['nace_r2'] != 'Total - all NACE activities'].copy()
        df_time['value'] = pd.to_numeric(df_time['OBS_VALUE'], errors='coerce')
        df_time['TIME_PERIOD'] = pd.to_numeric(df_time['TIME_PERIOD'], errors='coerce')
        df_time = df_time.dropna(subset=['value', 'TIME_PERIOD'])
        
        if df_time.empty:
            print(f"  No data for France - {unit_label}")
            continue
        
        # Get unique NACE sectors
        sectors = df_time['nace_r2'].unique()
        print(f"  Plotting {len(sectors)} NACE sectors for France ({unit_label})")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define color palette for sectors
        colors_nace = {
            'Manufacturing': '#fb8072',
            'Construction': '#ffd558',
            'Services': '#8dd3c7',
            'Agriculture, forestry and fishing': '#a6d854',
            'Mining and quarrying': '#998ec3',
            'Electricity, gas, steam and air conditioning supply; water supply; sewerage, waste management and remediation activities': '#80b1d3'
        }
        
        # Plot each NACE sector as a line
        for sector in sorted(sectors):
            df_sector = df_time[df_time['nace_r2'] == sector].sort_values('TIME_PERIOD')
            if not df_sector.empty:
                color = colors_nace.get(sector, '#999999')
                ax.plot(df_sector['TIME_PERIOD'].astype(int), df_sector['value'], 
                       marker='o', linewidth=2.5, markersize=6, label=sector, 
                       color=color, alpha=0.8)
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'BERD ({unit_label})', fontsize=12, fontweight='bold')
        ax.set_title(f'France: Business Enterprise R&D Expenditure by NACE Sector\\n({unit_label}) - All Available Data', 
                    fontsize=13, fontweight='bold', pad=15)
        
        # Simplify legend labels
        handles, labels = ax.get_legend_handles_labels()
        short_labels = [l.replace('Electricity, gas, steam and air conditioning supply; water supply; sewerage, waste management and remediation activities', 
                                 'Electricity, gas, water & utilities')
                       .replace('Agriculture, forestry and fishing', 'Agriculture & forestry')
                       for l in labels]
        ax.legend(handles, short_labels, fontsize=10, loc='best', framealpha=0.9)
        
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('white')
        
        # Format x-axis to show every other year
        all_years = sorted(df_time['TIME_PERIOD'].unique())
        ax.set_xticks(all_years[::2])
        ax.set_xticklabels([int(y) for y in all_years[::2]])
        
        plt.tight_layout()
        file_suffix = 'pps' if 'PPS' in unit_val else 'pct_gdp'
        plt.savefig(os.path.join(OUTPUT_DIR, f'13_berd_france_timeseries_{file_suffix}.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] 13_berd_france_timeseries_{file_suffix}.png")

def create_under_occupied_dwellings_graphs():
    """Create visualizations for under-occupied dwellings by age"""
    print("\n[5] Creating under-occupied dwellings graphs...")
    
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_under-occupied dwellings.csv')
    if not os.path.exists(file_path):
        print("  File not found:", file_path)
        return
    
    df = pd.read_csv(file_path)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    eu27_label = 'European Union - 27 countries (from 2020)'
    
    # EU27 by age (latest year)
    latest_year = int(df['TIME_PERIOD'].max())
    df_eu27 = df[df['geo'] == eu27_label].copy()
    if not df_eu27.empty:
        age_order = ['Total', 'Less than 18 years', 'From 18 to 64 years', '65 years or over']
        df_eu27_age = df_eu27[df_eu27['age'].isin(age_order)].copy()
        df_eu27_age['age'] = pd.Categorical(df_eu27_age['age'], categories=age_order, ordered=True)
        df_eu27_age = df_eu27_age.sort_values('age')
        
        if not df_eu27_age.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = COMPONENT_COLORS[:len(df_eu27_age)]
            bars = ax.bar(range(len(df_eu27_age)), df_eu27_age['value'], color=colors, edgecolor='white', linewidth=1.5)
            
            ax.set_xticks(range(len(df_eu27_age)))
            ax.set_xticklabels(df_eu27_age['age'], fontsize=10, rotation=15, ha='right')
            ax.set_ylabel('Share (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'EU27: Share of People in Under-occupied Dwellings by Age ({latest_year})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            
            ax.grid(axis='y', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'14_under_occupied_eu27_age.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 14_under_occupied_eu27_age.png")
    
    # Countries by age groups - using 2024 for better coverage
    year_for_countries = 2024
    df_2024 = df[df['TIME_PERIOD'] == year_for_countries].copy()
    df_countries = df_2024[df_2024['geo'] != eu27_label].copy()
    df_countries['country_name'] = df_countries['geo'].apply(standardize_country_name)
    df_countries = df_countries[df_countries['country_name'].isin(TARGET_COUNTRIES)]
    
    # Plot for each age group except Total
    age_order = ['Less than 18 years', 'From 18 to 64 years', '65 years or over']
    colors_age = ['#ff7f0e', '#2ca02c', '#d62728']
    
    for age_idx, age_group in enumerate(age_order):
        df_age = df_countries[df_countries['age'] == age_group].copy()
        df_age = df_age.dropna(subset=['value'])
        
        if not df_age.empty:
            df_age = df_age.sort_values('value', ascending=True)
            
            fig, ax = plt.subplots(figsize=(11, 10))
            color = colors_age[age_idx]
            bars = ax.barh(range(len(df_age)), df_age['value'], color=color, edgecolor='white', linewidth=1.5)
            
            ax.set_yticks(range(len(df_age)))
            ax.set_yticklabels(df_age['country_name'], fontsize=9)
            ax.set_xlabel('Share (%)', fontsize=12, fontweight='bold')
            
            age_label = age_group.replace('Less than ', '<').replace('From ', '').replace(' years', '').replace('or over', '+')
            ax.set_title(f'Under-occupied Dwellings by Country - Age {age_label} ({year_for_countries})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            ax.grid(axis='x', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            viz_num = 15 + age_idx
            filename = age_group.replace(' ', '_').replace('-', '').lower()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{viz_num}_under_occupied_countries_{filename}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] {viz_num}_under_occupied_countries_{filename}.png")

def create_renting_difficulties_graphs():
    """Create visualizations for renting difficulties by urbanization and income"""
    print("\n[6] Creating renting difficulties graphs...")
    
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_renting difficulties.csv')
    if not os.path.exists(file_path):
        print("  File not found:", file_path)
        return
    
    df = pd.read_csv(file_path)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    eu27_label = 'European Union - 27 countries (from 2020)'
    
    # Latest year (should be 2023)
    latest_year = int(df['TIME_PERIOD'].max())
    df_data = df[df['TIME_PERIOD'] == latest_year].copy()
    
    # EU27 by degree of urbanization
    df_eu27 = df_data[df_data['geo'] == eu27_label].copy()
    if not df_eu27.empty:
        deg_urb_order = ['Total', 'Cities', 'Towns and suburbs', 'Rural areas']
        df_eu27_urb = df_eu27[df_eu27['deg_urb'].isin(deg_urb_order) & (df_eu27['quant_inc'] == 'Total')]
        df_eu27_urb['deg_urb'] = pd.Categorical(df_eu27_urb['deg_urb'], categories=deg_urb_order, ordered=True)
        df_eu27_urb = df_eu27_urb.sort_values('deg_urb')
        
        if not df_eu27_urb.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = COMPONENT_COLORS[:len(df_eu27_urb)]
            bars = ax.bar(range(len(df_eu27_urb)), df_eu27_urb['value'], color=colors, edgecolor='white', linewidth=1.5)
            
            ax.set_xticks(range(len(df_eu27_urb)))
            ax.set_xticklabels(df_eu27_urb['deg_urb'], fontsize=10, rotation=15, ha='right')
            ax.set_ylabel('Share (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'EU27: Renting Difficulties by Degree of Urbanization ({latest_year})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            
            ax.grid(axis='y', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'16_renting_difficulties_eu27_urbanization.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 16_renting_difficulties_eu27_urbanization.png")
        
        # EU27 by income quintile
        quant_order = ['Total', 'First quintile', 'Second quintile', 'Third quintile', 'Fourth quintile', 'Fifth quintile']
        df_eu27_inc = df_eu27[df_eu27['quant_inc'].isin(quant_order) & (df_eu27['deg_urb'] == 'Total')]
        df_eu27_inc['quant_inc'] = pd.Categorical(df_eu27_inc['quant_inc'], categories=quant_order, ordered=True)
        df_eu27_inc = df_eu27_inc.sort_values('quant_inc')
        
        if not df_eu27_inc.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = COMPONENT_COLORS[:len(df_eu27_inc)]
            bars = ax.bar(range(len(df_eu27_inc)), df_eu27_inc['value'], color=colors, edgecolor='white', linewidth=1.5)
            
            ax.set_xticks(range(len(df_eu27_inc)))
            ax.set_xticklabels([l.replace('quintile', 'Q') for l in df_eu27_inc['quant_inc']], fontsize=10, rotation=15, ha='right')
            ax.set_ylabel('Share (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'EU27: Renting Difficulties by Income Quintile ({latest_year})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            
            ax.grid(axis='y', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'17_renting_difficulties_eu27_income.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 17_renting_difficulties_eu27_income.png")
    
    # Countries by total renting difficulties
    df_countries = df_data[df_data['geo'] != eu27_label].copy()
    df_countries['country_name'] = df_countries['geo'].apply(standardize_country_name)
    df_countries = df_countries[df_countries['country_name'].isin(TARGET_COUNTRIES)]
    df_countries = df_countries[(df_countries['deg_urb'] == 'Total') & (df_countries['quant_inc'] == 'Total')]
    df_countries = df_countries.dropna(subset=['value'])
    
    if not df_countries.empty:
        df_countries = df_countries.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(11, 10))
        colors = [get_country_color(country) for country in df_countries['country_name']]
        bars = ax.barh(range(len(df_countries)), df_countries['value'], color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_countries)))
        ax.set_yticklabels(df_countries['country_name'], fontsize=9)
        ax.set_xlabel('Share (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Renting Difficulties by Country ({latest_year})', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'18_renting_difficulties_countries.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] 18_renting_difficulties_countries.png")
    
    # France by degree of urbanization
    df_france = df_data[df_data['geo'] == 'France'].copy()
    if not df_france.empty:
        deg_urb_order = ['Total', 'Cities', 'Towns and suburbs', 'Rural areas']
        df_france_urb = df_france[df_france['deg_urb'].isin(deg_urb_order) & (df_france['quant_inc'] == 'Total')]
        df_france_urb['deg_urb'] = pd.Categorical(df_france_urb['deg_urb'], categories=deg_urb_order, ordered=True)
        df_france_urb = df_france_urb.sort_values('deg_urb')
        
        if not df_france_urb.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = COMPONENT_COLORS[:len(df_france_urb)]
            bars = ax.bar(range(len(df_france_urb)), df_france_urb['value'], color=colors, edgecolor='white', linewidth=1.5)
            
            ax.set_xticks(range(len(df_france_urb)))
            ax.set_xticklabels(df_france_urb['deg_urb'], fontsize=10, rotation=15, ha='right')
            ax.set_ylabel('Share (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'France: Renting Difficulties by Degree of Urbanization ({latest_year})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            
            ax.grid(axis='y', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'16b_renting_difficulties_france_urbanization.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 16b_renting_difficulties_france_urbanization.png")
        
        # France by income quintile
        quant_order = ['Total', 'First quintile', 'Second quintile', 'Third quintile', 'Fourth quintile', 'Fifth quintile']
        df_france_inc = df_france[df_france['quant_inc'].isin(quant_order) & (df_france['deg_urb'] == 'Total')]
        df_france_inc['quant_inc'] = pd.Categorical(df_france_inc['quant_inc'], categories=quant_order, ordered=True)
        df_france_inc = df_france_inc.sort_values('quant_inc')
        
        if not df_france_inc.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = COMPONENT_COLORS[:len(df_france_inc)]
            bars = ax.bar(range(len(df_france_inc)), df_france_inc['value'], color=colors, edgecolor='white', linewidth=1.5)
            
            ax.set_xticks(range(len(df_france_inc)))
            ax.set_xticklabels([l.replace('quintile', 'Q') for l in df_france_inc['quant_inc']], fontsize=10, rotation=15, ha='right')
            ax.set_ylabel('Share (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'France: Renting Difficulties by Income Quintile ({latest_year})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            
            ax.grid(axis='y', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'17b_renting_difficulties_france_income.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 17b_renting_difficulties_france_income.png")

def create_building_permits_graphs():
    """Create visualizations for building permits over time"""
    print("\n[7] Creating building permits graphs...")
    
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_Building permits.csv')
    if not os.path.exists(file_path):
        print("  File not found:", file_path)
        return
    
    df = pd.read_csv(file_path)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    eu27_label = 'European Union - 27 countries (from 2020)'
    
    # EU27 time series
    df_eu27 = df[df['geo'] == eu27_label].copy()
    if not df_eu27.empty:
        df_eu27 = df_eu27.sort_values('TIME_PERIOD')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_eu27['TIME_PERIOD'].astype(int), df_eu27['value'], 
               marker='o', linewidth=2.5, markersize=6, color='#1f77b4', label='EU27')
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Building Permits (m²/1000 inhabitants)', fontsize=12, fontweight='bold')
        ax.set_title(f'EU27: Building Permits Trends (2005-2024)', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'19_building_permits_eu27_timeseries.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] 19_building_permits_eu27_timeseries.png")
    
    # Countries latest year
    latest_year = int(df['TIME_PERIOD'].max())
    df_latest = df[(df['TIME_PERIOD'] == latest_year) & (df['geo'] != eu27_label)].copy()
    df_latest['country_name'] = df_latest['geo'].apply(standardize_country_name)
    df_latest = df_latest[df_latest['country_name'].isin(TARGET_COUNTRIES)]
    df_latest = df_latest.dropna(subset=['value'])
    
    if not df_latest.empty:
        df_latest = df_latest.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(11, 10))
        colors = [get_country_color(country) for country in df_latest['country_name']]
        bars = ax.barh(range(len(df_latest)), df_latest['value'], color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_latest)))
        ax.set_yticklabels(df_latest['country_name'], fontsize=9)
        ax.set_xlabel('Building Permits (m²/1000 inhabitants)', fontsize=12, fontweight='bold')
        ax.set_title(f'Building Permits by Country ({latest_year})', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'20_building_permits_countries.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] 20_building_permits_countries.png")

def create_building_permits_demographic_graphs():
    """Create visualizations for building permits per demographic change"""
    print("\n[7b] Creating building permits graphs (per demographic change)...")
    
    # Load building permits data
    permits_file = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_Building permits.csv')
    if not os.path.exists(permits_file):
        print("  File not found:", permits_file)
        return
    
    df_permits = pd.read_csv(permits_file)
    df_permits['value'] = pd.to_numeric(df_permits['OBS_VALUE'], errors='coerce')
    df_permits = df_permits.dropna(subset=['value'])
    
    # Load demographic data (population)
    demo_file = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_demo.csv')
    if not os.path.exists(demo_file):
        print("  File not found:", demo_file)
        return
    
    df_demo = pd.read_csv(demo_file)
    df_demo['pop'] = pd.to_numeric(df_demo['OBS_VALUE'], errors='coerce')
    df_demo = df_demo.dropna(subset=['pop'])
    
    # Calculate population change per country and year
    # For year Y, demographic change = POP(Y+1) - POP(Y)
    df_demo = df_demo.sort_values(['geo', 'TIME_PERIOD'])
    df_demo['demo_change'] = -df_demo.groupby('geo')['pop'].diff(-1)  # Negative diff to go forward, then negate
    
    # Merge permits and demographic data
    df_merged = df_permits.merge(
        df_demo[['geo', 'TIME_PERIOD', 'demo_change']],
        on=['geo', 'TIME_PERIOD'],
        how='left'
    )
    
    # Apply 5-year rolling average to demographic change to smooth variations
    df_merged['demo_change_smoothed'] = df_merged.groupby('geo')['demo_change'].transform(
        lambda x: x.rolling(window=5, center=True).mean()
    )
    
    # Calculate m² per new inhabitant: (permits value * 1000) / demographic_change
    # Building permits data is in m²/1000 inhabitants, convert to m² then divide by new inhabitants
    # Only calculate where demo_change_smoothed > 100 (significant population growth, at least 100 people)
    df_merged['permits_per_new_inhabitant'] = np.where(
        (df_merged['demo_change_smoothed'] > 100) & (df_merged['demo_change_smoothed'].notna()),
        (df_merged['value'] * 1000) / df_merged['demo_change_smoothed'],
        np.nan
    )
    
    # Remove extreme outliers (keep only values within reasonable range)
    valid_values = df_merged['permits_per_new_inhabitant'].dropna()
    if len(valid_values) > 10:
        # Use 5th and 95th percentiles for more robust filtering
        q5 = valid_values.quantile(0.05)
        q95 = valid_values.quantile(0.95)
        
        df_merged['permits_per_new_inhabitant'] = np.where(
            (df_merged['permits_per_new_inhabitant'] >= q5) & 
            (df_merged['permits_per_new_inhabitant'] <= q95),
            df_merged['permits_per_new_inhabitant'],
            np.nan
        )
    
    eu27_label = 'European Union - 27 countries (from 2020)'
    
    # EU27 time series
    df_eu27 = df_merged[df_merged['geo'] == eu27_label].copy()
    if not df_eu27.empty:
        df_eu27 = df_eu27.dropna(subset=['permits_per_new_inhabitant']).sort_values('TIME_PERIOD')
        
        if len(df_eu27) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_eu27['TIME_PERIOD'].astype(int), df_eu27['permits_per_new_inhabitant'], 
                   marker='o', linewidth=2.5, markersize=6, color='#1f77b4', label='EU27')
            
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('Building Permits (m² per new inhabitant)', fontsize=12, fontweight='bold')
            ax.set_title(f'EU27: Building Permits per Demographic Change Trends', 
                        fontsize=13, fontweight='bold', pad=15)
            
            ax.grid(True, alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'19_building_permits_eu27_timeseries_demographic.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 19_building_permits_eu27_timeseries_demographic.png")
    
    # Countries latest year (excluding last year since demographic change is NaN for last year)
    # Get second-to-last year
    latest_years = sorted(df_merged['TIME_PERIOD'].dropna().unique())
    if len(latest_years) > 1:
        analysis_year = latest_years[-2]  # Use second-to-last year
    else:
        analysis_year = latest_years[-1] if len(latest_years) > 0 else None
    
    if analysis_year is not None:
        df_latest = df_merged[(df_merged['TIME_PERIOD'] == analysis_year) & (df_merged['geo'] != eu27_label)].copy()
        df_latest['country_name'] = df_latest['geo'].apply(standardize_country_name)
        df_latest = df_latest[df_latest['country_name'].isin(TARGET_COUNTRIES)]
        df_latest = df_latest.dropna(subset=['permits_per_new_inhabitant'])
    
    if not df_latest.empty:
        df_latest = df_latest.sort_values('permits_per_new_inhabitant', ascending=True)
        
        fig, ax = plt.subplots(figsize=(11, 10))
        colors = [get_country_color(country) for country in df_latest['country_name']]
        bars = ax.barh(range(len(df_latest)), df_latest['permits_per_new_inhabitant'], color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_latest)))
        ax.set_yticklabels(df_latest['country_name'], fontsize=9)
        ax.set_xlabel('Building Permits (m² per new inhabitant)', fontsize=12, fontweight='bold')
        ax.set_title(f'Building Permits per Demographic Change by Country ({int(analysis_year)})', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'20_building_permits_countries_demographic.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] 20_building_permits_countries_demographic.png")


def create_building_permits_specific_countries_demographic():
    """Create visualizations for building permits per demographic change for specific countries"""
    print("\n[7c] Creating building permits graphs (specific countries per demographic change)...")
    
    target_countries = ['France', 'Germany', 'Denmark']
    
    # Load building permits data
    permits_file = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_Building permits.csv')
    if not os.path.exists(permits_file):
        print("  File not found:", permits_file)
        return
    
    df_permits = pd.read_csv(permits_file)
    df_permits['value'] = pd.to_numeric(df_permits['OBS_VALUE'], errors='coerce')
    df_permits = df_permits.dropna(subset=['value'])
    
    # Load demographic data (population)
    demo_file = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_demo.csv')
    if not os.path.exists(demo_file):
        print("  File not found:", demo_file)
        return
    
    df_demo = pd.read_csv(demo_file)
    df_demo['pop'] = pd.to_numeric(df_demo['OBS_VALUE'], errors='coerce')
    df_demo = df_demo.dropna(subset=['pop'])
    
    # Calculate population change per country and year with anticipation (POP(t-1) - POP(t-2))
    df_demo = df_demo.sort_values(['geo', 'TIME_PERIOD'])
    df_demo['demo_change'] = df_demo.groupby('geo')['pop'].shift(1).diff()
    
    # Merge permits and demographic data
    df_merged = df_permits.merge(
        df_demo[['geo', 'TIME_PERIOD', 'demo_change']],
        on=['geo', 'TIME_PERIOD'],
        how='left'
    )
    
    # Apply 5-year rolling average to demographic change to smooth variations
    df_merged['demo_change_smoothed'] = df_merged.groupby('geo')['demo_change'].transform(
        lambda x: x.rolling(window=5, center=True).mean()
    )
    
    # Calculate m² per new inhabitant
    df_merged['permits_per_new_inhabitant'] = np.where(
        (df_merged['demo_change_smoothed'] > 100) & (df_merged['demo_change_smoothed'].notna()),
        (df_merged['value'] * 1000) / df_merged['demo_change_smoothed'],
        np.nan
    )
    
    # Get only target countries data
    df_countries = df_merged[df_merged['geo'].isin(target_countries)].copy()
    df_countries = df_countries.dropna(subset=['permits_per_new_inhabitant']).sort_values(['geo', 'TIME_PERIOD'])
    
    if df_countries.empty:
        print("  No data found for target countries")
        return
    
    # Create comparison chart only (no individual country graphs)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for country in target_countries:
        df_country = df_countries[df_countries['geo'] == country].copy()
        
        if not df_country.empty:
            # Remove outliers for cleaner comparison
            valid_values = df_country['permits_per_new_inhabitant'].dropna()
            if len(valid_values) > 2:
                q5 = valid_values.quantile(0.05)
                q95 = valid_values.quantile(0.95)
                df_country['permits_per_new_inhabitant'] = np.where(
                    (df_country['permits_per_new_inhabitant'] >= q5) & 
                    (df_country['permits_per_new_inhabitant'] <= q95),
                    df_country['permits_per_new_inhabitant'],
                    np.nan
                )
            
            df_country = df_country.dropna(subset=['permits_per_new_inhabitant'])
            
            if len(df_country) > 0:
                ax.plot(df_country['TIME_PERIOD'].astype(int), df_country['permits_per_new_inhabitant'], 
                       marker='o', linewidth=2.5, markersize=6, label=country, 
                       color=get_country_color(country))
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Building Permits (m² per new inhabitant)', fontsize=12, fontweight='bold')
    ax.set_title('Building Permits per Demographic Change: France, Germany, Denmark', 
                fontsize=13, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('white')
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'21_building_permits_comparison_demographic.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVED] 21_building_permits_comparison_demographic.png")


def create_building_permits_growth_scatter():
    """Create scatter plot: mean population growth vs mean total m² built (2017-2022)"""
    print("\n[7d] Creating building permits vs population growth scatter plot...")
    
    # Load building permits data (in m²/1000 inhabitants)
    permits_file = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_Building permits.csv')
    if not os.path.exists(permits_file):
        print("  File not found:", permits_file)
        return
    
    df_permits = pd.read_csv(permits_file)
    df_permits['value'] = pd.to_numeric(df_permits['OBS_VALUE'], errors='coerce')
    df_permits = df_permits.dropna(subset=['value'])
    df_permits['TIME_PERIOD'] = pd.to_numeric(df_permits['TIME_PERIOD'], errors='coerce')
    
    # Load demographic data (population)
    demo_file = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_demo.csv')
    if not os.path.exists(demo_file):
        print("  File not found:", demo_file)
        return
    
    df_demo = pd.read_csv(demo_file)
    df_demo['pop'] = pd.to_numeric(df_demo['OBS_VALUE'], errors='coerce')
    df_demo = df_demo.dropna(subset=['pop'])
    df_demo['TIME_PERIOD'] = pd.to_numeric(df_demo['TIME_PERIOD'], errors='coerce')
    
    # Filter to 2017-2022
    year_min = 2017
    year_max = 2022
    
    # Population growth calculation
    df_demo_filtered = df_demo[(df_demo['TIME_PERIOD'] >= year_min) & (df_demo['TIME_PERIOD'] <= year_max)].copy()
    df_demo_filtered = df_demo_filtered.sort_values(['geo', 'TIME_PERIOD'])
    
    # Calculate annual growth rates
    df_demo_filtered['pop_growth'] = df_demo_filtered.groupby('geo')['pop'].pct_change() * 100
    
    # Get mean population growth per country
    mean_growth = df_demo_filtered.groupby('geo')['pop_growth'].mean().reset_index()
    mean_growth.columns = ['geo', 'mean_pop_growth']
    
    # Building permits: get mean m²/1000 inhabitants (original metric)
    df_permits_filtered = df_permits[(df_permits['TIME_PERIOD'] >= year_min) & (df_permits['TIME_PERIOD'] <= year_max)].copy()
    
    # Get mean permits value per country (mean over 5 years, already in m²/1000 inhabitants)
    mean_permits = df_permits_filtered.groupby('geo')['value'].mean().reset_index()
    mean_permits.columns = ['geo', 'mean_permits_per_1000_inh']
    
    # Merge the two datasets
    df_scatter = mean_growth.merge(mean_permits, on='geo', how='inner')
    
    # Standardize country names
    df_scatter['country_name'] = df_scatter['geo'].apply(standardize_country_name)
    df_scatter = df_scatter[df_scatter['country_name'].notna()]
    
    # Filter to countries we have colors for
    df_scatter = df_scatter[df_scatter['country_name'].isin(TARGET_COUNTRIES)]
    
    if df_scatter.empty:
        print("  No data found for scatter plot")
        return
    
    # Create scatter plot (30% smaller: from 14x8 to ~10x5.6)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate France from other countries
    df_france = df_scatter[df_scatter['country_name'] == 'France']
    df_other = df_scatter[df_scatter['country_name'] != 'France']
    
    # Plot other countries first
    for idx, row in df_other.iterrows():
        ax.scatter(row['mean_pop_growth'], row['mean_permits_per_1000_inh'], 
                  s=200, alpha=0.7, color=get_country_color(row['country_name']),
                  edgecolors='black', linewidth=1.5, label=row['country_name'])
        
        # Add country labels
        ax.annotate(row['country_name'], 
                   (row['mean_pop_growth'], row['mean_permits_per_1000_inh']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # Plot France last (on top) with yellow color
    for idx, row in df_france.iterrows():
        ax.scatter(row['mean_pop_growth'], row['mean_permits_per_1000_inh'], 
                  s=200, alpha=0.9, color='#ffd558',
                  edgecolors='black', linewidth=1.5, label=row['country_name'], zorder=10)
        
        # Add country label
        ax.annotate(row['country_name'], 
                   (row['mean_pop_growth'], row['mean_permits_per_1000_inh']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold', zorder=11)
    
    ax.set_xlabel('Mean Annual Population Growth (%) [2017-2022]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Building Permits (m²/1000 inhabitants) [2017-2022]', fontsize=12, fontweight='bold')
    ax.set_title('Building Permits vs Population Growth: Mean Annual Values (2017-2022)', 
                fontsize=13, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'22_building_permits_vs_growth_scatter.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVED] 22_building_permits_vs_growth_scatter.png")


def _plot_dwellings_vs_price_scatter(countries_data, filename, title, include_regression=True, show_origin=False, x_from_zero=False, single_country=None, fixed_xlim=None, fixed_ylim=None, x_label='Dwelling Stock variation (%)'):
    """Helper function to plot dwellings vs price-to-income scatter plot
    
    show_origin: if True, center the plot around (0,0) with equal scaling
    x_from_zero: if True, start x-axis at 0 but keep y-axis data-driven
    fixed_xlim: tuple (xmin, xmax) to freeze x-axis
    fixed_ylim: tuple (ymin, ymax) to freeze y-axis
    x_label: label for x-axis (default: 'Dwelling Stock variation (%)')
    """
    # Filter to single country if specified
    if single_country:
        countries_data = {single_country: countries_data[single_country]} if single_country in countries_data else {}
    
    if not countries_data:
        print(f"  No data found for plotting")
        return
    
    # Collect all data points
    all_x = []
    all_y = []
    for country, periods in countries_data.items():
        if '0111' in periods:
            all_x.append(periods['0111']['x'])
            all_y.append(periods['0111']['y'])
        if '1121' in periods:
            if '0111' in periods:
                x1 = periods['0111']['x'] + periods['1121']['x']
                y1 = periods['0111']['y'] + periods['1121']['y']
            else:
                x1 = periods['1121']['x']
                y1 = periods['1121']['y']
            all_x.append(x1)
            all_y.append(y1)
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # Calculate regression if needed
    regression_text = None
    if include_regression and len(all_x) > 0:
        slope = np.sum(all_x * all_y) / np.sum(all_x ** 2)
        y_pred = slope * all_x
        ss_res = np.sum((all_y - y_pred) ** 2)
        ss_tot = np.sum((all_y - np.mean(all_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        regression_text = f'Regression: y={slope:.3f}x (R²={r_squared:.3f})'
    
    # Create scatter plot with frozen axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set axis limits based on configuration
    if fixed_xlim and fixed_ylim:
        # Use provided frozen limits
        ax.set_xlim(fixed_xlim)
        ax.set_ylim(fixed_ylim)
    elif show_origin:
        # Ensure origin is visible and axis is square/frozen
        max_abs_x = max(abs(all_x)) if len(all_x) > 0 else 1
        max_abs_y = max(abs(all_y)) if len(all_y) > 0 else 1
        max_val = max(max_abs_x, max_abs_y) * 1.15  # 15% padding
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
    elif x_from_zero:
        # X-axis starts at 0, Y-axis data-driven
        padding = 0.1
        x_max_data = max(all_x)
        y_min_data, y_max_data = min(all_y), max(all_y)
        x_range = x_max_data  # From 0 to max
        y_range = y_max_data - y_min_data
        x_max = x_max_data + padding * x_range
        y_min = y_min_data - padding * y_range
        y_max = y_max_data + padding * y_range
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        # Original padding-based approach
        padding = 0.1
        x_min_data, x_max_data = min(all_x), max(all_x)
        y_min_data, y_max_data = min(all_y), max(all_y)
        x_range = x_max_data - x_min_data
        y_range = y_max_data - y_min_data
        x_min = x_min_data - padding * x_range
        x_max = x_max_data + padding * x_range
        y_min = y_min_data - padding * y_range
        y_max = y_max_data + padding * y_range
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    # Draw regression line if needed
    if include_regression and regression_text:
        x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
        y_line = slope * x_line
        ax.plot(x_line, y_line, color='grey', linestyle='--', linewidth=2, alpha=0.6, zorder=1)
    
    # Plot all circles and squares with arrows
    for country, periods in countries_data.items():
        color = get_country_color(country)
        
        # Plot 2001-2011 circle
        if '0111' in periods:
            x0 = periods['0111']['x']
            y0 = periods['0111']['y']
            ax.scatter(x0, y0, s=150, alpha=0.7, color=color, marker='o', zorder=4)
        
        # Plot 2011-2021 square and draw arrow if both periods exist
        if '1121' in periods and '0111' in periods:
            x1 = periods['0111']['x'] + periods['1121']['x']
            y1 = periods['0111']['y'] + periods['1121']['y']
            ax.scatter(x1, y1, s=150, alpha=0.9, color=color, marker='s', zorder=5)
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color=color, alpha=0.7))
            ax.annotate(country, (x1, y1), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, fontweight='bold')
        elif '1121' in periods and '0111' not in periods:
            x1 = periods['1121']['x']
            y1 = periods['1121']['y']
            ax.scatter(x1, y1, s=150, alpha=0.9, color=color, marker='s', zorder=5)
            ax.annotate(country, (x1, y1), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, fontweight='bold')
        elif '0111' in periods:
            ax.annotate(country, (x0, y0), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666', markersize=10, label='2001-2011 (circles)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#666666', markersize=10, label='2011-2021 (squares)')
    ]
    if include_regression and regression_text:
        legend_elements.append(Line2D([0], [0], color='grey', linestyle='--', linewidth=2, label=regression_text))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Price-to-Income variation (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVED] {filename}")


def _plot_dwellings_vs_price_dual_regression(countries_data, filename, title, x_label='Dwelling Stock variation (%)'):
    """Helper function to plot dwellings vs price-to-income with two separate population-weighted regressions (one per period)
    
    Regressions are standard linear (y = ax + b), not constrained through origin, weighted by population
    """
    if not countries_data:
        print(f"  No data found for plotting")
        return
    
    # Load population data
    pop_2001 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2001.csv'))
    pop_2011 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2011.csv'))
    pop_2021 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2021.csv'))
    
    # Create population dictionaries
    populations = {}
    for _, row in pop_2001.iterrows():
        populations[(row['geo'], 2001)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
    for _, row in pop_2011.iterrows():
        populations[(row['geo'], 2011)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
    for _, row in pop_2021.iterrows():
        populations[(row['geo'], 2021)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
    
    # Collect data points separately for each period with country info
    period_0111_data = []  # List of (country, x, y)
    period_1121_data = []  # List of (country, x, y)
    
    all_x = []
    all_y = []
    
    for country, periods in countries_data.items():
        if '0111' in periods:
            x = periods['0111']['x']
            y = periods['0111']['y']
            period_0111_data.append((country, x, y))
            all_x.append(x)
            all_y.append(y)
        
        if '1121' in periods:
            if '0111' in periods:
                # For cumulative position
                x1 = periods['0111']['x'] + periods['1121']['x']
                y1 = periods['0111']['y'] + periods['1121']['y']
            else:
                x1 = periods['1121']['x']
                y1 = periods['1121']['y']
            period_1121_data.append((country, x1, y1))
            all_x.append(x1)
            all_y.append(y1)
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # Calculate population-weighted linear regressions (y = ax + b) for each period
    regression_0111_text = None
    regression_1121_text = None
    slope_0111, intercept_0111 = None, None
    slope_1121, intercept_1121 = None, None
    
    if len(period_0111_data) > 1:
        # Extract data and calculate weights (average population 2001-2011)
        period_0111_x = []
        period_0111_y = []
        weights_0111 = []
        
        for country, x, y in period_0111_data:
            pop_2001_val = populations.get((country, 2001))
            pop_2011_val = populations.get((country, 2011))
            
            if pop_2001_val and pop_2011_val and not pd.isna(pop_2001_val) and not pd.isna(pop_2011_val):
                avg_pop = (pop_2001_val + pop_2011_val) / 2
                period_0111_x.append(x)
                period_0111_y.append(y)
                weights_0111.append(avg_pop)
        
        period_0111_x = np.array(period_0111_x)
        period_0111_y = np.array(period_0111_y)
        weights_0111 = np.array(weights_0111)
        
        if len(period_0111_x) > 1 and np.sum(weights_0111) > 0:
            # Weighted linear regression
            w_sum = np.sum(weights_0111)
            x_mean_w = np.sum(weights_0111 * period_0111_x) / w_sum
            y_mean_w = np.sum(weights_0111 * period_0111_y) / w_sum
            
            numerator = np.sum(weights_0111 * (period_0111_x - x_mean_w) * (period_0111_y - y_mean_w))
            denominator = np.sum(weights_0111 * (period_0111_x - x_mean_w) ** 2)
            
            if denominator > 0:
                slope_0111 = numerator / denominator
                intercept_0111 = y_mean_w - slope_0111 * x_mean_w
                
                # Calculate weighted R²
                y_pred_0111 = slope_0111 * period_0111_x + intercept_0111
                ss_res_w = np.sum(weights_0111 * (period_0111_y - y_pred_0111) ** 2)
                ss_tot_w = np.sum(weights_0111 * (period_0111_y - y_mean_w) ** 2)
                r_squared_0111 = 1 - (ss_res_w / ss_tot_w) if ss_tot_w > 0 else 0
                
                regression_0111_text = f'2001-2011: y={slope_0111:.3f}x+{intercept_0111:.2f} (R²={r_squared_0111:.3f}) [pop-weighted]'
    
    if len(period_1121_data) > 1:
        # Extract data and calculate weights (average population 2011-2021)
        period_1121_x = []
        period_1121_y = []
        weights_1121 = []
        
        for country, x, y in period_1121_data:
            pop_2011_val = populations.get((country, 2011))
            pop_2021_val = populations.get((country, 2021))
            
            if pop_2011_val and pop_2021_val and not pd.isna(pop_2011_val) and not pd.isna(pop_2021_val):
                avg_pop = (pop_2011_val + pop_2021_val) / 2
                period_1121_x.append(x)
                period_1121_y.append(y)
                weights_1121.append(avg_pop)
        
        period_1121_x = np.array(period_1121_x)
        period_1121_y = np.array(period_1121_y)
        weights_1121 = np.array(weights_1121)
        
        if len(period_1121_x) > 1 and np.sum(weights_1121) > 0:
            # Weighted linear regression
            w_sum = np.sum(weights_1121)
            x_mean_w = np.sum(weights_1121 * period_1121_x) / w_sum
            y_mean_w = np.sum(weights_1121 * period_1121_y) / w_sum
            
            numerator = np.sum(weights_1121 * (period_1121_x - x_mean_w) * (period_1121_y - y_mean_w))
            denominator = np.sum(weights_1121 * (period_1121_x - x_mean_w) ** 2)
            
            if denominator > 0:
                slope_1121 = numerator / denominator
                intercept_1121 = y_mean_w - slope_1121 * x_mean_w
                
                # Calculate weighted R²
                y_pred_1121 = slope_1121 * period_1121_x + intercept_1121
                ss_res_w = np.sum(weights_1121 * (period_1121_y - y_pred_1121) ** 2)
                ss_tot_w = np.sum(weights_1121 * (period_1121_y - y_mean_w) ** 2)
                r_squared_1121 = 1 - (ss_res_w / ss_tot_w) if ss_tot_w > 0 else 0
                
                regression_1121_text = f'2011-2021: y={slope_1121:.3f}x+{intercept_1121:.2f} (R²={r_squared_1121:.3f}) [pop-weighted]'
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate axis limits with padding
    padding = 0.1
    x_min_data, x_max_data = min(all_x), max(all_x)
    y_min_data, y_max_data = min(all_y), max(all_y)
    x_range = x_max_data - x_min_data
    y_range = y_max_data - y_min_data
    x_min = x_min_data - padding * x_range
    x_max = x_max_data + padding * x_range
    y_min = y_min_data - padding * y_range
    y_max = y_max_data + padding * y_range
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Draw regression lines
    if slope_0111 is not None and intercept_0111 is not None:
        x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
        y_line = slope_0111 * x_line + intercept_0111
        ax.plot(x_line, y_line, color='#e74c3c', linestyle='--', linewidth=2.5, alpha=0.7, 
                label=regression_0111_text, zorder=2)
    
    if slope_1121 is not None and intercept_1121 is not None:
        x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
        y_line = slope_1121 * x_line + intercept_1121
        ax.plot(x_line, y_line, color='#3498db', linestyle='--', linewidth=2.5, alpha=0.7, 
                label=regression_1121_text, zorder=2)
    
    # Plot all circles and squares with arrows
    for country, periods in countries_data.items():
        color = get_country_color(country)
        
        # Plot 2001-2011 circle
        if '0111' in periods:
            x0 = periods['0111']['x']
            y0 = periods['0111']['y']
            ax.scatter(x0, y0, s=150, alpha=0.7, color=color, marker='o', zorder=4)
        
        # Plot 2011-2021 square and draw arrow if both periods exist
        if '1121' in periods and '0111' in periods:
            x1 = periods['0111']['x'] + periods['1121']['x']
            y1 = periods['0111']['y'] + periods['1121']['y']
            ax.scatter(x1, y1, s=150, alpha=0.9, color=color, marker='s', zorder=5)
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color=color, alpha=0.7))
            ax.annotate(country, (x1, y1), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, fontweight='bold')
        elif '1121' in periods and '0111' not in periods:
            x1 = periods['1121']['x']
            y1 = periods['1121']['y']
            ax.scatter(x1, y1, s=150, alpha=0.9, color=color, marker='s', zorder=5)
            ax.annotate(country, (x1, y1), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, fontweight='bold')
        elif '0111' in periods:
            ax.annotate(country, (x0, y0), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666', markersize=10, label='2001-2011 (circles)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#666666', markersize=10, label='2011-2021 (squares)')
    ]
    
    # Add regression lines to legend
    if regression_0111_text:
        legend_elements.append(Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=2.5, label=regression_0111_text))
    if regression_1121_text:
        legend_elements.append(Line2D([0], [0], color='#3498db', linestyle='--', linewidth=2.5, label=regression_1121_text))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95)
    
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Price-to-Income variation (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVED] {filename}")


def create_dwellings_vs_price_to_income():
    """Create scatter plot with arrows: dwelling variation vs price-to-income ratio change"""
    print("\n[7e] Creating dwellings vs price-to-income scatter plots...")
    
    # ISO3 to country name mapping for price-to-income data
    ISO3_TO_COUNTRY = {
        'AUS': 'Austria', 'BEL': 'Belgium', 'BGR': 'Bulgaria', 'HRV': 'Croatia',
        'CYP': 'Cyprus', 'CZE': 'Czech Republic', 'DNK': 'Denmark', 'EST': 'Estonia',
        'FIN': 'Finland', 'FRA': 'France', 'DEU': 'Germany', 'GRC': 'Greece',
        'HUN': 'Hungary', 'ISL': 'Iceland', 'IRL': 'Ireland', 'ITA': 'Italy',
        'LVA': 'Latvia', 'LTU': 'Lithuania', 'LUX': 'Luxembourg', 'MLT': 'Malta',
        'NLD': 'Netherlands', 'NOR': 'Norway', 'POL': 'Poland', 'PRT': 'Portugal',
        'ROU': 'Romania', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'ESP': 'Spain',
        'SWE': 'Sweden', 'CHE': 'Switzerland', 'LIE': 'Liechtenstein'
    }
    
    # Load dwelling data
    dwellings_2001 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2001.csv'))
    dwellings_2011 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2011.csv'))
    dwellings_2021 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2021.csv'))
    
    # Load price-to-income data
    df_price = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'ocde_house price-to-income ratio.csv'))
    df_price['year'] = df_price['TIME_PERIOD'].str.extract(r'(\d{4})')[0].astype(int)
    df_price['country'] = df_price['REF_AREA'].map(ISO3_TO_COUNTRY)
    df_price['ratio'] = pd.to_numeric(df_price['OBS_VALUE'], errors='coerce')
    df_price = df_price[df_price['year'].isin([2001, 2011, 2021])].dropna(subset=['country', 'ratio'])
    df_price_yearly = df_price.groupby(['year', 'country'])['ratio'].mean().reset_index()
    
    # Create dictionaries for dwelling and price data by country and year
    dwellings = {}
    for _, row in pd.concat([dwellings_2001.assign(year=2001), 
                            dwellings_2011.assign(year=2011),
                            dwellings_2021.assign(year=2021)]).iterrows():
        key = (row['geo'], int(row['year']))
        dwellings[key] = row['OBS_VALUE']
    
    prices = {}
    for _, row in df_price_yearly.iterrows():
        key = (row['country'], int(row['year']))
        prices[key] = row['ratio']
    
    # Build data structure for plotting
    countries_data = {}
    
    for _, row in dwellings_2001.iterrows():
        country = row['geo']
        d2001 = dwellings.get((country, 2001))
        d2011 = dwellings.get((country, 2011))
        d2021 = dwellings.get((country, 2021))
        p2001 = prices.get((country, 2001))
        p2011 = prices.get((country, 2011))
        p2021 = prices.get((country, 2021))
        
        if d2001 and d2011 and p2001 and p2011:
            dwell_change_0111 = (d2011 - d2001) / d2001 * 100
            price_change_0111 = (p2011 - p2001) / p2001 * 100
            countries_data[country] = {
                '0111': {'x': dwell_change_0111, 'y': price_change_0111}
            }
        
        if d2011 and d2021 and p2011 and p2021:
            dwell_change_1121 = (d2021 - d2011) / d2011 * 100
            price_change_1121 = (p2021 - p2011) / p2011 * 100
            if country not in countries_data:
                countries_data[country] = {}
            countries_data[country]['1121'] = {'x': dwell_change_1121, 'y': price_change_1121}
    
    if not countries_data:
        print("  No data found for comparison")
        return
    
    # Calculate axis limits from all countries data (x_from_zero mode)
    all_x_all = []
    all_y_all = []
    for country, periods in countries_data.items():
        if '0111' in periods:
            all_x_all.append(periods['0111']['x'])
            all_y_all.append(periods['0111']['y'])
        if '1121' in periods:
            if '0111' in periods:
                x1 = periods['0111']['x'] + periods['1121']['x']
                y1 = periods['0111']['y'] + periods['1121']['y']
            else:
                x1 = periods['1121']['x']
                y1 = periods['1121']['y']
            all_x_all.append(x1)
            all_y_all.append(y1)
    
    # Calculate frozen axis limits
    padding = 0.1
    x_max_data = max(all_x_all)
    y_min_data, y_max_data = min(all_y_all), max(all_y_all)
    x_range = x_max_data  # From 0 to max
    y_range = y_max_data - y_min_data
    x_max = x_max_data + padding * x_range
    y_min = y_min_data - padding * y_range
    y_max = y_max_data + padding * y_range
    
    frozen_xlim = (0, x_max)
    frozen_ylim = (y_min, y_max)
    
    # Version 1: With regression, x-axis from 0, y-axis data-driven (using frozen limits)
    _plot_dwellings_vs_price_scatter(
        countries_data,
        '23_dwellings_vs_price_to_income_arrows.png',
        'Price-to-Income vs Dwelling Stock:\n2001-2011 (circles) → 2011-2021 (squares)',
        include_regression=True,
        x_from_zero=True,
        fixed_xlim=frozen_xlim,
        fixed_ylim=frozen_ylim
    )
    
    # Version 2: Austria only, no regression, using same frozen axis
    _plot_dwellings_vs_price_scatter(
        countries_data,
        '23_dwellings_vs_price_to_income_austria_only.png',
        'Austria: Price-to-Income vs Dwelling Stock\n2001-2011 (circle) → 2011-2021 (square)',
        include_regression=False,
        single_country='Austria',
        fixed_xlim=frozen_xlim,
        fixed_ylim=frozen_ylim
    )
    
    # Version 3: All countries, regression, data-driven axes (original)
    _plot_dwellings_vs_price_scatter(
        countries_data,
        '23_dwellings_vs_price_to_income_arrows_datadriven.png',
        'Price-to-Income vs Dwelling Stock (Data-Driven Axis):\n2001-2011 (circles) → 2011-2021 (squares)',
        include_regression=True,
        show_origin=False
    )


def create_dwellings_per_household_vs_price_to_income():
    """Create scatter plot with arrows: dwelling per household variation vs price-to-income ratio change"""
    print("\n[7f] Creating dwellings per household vs price-to-income scatter plots...")
    
    # ISO3 to country name mapping for price-to-income data
    ISO3_TO_COUNTRY = {
        'AUS': 'Austria', 'BEL': 'Belgium', 'BGR': 'Bulgaria', 'HRV': 'Croatia',
        'CYP': 'Cyprus', 'CZE': 'Czech Republic', 'DNK': 'Denmark', 'EST': 'Estonia',
        'FIN': 'Finland', 'FRA': 'France', 'DEU': 'Germany', 'GRC': 'Greece',
        'HUN': 'Hungary', 'ISL': 'Iceland', 'IRL': 'Ireland', 'ITA': 'Italy',
        'LVA': 'Latvia', 'LTU': 'Lithuania', 'LUX': 'Luxembourg', 'MLT': 'Malta',
        'NLD': 'Netherlands', 'NOR': 'Norway', 'POL': 'Poland', 'PRT': 'Portugal',
        'ROU': 'Romania', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'ESP': 'Spain',
        'SWE': 'Sweden', 'CHE': 'Switzerland', 'LIE': 'Liechtenstein'
    }
    
    # Load dwelling data
    dwellings_2001 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2001.csv'))
    dwellings_2011 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2011.csv'))
    dwellings_2021 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2021.csv'))
    
    # Load household data
    households_2001 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_households_2001.csv'))
    households_2011 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_households_2011.csv'))
    households_2021 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_households_2021.csv'))
    
    # Load price-to-income data
    df_price = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'ocde_house price-to-income ratio.csv'))
    df_price['year'] = df_price['TIME_PERIOD'].str.extract(r'(\d{4})')[0].astype(int)
    df_price['country'] = df_price['REF_AREA'].map(ISO3_TO_COUNTRY)
    df_price['ratio'] = pd.to_numeric(df_price['OBS_VALUE'], errors='coerce')
    df_price = df_price[df_price['year'].isin([2001, 2011, 2021])].dropna(subset=['country', 'ratio'])
    df_price_yearly = df_price.groupby(['year', 'country'])['ratio'].mean().reset_index()
    
    # Create dictionaries for dwelling, household, and price data by country and year
    dwellings = {}
    for _, row in pd.concat([dwellings_2001.assign(year=2001), 
                            dwellings_2011.assign(year=2011),
                            dwellings_2021.assign(year=2021)]).iterrows():
        key = (row['geo'], int(row['year']))
        dwellings[key] = row['OBS_VALUE']
    
    households = {}
    for _, row in pd.concat([households_2001.assign(year=2001), 
                            households_2011.assign(year=2011),
                            households_2021.assign(year=2021)]).iterrows():
        key = (row['geo'], int(row['year']))
        households[key] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
    
    prices = {}
    for _, row in df_price_yearly.iterrows():
        key = (row['country'], int(row['year']))
        prices[key] = row['ratio']
    
    # Build data structure for plotting
    countries_data = {}
    
    for _, row in dwellings_2001.iterrows():
        country = row['geo']
        
        # Get dwellings and households for all years
        d2001 = dwellings.get((country, 2001))
        d2011 = dwellings.get((country, 2011))
        d2021 = dwellings.get((country, 2021))
        h2001 = households.get((country, 2001))
        h2011 = households.get((country, 2011))
        h2021 = households.get((country, 2021))
        p2001 = prices.get((country, 2001))
        p2011 = prices.get((country, 2011))
        p2021 = prices.get((country, 2021))
        
        # Calculate dwellings per household for each year
        if d2001 and h2001 and not pd.isna(h2001) and h2001 > 0:
            dph2001 = d2001 / h2001
        else:
            dph2001 = None
            
        if d2011 and h2011 and not pd.isna(h2011) and h2011 > 0:
            dph2011 = d2011 / h2011
        else:
            dph2011 = None
            
        if d2021 and h2021 and not pd.isna(h2021) and h2021 > 0:
            dph2021 = d2021 / h2021
        else:
            dph2021 = None
        
        # Calculate variations for 2001-2011 period
        if dph2001 and dph2011 and p2001 and p2011:
            dph_change_0111 = (dph2011 - dph2001) / dph2001 * 100
            price_change_0111 = (p2011 - p2001) / p2001 * 100
            countries_data[country] = {
                '0111': {'x': dph_change_0111, 'y': price_change_0111}
            }
        
        # Calculate variations for 2011-2021 period
        if dph2011 and dph2021 and p2011 and p2021:
            dph_change_1121 = (dph2021 - dph2011) / dph2011 * 100
            price_change_1121 = (p2021 - p2011) / p2011 * 100
            if country not in countries_data:
                countries_data[country] = {}
            countries_data[country]['1121'] = {'x': dph_change_1121, 'y': price_change_1121}
    
    if not countries_data:
        print("  No data found for comparison")
        return
    
    # Calculate axis limits from all countries data (x_from_zero mode)
    all_x_all = []
    all_y_all = []
    for country, periods in countries_data.items():
        if '0111' in periods:
            all_x_all.append(periods['0111']['x'])
            all_y_all.append(periods['0111']['y'])
        if '1121' in periods:
            if '0111' in periods:
                x1 = periods['0111']['x'] + periods['1121']['x']
                y1 = periods['0111']['y'] + periods['1121']['y']
            else:
                x1 = periods['1121']['x']
                y1 = periods['1121']['y']
            all_x_all.append(x1)
            all_y_all.append(y1)
    
    # Calculate frozen axis limits
    padding = 0.1
    x_max_data = max(all_x_all)
    y_min_data, y_max_data = min(all_y_all), max(all_y_all)
    x_range = x_max_data  # From 0 to max
    y_range = y_max_data - y_min_data
    x_max = x_max_data + padding * x_range
    y_min = y_min_data - padding * y_range
    y_max = y_max_data + padding * y_range
    
    frozen_xlim = (0, x_max)
    frozen_ylim = (y_min, y_max)
    
    # Version 1: With regression, x-axis from 0, y-axis data-driven (using frozen limits)
    _plot_dwellings_vs_price_scatter(
        countries_data,
        '23b_dwellings_per_household_vs_price_to_income_arrows.png',
        'Price-to-Income vs Dwelling per Household:\n2001-2011 (circles) → 2011-2021 (squares)',
        include_regression=True,
        x_from_zero=True,
        fixed_xlim=frozen_xlim,
        fixed_ylim=frozen_ylim,
        x_label='Dwelling per Household variation (%)'
    )
    
    # Version 2: Austria only, no regression, using same frozen axis
    _plot_dwellings_vs_price_scatter(
        countries_data,
        '23b_dwellings_per_household_vs_price_to_income_austria_only.png',
        'Austria: Price-to-Income vs Dwelling per Household\n2001-2011 (circle) → 2011-2021 (square)',
        include_regression=False,
        single_country='Austria',
        fixed_xlim=frozen_xlim,
        fixed_ylim=frozen_ylim,
        x_label='Dwelling per Household variation (%)'
    )
    
    # Version 3: All countries, regression, data-driven axes (original)
    _plot_dwellings_vs_price_scatter(
        countries_data,
        '23b_dwellings_per_household_vs_price_to_income_arrows_datadriven.png',
        'Price-to-Income vs Dwelling per Household (Data-Driven Axis):\n2001-2011 (circles) → 2011-2021 (squares)',
        include_regression=True,
        show_origin=False,
        x_label='Dwelling per Household variation (%)'
    )
    
    # Version 4: Dual regressions (one for each period), not constrained through origin
    _plot_dwellings_vs_price_dual_regression(
        countries_data,
        '23b_dwellings_per_household_vs_price_dual_regression.png',
        'Price-to-Income vs Dwelling per Household (Dual Regression):\n2001-2011 (circles) → 2011-2021 (squares)',
        x_label='Dwelling per Household variation (%)'
    )
    
    # Now create variant with % unoccupied dwellings variation
    print("\n[7f-variant] Creating unoccupied dwellings variation vs price-to-income scatter plot...")
    
    countries_data_unoccupied = {}
    
    for _, row in dwellings_2001.iterrows():
        country = row['geo']
        
        # Get dwellings and households for all years
        d2001 = dwellings.get((country, 2001))
        d2011 = dwellings.get((country, 2011))
        d2021 = dwellings.get((country, 2021))
        h2001 = households.get((country, 2001))
        h2011 = households.get((country, 2011))
        h2021 = households.get((country, 2021))
        p2001 = prices.get((country, 2001))
        p2011 = prices.get((country, 2011))
        p2021 = prices.get((country, 2021))
        
        # Calculate % unoccupied dwellings for each year: (dwellings - households) / dwellings * 100
        if d2001 and h2001 and not pd.isna(h2001) and d2001 > 0:
            unocc2001 = ((d2001 - h2001) / d2001) * 100
        else:
            unocc2001 = None
            
        if d2011 and h2011 and not pd.isna(h2011) and d2011 > 0:
            unocc2011 = ((d2011 - h2011) / d2011) * 100
        else:
            unocc2011 = None
            
        if d2021 and h2021 and not pd.isna(h2021) and d2021 > 0:
            unocc2021 = ((d2021 - h2021) / d2021) * 100
        else:
            unocc2021 = None
        
        # Calculate variation in percentage points for 2001-2011 period
        if unocc2001 is not None and unocc2011 is not None and p2001 and p2011:
            unocc_change_0111 = unocc2011 - unocc2001  # Percentage point change
            price_change_0111 = (p2011 - p2001) / p2001 * 100
            countries_data_unoccupied[country] = {
                '0111': {'x': unocc_change_0111, 'y': price_change_0111}
            }
        
        # Calculate variation in percentage points for 2011-2021 period
        if unocc2011 is not None and unocc2021 is not None and p2011 and p2021:
            unocc_change_1121 = unocc2021 - unocc2011  # Percentage point change
            price_change_1121 = (p2021 - p2011) / p2011 * 100
            if country not in countries_data_unoccupied:
                countries_data_unoccupied[country] = {}
            countries_data_unoccupied[country]['1121'] = {'x': unocc_change_1121, 'y': price_change_1121}
    
    if countries_data_unoccupied:
        _plot_dwellings_vs_price_scatter(
            countries_data_unoccupied,
            '23c_unoccupied_dwellings_variation_vs_price_to_income_arrows_datadriven.png',
            'Price-to-Income vs Unoccupied Dwellings Share (Data-Driven Axis):\n2001-2011 (circles) → 2011-2021 (squares)',
            include_regression=True,
            show_origin=False,
            x_label='Unoccupied Dwellings Share variation (percentage points)'
        )
    else:
        print("  No data found for unoccupied dwellings variation")


def _plot_raw_timeseries_scatter(countries_data, filename, title, x_label, y_label='Price-to-Income Index', populations=None):
    """Helper function to plot raw indicators over time with 3 points and 2 arrows per country
    
    Args:
        countries_data: dict with structure {country: {2001: {'x': val, 'y': val}, 2011: {...}, 2021: {...}}}
        filename: output filename
        title: plot title
        x_label: x-axis label
        y_label: y-axis label
        populations: dict with structure {(country, year): population} for population weighting
    """
    if not countries_data:
        print(f"  No data found for plotting")
        return
    
    # Collect all data points for axis limits
    all_x = []
    all_y = []
    for country, years in countries_data.items():
        for year, values in years.items():
            if values:
                all_x.append(values['x'])
                all_y.append(values['y'])
    
    if not all_x:
        print(f"  No valid data points for plotting")
        return
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # Calculate population-weighted regressions if population data provided - one for each year
    regression_2001_text = None
    regression_2011_text = None
    regression_2021_text = None
    slope_2001, intercept_2001 = None, None
    slope_2011, intercept_2011 = None, None
    slope_2021, intercept_2021 = None, None
    
    if populations:
        # Collect data for 2001
        year_2001_data = []
        for country, years in countries_data.items():
            if 2001 in years and years[2001]:
                x = years[2001]['x']
                y = years[2001]['y']
                pop = populations.get((country, 2001))
                if pop and not pd.isna(pop):
                    year_2001_data.append((x, y, pop))
        
        # Calculate weighted regression for 2001
        if len(year_2001_data) > 1:
            x_arr = np.array([d[0] for d in year_2001_data])
            y_arr = np.array([d[1] for d in year_2001_data])
            w_arr = np.array([d[2] for d in year_2001_data])
            
            w_sum = np.sum(w_arr)
            x_mean_w = np.sum(w_arr * x_arr) / w_sum
            y_mean_w = np.sum(w_arr * y_arr) / w_sum
            
            numerator = np.sum(w_arr * (x_arr - x_mean_w) * (y_arr - y_mean_w))
            denominator = np.sum(w_arr * (x_arr - x_mean_w) ** 2)
            
            if denominator > 0:
                slope_2001 = numerator / denominator
                intercept_2001 = y_mean_w - slope_2001 * x_mean_w
                
                y_pred = slope_2001 * x_arr + intercept_2001
                ss_res_w = np.sum(w_arr * (y_arr - y_pred) ** 2)
                ss_tot_w = np.sum(w_arr * (y_arr - y_mean_w) ** 2)
                r_squared = 1 - (ss_res_w / ss_tot_w) if ss_tot_w > 0 else 0
                
                regression_2001_text = f'2001: y={slope_2001:.3f}x+{intercept_2001:.2f} (R²={r_squared:.3f})'
        
        # Collect data for 2011
        year_2011_data = []
        for country, years in countries_data.items():
            if 2011 in years and years[2011]:
                x = years[2011]['x']
                y = years[2011]['y']
                pop = populations.get((country, 2011))
                if pop and not pd.isna(pop):
                    year_2011_data.append((x, y, pop))
        
        # Calculate weighted regression for 2011
        if len(year_2011_data) > 1:
            x_arr = np.array([d[0] for d in year_2011_data])
            y_arr = np.array([d[1] for d in year_2011_data])
            w_arr = np.array([d[2] for d in year_2011_data])
            
            w_sum = np.sum(w_arr)
            x_mean_w = np.sum(w_arr * x_arr) / w_sum
            y_mean_w = np.sum(w_arr * y_arr) / w_sum
            
            numerator = np.sum(w_arr * (x_arr - x_mean_w) * (y_arr - y_mean_w))
            denominator = np.sum(w_arr * (x_arr - x_mean_w) ** 2)
            
            if denominator > 0:
                slope_2011 = numerator / denominator
                intercept_2011 = y_mean_w - slope_2011 * x_mean_w
                
                y_pred = slope_2011 * x_arr + intercept_2011
                ss_res_w = np.sum(w_arr * (y_arr - y_pred) ** 2)
                ss_tot_w = np.sum(w_arr * (y_arr - y_mean_w) ** 2)
                r_squared = 1 - (ss_res_w / ss_tot_w) if ss_tot_w > 0 else 0
                
                regression_2011_text = f'2011: y={slope_2011:.3f}x+{intercept_2011:.2f} (R²={r_squared:.3f})'
        
        # Collect data for 2021
        year_2021_data = []
        for country, years in countries_data.items():
            if 2021 in years and years[2021]:
                x = years[2021]['x']
                y = years[2021]['y']
                pop = populations.get((country, 2021))
                if pop and not pd.isna(pop):
                    year_2021_data.append((x, y, pop))
        
        # Calculate weighted regression for 2021
        if len(year_2021_data) > 1:
            x_arr = np.array([d[0] for d in year_2021_data])
            y_arr = np.array([d[1] for d in year_2021_data])
            w_arr = np.array([d[2] for d in year_2021_data])
            
            w_sum = np.sum(w_arr)
            x_mean_w = np.sum(w_arr * x_arr) / w_sum
            y_mean_w = np.sum(w_arr * y_arr) / w_sum
            
            numerator = np.sum(w_arr * (x_arr - x_mean_w) * (y_arr - y_mean_w))
            denominator = np.sum(w_arr * (x_arr - x_mean_w) ** 2)
            
            if denominator > 0:
                slope_2021 = numerator / denominator
                intercept_2021 = y_mean_w - slope_2021 * x_mean_w
                
                y_pred = slope_2021 * x_arr + intercept_2021
                ss_res_w = np.sum(w_arr * (y_arr - y_pred) ** 2)
                ss_tot_w = np.sum(w_arr * (y_arr - y_mean_w) ** 2)
                r_squared = 1 - (ss_res_w / ss_tot_w) if ss_tot_w > 0 else 0
                
                regression_2021_text = f'2021: y={slope_2021:.3f}x+{intercept_2021:.2f} (R²={r_squared:.3f})'
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate axis limits with padding
    padding = 0.1
    x_min_data, x_max_data = min(all_x), max(all_x)
    y_min_data, y_max_data = min(all_y), max(all_y)
    x_range = x_max_data - x_min_data
    y_range = y_max_data - y_min_data
    x_min = x_min_data - padding * x_range
    x_max = x_max_data + padding * x_range
    y_min = y_min_data - padding * y_range
    y_max = y_max_data + padding * y_range
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Draw regression lines if calculated
    if slope_2001 is not None and intercept_2001 is not None:
        x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
        y_line = slope_2001 * x_line + intercept_2001
        ax.plot(x_line, y_line, color='#999999', linestyle='--', linewidth=2.5, alpha=0.7, zorder=2)
    
    if slope_2011 is not None and intercept_2011 is not None:
        x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
        y_line = slope_2011 * x_line + intercept_2011
        ax.plot(x_line, y_line, color='#666666', linestyle='--', linewidth=2.5, alpha=0.7, zorder=2)
    
    if slope_2021 is not None and intercept_2021 is not None:
        x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
        y_line = slope_2021 * x_line + intercept_2021
        ax.plot(x_line, y_line, color='#333333', linestyle='--', linewidth=2.5, alpha=0.7, zorder=2)
    
    # Plot points and arrows for each country
    for country, years in countries_data.items():
        color = get_country_color(country)
        
        # Plot points for each year
        years_sorted = sorted(years.keys())
        points = []
        for year in years_sorted:
            if years[year]:
                x = years[year]['x']
                y = years[year]['y']
                # Different marker sizes for visual distinction
                size = 100 if year == 2001 else (125 if year == 2011 else 150)
                alpha = 0.6 if year == 2001 else (0.75 if year == 2011 else 0.9)
                ax.scatter(x, y, s=size, alpha=alpha, color=color, marker='o', zorder=4)
                points.append((x, y, year))
        
        # Draw arrows connecting consecutive years
        for i in range(len(points) - 1):
            x0, y0, year0 = points[i]
            x1, y1, year1 = points[i + 1]
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color=color, alpha=0.7))
        
        # Label the country at the last point (2021 or most recent)
        if points:
            last_x, last_y, last_year = points[-1]
            ax.annotate(country, (last_x, last_y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', markersize=8, label='2001', alpha=0.6),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666', markersize=9, label='2011', alpha=0.75),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333', markersize=10, label='2021', alpha=0.9)
    ]
    
    # Add regression lines to legend if calculated
    if regression_2001_text:
        legend_elements.append(Line2D([0], [0], color='#999999', linestyle='--', linewidth=2.5, label=regression_2001_text))
    if regression_2011_text:
        legend_elements.append(Line2D([0], [0], color='#666666', linestyle='--', linewidth=2.5, label=regression_2011_text))
    if regression_2021_text:
        legend_elements.append(Line2D([0], [0], color='#333333', linestyle='--', linewidth=2.5, label=regression_2021_text))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95)
    
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVED] {filename}")


def create_raw_housing_indicators_vs_price():
    """Create scatter plots with raw indicators: dwelling metrics vs price-to-income index over time"""
    print("\n[7g] Creating raw housing indicators vs price-to-income scatter plots...")
    
    # ISO3 to country name mapping for price-to-income data
    ISO3_TO_COUNTRY = {
        'AUS': 'Austria', 'BEL': 'Belgium', 'BGR': 'Bulgaria', 'HRV': 'Croatia',
        'CYP': 'Cyprus', 'CZE': 'Czech Republic', 'DNK': 'Denmark', 'EST': 'Estonia',
        'FIN': 'Finland', 'FRA': 'France', 'DEU': 'Germany', 'GRC': 'Greece',
        'HUN': 'Hungary', 'ISL': 'Iceland', 'IRL': 'Ireland', 'ITA': 'Italy',
        'LVA': 'Latvia', 'LTU': 'Lithuania', 'LUX': 'Luxembourg', 'MLT': 'Malta',
        'NLD': 'Netherlands', 'NOR': 'Norway', 'POL': 'Poland', 'PRT': 'Portugal',
        'ROU': 'Romania', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'ESP': 'Spain',
        'SWE': 'Sweden', 'CHE': 'Switzerland', 'LIE': 'Liechtenstein'
    }
    
    # Load dwelling data
    dwellings_2001 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2001.csv'))
    dwellings_2011 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2011.csv'))
    dwellings_2021 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2021.csv'))
    
    # Load household data
    households_2001 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_households_2001.csv'))
    households_2011 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_households_2011.csv'))
    households_2021 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_households_2021.csv'))
    
    # Load population data for weighted regressions
    pop_2001 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2001.csv'))
    pop_2011 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2011.csv'))
    pop_2021 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2021.csv'))
    
    # Load price-to-income data
    df_price = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'ocde_house price-to-income ratio.csv'))
    df_price['year'] = df_price['TIME_PERIOD'].str.extract(r'(\d{4})')[0].astype(int)
    df_price['country'] = df_price['REF_AREA'].map(ISO3_TO_COUNTRY)
    df_price['ratio'] = pd.to_numeric(df_price['OBS_VALUE'], errors='coerce')
    df_price = df_price[df_price['year'].isin([2001, 2011, 2021])].dropna(subset=['country', 'ratio'])
    df_price_yearly = df_price.groupby(['year', 'country'])['ratio'].mean().reset_index()
    
    # Create dictionaries for dwelling, household, and price data by country and year
    dwellings = {}
    for _, row in pd.concat([dwellings_2001.assign(year=2001), 
                            dwellings_2011.assign(year=2011),
                            dwellings_2021.assign(year=2021)]).iterrows():
        key = (row['geo'], int(row['year']))
        dwellings[key] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
    
    households = {}
    for _, row in pd.concat([households_2001.assign(year=2001), 
                            households_2011.assign(year=2011),
                            households_2021.assign(year=2021)]).iterrows():
        key = (row['geo'], int(row['year']))
        households[key] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
    
    prices = {}
    for _, row in df_price_yearly.iterrows():
        key = (row['country'], int(row['year']))
        prices[key] = row['ratio']
    
    # Create population dictionary
    populations = {}
    for _, row in pop_2001.iterrows():
        populations[(row['geo'], 2001)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
    for _, row in pop_2011.iterrows():
        populations[(row['geo'], 2011)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
    for _, row in pop_2021.iterrows():
        populations[(row['geo'], 2021)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
    
    # Build data structures for both variants (A and B)
    countries_data_dph = {}  # Variant A: Dwelling per household
    countries_data_unoccupied = {}  # Variant B: % unoccupied dwellings
    
    # Get all unique countries from dwelling data
    all_countries = set()
    for key in dwellings.keys():
        all_countries.add(key[0])
    
    for country in all_countries:
        country_dph = {}
        country_unoccupied = {}
        
        for year in [2001, 2011, 2021]:
            d = dwellings.get((country, year))
            h = households.get((country, year))
            p = prices.get((country, year))
            
            # Check if all required data is available and valid
            if d and h and p and not pd.isna(d) and not pd.isna(h) and h > 0:
                # Variant A: Dwelling per household
                dph = d / h
                country_dph[year] = {'x': dph, 'y': p}
                
                # Variant B: % unoccupied dwellings
                pct_unoccupied = ((d - h) / d) * 100
                country_unoccupied[year] = {'x': pct_unoccupied, 'y': p}
        
        # Only include countries with at least 2 data points
        if len(country_dph) >= 2:
            countries_data_dph[country] = country_dph
        if len(country_unoccupied) >= 2:
            countries_data_unoccupied[country] = country_unoccupied
    
    # Create plots
    if countries_data_dph:
        _plot_raw_timeseries_scatter(
            countries_data_dph,
            '24a_raw_dwelling_per_household_vs_price.png',
            'Price-to-Income Index vs Dwelling per Household\n2001 → 2011 → 2021',
            'Dwelling per Household',
            populations=populations
        )
    else:
        print("  No data available for dwelling per household variant")
    
    if countries_data_unoccupied:
        _plot_raw_timeseries_scatter(
            countries_data_unoccupied,
            '24b_raw_unoccupied_dwellings_vs_price.png',
            'Price-to-Income Index vs Unoccupied Dwellings\n2001 → 2011 → 2021',
            'Unoccupied Dwellings (%)',
            populations=populations
        )
    else:
        print("  No data available for unoccupied dwellings variant")


def create_tenure_status_graphs():
    """Create visualizations for tenure status"""
    print("\n[8] Creating tenure status graphs...")
    
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_tenure status.csv')
    if not os.path.exists(file_path):
        print("  File not found:", file_path)
        return
    
    df = pd.read_csv(file_path)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    eu27_label = 'European Union - 27 countries (from 2020)'
    latest_year = int(df['TIME_PERIOD'].max())
    df_data = df[df['TIME_PERIOD'] == latest_year].copy()
    
    # EU27 by tenure type (Total income group)
    df_eu27 = df_data[(df_data['geo'] == eu27_label) & (df_data['incgrp'] == 'Total')].copy()
    if not df_eu27.empty:
        tenure_order = ['Total', 'Owner', 'Owner, with mortgage or loan', 'Owner, no outstanding mortgage or housing loan',
                       'Tenant', 'Tenant, rent at market price', 'Tenant, rent at reduced price or free']
        df_eu27_tenure = df_eu27[df_eu27['tenure'].isin(tenure_order) & (df_eu27['hhtyp'] == 'Total')]
        df_eu27_tenure['tenure'] = pd.Categorical(df_eu27_tenure['tenure'], categories=tenure_order, ordered=True)
        df_eu27_tenure = df_eu27_tenure.sort_values('tenure')
        
        if not df_eu27_tenure.empty and len(df_eu27_tenure) > 1:
            # Exclude Total for cleaner visualization
            df_eu27_tenure = df_eu27_tenure[df_eu27_tenure['tenure'] != 'Total']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = COMPONENT_COLORS[:len(df_eu27_tenure)]
            bars = ax.bar(range(len(df_eu27_tenure)), df_eu27_tenure['value'], color=colors, edgecolor='white', linewidth=1.5)
            
            ax.set_xticks(range(len(df_eu27_tenure)))
            short_labels = [l.replace('Owner, with mortgage or loan', 'Owner w/ mortgage')
                          .replace('Owner, no outstanding mortgage or housing loan', 'Owner no mortgage')
                          .replace('Tenant, rent at market price', 'Tenant market')
                          .replace('Tenant, rent at reduced price or free', 'Tenant reduced/free')
                          for l in df_eu27_tenure['tenure']]
            ax.set_xticklabels(short_labels, fontsize=9, rotation=15, ha='right')
            ax.set_ylabel('Share (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'EU27: Distribution by Tenure Status ({latest_year})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax.grid(axis='y', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'21_tenure_status_eu27.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 21_tenure_status_eu27.png")
    
    # Countries by Owner share (main tenure indicator)
    df_countries = df_data[(df_data['geo'] != eu27_label) & (df_data['incgrp'] == 'Total')].copy()
    df_countries['country_name'] = df_countries['geo'].apply(standardize_country_name)
    df_countries = df_countries[df_countries['country_name'].isin(TARGET_COUNTRIES)]
    df_countries = df_countries[(df_countries['tenure'] == 'Owner') & (df_countries['hhtyp'] == 'Total')]
    df_countries = df_countries.dropna(subset=['value'])
    
    if not df_countries.empty:
        df_countries = df_countries.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(11, 10))
        colors = [get_country_color(country) for country in df_countries['country_name']]
        bars = ax.barh(range(len(df_countries)), df_countries['value'], color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_countries)))
        ax.set_yticklabels(df_countries['country_name'], fontsize=9)
        ax.set_xlabel('Share (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Owner Occupancy by Country ({latest_year})', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'22_tenure_status_countries_owner.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] 22_tenure_status_countries_owner.png")
    
    # EU27 by tenure type with poverty threshold [A_MD60] (Above 60% median income)
    df_eu27_above = df_data[(df_data['geo'] == eu27_label) & (df_data['incgrp'] == 'Above 60% of median equivalised income')].copy()
    if not df_eu27_above.empty:
        tenure_order = ['Total', 'Owner', 'Owner, with mortgage or loan', 'Owner, no outstanding mortgage or housing loan',
                       'Tenant', 'Tenant, rent at market price', 'Tenant, rent at reduced price or free']
        df_eu27_above_tenure = df_eu27_above[df_eu27_above['tenure'].isin(tenure_order) & (df_eu27_above['hhtyp'] == 'Total')]
        df_eu27_above_tenure['tenure'] = pd.Categorical(df_eu27_above_tenure['tenure'], categories=tenure_order, ordered=True)
        df_eu27_above_tenure = df_eu27_above_tenure.sort_values('tenure')
        
        if not df_eu27_above_tenure.empty and len(df_eu27_above_tenure) > 1:
            # Exclude Total for cleaner visualization
            df_eu27_above_tenure = df_eu27_above_tenure[df_eu27_above_tenure['tenure'] != 'Total']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = COMPONENT_COLORS[:len(df_eu27_above_tenure)]
            bars = ax.bar(range(len(df_eu27_above_tenure)), df_eu27_above_tenure['value'], color=colors, edgecolor='white', linewidth=1.5)
            
            ax.set_xticks(range(len(df_eu27_above_tenure)))
            short_labels = [l.replace('Owner, with mortgage or loan', 'Owner w/ mortgage')
                          .replace('Owner, no outstanding mortgage or housing loan', 'Owner no mortgage')
                          .replace('Tenant, rent at market price', 'Tenant market')
                          .replace('Tenant, rent at reduced price or free', 'Tenant reduced/free')
                          for l in df_eu27_above_tenure['tenure']]
            ax.set_xticklabels(short_labels, fontsize=9, rotation=15, ha='right')
            ax.set_ylabel('Share (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'EU27: Distribution by Tenure Status - Above 60% of Median Income ({latest_year})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax.grid(axis='y', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'21b_tenure_status_eu27_above60.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 21b_tenure_status_eu27_above60.png")
    
    # France by tenure type (Total income group)
    df_france = df_data[(df_data['geo'] == 'France') & (df_data['incgrp'] == 'Total')].copy()
    if not df_france.empty:
        tenure_order = ['Total', 'Owner', 'Owner, with mortgage or loan', 'Owner, no outstanding mortgage or housing loan',
                       'Tenant', 'Tenant, rent at market price', 'Tenant, rent at reduced price or free']
        df_france_tenure = df_france[df_france['tenure'].isin(tenure_order) & (df_france['hhtyp'] == 'Total')]
        df_france_tenure['tenure'] = pd.Categorical(df_france_tenure['tenure'], categories=tenure_order, ordered=True)
        df_france_tenure = df_france_tenure.sort_values('tenure')
        
        if not df_france_tenure.empty and len(df_france_tenure) > 1:
            # Exclude Total for cleaner visualization
            df_france_tenure = df_france_tenure[df_france_tenure['tenure'] != 'Total']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = COMPONENT_COLORS[:len(df_france_tenure)]
            bars = ax.bar(range(len(df_france_tenure)), df_france_tenure['value'], color=colors, edgecolor='white', linewidth=1.5)
            
            ax.set_xticks(range(len(df_france_tenure)))
            short_labels = [l.replace('Owner, with mortgage or loan', 'Owner w/ mortgage')
                          .replace('Owner, no outstanding mortgage or housing loan', 'Owner no mortgage')
                          .replace('Tenant, rent at market price', 'Tenant market')
                          .replace('Tenant, rent at reduced price or free', 'Tenant reduced/free')
                          for l in df_france_tenure['tenure']]
            ax.set_xticklabels(short_labels, fontsize=9, rotation=15, ha='right')
            ax.set_ylabel('Share (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'France: Distribution by Tenure Status ({latest_year})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax.grid(axis='y', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'21c_tenure_status_france.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 21c_tenure_status_france.png")

def create_house_sales_graphs():
    """Create visualizations for house sales by purchase type"""
    print("\n[9] Creating house sales graphs...")
    
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_House sales.csv')
    if not os.path.exists(file_path):
        print("  File not found:", file_path)
        return
    
    df = pd.read_csv(file_path)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    eu27_label = 'European Union - 27 countries (from 2020)'
    
    # EU27 time series by purchase type
    df_eu27 = df[df['geo'] == eu27_label].copy()
    if not df_eu27.empty:
        purchase_order = ['Total', 'Purchases of newly built dwellings', 'Purchases of existing dwellings']
        df_eu27_time = df_eu27[df_eu27['purchase'].isin(purchase_order)]
        
        if not df_eu27_time.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Plot each purchase type as a line
            colors_purchase = COMPONENT_COLORS[:len(purchase_order)-1]  # Skip Total color
            labels_short = {
                'Purchases of newly built dwellings': 'Newly built',
                'Purchases of existing dwellings': 'Existing'
            }
            
            for idx, ptype in enumerate(purchase_order):
                if ptype == 'Total':
                    continue
                df_ptype = df_eu27_time[df_eu27_time['purchase'] == ptype].sort_values('TIME_PERIOD')
                if not df_ptype.empty:
                    ax.plot(df_ptype['TIME_PERIOD'].astype(int), df_ptype['value'],
                           marker='o', linewidth=2.5, markersize=6,
                           label=labels_short.get(ptype, ptype),
                           color=colors_purchase[idx-1], alpha=0.8)
            
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('House Sales (Number)', fontsize=12, fontweight='bold')
            ax.set_title(f'EU27: House Sales by Purchase Type (2000-2024)', 
                        fontsize=13, fontweight='bold', pad=15)
            
            ax.legend(fontsize=11, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'23_house_sales_eu27_timeseries.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 23_house_sales_eu27_timeseries.png")
    
    # Countries latest year - Total purchases
    latest_year = int(df['TIME_PERIOD'].max())
    df_latest = df[(df['TIME_PERIOD'] == latest_year) & (df['geo'] != eu27_label) & (df['purchase'] == 'Total')].copy()
    df_latest['country_name'] = df_latest['geo'].apply(standardize_country_name)
    df_latest = df_latest[df_latest['country_name'].isin(TARGET_COUNTRIES)]
    df_latest = df_latest.dropna(subset=['value'])
    
    if not df_latest.empty:
        df_latest = df_latest.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(11, 10))
        colors = [get_country_color(country) for country in df_latest['country_name']]
        bars = ax.barh(range(len(df_latest)), df_latest['value'], color=colors, edgecolor='white', linewidth=1.5)
        
        ax.set_yticks(range(len(df_latest)))
        ax.set_yticklabels(df_latest['country_name'], fontsize=9)
        ax.set_xlabel('House Sales (Number)', fontsize=12, fontweight='bold')
        ax.set_title(f'House Sales by Country ({latest_year})', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.grid(axis='x', alpha=0.2)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'24_house_sales_countries.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] 24_house_sales_countries.png")
    
    # Countries time series - Total purchases
    df_countries_time = df[(df['geo'] != eu27_label) & (df['purchase'] == 'Total')].copy()
    df_countries_time['country_name'] = df_countries_time['geo'].apply(standardize_country_name)
    df_countries_time = df_countries_time[df_countries_time['country_name'].isin(TARGET_COUNTRIES)]
    df_countries_time = df_countries_time.dropna(subset=['value'])
    
    # Get countries with good time series coverage
    country_counts = df_countries_time.groupby('country_name').size()
    countries_with_data = country_counts[country_counts >= 10].index.tolist()
    
    if len(countries_with_data) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Select top 6 countries by latest value for clarity
        df_latest_vals = df_countries_time[df_countries_time['TIME_PERIOD'] == df_countries_time['TIME_PERIOD'].max()]
        top_countries = df_latest_vals.nlargest(6, 'value')['country_name'].unique().tolist()
        
        colors_countries = COMPONENT_COLORS[:len(top_countries)]
        
        for idx, country in enumerate(top_countries):
            df_country = df_countries_time[df_countries_time['country_name'] == country].sort_values('TIME_PERIOD')
            if not df_country.empty:
                ax.plot(df_country['TIME_PERIOD'].astype(int), df_country['value'],
                       marker='o', linewidth=2.5, markersize=6, label=country,
                       color=colors_countries[idx], alpha=0.8)
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('House Sales (Number)', fontsize=12, fontweight='bold')
        ax.set_title(f'House Sales Trends by Country (Top 6, 2000-2024)', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'24b_house_sales_countries_timeseries.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] 24b_house_sales_countries_timeseries.png")

def create_air_emissions_graphs():
    """Create visualizations for air emissions by NACE sector"""
    print("\n[10] Creating air emissions graphs...")
    
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_Air emissions.csv')
    if not os.path.exists(file_path):
        print("  File not found:", file_path)
        return
    
    df = pd.read_csv(file_path)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    eu27_label = 'European Union - 27 countries (from 2020)'
    latest_year = int(df['TIME_PERIOD'].max())
    df_latest = df[df['TIME_PERIOD'] == latest_year].copy()
    
    # Process for each unit separately
    for unit_val in sorted(df_latest['unit'].unique()):
        df_unit = df_latest[df_latest['unit'] == unit_val].copy()
        unit_label = 'kg per capita' if 'Kilogram' in unit_val else 'tonnes'
        
        # EU27 by NACE sector
        df_eu27 = df_unit[df_unit['geo'] == eu27_label].copy()
        if not df_eu27.empty:
            # Exclude Total NACE
            df_eu27 = df_eu27[df_eu27['nace_r2'] != 'Total - all NACE activities'].copy()
            
            # Sort by value descending to show biggest emitters first
            df_eu27 = df_eu27.sort_values('value', ascending=False)
            
            # Limit to top 12 NACE sectors for readability
            if len(df_eu27) > 12:
                df_eu27 = df_eu27.head(12)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = COMPONENT_COLORS[:len(df_eu27)]
            bars = ax.barh(range(len(df_eu27)), df_eu27['value'], color=colors, edgecolor='white', linewidth=1.5)
            
            ax.set_yticks(range(len(df_eu27)))
            # Shorten NACE labels for display
            short_labels = [label[:40] + '...' if len(label) > 40 else label for label in df_eu27['nace_r2']]
            ax.set_yticklabels(short_labels, fontsize=9)
            
            ax.set_xlabel(f'Greenhouse Gas Emissions ({unit_label})', fontsize=12, fontweight='bold')
            ax.set_title(f'EU27: GHG Emissions by NACE Sector ({latest_year})\n({unit_label})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            ax.grid(axis='x', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            file_suffix = 'kgcap' if 'Kilogram' in unit_val else 'tonnes'
            plt.savefig(os.path.join(OUTPUT_DIR, f'25_air_emissions_eu27_nace_{file_suffix}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 25_air_emissions_eu27_nace_{file_suffix}.png")
        
        # Countries - Total NACE (all sectors combined)
        df_countries = df_unit[(df_unit['geo'] != eu27_label) & (df_unit['nace_r2'] == 'Total - all NACE activities')].copy()
        df_countries['country_name'] = df_countries['geo'].apply(standardize_country_name)
        df_countries = df_countries[df_countries['country_name'].isin(TARGET_COUNTRIES)]
        df_countries = df_countries.dropna(subset=['value'])
        
        if not df_countries.empty:
            df_countries = df_countries.sort_values('value', ascending=True)
            
            fig, ax = plt.subplots(figsize=(11, 10))
            colors = [get_country_color(country) for country in df_countries['country_name']]
            bars = ax.barh(range(len(df_countries)), df_countries['value'], color=colors, edgecolor='white', linewidth=1.5)
            
            ax.set_yticks(range(len(df_countries)))
            ax.set_yticklabels(df_countries['country_name'], fontsize=9)
            ax.set_xlabel(f'GHG Emissions ({unit_label})', fontsize=12, fontweight='bold')
            ax.set_title(f'GHG Emissions (All NACE) by Country ({latest_year})\n({unit_label})', 
                        fontsize=13, fontweight='bold', pad=15)
            
            ax.grid(axis='x', alpha=0.2)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            file_suffix = 'kgcap' if 'Kilogram' in unit_val else 'tonnes'
            plt.savefig(os.path.join(OUTPUT_DIR, f'26_air_emissions_countries_{file_suffix}.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  [SAVED] 26_air_emissions_countries_{file_suffix}.png")

def create_dwellings_per_household_2011_2022():
    """Create scatter plot: dwellings per inhabitant variation (2001-2011 & 2011-2021) vs price-to-income change
    Eurostat-only: dwellings per inhabitant = dwellings / population
    """
    print("\n[23_new] Creating dwellings per inhabitant (2001-2011 & 2011-2021) vs price-to-income scatter plot (Eurostat-only)...")

    EUROSTAT_TO_STANDARD = {
        'Austria': 'Austria', 'Belgium': 'Belgium', 'Bulgaria': 'Bulgaria',
        'Croatia': 'Croatia', 'Cyprus': 'Cyprus', 'Czechia': 'Czech Republic',
        'Denmark': 'Denmark', 'Estonia': 'Estonia', 'Finland': 'Finland',
        'France': 'France', 'Germany': 'Germany', 'Greece': 'Greece',
        'Hungary': 'Hungary', 'Iceland': 'Iceland', 'Ireland': 'Ireland',
        'Italy': 'Italy', 'Latvia': 'Latvia', 'Lithuania': 'Lithuania',
        'Luxembourg': 'Luxembourg', 'Malta': 'Malta', 'Netherlands': 'Netherlands',
        'Norway': 'Norway', 'Poland': 'Poland', 'Portugal': 'Portugal',
        'Romania': 'Romania', 'Slovakia': 'Slovakia', 'Slovenia': 'Slovenia',
        'Spain': 'Spain', 'Sweden': 'Sweden', 'Switzerland': 'Switzerland'
    }
    
    ISO3_TO_COUNTRY = {
        'AUS': 'Austria', 'BEL': 'Belgium', 'BGR': 'Bulgaria', 'HRV': 'Croatia',
        'CYP': 'Cyprus', 'CZE': 'Czech Republic', 'DNK': 'Denmark', 'EST': 'Estonia',
        'FIN': 'Finland', 'FRA': 'France', 'DEU': 'Germany', 'GRC': 'Greece',
        'HUN': 'Hungary', 'ISL': 'Iceland', 'IRL': 'Ireland', 'ITA': 'Italy',
        'LVA': 'Latvia', 'LTU': 'Lithuania', 'LUX': 'Luxembourg', 'MLT': 'Malta',
        'NLD': 'Netherlands', 'NOR': 'Norway', 'POL': 'Poland', 'PRT': 'Portugal',
        'ROU': 'Romania', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'ESP': 'Spain',
        'SWE': 'Sweden', 'CHE': 'Switzerland'
    }
    
    try:
        # Load population data for weighted regressions
        pop_2001 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2001.csv'))
        pop_2011 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2011.csv'))
        pop_2021 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2021.csv'))
        
        # Create population dictionaries
        populations = {}
        for _, row in pop_2001.iterrows():
            country = EUROSTAT_TO_STANDARD.get(row['geo'], row['geo'])
            populations[(country, 2001)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
        for _, row in pop_2011.iterrows():
            country = EUROSTAT_TO_STANDARD.get(row['geo'], row['geo'])
            populations[(country, 2011)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
        for _, row in pop_2021.iterrows():
            country = EUROSTAT_TO_STANDARD.get(row['geo'], row['geo'])
            populations[(country, 2021)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
        
        # Load price-to-income data
        df_price = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'ocde_house price-to-income ratio.csv'))
        df_price['year'] = df_price['TIME_PERIOD'].str.extract(r'(\d{4})')[0].astype(int)
        df_price['country'] = df_price['REF_AREA'].map(ISO3_TO_COUNTRY)
        df_price['ratio'] = pd.to_numeric(df_price['OBS_VALUE'], errors='coerce')
        df_price = df_price[df_price['year'].isin([2001, 2011, 2021])].dropna(subset=['country', 'ratio'])
        df_price_yearly = df_price.groupby(['year', 'country'])['ratio'].mean().reset_index()
        
        # Build data structures
        countries_data_1121_hh = {}
        countries_data_0111_hh = {}
        countries_data_1121_inh = {}
        countries_data_0111_inh = {}
        
        # Build 2001-2011 and 2011-2021 series using Eurostat dwellings, households, and population
        dwellings_2001 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2001.csv'))
        dwellings_2011 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2011.csv'))
        dwellings_2021 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_dwellings_2021.csv'))
        households_2001 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_households_2001.csv'))
        households_2011 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_households_2011.csv'))
        households_2021 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_households_2021.csv'))
        pop_2001_df = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2001.csv'))
        pop_2011_df = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2011.csv'))
        pop_2021_df = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2021.csv'))

        def _build_series(df_values):
            df_values = df_values.copy()
            df_values['country'] = df_values['geo'].map(EUROSTAT_TO_STANDARD).fillna(df_values['geo'])
            df_values['value'] = pd.to_numeric(df_values['OBS_VALUE'], errors='coerce')
            return df_values.groupby('country')['value'].mean()

        dw_2001 = _build_series(dwellings_2001)
        dw_2011 = _build_series(dwellings_2011)
        dw_2021 = _build_series(dwellings_2021)
        hh_2001 = _build_series(households_2001)
        hh_2011 = _build_series(households_2011)
        hh_2021 = _build_series(households_2021)
        pop_2001 = _build_series(pop_2001_df)
        pop_2011 = _build_series(pop_2011_df)
        pop_2021 = _build_series(pop_2021_df)

        # --- Per household (dwellings / households) ---
        candidate_countries_hh_0111 = set(dw_2001.index) & set(dw_2011.index) & set(hh_2001.index) & set(hh_2011.index)
        for country in candidate_countries_hh_0111:
            d2001 = dw_2001.get(country)
            d2011 = dw_2011.get(country)
            h2001 = hh_2001.get(country)
            h2011 = hh_2011.get(country)
            if (pd.isna(d2001) or pd.isna(d2011) or pd.isna(h2001) or pd.isna(h2011)
                or h2001 <= 0 or h2011 <= 0):
                continue

            dph_2001 = d2001 / h2001
            dph_2011 = d2011 / h2011

            price_2001 = df_price_yearly[(df_price_yearly['country'] == country) & (df_price_yearly['year'] == 2001)]
            price_2011 = df_price_yearly[(df_price_yearly['country'] == country) & (df_price_yearly['year'] == 2011)]
            if price_2001.empty or price_2011.empty:
                continue

            p2001 = price_2001['ratio'].values[0]
            p2011 = price_2011['ratio'].values[0]

            dw_variation_0111 = ((dph_2011 - dph_2001) / dph_2001) * 100
            price_variation_0111 = ((p2011 - p2001) / p2001) * 100

            countries_data_0111_hh[country] = {
                'x_2001': dph_2001,
                'x_2011': dph_2011,
                'price_2001': p2001,
                'price_2011': p2011,
                'dw_variation': dw_variation_0111,
                'price_variation': price_variation_0111
            }

        candidate_countries_hh_1121 = set(dw_2011.index) & set(dw_2021.index) & set(hh_2011.index) & set(hh_2021.index)
        for country in candidate_countries_hh_1121:
            d2011 = dw_2011.get(country)
            d2021 = dw_2021.get(country)
            h2011 = hh_2011.get(country)
            h2021 = hh_2021.get(country)
            if (pd.isna(d2011) or pd.isna(d2021) or pd.isna(h2011) or pd.isna(h2021)
                or h2011 <= 0 or h2021 <= 0):
                continue

            dph_2011 = d2011 / h2011
            dph_2021 = d2021 / h2021

            price_2011 = df_price_yearly[(df_price_yearly['country'] == country) & (df_price_yearly['year'] == 2011)]
            price_2021 = df_price_yearly[(df_price_yearly['country'] == country) & (df_price_yearly['year'] == 2021)]
            if price_2011.empty or price_2021.empty:
                continue

            p2011 = price_2011['ratio'].values[0]
            p2021 = price_2021['ratio'].values[0]

            dw_variation_1121 = ((dph_2021 - dph_2011) / dph_2011) * 100
            price_variation_1121 = ((p2021 - p2011) / p2011) * 100

            countries_data_1121_hh[country] = {
                'x_2011': dph_2011,
                'x_2021': dph_2021,
                'price_2011': p2011,
                'price_2021': p2021,
                'dw_variation': dw_variation_1121,
                'price_variation': price_variation_1121
            }

        # --- Per inhabitant (dwellings / population) ---
        candidate_countries_inh_0111 = set(dw_2001.index) & set(dw_2011.index) & set(pop_2001.index) & set(pop_2011.index)
        for country in candidate_countries_inh_0111:
            d2001 = dw_2001.get(country)
            d2011 = dw_2011.get(country)
            p2001_pop = pop_2001.get(country)
            p2011_pop = pop_2011.get(country)
            if (pd.isna(d2001) or pd.isna(d2011) or pd.isna(p2001_pop) or pd.isna(p2011_pop)
                or p2001_pop <= 0 or p2011_pop <= 0):
                continue

            dpi_2001 = d2001 / p2001_pop
            dpi_2011 = d2011 / p2011_pop

            price_2001 = df_price_yearly[(df_price_yearly['country'] == country) & (df_price_yearly['year'] == 2001)]
            price_2011 = df_price_yearly[(df_price_yearly['country'] == country) & (df_price_yearly['year'] == 2011)]
            if price_2001.empty or price_2011.empty:
                continue

            p2001 = price_2001['ratio'].values[0]
            p2011 = price_2011['ratio'].values[0]

            dw_variation_0111 = ((dpi_2011 - dpi_2001) / dpi_2001) * 100
            price_variation_0111 = ((p2011 - p2001) / p2001) * 100

            countries_data_0111_inh[country] = {
                'x_2001': dpi_2001,
                'x_2011': dpi_2011,
                'price_2001': p2001,
                'price_2011': p2011,
                'dw_variation': dw_variation_0111,
                'price_variation': price_variation_0111
            }

        candidate_countries_inh_1121 = set(dw_2011.index) & set(dw_2021.index) & set(pop_2011.index) & set(pop_2021.index)
        for country in candidate_countries_inh_1121:
            d2011 = dw_2011.get(country)
            d2021 = dw_2021.get(country)
            p2011_pop = pop_2011.get(country)
            p2021_pop = pop_2021.get(country)
            if (pd.isna(d2011) or pd.isna(d2021) or pd.isna(p2011_pop) or pd.isna(p2021_pop)
                or p2011_pop <= 0 or p2021_pop <= 0):
                continue

            dpi_2011 = d2011 / p2011_pop
            dpi_2021 = d2021 / p2021_pop

            price_2011 = df_price_yearly[(df_price_yearly['country'] == country) & (df_price_yearly['year'] == 2011)]
            price_2021 = df_price_yearly[(df_price_yearly['country'] == country) & (df_price_yearly['year'] == 2021)]
            if price_2011.empty or price_2021.empty:
                continue

            p2011 = price_2011['ratio'].values[0]
            p2021 = price_2021['ratio'].values[0]

            dw_variation_1121 = ((dpi_2021 - dpi_2011) / dpi_2011) * 100
            price_variation_1121 = ((p2021 - p2011) / p2011) * 100

            countries_data_1121_inh[country] = {
                'x_2011': dpi_2011,
                'x_2021': dpi_2021,
                'price_2011': p2011,
                'price_2021': p2021,
                'dw_variation': dw_variation_1121,
                'price_variation': price_variation_1121
            }

        def _plot_variation_and_raw(countries_data_0111, countries_data_1121, variation_filename,
                                    variation_title, x_label_variation, raw_filename, raw_title, x_label_raw):
            if not countries_data_1121 and not countries_data_0111:
                print("    No data available for plotting")
                return

            # ===== GRAPH 1: Variation scatter with population-weighted regression =====
            fig, ax = plt.subplots(figsize=(12, 8))

            x_data_1122 = []
            y_data_1122 = []
            weights_1122 = []
            x_data_0111 = []
            y_data_0111 = []
            weights_0111 = []

            for country, data in countries_data_1121.items():
                x = data['dw_variation']
                y = data['price_variation']

                pop_2011 = populations.get((country, 2011))
                pop_2021 = populations.get((country, 2021))

                if pop_2011 and pop_2021 and not pd.isna(pop_2011) and not pd.isna(pop_2021):
                    avg_pop = (pop_2011 + pop_2021) / 2
                    x_data_1122.append(x)
                    y_data_1122.append(y)
                    weights_1122.append(avg_pop)

                ax.scatter(x, y, s=100, alpha=0.6, color='#80b1d3', edgecolors='none', marker='o')
                ax.annotate(country, (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

            for country, data in countries_data_0111.items():
                x = data['dw_variation']
                y = data['price_variation']

                pop_2001 = populations.get((country, 2001))
                pop_2011 = populations.get((country, 2011))

                if pop_2001 and pop_2011 and not pd.isna(pop_2001) and not pd.isna(pop_2011):
                    avg_pop = (pop_2001 + pop_2011) / 2
                    x_data_0111.append(x)
                    y_data_0111.append(y)
                    weights_0111.append(avg_pop)

                ax.scatter(x, y, s=95, alpha=0.6, color='#fdb462', edgecolors='none', marker='s')

            regression_labels = []

            if len(x_data_1122) > 1 and np.sum(weights_1122) > 0:
                x_arr = np.array(x_data_1122)
                y_arr = np.array(y_data_1122)
                w_arr = np.array(weights_1122)

                w_sum = np.sum(w_arr)
                x_mean_w = np.sum(w_arr * x_arr) / w_sum
                y_mean_w = np.sum(w_arr * y_arr) / w_sum

                numerator = np.sum(w_arr * (x_arr - x_mean_w) * (y_arr - y_mean_w))
                denominator = np.sum(w_arr * (x_arr - x_mean_w) ** 2)

                if denominator > 0:
                    slope = numerator / denominator
                    intercept = y_mean_w - slope * x_mean_w

                    y_pred = slope * x_arr + intercept
                    ss_res_w = np.sum(w_arr * (y_arr - y_pred) ** 2)
                    ss_tot_w = np.sum(w_arr * (y_arr - y_mean_w) ** 2)
                    r_squared = 1 - (ss_res_w / ss_tot_w) if ss_tot_w > 0 else 0

                    x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color='#80b1d3', linestyle='--', linewidth=2.5, alpha=0.7)
                    regression_labels.append((
                        f'2011-2021: y={slope:.3f}x + {intercept:.2f} (R² = {r_squared:.3f}) [pop-weighted]',
                        '#80b1d3'
                    ))

            if len(x_data_0111) > 1 and np.sum(weights_0111) > 0:
                x_arr = np.array(x_data_0111)
                y_arr = np.array(y_data_0111)
                w_arr = np.array(weights_0111)

                w_sum = np.sum(w_arr)
                x_mean_w = np.sum(w_arr * x_arr) / w_sum
                y_mean_w = np.sum(w_arr * y_arr) / w_sum

                numerator = np.sum(w_arr * (x_arr - x_mean_w) * (y_arr - y_mean_w))
                denominator = np.sum(w_arr * (x_arr - x_mean_w) ** 2)

                if denominator > 0:
                    slope = numerator / denominator
                    intercept = y_mean_w - slope * x_mean_w

                    y_pred = slope * x_arr + intercept
                    ss_res_w = np.sum(w_arr * (y_arr - y_pred) ** 2)
                    ss_tot_w = np.sum(w_arr * (y_arr - y_mean_w) ** 2)
                    r_squared = 1 - (ss_res_w / ss_tot_w) if ss_tot_w > 0 else 0

                    x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color='#fdb462', linestyle='--', linewidth=2.5, alpha=0.7)
                    regression_labels.append((
                        f'2001-2011: y={slope:.3f}x + {intercept:.2f} (R² = {r_squared:.3f}) [pop-weighted]',
                        '#fdb462'
                    ))

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

            ax.set_xlabel(x_label_variation, fontsize=12, fontweight='bold')
            ax.set_ylabel('Variation in Price-to-Income Ratio (%)', fontsize=12, fontweight='bold')
            ax.set_title(variation_title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#fdb462', markersize=8, label='2001-2011', alpha=0.6),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#80b1d3', markersize=8, label='2011-2021', alpha=0.6)
            ]
            for label, color in regression_labels:
                legend_elements.append(Line2D([0], [0], color=color, linestyle='--', linewidth=2.5, label=label))
            ax.legend(handles=legend_elements, loc='best', fontsize=9, framealpha=0.95)

            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, variation_filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    OK Saved: {variation_filename}")
            plt.close()

            # ===== GRAPH 2: Raw values with arrows (2011 -> 2021) =====
            fig, ax = plt.subplots(figsize=(14, 10))

            for country, data in countries_data_1121.items():
                x_2011 = data['x_2011']
                y_2011 = data['price_2011']
                x_2021 = data['x_2021']
                y_2021 = data['price_2021']

                ax.scatter(x_2011, y_2011, s=100, alpha=0.6, color='#80b1d3', edgecolors='none', zorder=4)
                ax.scatter(x_2021, y_2021, s=125, alpha=0.75, color='#80b1d3', edgecolors='none', zorder=4)

                ax.annotate('', xy=(x_2021, y_2021), xytext=(x_2011, y_2011),
                           arrowprops=dict(arrowstyle='->', lw=2, color='#80b1d3', alpha=0.6))

                ax.annotate(country, (x_2021, y_2021), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', alpha=0.7)

            ax.set_xlabel(x_label_raw, fontsize=12, fontweight='bold')
            ax.set_ylabel('Price-to-Income Index', fontsize=12, fontweight='bold')
            ax.set_title(raw_title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#80b1d3', markersize=8,
                       label='2011', alpha=0.6, markeredgewidth=0),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#80b1d3', markersize=9,
                       label='2021', alpha=0.75, markeredgewidth=0)
            ]
            ax.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.95)

            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, raw_filename)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    OK Saved: {raw_filename}")
            plt.close()

            print(f"    Countries plotted (2011-2021): {len(countries_data_1121)}")

        _plot_variation_and_raw(
            countries_data_0111_hh,
            countries_data_1121_hh,
            '23_dwellings_per_household_2011_2022_vs_price.png',
            'Dwellings per Household vs Price-to-Income Changes (2001-2011 & 2011-2021)',
            'Variation in Dwellings per Household (%)',
            '23b_dwellings_per_household_2011_2022_raw_with_arrows.png',
            'Dwellings per Household vs Price-to-Income (2011 → 2021)',
            'Dwellings per Household'
        )

        _plot_variation_and_raw(
            countries_data_0111_inh,
            countries_data_1121_inh,
            '23_dwellings_per_inhabitant_2011_2022_vs_price.png',
            'Dwellings per Inhabitant vs Price-to-Income Changes (2001-2011 & 2011-2021)',
            'Variation in Dwellings per Inhabitant (%)',
            '23b_dwellings_per_inhabitant_2011_2022_raw_with_arrows.png',
            'Dwellings per Inhabitant vs Price-to-Income (2011 → 2021)',
            'Dwellings per Inhabitant'
        )
        
    except Exception as e:
        print(f"    ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def create_dwellings_per_household_annualized_2005_2022():
    """Create scatter plot: annualized change in dwelling per 1000 households vs annualized price-to-income change
    Two data series: [2005-2011] and [2011-2022], each with population-weighted regression
    """
    print("\n[23_annualized] Creating annualized dwelling per household vs price-to-income scatter plot (2005-2022)...")
    
    # ISO3 to country name mapping
    ISO3_TO_COUNTRY = {
        'AUT': 'Austria', 'BEL': 'Belgium', 'BGR': 'Bulgaria', 'HRV': 'Croatia',
        'CYP': 'Cyprus', 'CZE': 'Czech Republic', 'DNK': 'Denmark', 'EST': 'Estonia',
        'FIN': 'Finland', 'FRA': 'France', 'DEU': 'Germany', 'GRC': 'Greece',
        'HUN': 'Hungary', 'ISL': 'Iceland', 'IRL': 'Ireland', 'ITA': 'Italy',
        'LVA': 'Latvia', 'LTU': 'Lithuania', 'LUX': 'Luxembourg', 'MLT': 'Malta',
        'NLD': 'Netherlands', 'NOR': 'Norway', 'POL': 'Poland', 'PRT': 'Portugal',
        'ROU': 'Romania', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'ESP': 'Spain',
        'SWE': 'Sweden', 'CHE': 'Switzerland', 'USA': 'United States', 'CAN': 'Canada',
        'GBR': 'United Kingdom'
    }
    
    EUROSTAT_TO_STANDARD = {
        'Austria': 'Austria', 'Belgium': 'Belgium', 'Bulgaria': 'Bulgaria',
        'Croatia': 'Croatia', 'Cyprus': 'Cyprus', 'Czechia': 'Czech Republic',
        'Denmark': 'Denmark', 'Estonia': 'Estonia', 'Finland': 'Finland',
        'France': 'France', 'Germany': 'Germany', 'Greece': 'Greece',
        'Hungary': 'Hungary', 'Iceland': 'Iceland', 'Ireland': 'Ireland',
        'Italy': 'Italy', 'Latvia': 'Latvia', 'Lithuania': 'Lithuania',
        'Luxembourg': 'Luxembourg', 'Malta': 'Malta', 'Netherlands': 'Netherlands',
        'Norway': 'Norway', 'Poland': 'Poland', 'Portugal': 'Portugal',
        'Romania': 'Romania', 'Slovakia': 'Slovakia', 'Slovenia': 'Slovenia',
        'Spain': 'Spain', 'Sweden': 'Sweden', 'Switzerland': 'Switzerland'
    }
    
    try:
        # Load 2005 data (dwellings per 1000 households)
        df_2005 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'oecd_housing_households_2005.csv'), sep=';')
        df_2005['Country_Name'] = df_2005['Country'].map(ISO3_TO_COUNTRY)
        df_2005['Dw_per_1000_hh_2005'] = pd.to_numeric(df_2005['Dwelling stock'], errors='coerce')
        
        # Load household size data
        df_hh_size = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_household_size.csv'))
        df_hh_size = df_hh_size[df_hh_size['TIME_PERIOD'].isin([2011, 2022])].copy()
        df_hh_size['country'] = df_hh_size['geo'].map(EUROSTAT_TO_STANDARD)
        df_hh_size = df_hh_size.dropna(subset=['country', 'OBS_VALUE'])
        df_hh_size['household_size'] = pd.to_numeric(df_hh_size['OBS_VALUE'], errors='coerce')
        
        # Load population data
        pop_2005 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2001.csv'))  # Using 2001 as proxy for 2005
        pop_2011 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2011.csv'))
        pop_2021 = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population_2021.csv'))
        
        # Create population dictionaries
        populations = {}
        for _, row in pop_2005.iterrows():
            country = EUROSTAT_TO_STANDARD.get(row['geo'], row['geo'])
            populations[(country, 2005)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
        for _, row in pop_2011.iterrows():
            country = EUROSTAT_TO_STANDARD.get(row['geo'], row['geo'])
            populations[(country, 2011)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
        for _, row in pop_2021.iterrows():
            country = EUROSTAT_TO_STANDARD.get(row['geo'], row['geo'])
            populations[(country, 2021)] = pd.to_numeric(row['OBS_VALUE'], errors='coerce')
        
        # Load 2011 and 2022 dwellings per inhabitants data
        df_dw_per_inh_raw = pd.read_excel(os.path.join(EXTERNAL_DATA_DIR, 'oecd_dwellings_inhabitants.xlsx'))
        df_dw_per_inh = pd.DataFrame({
            'Country': df_dw_per_inh_raw.iloc[4:, 11].values,
            '2011': pd.to_numeric(df_dw_per_inh_raw.iloc[4:, 12], errors='coerce') / 1000,
            '2022': pd.to_numeric(df_dw_per_inh_raw.iloc[4:, 14], errors='coerce') / 1000
        }).dropna(subset=['Country'])
        
        # Load price-to-income data
        df_price = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'ocde_house price-to-income ratio.csv'))
        df_price['year'] = df_price['TIME_PERIOD'].str.extract(r'(\d{4})')[0].astype(int)
        df_price['country'] = df_price['REF_AREA'].map(ISO3_TO_COUNTRY)
        df_price['ratio'] = pd.to_numeric(df_price['OBS_VALUE'], errors='coerce')
        df_price = df_price[df_price['year'].isin([2005, 2011, 2022])].dropna(subset=['country', 'ratio'])
        df_price_yearly = df_price.groupby(['year', 'country'])['ratio'].mean().reset_index()
        
        # Build data structure for both periods
        countries_data_0511 = {}  # 2005-2011 period
        countries_data_1122 = {}  # 2011-2022 period
        
        # Get unique countries from household size data
        for country in df_hh_size['country'].unique():
            # Get household sizes
            hh_2011 = df_hh_size[(df_hh_size['country'] == country) & (df_hh_size['TIME_PERIOD'] == 2011)]
            hh_2022 = df_hh_size[(df_hh_size['country'] == country) & (df_hh_size['TIME_PERIOD'] == 2022)]
            
            if hh_2011.empty or hh_2022.empty:
                continue
            
            hh_size_2011 = hh_2011['household_size'].values[0]
            hh_size_2022 = hh_2022['household_size'].values[0]
            
            # Get dwellings per inhabitant for 2011 and 2022
            country_row = df_dw_per_inh[df_dw_per_inh['Country'] == country]
            if country_row.empty:
                continue
            
            dw_per_inh_2011 = country_row['2011'].values[0]
            dw_per_inh_2022 = country_row['2022'].values[0]
            
            if pd.isna(dw_per_inh_2011) or pd.isna(dw_per_inh_2022):
                continue
            
            # Calculate dwellings per 1000 households = (dwellings per inhabitant) * household_size * 1000
            dw_per_1000_hh_2011 = dw_per_inh_2011 * hh_size_2011 * 1000
            dw_per_1000_hh_2022 = dw_per_inh_2022 * hh_size_2022 * 1000
            
            # Get 2005 data
            dw_2005_row = df_2005[df_2005['Country_Name'] == country]
            if dw_2005_row.empty:
                dw_per_1000_hh_2005 = None
            else:
                dw_per_1000_hh_2005 = dw_2005_row['Dw_per_1000_hh_2005'].values[0]
            
            # Get price-to-income ratios
            price_2005 = df_price_yearly[(df_price_yearly['country'] == country) & (df_price_yearly['year'] == 2005)]
            price_2011 = df_price_yearly[(df_price_yearly['country'] == country) & (df_price_yearly['year'] == 2011)]
            price_2022 = df_price_yearly[(df_price_yearly['country'] == country) & (df_price_yearly['year'] == 2022)]
            
            # Period 2005-2011 (6 years)
            if (dw_per_1000_hh_2005 is not None and not pd.isna(dw_per_1000_hh_2005) and 
                not price_2005.empty and not price_2011.empty):
                p2005 = price_2005['ratio'].values[0]
                p2011 = price_2011['ratio'].values[0]
                
                # Annualized changes
                years_0511 = 6
                dw_annual_change_0511 = ((dw_per_1000_hh_2011 - dw_per_1000_hh_2005) / dw_per_1000_hh_2005 * 100) / years_0511
                price_annual_change_0511 = ((p2011 - p2005) / p2005 * 100) / years_0511
                
                countries_data_0511[country] = {
                    'x': dw_annual_change_0511,
                    'y': price_annual_change_0511
                }
            
            # Period 2011-2022 (11 years)
            if not price_2011.empty and not price_2022.empty:
                p2011 = price_2011['ratio'].values[0]
                p2022 = price_2022['ratio'].values[0]
                
                # Annualized changes
                years_1122 = 11
                dw_annual_change_1122 = ((dw_per_1000_hh_2022 - dw_per_1000_hh_2011) / dw_per_1000_hh_2011 * 100) / years_1122
                price_annual_change_1122 = ((p2022 - p2011) / p2011 * 100) / years_1122
                
                countries_data_1122[country] = {
                    'x': dw_annual_change_1122,
                    'y': price_annual_change_1122
                }
        
        if not countries_data_0511 and not countries_data_1122:
            print("    No data available for plotting")
            return
        
        EU_COUNTRIES = {
            'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
            'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
            'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
            'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia',
            'Slovenia', 'Spain', 'Sweden'
        }
        EFTA_COUNTRIES = {'Iceland', 'Norway', 'Switzerland', 'Liechtenstein'}
        EU_EFTA_COUNTRIES = EU_COUNTRIES | EFTA_COUNTRIES

        def _plot_annualized(c0511, c1122, output_name, title_suffix):
            # Create the scatter plot with two data series
            fig, ax = plt.subplots(figsize=(14, 10))

            # Collect data for regressions
            # Period 2005-2011
            x_data_0511 = []
            y_data_0511 = []
            weights_0511 = []

            for country, data in c0511.items():
                x = data['x']
                y = data['y']

                # Get average population for 2005-2011 period
                pop_2005 = populations.get((country, 2005))
                pop_2011 = populations.get((country, 2011))

                if pop_2005 and pop_2011 and not pd.isna(pop_2005) and not pd.isna(pop_2011):
                    avg_pop = (pop_2005 + pop_2011) / 2
                    x_data_0511.append(x)
                    y_data_0511.append(y)
                    weights_0511.append(avg_pop)

                # Plot point for 2005-2011 period (dots only)
                ax.scatter(x, y, s=120, alpha=0.65, color='#e74c3c', edgecolors='none', marker='o', zorder=4)
                ax.annotate(country, (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=7, alpha=0.7)

            # Period 2011-2022
            x_data_1122 = []
            y_data_1122 = []
            weights_1122 = []

            for country, data in c1122.items():
                x = data['x']
                y = data['y']

                # Get average population for 2011-2022 period
                pop_2011 = populations.get((country, 2011))
                pop_2021 = populations.get((country, 2021))

                if pop_2011 and pop_2021 and not pd.isna(pop_2011) and not pd.isna(pop_2021):
                    avg_pop = (pop_2011 + pop_2021) / 2
                    x_data_1122.append(x)
                    y_data_1122.append(y)
                    weights_1122.append(avg_pop)

                # Plot point for 2011-2022 period (dots only)
                ax.scatter(x, y, s=120, alpha=0.65, color='#3498db', edgecolors='none', marker='o', zorder=4)
                ax.annotate(country, (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=7, alpha=0.7)

            # Fix x-axis range to [-2%, 2%]
            ax.set_xlim(-2, 2)

            # Calculate and plot population-weighted regression for 2005-2011
            if len(x_data_0511) > 1 and np.sum(weights_0511) > 0:
                x_arr = np.array(x_data_0511)
                y_arr = np.array(y_data_0511)
                w_arr = np.array(weights_0511)

                w_sum = np.sum(w_arr)
                x_mean_w = np.sum(w_arr * x_arr) / w_sum
                y_mean_w = np.sum(w_arr * y_arr) / w_sum

                numerator = np.sum(w_arr * (x_arr - x_mean_w) * (y_arr - y_mean_w))
                denominator = np.sum(w_arr * (x_arr - x_mean_w) ** 2)

                if denominator > 0:
                    slope_0511 = numerator / denominator
                    intercept_0511 = y_mean_w - slope_0511 * x_mean_w

                    y_pred = slope_0511 * x_arr + intercept_0511
                    ss_res_w = np.sum(w_arr * (y_arr - y_pred) ** 2)
                    ss_tot_w = np.sum(w_arr * (y_arr - y_mean_w) ** 2)
                    r_squared_0511 = 1 - (ss_res_w / ss_tot_w) if ss_tot_w > 0 else 0

                    x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
                    y_line = slope_0511 * x_line + intercept_0511
                    ax.plot(x_line, y_line, color='#e74c3c', linestyle='--', linewidth=2.5, alpha=0.7,
                            label=f'2005-2011 (weighted): y={slope_0511:.3f}x+{intercept_0511:.2f} (R²={r_squared_0511:.3f})')

            # Calculate and plot unweighted regression for 2005-2011
            if len(x_data_0511) > 1:
                x_arr = np.array(x_data_0511)
                y_arr = np.array(y_data_0511)

                x_mean = np.mean(x_arr)
                y_mean = np.mean(y_arr)
                numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
                denominator = np.sum((x_arr - x_mean) ** 2)

                if denominator > 0:
                    slope_0511_unw = numerator / denominator
                    intercept_0511_unw = y_mean - slope_0511_unw * x_mean

                    y_pred = slope_0511_unw * x_arr + intercept_0511_unw
                    ss_res = np.sum((y_arr - y_pred) ** 2)
                    ss_tot = np.sum((y_arr - y_mean) ** 2)
                    r_squared_0511_unw = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
                    y_line = slope_0511_unw * x_line + intercept_0511_unw
                    ax.plot(x_line, y_line, color='#e74c3c', linestyle=':', linewidth=2.5, alpha=0.7,
                            label=f'2005-2011 (unweighted): y={slope_0511_unw:.3f}x+{intercept_0511_unw:.2f} (R²={r_squared_0511_unw:.3f})')

            # Calculate and plot population-weighted regression for 2011-2022
            if len(x_data_1122) > 1 and np.sum(weights_1122) > 0:
                x_arr = np.array(x_data_1122)
                y_arr = np.array(y_data_1122)
                w_arr = np.array(weights_1122)

                w_sum = np.sum(w_arr)
                x_mean_w = np.sum(w_arr * x_arr) / w_sum
                y_mean_w = np.sum(w_arr * y_arr) / w_sum

                numerator = np.sum(w_arr * (x_arr - x_mean_w) * (y_arr - y_mean_w))
                denominator = np.sum(w_arr * (x_arr - x_mean_w) ** 2)

                if denominator > 0:
                    slope_1122 = numerator / denominator
                    intercept_1122 = y_mean_w - slope_1122 * x_mean_w

                    y_pred = slope_1122 * x_arr + intercept_1122
                    ss_res_w = np.sum(w_arr * (y_arr - y_pred) ** 2)
                    ss_tot_w = np.sum(w_arr * (y_arr - y_mean_w) ** 2)
                    r_squared_1122 = 1 - (ss_res_w / ss_tot_w) if ss_tot_w > 0 else 0

                    x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
                    y_line = slope_1122 * x_line + intercept_1122
                    ax.plot(x_line, y_line, color='#3498db', linestyle='--', linewidth=2.5, alpha=0.7,
                            label=f'2011-2022 (weighted): y={slope_1122:.3f}x+{intercept_1122:.2f} (R²={r_squared_1122:.3f})')

            # Calculate and plot unweighted regression for 2011-2022
            if len(x_data_1122) > 1:
                x_arr = np.array(x_data_1122)
                y_arr = np.array(y_data_1122)

                x_mean = np.mean(x_arr)
                y_mean = np.mean(y_arr)
                numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
                denominator = np.sum((x_arr - x_mean) ** 2)

                if denominator > 0:
                    slope_1122_unw = numerator / denominator
                    intercept_1122_unw = y_mean - slope_1122_unw * x_mean

                    y_pred = slope_1122_unw * x_arr + intercept_1122_unw
                    ss_res = np.sum((y_arr - y_pred) ** 2)
                    ss_tot = np.sum((y_arr - y_mean) ** 2)
                    r_squared_1122_unw = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    x_line = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
                    y_line = slope_1122_unw * x_line + intercept_1122_unw
                    ax.plot(x_line, y_line, color='#3498db', linestyle=':', linewidth=2.5, alpha=0.7,
                            label=f'2011-2022 (unweighted): y={slope_1122_unw:.3f}x+{intercept_1122_unw:.2f} (R²={r_squared_1122_unw:.3f})')

            # Add reference lines
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

            # Labels and title
            ax.set_xlabel('Mean Annual Change in Dwellings per Household (%/year)',
                          fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Annual Change in Price-to-Income Index (%/year)',
                          fontsize=12, fontweight='bold')
            ax.set_title('Annualized Changes: Dwellings per Household vs Price-to-Income\n(2005-2011 & 2011-2022)' + title_suffix,
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=9,
                       label='2005-2011', alpha=0.65, markeredgewidth=0),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=9,
                       label='2011-2022', alpha=0.65, markeredgewidth=0)
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

            # Add regression lines to legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=9, framealpha=0.95)

            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, output_name)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    OK Saved: {output_name}")
            plt.close()

            # Print summary
            print(f"    Countries plotted (2005-2011): {len(c0511)}")
            print(f"    Countries plotted (2011-2022): {len(c1122)}")

        # EU + EFTA subset (single output since there are no non-EU/EFTA countries)
        countries_data_0511_eu = {k: v for k, v in countries_data_0511.items() if k in EU_EFTA_COUNTRIES}
        countries_data_1122_eu = {k: v for k, v in countries_data_1122.items() if k in EU_EFTA_COUNTRIES}

        plot_0511 = countries_data_0511_eu if countries_data_0511_eu else countries_data_0511
        plot_1122 = countries_data_1122_eu if countries_data_1122_eu else countries_data_1122
        title_suffix = ' (EU + EFTA)' if (countries_data_0511_eu or countries_data_1122_eu) else ' (All countries)'

        _plot_annualized(
            plot_0511,
            plot_1122,
            '23c_dwellings_per_household_annualized_2005_2022.png',
            title_suffix
        )
        
    except Exception as e:
        print(f"    ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def create_fixed_capital_formation_graphs():
    """Fixed capital formation by asset type analysis"""
    print("\n[Fixed Capital Formation by Asset Type]")
    
    filepath = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_fixed_capital.csv')
    
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  Error loading file: {e}")
        return
    
    print(f"  Loaded {len(df)} records")
    
    # Standardize country names
    df['geo'] = df['geo'].str.strip()
    df['geo'] = df['geo'].replace({
        'European Union - 27 countries (from 2020)': 'EU27',
        'Czechia': 'Czech Republic',
    })
    
    # Filter to percentage of GDP
    df = df[df['unit'] == 'Percentage of gross domestic product (GDP)'].copy()
    df['asset10'] = df['asset10'].astype(str).str.strip()
    
    # Define asset type mapping
    asset_mapping = {
        'Total fixed assets (gross)': 'N11G',
        'Dwellings (gross)': 'N111G',
        'Other buildings and structures (gross)': 'N112G',
        'Transport equipment (gross)': 'N1131G',
        'ICT equipment (gross)': 'N1132G',
        'Other machinery and equipment and weapons systems (gross)': 'N11OG'
    }
    
    # Filter relevant asset types
    df = df[df['asset10'].isin(asset_mapping.keys())].copy()
    df['asset_code'] = df['asset10'].map(asset_mapping)
    
    # Convert TIME_PERIOD to numeric
    df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['TIME_PERIOD', 'OBS_VALUE'])

    if df.empty:
        print("  No data available after filtering; skipping fixed capital formation graphs.")
        return
    
    # Get latest year available
    latest_year = int(df['TIME_PERIOD'].max())
    print(f"  Latest year available: {latest_year}")
    
    # Define colors for categories - using component colors for consistency
    category_colors = {
        'N111G': '#fb8072',    # Dwellings - red
        'N112G': '#80b1d3',    # Other buildings - blue
        'N1131G': '#fdb462',   # Transport - orange
        'N1132G': '#8dd3c7',   # ICT equipment - teal
        'N11OG': '#ffffb3',    # Other machinery & equipment - yellow
        'Other': '#bebada'     # Other - purple
    }
    
    # ====================================================================
    # VIS 1: Area graph for EU-27 over time
    # ====================================================================
    print("\n  Creating area graph for EU-27...")
    
    eu27_data = df[df['geo'] == 'EU27'].copy()
    
    if not eu27_data.empty:
        # Pivot data
        pivot_eu27 = eu27_data.pivot_table(
            index='TIME_PERIOD',
            columns='asset_code',
            values='OBS_VALUE',
            aggfunc='first'
        )
        
        # Check if we have all needed columns
        if 'N11G' in pivot_eu27.columns:
            # Calculate "Other" category
            components = ['N111G', 'N112G', 'N1131G', 'N1132G', 'N11OG']
            available_components = [c for c in components if c in pivot_eu27.columns]
            
            if available_components:
                pivot_eu27['Other'] = pivot_eu27['N11G'] - pivot_eu27[available_components].sum(axis=1)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # Prepare data for stacking
                plot_columns = available_components + ['Other']
                plot_data = pivot_eu27[plot_columns].fillna(0)
                
                # Create stackplot
                colors = [category_colors[col] for col in plot_columns]
                ax.stackplot(plot_data.index, plot_data.T, labels=plot_columns,
                           colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add percentage labels on the areas
                # Calculate midpoints of each area for label placement
                cumulative = np.zeros(len(plot_data))
                for i, col in enumerate(plot_columns):
                    values = plot_data[col].values
                    mid_heights = cumulative + values / 2
                    cumulative += values
                    
                    # Add label at the middle of the time series and middle of the area
                    mid_time_idx = len(plot_data) // 2
                    mid_time = plot_data.index[mid_time_idx]
                    mid_height = mid_heights[mid_time_idx]
                    
                    # Only add label if area is significant enough
                    if values[mid_time_idx] > 0.5:
                        label_text = col.replace('N111G', 'Dwellings').replace('N112G', 'Other Buildings').replace('N1131G', 'Transport').replace('N1132G', 'ICT').replace('N11OG', 'Other Machinery')
                        ax.text(mid_time, mid_height, label_text, 
                              fontsize=10, fontweight='bold', ha='center', va='center',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
                
                ax.set_xlabel('Year', fontsize=12, fontweight='bold')
                ax.set_ylabel('% of GDP', fontsize=12, fontweight='bold')
                ax.set_title('Gross Fixed Capital Formation by Asset Type - EU-27', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                plt.tight_layout()
                output_file = os.path.join(OUTPUT_DIR, 'fixed_capital_formation_eu27.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"    OK Saved: fixed_capital_formation_eu27.png")
                plt.close()
    
    # ====================================================================
    # VIS 2: Area graph for France over time
    # ====================================================================
    print("\n  Creating area graph for France...")
    
    fr_data = df[df['geo'] == 'France'].copy()
    
    if not fr_data.empty:
        # Pivot data
        pivot_fr = fr_data.pivot_table(
            index='TIME_PERIOD',
            columns='asset_code',
            values='OBS_VALUE',
            aggfunc='first'
        )
        
        # Check if we have all needed columns
        if 'N11G' in pivot_fr.columns:
            # Calculate "Other" category
            components = ['N111G', 'N112G', 'N1131G', 'N1132G', 'N11OG']
            available_components = [c for c in components if c in pivot_fr.columns]
            
            if available_components:
                pivot_fr['Other'] = pivot_fr['N11G'] - pivot_fr[available_components].sum(axis=1)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # Prepare data for stacking
                plot_columns = available_components + ['Other']
                plot_data = pivot_fr[plot_columns].fillna(0)
                
                # Create stackplot
                colors = [category_colors[col] for col in plot_columns]
                ax.stackplot(plot_data.index, plot_data.T, labels=plot_columns,
                           colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add percentage labels on the areas
                cumulative = np.zeros(len(plot_data))
                for i, col in enumerate(plot_columns):
                    values = plot_data[col].values
                    mid_heights = cumulative + values / 2
                    cumulative += values
                    
                    mid_time_idx = len(plot_data) // 2
                    mid_time = plot_data.index[mid_time_idx]
                    mid_height = mid_heights[mid_time_idx]
                    
                    if values[mid_time_idx] > 0.5:
                        label_text = col.replace('N111G', 'Dwellings').replace('N112G', 'Other Buildings').replace('N1131G', 'Transport').replace('N1132G', 'ICT').replace('N11OG', 'Other Machinery')
                        ax.text(mid_time, mid_height, label_text, 
                              fontsize=10, fontweight='bold', ha='center', va='center',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
                
                ax.set_xlabel('Year', fontsize=12, fontweight='bold')
                ax.set_ylabel('% of GDP', fontsize=12, fontweight='bold')
                ax.set_title('Gross Fixed Capital Formation by Asset Type - France', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                plt.tight_layout()
                output_file = os.path.join(OUTPUT_DIR, 'fixed_capital_formation_france.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"    OK Saved: fixed_capital_formation_france.png")
                plt.close()
    
    # ====================================================================
    # VIS 3: Area graph for Italy over time
    # ====================================================================
    print("\n  Creating area graph for Italy...")
    
    it_data = df[df['geo'] == 'Italy'].copy()
    
    if not it_data.empty:
        # Pivot data
        pivot_it = it_data.pivot_table(
            index='TIME_PERIOD',
            columns='asset_code',
            values='OBS_VALUE',
            aggfunc='first'
        )
        
        # Check if we have all needed columns
        if 'N11G' in pivot_it.columns:
            # Calculate "Other" category
            components = ['N111G', 'N112G', 'N1131G', 'N1132G', 'N11OG']
            available_components = [c for c in components if c in pivot_it.columns]
            
            if available_components:
                pivot_it['Other'] = pivot_it['N11G'] - pivot_it[available_components].sum(axis=1)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # Prepare data for stacking
                plot_columns = available_components + ['Other']
                plot_data = pivot_it[plot_columns].fillna(0)
                
                # Create stackplot
                colors = [category_colors[col] for col in plot_columns]
                ax.stackplot(plot_data.index, plot_data.T, labels=plot_columns,
                           colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add percentage labels on the areas
                cumulative = np.zeros(len(plot_data))
                for i, col in enumerate(plot_columns):
                    values = plot_data[col].values
                    mid_heights = cumulative + values / 2
                    cumulative += values
                    
                    mid_time_idx = len(plot_data) // 2
                    mid_time = plot_data.index[mid_time_idx]
                    mid_height = mid_heights[mid_time_idx]
                    
                    if values[mid_time_idx] > 0.5:
                        label_text = col.replace('N111G', 'Dwellings').replace('N112G', 'Other Buildings').replace('N1131G', 'Transport').replace('N1132G', 'ICT').replace('N11OG', 'Other Machinery')
                        ax.text(mid_time, mid_height, label_text, 
                              fontsize=10, fontweight='bold', ha='center', va='center',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
                
                ax.set_xlabel('Year', fontsize=12, fontweight='bold')
                ax.set_ylabel('% of GDP', fontsize=12, fontweight='bold')
                ax.set_title('Gross Fixed Capital Formation by Asset Type - Italy', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                
                plt.tight_layout()
                output_file = os.path.join(OUTPUT_DIR, 'fixed_capital_formation_italy.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"    OK Saved: fixed_capital_formation_italy.png")
                plt.close()
    
    # ====================================================================
    # VIS 4: Bar chart comparing countries for 2023
    # ====================================================================
    print(f"\n  Creating country comparison bar chart for 2023...")
    
    data_2023 = df[df['TIME_PERIOD'] == 2023].copy()
    
    if not data_2023.empty:
        # Get major EU countries
        major_countries = ['EU27', 'Germany', 'France', 'Italy', 'Spain',
                          'Netherlands', 'Belgium', 'Austria', 'Sweden', 'Denmark']
        data_2023_major = data_2023[data_2023['geo'].isin(major_countries)].copy()
        
        if not data_2023_major.empty:
            # Pivot data
            pivot_2023 = data_2023_major.pivot_table(
                index='geo',
                columns='asset_code',
                values='OBS_VALUE',
                aggfunc='first'
            )
            
            if 'N11G' in pivot_2023.columns:
                # Calculate "Other" category
                components = ['N111G', 'N112G', 'N1131G', 'N1132G', 'N11OG']
                available_components = [c for c in components if c in pivot_2023.columns]
                
                if available_components:
                    pivot_2023['Other'] = pivot_2023['N11G'] - pivot_2023[available_components].sum(axis=1)
                    
                    # Reorder columns for display
                    plot_columns = available_components + ['Other']
                    pivot_2023 = pivot_2023[plot_columns]
                    
                    # Reorder so EU27 is first, then sort by total
                    order = ['EU27'] + [x for x in major_countries[1:] if x in pivot_2023.index]
                    pivot_2023 = pivot_2023.reindex([x for x in order if x in pivot_2023.index])
                    
                    # Calculate total for sorting (excluding EU27)
                    pivot_2023_temp = pivot_2023.copy()
                    pivot_2023_temp['Total'] = pivot_2023_temp.sum(axis=1)
                    eu27_row = pivot_2023_temp.loc[['EU27']] if 'EU27' in pivot_2023_temp.index else None
                    other_rows = pivot_2023_temp.drop('EU27', errors='ignore').sort_values('Total', ascending=True)
                    
                    if eu27_row is not None:
                        pivot_2023_sorted = pd.concat([other_rows, eu27_row])
                    else:
                        pivot_2023_sorted = other_rows
                    
                    pivot_2023_sorted = pivot_2023_sorted.drop('Total', axis=1)
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    colors = [category_colors[col] for col in plot_columns]
                    
                    pivot_2023_sorted.plot(kind='barh', stacked=True, ax=ax, color=colors,
                                           edgecolor='black', linewidth=0.5)
                    
                    # Add percentage labels on bars
                    for i, country in enumerate(pivot_2023_sorted.index):
                        cumulative = 0
                        for col in plot_columns:
                            value = pivot_2023_sorted.loc[country, col]
                            if pd.notna(value) and value > 0.3:  # Only show if > 0.3% of GDP
                                x_pos = cumulative + value / 2
                                ax.text(x_pos, i, f'{value:.1f}%', 
                                       ha='center', va='center', fontsize=8, 
                                       fontweight='bold', color='white')
                            cumulative += value if pd.notna(value) else 0
                        total_value = pivot_2023_sorted.loc[country].sum()
                        if pd.notna(total_value):
                            ax.text(total_value + 0.2, i, f'{total_value:.1f}%',
                                   ha='left', va='center', fontsize=8, fontweight='bold', color='black')
                    
                    ax.set_xlabel('% of GDP', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Country', fontsize=12, fontweight='bold')
                    ax.set_title(f'Gross Fixed Capital Formation by Asset Type - Country Comparison (2023)', 
                               fontsize=13, fontweight='bold')
                    
                    # Update legend with readable names
                    handles, labels = ax.get_legend_handles_labels()
                    readable_labels = []
                    for label in labels:
                        if label == 'N111G':
                            readable_labels.append('Dwellings')
                        elif label == 'N112G':
                            readable_labels.append('Other Buildings')
                        elif label == 'N1131G':
                            readable_labels.append('Transport Equipment')
                        elif label == 'N1132G':
                            readable_labels.append('ICT Equipment')
                        elif label == 'N11OG':
                            readable_labels.append('Other Machinery & Equipment')
                        else:
                            readable_labels.append(label)
                    ax.legend(handles, readable_labels, title='Asset Type', 
                            bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    plt.tight_layout()
                    output_file = os.path.join(OUTPUT_DIR, f'fixed_capital_formation_countries_2023.png')
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    print(f"    OK Saved: fixed_capital_formation_countries_2023.png")
                    plt.close()
    
    data_2024 = df[df['TIME_PERIOD'] == 2024].copy()

    if not data_2024.empty:
        # Get major EU countries
        major_countries = ['EU27', 'Germany', 'France', 'Italy', 'Spain',
                          'Netherlands', 'Belgium', 'Austria', 'Sweden', 'Denmark']
        data_2024_major = data_2024[data_2024['geo'].isin(major_countries)].copy()
        
        if not data_2024_major.empty:
            # Pivot data
            pivot_2024 = data_2024_major.pivot_table(
                index='geo',
                columns='asset_code',
                values='OBS_VALUE',
                aggfunc='first'
            )
            
            if 'N11G' in pivot_2024.columns:
                # Calculate "Other" category
                components = ['N111G', 'N112G', 'N1131G', 'N1132G', 'N11OG']
                available_components = [c for c in components if c in pivot_2024.columns]
                
                if available_components:
                    pivot_2024['Other'] = pivot_2024['N11G'] - pivot_2024[available_components].sum(axis=1)
                    
                    # Reorder columns for display
                    plot_columns = available_components + ['Other']
                    pivot_2024 = pivot_2024[plot_columns]
                    
                    # Reorder so EU27 is first, then sort by total
                    order = ['EU27'] + [x for x in major_countries[1:] if x in pivot_2024.index]
                    pivot_2024 = pivot_2024.reindex([x for x in order if x in pivot_2024.index])
                    
                    # Calculate total for sorting (excluding EU27)
                    pivot_2024_temp = pivot_2024.copy()
                    pivot_2024_temp['Total'] = pivot_2024_temp.sum(axis=1)
                    eu27_row = pivot_2024_temp.loc[['EU27']] if 'EU27' in pivot_2024_temp.index else None
                    other_rows = pivot_2024_temp.drop('EU27', errors='ignore').sort_values('Total', ascending=True)
                    
                    if eu27_row is not None:
                        pivot_2024_sorted = pd.concat([other_rows, eu27_row])
                    else:
                        pivot_2024_sorted = other_rows
                    
                    pivot_2024_sorted = pivot_2024_sorted.drop('Total', axis=1)
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    colors = [category_colors[col] for col in plot_columns]
                    
                    pivot_2024_sorted.plot(kind='barh', stacked=True, ax=ax, color=colors,
                                           edgecolor='black', linewidth=0.5)
                    
                    # Add percentage labels on bars
                    for i, country in enumerate(pivot_2024_sorted.index):
                        cumulative = 0
                        for col in plot_columns:
                            value = pivot_2024_sorted.loc[country, col]
                            if pd.notna(value) and value > 0.3:  # Only show if > 0.3% of GDP
                                x_pos = cumulative + value / 2
                                ax.text(x_pos, i, f'{value:.1f}%', 
                                       ha='center', va='center', fontsize=8, 
                                       fontweight='bold', color='white')
                            cumulative += value if pd.notna(value) else 0
                        total_value = pivot_2024_sorted.loc[country].sum()
                        if pd.notna(total_value):
                            ax.text(total_value + 0.2, i, f'{total_value:.1f}%',
                                   ha='left', va='center', fontsize=8, fontweight='bold', color='black')
                    
                    ax.set_xlabel('% of GDP', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Country', fontsize=12, fontweight='bold')
                    ax.set_title(f'Gross Fixed Capital Formation by Asset Type - Country Comparison (2024)', 
                               fontsize=13, fontweight='bold')
                    
                    # Update legend with readable names
                    handles, labels = ax.get_legend_handles_labels()
                    readable_labels = []
                    for label in labels:
                        if label == 'N111G':
                            readable_labels.append('Dwellings')
                        elif label == 'N112G':
                            readable_labels.append('Other Buildings')
                        elif label == 'N1131G':
                            readable_labels.append('Transport Equipment')
                        elif label == 'N1132G':
                            readable_labels.append('ICT Equipment')
                        elif label == 'N11OG':
                            readable_labels.append('Other Machinery & Equipment')
                        else:
                            readable_labels.append(label)
                    ax.legend(handles, readable_labels, title='Asset Type', 
                            bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    plt.tight_layout()
                    output_file = os.path.join(OUTPUT_DIR, f'fixed_capital_formation_countries_2024.png')
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    print(f"    OK Saved: fixed_capital_formation_countries_2024.png")
                    plt.close()


def create_capital_stocks_graphs():
    """Capital stocks by sector and asset type (France, Germany, Austria)"""
    print("\n[Capital Stocks by Sector - France, Germany, Austria]")

    filepath = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_capital_stocks.csv')

    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  Error loading file: {e}")
        return

    print(f"  Loaded {len(df)} records")

    # Standardize text fields
    df['geo'] = df['geo'].astype(str).str.strip()
    df['sector'] = df['sector'].astype(str).str.strip()
    df['asset10'] = df['asset10'].astype(str).str.strip()

    # Filter to current prices (million euro)
    df = df[df['unit'] == 'Current prices, million euro'].copy()

    # Filter to target countries and prepare values
    countries = ['France', 'Germany', 'Austria']
    df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['TIME_PERIOD', 'OBS_VALUE'])

    asset_labels = {
        'Dwellings (net)': 'dwellings',
        'Land (net)': 'land'
    }
    total_label = 'Produced non-financial assets (net)'

    # Define sector order and colors
    sector_order = [
        'Non-financial corporations',
        'Financial corporations',
        'General government',
        'Households; non-profit institutions serving households'
    ]
    sector_colors = {
        'Non-financial corporations': '#80b1d3',
        'Financial corporations': '#fdb462',
        'General government': '#bebada',
        'Households; non-profit institutions serving households': '#8dd3c7'
    }

    # Helper to plot stacked area
    def plot_area(data, ylabel, title, filename):
        plot_data = data.fillna(0)
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = [sector_colors.get(col, '#cccccc') for col in plot_data.columns]
        ax.stackplot(plot_data.index, plot_data.T, labels=plot_data.columns,
                    colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    OK Saved: {filename}")
        plt.close()

    gdp_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_gdp_current_price.csv')
    gdp_df = None
    if os.path.exists(gdp_path):
        gdp_df = pd.read_csv(gdp_path)
        gdp_df['geo'] = gdp_df['geo'].astype(str).str.strip()
        gdp_df['TIME_PERIOD'] = pd.to_numeric(gdp_df['TIME_PERIOD'], errors='coerce')
        gdp_df['OBS_VALUE'] = pd.to_numeric(gdp_df['OBS_VALUE'], errors='coerce')
        gdp_df = gdp_df[(gdp_df['unit'] == 'Current prices, million euro') &
                        (gdp_df['na_item'] == 'Gross domestic product at market prices')].copy()
        gdp_df = gdp_df.dropna(subset=['TIME_PERIOD', 'OBS_VALUE'])

    for country in countries:
        country_df = df[df['geo'] == country].copy()
        if country_df.empty:
            print(f"  No data for {country}; skipping.")
            continue

        total_df = country_df[country_df['asset10'] == total_label].copy()
        country_slug = country.lower().replace(' ', '_')

        for asset_label, asset_slug in asset_labels.items():
            asset_df = country_df[country_df['asset10'] == asset_label].copy()

            if asset_df.empty:
                print(f"  No {asset_label} data for {country}; skipping.")
                continue

            # Pivot asset by sector
            pivot_asset = asset_df.pivot_table(
                index='TIME_PERIOD',
                columns='sector',
                values='OBS_VALUE',
                aggfunc='first'
            ).sort_index()

            # Keep desired sector order
            available_sectors = [s for s in sector_order if s in pivot_asset.columns]
            pivot_asset = pivot_asset[available_sectors]

            if pivot_asset.empty:
                print(f"  No {asset_label} data for selected sectors in {country}; skipping.")
                continue

            # 1) Share of N1N (Produced non-financial assets net)
            if not total_df.empty:
                pivot_total = total_df.pivot_table(
                    index='TIME_PERIOD',
                    columns='sector',
                    values='OBS_VALUE',
                    aggfunc='first'
                ).sort_index()

                total_series = pivot_total.get('Total economy')
                if total_series is None:
                    total_series = pivot_total.sum(axis=1, min_count=1)

                share_n11n = pivot_asset.div(total_series, axis=0) * 100
                plot_area(
                    share_n11n,
                    '% of Produced Non-financial Assets (N1N)',
                    f'{country}: {asset_label} Share of Produced Non-financial Assets by Sector',
                    f'capital_stocks_{country_slug}_{asset_slug}_share_n11n.png'
                )
            else:
                print(f"  No Produced non-financial assets (net) data found for {country}; skipping share of N1N graph.")

            # 2) Share of GDP (Current prices)
            if gdp_df is not None:
                gdp_country = gdp_df[gdp_df['geo'] == country].copy()
                if not gdp_country.empty:
                    gdp_series = gdp_country.set_index('TIME_PERIOD')['OBS_VALUE']
                    share_gdp = pivot_asset.div(gdp_series, axis=0) * 100
                    plot_area(
                        share_gdp,
                        '% of GDP',
                        f'{country}: {asset_label} Share of GDP by Sector',
                        f'capital_stocks_{country_slug}_{asset_slug}_share_gdp.png'
                    )
                else:
                    print(f"  No GDP data found for {country}; skipping share of GDP graph.")
            else:
                print(f"  GDP file not found: {gdp_path}")

            # 3) Absolute value (current prices, million euro)
            plot_area(
                pivot_asset,
                'Current prices (million euro)',
                f'{country}: {asset_label} by Sector (Current Prices)',
                f'capital_stocks_{country_slug}_{asset_slug}_absolute.png'
            )


def create_capital_stocks_share_gdp_panels():
    """Capital stocks share of GDP by sector for selected countries (dwellings & land)."""
    print("\n[Capital Stocks Share of GDP Panels]")

    filepath = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_capital_stocks.csv')
    gdp_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_gdp_current_price.csv')

    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return

    if not os.path.exists(gdp_path):
        print(f"  GDP file not found: {gdp_path}")
        return

    try:
        df = pd.read_csv(filepath)
        gdp_df = pd.read_csv(gdp_path)
    except Exception as e:
        print(f"  Error loading files: {e}")
        return

    # Standardize fields
    df['geo'] = df['geo'].astype(str).str.strip()
    df['sector'] = df['sector'].astype(str).str.strip()
    df['asset10'] = df['asset10'].astype(str).str.strip()
    df['TIME_PERIOD'] = pd.to_numeric(df['TIME_PERIOD'], errors='coerce')
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['TIME_PERIOD', 'OBS_VALUE'])

    gdp_df['geo'] = gdp_df['geo'].astype(str).str.strip()
    gdp_df['TIME_PERIOD'] = pd.to_numeric(gdp_df['TIME_PERIOD'], errors='coerce')
    gdp_df['OBS_VALUE'] = pd.to_numeric(gdp_df['OBS_VALUE'], errors='coerce')
    gdp_df = gdp_df[(gdp_df['unit'] == 'Current prices, million euro') &
                    (gdp_df['na_item'] == 'Gross domestic product at market prices')].copy()
    gdp_df = gdp_df.dropna(subset=['TIME_PERIOD', 'OBS_VALUE'])

    # Filter capital stocks to current prices (million euro)
    df = df[df['unit'] == 'Current prices, million euro'].copy()

    countries = ['Austria', 'Spain', 'Sweden', 'France', 'Germany', 'Czechia']
    country_display = {
        'Austria': 'Austria',
        'Spain': 'Spain',
        'Sweden': 'Sweden',
        'France': 'France',
        'Germany': 'Germany',
        'Czechia': 'Czechia'
    }

    sector_order = [
        'Non-financial corporations',
        'Financial corporations',
        'General government',
        'Households; non-profit institutions serving households'
    ]
    sector_colors = {
        'Non-financial corporations': '#80b1d3',
        'Financial corporations': '#fdb462',
        'General government': '#bebada',
        'Households; non-profit institutions serving households': '#8dd3c7'
    }

    def build_share_gdp_series(asset_label):
        series_by_country = {}
        min_year = 1990
        max_year = None
        global_max = 0

        for country in countries:
            country_df = df[(df['geo'] == country) & (df['asset10'] == asset_label)].copy()
            if country_df.empty:
                print(f"  No {asset_label} data for {country}; skipping.")
                continue

            pivot_asset = country_df.pivot_table(
                index='TIME_PERIOD',
                columns='sector',
                values='OBS_VALUE',
                aggfunc='first'
            ).sort_index()

            pivot_asset = pivot_asset.reindex(columns=sector_order)

            if pivot_asset.empty:
                print(f"  No {asset_label} sector data for {country}; skipping.")
                continue

            gdp_country = gdp_df[gdp_df['geo'] == country].copy()
            if gdp_country.empty:
                print(f"  No GDP data for {country}; skipping.")
                continue

            gdp_series = gdp_country.set_index('TIME_PERIOD')['OBS_VALUE']
            share_gdp = pivot_asset.div(gdp_series, axis=0) * 100

            # Do not trace zeros
            share_gdp = share_gdp.replace(0, np.nan)

            if share_gdp.empty:
                print(f"  No share of GDP data for {country}; skipping.")
                continue

            series_by_country[country] = share_gdp

            if not share_gdp.index.empty:
                max_year = int(share_gdp.index.max()) if max_year is None else max(max_year, int(share_gdp.index.max()))
            current_max = share_gdp.sum(axis=1, min_count=1).max()
            if pd.notna(current_max):
                global_max = max(global_max, current_max)

        if max_year is None:
            max_year = min_year

        return series_by_country, min_year, max_year, global_max

    def plot_panel(asset_label, filename, title_prefix, ylabel, start_year=None):
        data_by_country, min_year, max_year, global_max = build_share_gdp_series(asset_label)
        if not data_by_country:
            print(f"  No data available for {asset_label}; skipping panel.")
            return

        if start_year is not None:
            min_year = start_year

        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=True)
        axes = axes.flatten()

        y_max = global_max * 1.1 if global_max > 0 else 1

        for idx, country in enumerate(countries):
            ax = axes[idx]
            if country not in data_by_country:
                ax.axis('off')
                continue

            plot_data = data_by_country[country]
            plot_data = plot_data[plot_data.notna().any(axis=1)].fillna(0)
            colors = [sector_colors.get(col, '#cccccc') for col in plot_data.columns]
            ax.stackplot(plot_data.index, plot_data.T,
                         labels=plot_data.columns,
                         colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_title(country_display.get(country, country), fontsize=11, fontweight='bold')
            ax.set_xlim(min_year, max_year)
            ax.set_ylim(0, y_max)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title_prefix, fontsize=16, fontweight='bold', y=0.92)
        fig.text(0.5, 0.06, 'Year', ha='center', fontsize=12, fontweight='bold')
        fig.text(0.06, 0.5, ylabel, va='center', rotation='vertical', fontsize=12, fontweight='bold')

        # Use legend from first available subplot (placed to the right, aligned with top row)
        legend_added = False
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_labels = [
                    'Household and non-profit Institutions'
                    if label == 'Households; non-profit institutions serving households'
                    else label
                    for label in labels
                ]
                fig.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(0.88, 0.84),
                           ncol=1, fontsize=9, framealpha=0.95)
                legend_added = True
                break

        right_margin = 0.88 if legend_added else 1
        plt.tight_layout(rect=[0.08, 0.08, right_margin, 0.9])
        output_file = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    OK Saved: {filename}")
        plt.close()

    plot_panel(
        'Dwellings (net)',
        'capital_stocks_dwellings_share_gdp_6countries.png',
        'Dwellings capital stock: Share of GDP by sector',
        '% of GDP',
        start_year=1995
    )

    plot_panel(
        'Land (net)',
        'capital_stocks_land_share_gdp_6countries.png',
        'Land capital stock: Share of GDP by sector',
        '% of GDP'
    )


def main():
    """Generate all Eurostat visualizations"""
    print("=" * 70)
    print("EUROSTAT Analysis - Housing Data Visualizations")
    print("=" * 70)
    
    print(f"\nData location: {EXTERNAL_DATA_DIR}")
    print(f"Output location: {OUTPUT_DIR}")
    
    print("\nGenerating visualizations...")
    create_real_estate_graphs()
    create_rooms_graphs()
    create_energy_efficiency_graphs()
    create_berd_graphs()
    create_hrst_graphs()
    create_berd_timeseries()
    create_france_berd_timeseries()
    create_under_occupied_dwellings_graphs()
    create_renting_difficulties_graphs()
    create_building_permits_graphs()
    create_building_permits_demographic_graphs()
    create_building_permits_specific_countries_demographic()
    create_building_permits_growth_scatter()
    create_dwellings_vs_price_to_income()
    create_dwellings_per_household_vs_price_to_income()
    create_dwellings_per_household_2011_2022()
    create_dwellings_per_household_annualized_2005_2022()
    create_raw_housing_indicators_vs_price()
    create_tenure_status_graphs()
    create_house_sales_graphs()
    create_air_emissions_graphs()
    create_government_expenditure_housing_graphs()
    create_fixed_capital_formation_graphs()
    create_capital_stocks_graphs()
    create_capital_stocks_share_gdp_panels()
    
    print("\n" + "=" * 70)
    print("All visualizations completed!")
    print("=" * 70)


def create_government_expenditure_housing_graphs():
    """Government expenditure on housing analysis"""
    print("\n[Government Expenditure on Housing]")
    
    filepath = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_government expenditure housing.csv')
    
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  Error loading file: {e}")
        return
    
    print(f"  Loaded {len(df)} records")
    
    # Standardize country names
    df['geo'] = df['geo'].str.strip()
    df['geo'] = df['geo'].replace({
        'European Union - 27 countries (from 2020)': 'EU27',
        'Czechia': 'Czech Republic',
    })
    
    # Colors for categories - using component colors for consistency
    category_colors = {
        'Housing social protection': '#fb8072',        # Red
        'Housing development': '#80b1d3',              # Blue
        'Community development': '#fdb462',            # Orange
        'Water supply': '#bebada',                     # Purple
        'Street lighting': '#8dd3c7',                  # Teal
        'R&D Housing and community amenities': '#ffffb3',  # Yellow
        'Housing and community amenities n.e.c.': '#a6d854'  # Green
    }
    
    # Define category order for stacking (bottom to top) - excludes the aggregate category
    category_order = [
        'Housing social protection',
        'Housing development',
        'Community development',
        'Water supply',
        'Street lighting',
        'R&D Housing and community amenities',
        'Housing and community amenities n.e.c.'
    ]
    
    # Filter data for GDP percentage
    df_filtered = df[df['unit'] == 'Percentage of gross domestic product (GDP)'].copy()
    df_filtered['TIME_PERIOD'] = pd.to_numeric(df_filtered['TIME_PERIOD'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['TIME_PERIOD', 'OBS_VALUE'])
    
    # Rename "Housing" to "Housing social protection"
    df_filtered['cofog99'] = df_filtered['cofog99'].replace({'Housing': 'Housing social protection'})
    
    # Get Total values for % of total calculation
    df_total = df[df['cofog99'] == 'Total'].copy()
    df_total['TIME_PERIOD'] = pd.to_numeric(df_total['TIME_PERIOD'], errors='coerce')
    df_total = df_total.dropna(subset=['TIME_PERIOD', 'OBS_VALUE'])
    
    # Helper function to reorder pivot table columns
    def reorder_columns(pivot_table, category_order):
        """Reorder pivot table columns according to category_order, handling missing categories."""
        cols = [col for col in category_order if col in pivot_table.columns]
        return pivot_table[cols]
    
    # ====================================================================
    # VIS 1: EU Area graph - % of GDP across time
    # ====================================================================
    print("\n  Creating area graph - EU % of GDP...")
    
    # Get EU27 data
    eu_data = df_filtered[(df_filtered['geo'] == 'EU27')].copy()
    
    if not eu_data.empty:
        # Pivot for area chart
        pivot_gdp = eu_data.pivot_table(
            index='TIME_PERIOD',
            columns='cofog99',
            values='OBS_VALUE',
            aggfunc='first'
        )
        pivot_gdp = reorder_columns(pivot_gdp, category_order)
        pivot_gdp = pivot_gdp.sort_index()
        
        if not pivot_gdp.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            colors = [category_colors.get(cat, '#cccccc') for cat in pivot_gdp.columns]
            ax.stackplot(pivot_gdp.index, pivot_gdp.T, labels=pivot_gdp.columns,
                        colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('% of GDP', fontsize=12, fontweight='bold')
            ax.set_title('EU27: Government Expenditure on Housing and Community Amenities (% of GDP)', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, 'government_expenditure_housing_eu_area_gdp.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    OK Saved: government_expenditure_housing_eu_area_gdp.png")
            plt.close()
    else:
        print("    No EU27 data found")
    
    # ====================================================================
    # VIS 2: EU Area graph - % of total government expenditure across time
    # ====================================================================
    print("\n  Creating area graph - EU % of total expenditure...")
    
    # Filter for EU27
    eu_total = df_total[df_total['geo'] == 'EU27'][['TIME_PERIOD', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'total_value'})
    
    if not eu_data.empty and not eu_total.empty:
        # Calculate percentage of total
        eu_data_calc = eu_data.copy().merge(eu_total, on='TIME_PERIOD', how='left')
        eu_data_calc['pct_total'] = (eu_data_calc['OBS_VALUE'] / eu_data_calc['total_value']) * 100
        
        pivot_pct = eu_data_calc.pivot_table(
            index='TIME_PERIOD',
            columns='cofog99',
            values='pct_total',
            aggfunc='first'
        )
        pivot_pct = reorder_columns(pivot_pct, category_order)
        pivot_pct = pivot_pct.sort_index()
        
        if not pivot_pct.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            colors = [category_colors.get(cat, '#cccccc') for cat in pivot_pct.columns]
            ax.stackplot(pivot_pct.index, pivot_pct.T, labels=pivot_pct.columns,
                        colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('% of Total Government Expenditure', fontsize=12, fontweight='bold')
            ax.set_title('EU27: Government Expenditure on Housing and Community Amenities (% of Total)', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, 'government_expenditure_housing_eu_area_total.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    OK Saved: government_expenditure_housing_eu_area_total.png")
            plt.close()
    else:
        print("    No EU27 total data found")
    
    # ====================================================================
    # VIS 3: France Area graph - % of GDP across time
    # ====================================================================
    print("\n  Creating area graph - France % of GDP...")
    
    # Get France data
    fr_data = df_filtered[(df_filtered['geo'] == 'France')].copy()
    
    if not fr_data.empty:
        # Pivot for area chart
        pivot_gdp_fr = fr_data.pivot_table(
            index='TIME_PERIOD',
            columns='cofog99',
            values='OBS_VALUE',
            aggfunc='first'
        )
        pivot_gdp_fr = reorder_columns(pivot_gdp_fr, category_order)
        pivot_gdp_fr = pivot_gdp_fr.sort_index()
        
        if not pivot_gdp_fr.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            colors = [category_colors.get(cat, '#cccccc') for cat in pivot_gdp_fr.columns]
            ax.stackplot(pivot_gdp_fr.index, pivot_gdp_fr.T, labels=pivot_gdp_fr.columns,
                        colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('% of GDP', fontsize=12, fontweight='bold')
            ax.set_title('France: Government Expenditure on Housing and Community Amenities (% of GDP)', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, 'government_expenditure_housing_france_area_gdp.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    OK Saved: government_expenditure_housing_france_area_gdp.png")
            plt.close()
    else:
        print("    No France data found")

    # ====================================================================
    # VIS 3b: 6-country panel - % of GDP across time (France colors)
    # ====================================================================
    print("\n  Creating 6-country panel - % of GDP...")

    panel_countries = ['Austria', 'Spain', 'Sweden', 'France', 'Germany', 'Italy']
    panel_display = {
        'Austria': 'Austria',
        'Spain': 'Spain',
        'Sweden': 'Sweden',
        'France': 'France',
        'Germany': 'Germany',
        'Italy': 'Italy'
    }

    panel_data = {}
    panel_min_year = None
    panel_max_year = None
    panel_global_max = 0

    for country in panel_countries:
        country_data = df_filtered[df_filtered['geo'] == country].copy()
        if country_data.empty:
            print(f"    No {country} data found for panel")
            continue

        pivot_country = country_data.pivot_table(
            index='TIME_PERIOD',
            columns='cofog99',
            values='OBS_VALUE',
            aggfunc='first'
        )
        pivot_country = reorder_columns(pivot_country, category_order)
        pivot_country = pivot_country.sort_index()
        zero_rows = pivot_country.fillna(0).sum(axis=1) == 0
        pivot_country.loc[zero_rows] = np.nan

        if pivot_country.empty:
            print(f"    No usable {country} data for panel")
            continue

        panel_data[country] = pivot_country

        if panel_min_year is None or int(pivot_country.index.min()) < panel_min_year:
            panel_min_year = int(pivot_country.index.min())
        if panel_max_year is None or int(pivot_country.index.max()) > panel_max_year:
            panel_max_year = int(pivot_country.index.max())

        current_max = pivot_country.sum(axis=1, min_count=1).max()
        if pd.notna(current_max):
            panel_global_max = max(panel_global_max, current_max)

    if panel_data:
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=True)
        axes = axes.flatten()

        y_max = panel_global_max * 1.1 if panel_global_max > 0 else 1

        for idx, country in enumerate(panel_countries):
            ax = axes[idx]
            if country not in panel_data:
                ax.axis('off')
                continue

            plot_data = panel_data[country]
            colors = [category_colors.get(cat, '#cccccc') for cat in plot_data.columns]
            ax.stackplot(plot_data.index, plot_data.T, labels=plot_data.columns,
                         colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_title(panel_display.get(country, country), fontsize=11, fontweight='bold')
            ax.set_xlim(panel_min_year, panel_max_year)
            ax.set_ylim(0, y_max)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Government Expenditure on Housing and Community Amenities (% of GDP)',
                     fontsize=16, fontweight='bold', y=0.92)
        fig.text(0.5, 0.06, 'Year', ha='center', fontsize=12, fontweight='bold')
        fig.text(0.06, 0.5, '% of GDP', va='center', rotation='vertical', fontsize=12, fontweight='bold')

        legend_added = False
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.88, 0.88),
                           ncol=1, fontsize=9, framealpha=0.95)
                legend_added = True
                break

        right_margin = 0.88 if legend_added else 1
        plt.tight_layout(rect=[0.08, 0.08, right_margin, 0.9])
        output_file = os.path.join(OUTPUT_DIR, 'government_expenditure_housing_gdp_panel_6countries.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print("    OK Saved: government_expenditure_housing_gdp_panel_6countries.png")
        plt.close()
    else:
        print("    No data available for 6-country panel")
    
    # ====================================================================
    # VIS 4: France Area graph - % of total government expenditure across time
    # ====================================================================
    print("\n  Creating area graph - France % of total expenditure...")
    
    # Filter for France
    fr_total = df_total[df_total['geo'] == 'France'][['TIME_PERIOD', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'total_value'})
    
    if not fr_data.empty and not fr_total.empty:
        # Calculate percentage of total
        fr_data_calc = fr_data.copy().merge(fr_total, on='TIME_PERIOD', how='left')
        fr_data_calc['pct_total'] = (fr_data_calc['OBS_VALUE'] / fr_data_calc['total_value']) * 100
        
        pivot_pct_fr = fr_data_calc.pivot_table(
            index='TIME_PERIOD',
            columns='cofog99',
            values='pct_total',
            aggfunc='first'
        )
        pivot_pct_fr = reorder_columns(pivot_pct_fr, category_order)
        pivot_pct_fr = pivot_pct_fr.sort_index()
        
        if not pivot_pct_fr.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            colors = [category_colors.get(cat, '#cccccc') for cat in pivot_pct_fr.columns]
            ax.stackplot(pivot_pct_fr.index, pivot_pct_fr.T, labels=pivot_pct_fr.columns,
                        colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('% of Total Government Expenditure', fontsize=12, fontweight='bold')
            ax.set_title('France: Government Expenditure on Housing and Community Amenities (% of Total)', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, 'government_expenditure_housing_france_area_total.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    OK Saved: government_expenditure_housing_france_area_total.png")
            plt.close()
    else:
        print("    No France total data found")
    
    # ====================================================================
    # VIS 5: Country comparison - % of GDP (2022) - Horizontal bars
    # ====================================================================
    print("\n  Creating country comparison - % of GDP (2022, horizontal bars)...")
    
    year_2022 = 2022.0
    data_2022 = df_filtered[df_filtered['TIME_PERIOD'] == year_2022].copy()
    
    if not data_2022.empty:
        # Get major EU27 countries + EU27 aggregate
        major_countries = ['EU27', 'Germany', 'France', 'Italy', 'Spain',
                          'Netherlands', 'Belgium', 'Austria', 'Sweden', 'Denmark']
        data_2022_major = data_2022[data_2022['geo'].isin(major_countries)].copy()
        
        if not data_2022_major.empty:
            pivot_countries_gdp = data_2022_major.pivot_table(
                index='geo',
                columns='cofog99',
                values='OBS_VALUE',
                aggfunc='first'
            )
            pivot_countries_gdp = reorder_columns(pivot_countries_gdp, category_order)
            
            # Reorder so EU27 is first
            order = ['EU27'] + [x for x in major_countries[1:] if x in pivot_countries_gdp.index]
            pivot_countries_gdp = pivot_countries_gdp.reindex(order)
            
            # Calculate total for sorting
            pivot_countries_gdp['Total'] = pivot_countries_gdp.sum(axis=1)
            pivot_countries_gdp = pivot_countries_gdp.sort_values('Total', ascending=True)
            pivot_countries_gdp = pivot_countries_gdp.drop('Total', axis=1)
            
            if not pivot_countries_gdp.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                colors = [category_colors.get(cat, '#cccccc') for cat in pivot_countries_gdp.columns]
                
                pivot_countries_gdp.plot(kind='barh', stacked=True, ax=ax, color=colors,
                                        edgecolor='black', linewidth=0.5)
                
                ax.set_xlabel('% of GDP', fontsize=12, fontweight='bold')
                ax.set_ylabel('Country', fontsize=12, fontweight='bold')
                ax.set_title(f'Government Expenditure on Housing and Community Amenities by Country - % of GDP (2022)', 
                            fontsize=13, fontweight='bold')
                ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                output_file = os.path.join(OUTPUT_DIR, 'government_expenditure_housing_countries_gdp_2022.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"    OK Saved: government_expenditure_housing_countries_gdp_2022.png")
                plt.close()
    
    # ====================================================================
    # VIS 6: Country comparison - % of total government expenditure (2022) - Horizontal bars
    # ====================================================================
    print("\n  Creating country comparison - % of total expenditure (2022, horizontal bars)...")
    
    if not data_2022.empty and not df_total.empty:
        # Get totals for 2022
        total_2022 = df_total[df_total['TIME_PERIOD'] == year_2022].copy()
        total_2022_major = total_2022[total_2022['geo'].isin(major_countries)].copy()
        
        if not total_2022_major.empty:
            # Merge and calculate percentage
            data_2022_calc = data_2022_major.copy().merge(
                total_2022_major[['geo', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'total_value'}),
                on='geo',
                how='left'
            )
            data_2022_calc = data_2022_calc.dropna(subset=['total_value'])
            data_2022_calc['pct_total'] = (data_2022_calc['OBS_VALUE'] / data_2022_calc['total_value']) * 100
            
            pivot_countries_pct = data_2022_calc.pivot_table(
                index='geo',
                columns='cofog99',
                values='pct_total',
                aggfunc='first'
            )
            pivot_countries_pct = reorder_columns(pivot_countries_pct, category_order)
            
            # Reorder so EU27 is first
            order = ['EU27'] + [x for x in major_countries[1:] if x in pivot_countries_pct.index]
            pivot_countries_pct = pivot_countries_pct.reindex(order)
            
            # Calculate total for sorting
            pivot_countries_pct['Total'] = pivot_countries_pct.sum(axis=1)
            pivot_countries_pct = pivot_countries_pct.sort_values('Total', ascending=True)
            pivot_countries_pct = pivot_countries_pct.drop('Total', axis=1)
            
            if not pivot_countries_pct.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                colors = [category_colors.get(cat, '#cccccc') for cat in pivot_countries_pct.columns]
                
                pivot_countries_pct.plot(kind='barh', stacked=True, ax=ax, color=colors,
                                        edgecolor='black', linewidth=0.5)
                
                ax.set_xlabel('% of Total Government Expenditure', fontsize=12, fontweight='bold')
                ax.set_ylabel('Country', fontsize=12, fontweight='bold')
                ax.set_title(f'Government Expenditure on Housing and Community Amenities by Country - % of Total (2022)', 
                            fontsize=13, fontweight='bold')
                ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                output_file = os.path.join(OUTPUT_DIR, 'government_expenditure_housing_countries_total_2022.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"    OK Saved: government_expenditure_housing_countries_total_2022.png")
                plt.close()
        else:
            print("    No total data for major countries in 2022")
    
    # ====================================================================
    # VIS 7: Germany Area graph - % of GDP across time
    # ====================================================================
    print("\n  Creating area graph - Germany % of GDP...")
    
    # Get Germany data
    de_data = df_filtered[(df_filtered['geo'] == 'Germany')].copy()
    
    if not de_data.empty:
        # Pivot for area chart
        pivot_gdp_de = de_data.pivot_table(
            index='TIME_PERIOD',
            columns='cofog99',
            values='OBS_VALUE',
            aggfunc='first'
        )
        pivot_gdp_de = reorder_columns(pivot_gdp_de, category_order)
        pivot_gdp_de = pivot_gdp_de.sort_index()
        
        if not pivot_gdp_de.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            colors = [category_colors.get(cat, '#cccccc') for cat in pivot_gdp_de.columns]
            ax.stackplot(pivot_gdp_de.index, pivot_gdp_de.T, labels=pivot_gdp_de.columns,
                        colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('% of GDP', fontsize=12, fontweight='bold')
            ax.set_title('Germany: Government Expenditure on Housing and Community Amenities (% of GDP)', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, 'government_expenditure_housing_germany_area_gdp.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    OK Saved: government_expenditure_housing_germany_area_gdp.png")
            plt.close()
    else:
        print("    No Germany data found")
    
    # ====================================================================
    # VIS 8: Germany Area graph - % of total government expenditure across time
    # ====================================================================
    print("\n  Creating area graph - Germany % of total expenditure...")
    
    # Filter for Germany
    de_total = df_total[df_total['geo'] == 'Germany'][['TIME_PERIOD', 'OBS_VALUE']].rename(columns={'OBS_VALUE': 'total_value'})
    
    if not de_data.empty and not de_total.empty:
        # Calculate percentage of total
        de_data_calc = de_data.copy().merge(de_total, on='TIME_PERIOD', how='left')
        de_data_calc['pct_total'] = (de_data_calc['OBS_VALUE'] / de_data_calc['total_value']) * 100
        
        pivot_pct_de = de_data_calc.pivot_table(
            index='TIME_PERIOD',
            columns='cofog99',
            values='pct_total',
            aggfunc='first'
        )
        pivot_pct_de = reorder_columns(pivot_pct_de, category_order)
        pivot_pct_de = pivot_pct_de.sort_index()
        
        if not pivot_pct_de.empty:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            colors = [category_colors.get(cat, '#cccccc') for cat in pivot_pct_de.columns]
            ax.stackplot(pivot_pct_de.index, pivot_pct_de.T, labels=pivot_pct_de.columns,
                        colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('% of Total Government Expenditure', fontsize=12, fontweight='bold')
            ax.set_title('Germany: Government Expenditure on Housing and Community Amenities (% of Total)', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, 'government_expenditure_housing_germany_area_total.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    OK Saved: government_expenditure_housing_germany_area_total.png")
            plt.close()
    else:
        print("    No Germany total data found")
    
    # ====================================================================
    # VIS 9: France vs Germany comparison - % of GDP (same y-axis scale)
    # ====================================================================
    print("\n  Creating France vs Germany comparison - % of GDP (aligned axes)...")
    
    if not fr_data.empty and not de_data.empty:
        # Create side-by-side plots with same y-axis scale
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10.5))
        
        # Pivot for both countries
        pivot_gdp_fr_comp = fr_data.pivot_table(
            index='TIME_PERIOD',
            columns='cofog99',
            values='OBS_VALUE',
            aggfunc='first'
        )
        pivot_gdp_fr_comp = reorder_columns(pivot_gdp_fr_comp, category_order)
        pivot_gdp_fr_comp = pivot_gdp_fr_comp.sort_index()
        
        pivot_gdp_de_comp = de_data.pivot_table(
            index='TIME_PERIOD',
            columns='cofog99',
            values='OBS_VALUE',
            aggfunc='first'
        )
        pivot_gdp_de_comp = reorder_columns(pivot_gdp_de_comp, category_order)
        pivot_gdp_de_comp = pivot_gdp_de_comp.sort_index()
        
        # Get max value from France for consistent scale
        max_fr = pivot_gdp_fr_comp.sum(axis=1).max() if not pivot_gdp_fr_comp.empty else 0
        y_max = max_fr * 1.1  # 10% margin
        
        # France plot
        if not pivot_gdp_fr_comp.empty:
            colors = [category_colors.get(cat, '#cccccc') for cat in pivot_gdp_fr_comp.columns]
            ax1.stackplot(pivot_gdp_fr_comp.index, pivot_gdp_fr_comp.T, labels=pivot_gdp_fr_comp.columns,
                         colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax1.set_ylabel('% of GDP', fontsize=12, fontweight='bold')
            ax1.set_title('France', fontsize=12, fontweight='bold')
            ax1.set_xlim(2000, 2022)
            ax1.set_ylim(0, y_max)
            ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax1.grid(True, alpha=0.3)
        
        # Germany plot
        if not pivot_gdp_de_comp.empty:
            colors = [category_colors.get(cat, '#cccccc') for cat in pivot_gdp_de_comp.columns]
            ax2.stackplot(pivot_gdp_de_comp.index, pivot_gdp_de_comp.T, labels=pivot_gdp_de_comp.columns,
                         colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax2.set_ylabel('% of GDP', fontsize=12, fontweight='bold')
            ax2.set_title('Germany', fontsize=12, fontweight='bold')
            ax2.set_xlim(2000, 2022)
            ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax2.set_ylim(0, y_max)
            ax2.legend(loc='upper left', fontsize=9, framealpha=0.95)
            ax2.grid(True, alpha=0.3)
        
        # Add figure title at top
        fig.suptitle('Government Expenditure on Housing and Community Amenities (% of GDP)', 
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_file = os.path.join(OUTPUT_DIR, 'government_expenditure_housing_france_vs_germany_gdp.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    OK Saved: government_expenditure_housing_france_vs_germany_gdp.png")
        plt.close()
    else:
        print("    Missing France or Germany data for comparison")
    
    # ====================================================================
    # VIS 10: France vs Germany comparison - % of total (same y-axis scale)
    # ====================================================================
    print("\n  Creating France vs Germany comparison - % of total (aligned axes)...")
    
    if not fr_data.empty and not de_data.empty and not fr_total.empty and not de_total.empty:
        # Calculate percentage of total for both countries
        fr_data_calc_comp = fr_data.copy().merge(fr_total, on='TIME_PERIOD', how='left')
        fr_data_calc_comp['pct_total'] = (fr_data_calc_comp['OBS_VALUE'] / fr_data_calc_comp['total_value']) * 100
        
        de_data_calc_comp = de_data.copy().merge(de_total, on='TIME_PERIOD', how='left')
        de_data_calc_comp['pct_total'] = (de_data_calc_comp['OBS_VALUE'] / de_data_calc_comp['total_value']) * 100
        
        # Pivot for both countries
        pivot_pct_fr_comp = fr_data_calc_comp.pivot_table(
            index='TIME_PERIOD',
            columns='cofog99',
            values='pct_total',
            aggfunc='first'
        )
        pivot_pct_fr_comp = reorder_columns(pivot_pct_fr_comp, category_order)
        pivot_pct_fr_comp = pivot_pct_fr_comp.sort_index()
        
        pivot_pct_de_comp = de_data_calc_comp.pivot_table(
            index='TIME_PERIOD',
            columns='cofog99',
            values='pct_total',
            aggfunc='first'
        )
        pivot_pct_de_comp = reorder_columns(pivot_pct_de_comp, category_order)
        pivot_pct_de_comp = pivot_pct_de_comp.sort_index()
        
        # Get max value from both countries for consistent scale
        max_fr_pct = pivot_pct_fr_comp.sum(axis=1).max() if not pivot_pct_fr_comp.empty else 0
        max_de_pct = pivot_pct_de_comp.sum(axis=1).max() if not pivot_pct_de_comp.empty else 0
        y_max_pct = max(max_fr_pct, max_de_pct) * 1.1  # 10% margin
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # France plot
        if not pivot_pct_fr_comp.empty:
            colors = [category_colors.get(cat, '#cccccc') for cat in pivot_pct_fr_comp.columns]
            ax1.stackplot(pivot_pct_fr_comp.index, pivot_pct_fr_comp.T, labels=pivot_pct_fr_comp.columns,
                         colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax1.set_ylabel('% of Total Government Expenditure', fontsize=12, fontweight='bold')
            ax1.set_title('France', fontsize=12, fontweight='bold')
            ax1.set_ylim(0, y_max_pct)
            ax1.grid(True, alpha=0.3)
        
        # Germany plot
        if not pivot_pct_de_comp.empty:
            colors = [category_colors.get(cat, '#cccccc') for cat in pivot_pct_de_comp.columns]
            ax2.stackplot(pivot_pct_de_comp.index, pivot_pct_de_comp.T, labels=pivot_pct_de_comp.columns,
                         colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax2.set_ylabel('% of Total Government Expenditure', fontsize=12, fontweight='bold')
            ax2.set_title('Germany', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, y_max_pct)
            ax2.legend(loc='upper left', fontsize=9, framealpha=0.95)
            ax2.grid(True, alpha=0.3)
        
        # Add figure title at top
        fig.suptitle('Government Expenditure on Housing and Community Amenities (% of Total)', 
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_file = os.path.join(OUTPUT_DIR, 'government_expenditure_housing_france_vs_germany_total.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    OK Saved: government_expenditure_housing_france_vs_germany_total.png")
        plt.close()
    else:
        print("    Missing data for France vs Germany comparison")

if __name__ == '__main__':
    main()
