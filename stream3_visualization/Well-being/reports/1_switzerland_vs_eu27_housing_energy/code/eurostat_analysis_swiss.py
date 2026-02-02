"""
EUROSTAT Analysis for Switzerland - Generate graphs from Eurostat housing data
This script creates visualizations adapted from the EU analysis, with visual styling consistent with oecd_graphs_generator.py
Focus on Switzerland vs EU27 comparison
Outputs saved as PNG files in outputs/graphs/EUROSTAT folder
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import mapclassify
    MAPCLASSIFY_AVAILABLE = True
except ImportError:
    MAPCLASSIFY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Base paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'graphs', 'EUROSTAT')

# External data location - use the EU analysis datasets
EU_ANALYSIS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../3_eu_analysis_with_examples'))
EXTERNAL_DATA_DIR = os.path.join(EU_ANALYSIS_DIR, 'external_data')

# Add code directory to path to import plot_functions
sys.path.insert(0, CURRENT_DIR)

# Import plot_functions
try:
    from plot_functions import plot_europe_map
    PLOT_FUNCTIONS_AVAILABLE = True
except ImportError:
    PLOT_FUNCTIONS_AVAILABLE = False

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette - consistent with oecd_graphs_generator.py
SWITZERLAND_COLOR = '#ffd558'  # Yellow for Switzerland
EU_AGGREGATE_COLOR = '#80b1d3'  # Blue for EU27
COMPONENT_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3', 
                    '#80b1d3', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']

# EU27 and target countries
EU27_COUNTRIES = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
    'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
    'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
    'Spain', 'Sweden'
]

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

# Create comprehensive color map for all countries
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
COUNTRY_COLOR_MAP = {country: _color_palette[i % len(_color_palette)] for i, country in enumerate(_all_countries)}
DEFAULT_COUNTRY_COLOR = '#8dd3c7'

# Country code mapping (ISO 3166-1 alpha-2)
COUNTRY_CODE_MAP = {
    'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR', 'Cyprus': 'CY',
    'Czech Republic': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE', 'Finland': 'FI', 'France': 'FR',
    'Germany': 'DE', 'Greece': 'GR', 'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE',
    'Italy': 'IT', 'Latvia': 'LV', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT',
    'Netherlands': 'NL', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO',
    'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES', 'Sweden': 'SE', 'Switzerland': 'CH',
    'Liechtenstein': 'LI'
}

def get_country_label(country):
    """Return 2-digit country code for display"""
    return COUNTRY_CODE_MAP.get(country, country)
    """Return color for country - Switzerland in yellow, EU27 in blue, others in light teal"""
    if country == 'Switzerland':
        return SWITZERLAND_COLOR
    elif country in ['EU27', 'EU-27', 'EU-28', 'OECD']:
        return EU_AGGREGATE_COLOR
    elif country in COUNTRY_COLOR_MAP:
        return COUNTRY_COLOR_MAP[country]
    return DEFAULT_COUNTRY_COLOR

def standardize_country_name(country):
    """Convert Eurostat country name to standard name"""
    return EUROSTAT_TO_STANDARD.get(country, None)

def get_country_color(country):
    """Return color for country - Switzerland in yellow, EU27 in blue, others in light teal"""
    if country == 'Switzerland':
        return SWITZERLAND_COLOR
    elif country in ['EU27', 'EU-27', 'EU-28', 'OECD']:
        return EU_AGGREGATE_COLOR
    elif country in COUNTRY_COLOR_MAP:
        return COUNTRY_COLOR_MAP[country]
    return DEFAULT_COUNTRY_COLOR

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

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
    
    # Filter to EU27 aggregate and Switzerland
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label, 'Switzerland'])].copy()
    
    # Standardize country names
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    
    # Convert value to numeric
    df_filtered['value'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['value'])
    
    return df_filtered

def load_real_estate_data():
    """Load and process real estate data"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_real estate other than main.csv')
    
    if not os.path.exists(file_path):
        print("  Missing data file: eurostat_real estate other than main.csv")
        return None
    
    df = pd.read_csv(file_path)
    
    # Keep only latest year
    df = df[df['TIME_PERIOD'] == df['TIME_PERIOD'].max()]
    
    # Filter to EU27 aggregate and all individual countries
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

def load_energy_efficiency_data():
    """Load and process energy efficiency data"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_energy efficiency.csv')
    
    if not os.path.exists(file_path):
        print("  Missing data file: eurostat_energy efficiency.csv")
        return None
    
    df = pd.read_csv(file_path)
    
    # Filter to EU27 aggregate and all individual countries
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
    
    # Filter to EU27 aggregate and Switzerland
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label, 'Switzerland'])].copy()
    
    # Standardize country names
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    
    # Convert value to numeric
    df_filtered['value'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['value'])
    
    return df_filtered

def load_under_occupied_dwellings_data():
    """Load and process under-occupied dwellings data"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_under-occupied dwellings.csv')
    
    if not os.path.exists(file_path):
        print("  Missing data file: eurostat_under-occupied dwellings.csv")
        return None
    
    df = pd.read_csv(file_path)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label, 'Switzerland'])].copy()
    
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    
    return df_filtered

def load_tenure_status_data():
    """Load and process tenure status data"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_tenure status.csv')
    
    if not os.path.exists(file_path):
        print("  Missing data file: eurostat_tenure status.csv")
        return None
    
    df = pd.read_csv(file_path)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label, 'Switzerland'])].copy()
    
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    
    return df_filtered

def load_tenure_status_countries_data():
    """Load tenure status data for all countries"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_tenure status.csv')
    
    if not os.path.exists(file_path):
        print("  Missing data file: eurostat_tenure status.csv")
        return None
    
    df = pd.read_csv(file_path)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    eu27_label = 'European Union - 27 countries (from 2020)'
    df_filtered = df[df['geo'].isin([eu27_label] + list(EUROSTAT_TO_STANDARD.keys()))].copy()
    
    df_filtered['country_name'] = df_filtered['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_filtered = df_filtered.dropna(subset=['country_name'])
    
    return df_filtered

def load_dwellings_and_price_data():
    """Load dwelling stock and price-to-income ratio data for all countries"""
    try:
        # ISO3 to country name mapping
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
        
        return countries_data
    except Exception as e:
        print(f"  Error loading data: {e}")
        return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_rooms_graphs():
    """Create average rooms visualization graphs - Switzerland vs EU27"""
    print("\n[1] Creating average rooms graphs...")
    
    df = load_rooms_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    latest_year = int(df['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df.columns else 'Latest'
    
    # Graph: Switzerland vs EU27 with tenure types comparison
    df_comparison = df.copy()
    
    tenure_order = ['Tenant', 'Owner', 'Total']
    df_comparison = df_comparison[df_comparison['tenure'].isin(tenure_order)]
    
    # Find common tenure types
    df_swiss = df_comparison[df_comparison['country_name'] == 'Switzerland'].copy()
    df_eu27_data = df_comparison[df_comparison['country_name'] == 'EU27'].copy()
    
    common_tenure = list(set(df_swiss['tenure'].unique()) & set(df_eu27_data['tenure'].unique()))
    common_tenure = [t for t in tenure_order if t in common_tenure]
    
    if not common_tenure:
        print("  No common tenure types found between Switzerland and EU27")
        return
    
    df_comparison = df_comparison[df_comparison['tenure'].isin(common_tenure)]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    countries = ['Switzerland', 'EU27']
    x = np.arange(len(common_tenure))
    width = 0.35
    
    for i, country in enumerate(countries):
        df_country = df_comparison[df_comparison['country_name'] == country].copy()
        df_country['tenure'] = pd.Categorical(df_country['tenure'], categories=common_tenure, ordered=True)
        df_country = df_country.sort_values('tenure')
        values = df_country['value'].tolist()
        
        color = get_country_color(country)
        bars = ax.bar(x + i*width, values, width, label=country, color=color, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Average number of rooms per person', fontsize=12, fontweight='bold')
    ax.set_title(f'Average Number of Rooms Per Person ({latest_year})\nSwitzerland vs EU27 by Tenure Status', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(common_tenure, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_rooms_switzerland_vs_eu27_tenure.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 1_rooms_switzerland_vs_eu27_tenure.png")

def create_real_estate_graphs():
    """Create real estate visualization graphs - Switzerland vs EU27"""
    print("\n[2] Creating real estate by income quantile graphs...")
    
    df = load_real_estate_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    latest_year = int(df['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df.columns else 'Latest'
    
    # Comparison with quintiles
    quintile_order = ['First quintile', 'Second quintile', 'Third quintile', 
                     'Fourth quintile', 'Fifth quintile', 'Total']
    df_filtered = df[df['quant_inc'].isin(quintile_order)].copy()
    
    # Find common quantiles
    df_swiss = df_filtered[df_filtered['country_name'] == 'Switzerland'].copy()
    df_eu27_data = df_filtered[df_filtered['country_name'] == 'EU27'].copy()
    
    common_quantiles = list(set(df_swiss['quant_inc'].unique()) & set(df_eu27_data['quant_inc'].unique()))
    common_quantiles = [q for q in quintile_order if q in common_quantiles]
    
    if not common_quantiles:
        print("  No common income quantiles found between Switzerland and EU27")
        return
    
    df_filtered = df_filtered[df_filtered['quant_inc'].isin(common_quantiles)]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    countries = ['Switzerland', 'EU27']
    x = np.arange(len(common_quantiles))
    width = 0.35
    
    for i, country in enumerate(countries):
        df_country = df_filtered[df_filtered['country_name'] == country].copy()
        df_country['quant_inc'] = pd.Categorical(df_country['quant_inc'], categories=common_quantiles, ordered=True)
        df_country = df_country.sort_values('quant_inc')
        values = df_country['value'].tolist()
        
        color = get_country_color(country)
        bars = ax.bar(x + i*width, values, width, label=country, color=color, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Persons owning real estate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Persons Owning Real Estate Other Than Main Residence ({latest_year})\nSwitzerland vs EU27 by Income Quintile', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x + width/2)
    short_labels = [l.replace('First ', 'Q1 ').replace('Second ', 'Q2 ').replace('Third ', 'Q3 ')
                    .replace('Fourth ', 'Q4 ').replace('Fifth ', 'Q5 ').replace(' quintile', '') 
                    for l in common_quantiles]
    ax.set_xticklabels(short_labels, fontsize=10, rotation=15, ha='right')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_real_estate_switzerland_vs_eu27_quintiles.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 2_real_estate_switzerland_vs_eu27_quintiles.png")

def create_energy_efficiency_graphs():
    """Create energy efficiency visualization graphs - Switzerland vs EU27 (overall population only)"""
    print("\n[3] Creating energy efficiency graphs...")
    
    df = load_energy_efficiency_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    latest_year = int(df['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df.columns else '2023'
    
    # Filter to overall population (Total or All population)
    df_total = df[df['age'].isin(['Total', 'All population'])].copy()
    
    if df_total.empty:
        print("  No overall population data available")
        return
    
    # Get Switzerland and EU27 only
    df_comparison = df_total[df_total['country_name'].isin(['Switzerland', 'EU27'])].copy()
    
    if len(df_comparison) < 2:
        print("  Data not available for both Switzerland and EU27")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    countries = ['Switzerland', 'EU27']
    values = []
    for country in countries:
        val = df_comparison[df_comparison['country_name'] == country]['value'].values
        values.append(val[0] if len(val) > 0 else 0)
    
    colors = [get_country_color(c) for c in countries]
    bars = ax.bar(range(len(countries)), values, color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries, fontsize=12, fontweight='bold')
    ax.set_ylabel('Persons living in improved dwellings (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Energy Efficiency: Persons in Improved Dwellings ({latest_year})\nSwitzerland vs EU27 - Overall Population', 
                fontsize=13, fontweight='bold', pad=15)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_energy_efficiency_switzerland_vs_eu27_overall.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 3_energy_efficiency_switzerland_vs_eu27_overall.png")

def create_berd_graphs():
    """Create Business Enterprise R&D Expenditure graphs - Switzerland vs EU27"""
    print("\n[4] Creating BERD graphs...")
    
    df = load_berd_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    latest_year = int(df['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df.columns else 'Latest'
    
    # NACE order
    nace_order = ['Total - all NACE activities', 'Manufacturing', 'Services', 'Construction',
                  'Agriculture, forestry and fishing', 'Mining and quarrying']
    
    # Create separate graphs for each unit
    for unit_val in df['unit'].unique():
        df_unit = df[df['unit'] == unit_val].copy()
        
        # Short label for unit
        unit_label = 'PPS per inhabitant' if 'PPS' in unit_val else 'Percentage of GDP'
        
        # Find common NACE sectors available in both countries
        df_swiss = df_unit[df_unit['country_name'] == 'Switzerland'].copy()
        df_eu27 = df_unit[df_unit['country_name'] == 'EU27'].copy()
        
        common_nace = list(set(df_swiss['nace_r2'].unique()) & set(df_eu27['nace_r2'].unique()))
        common_nace = [n for n in nace_order if n in common_nace]
        
        if not common_nace:
            print(f"  No common NACE sectors found for unit {unit_val}")
            continue
        
        # Filter data
        df_filtered = df_unit[df_unit['nace_r2'].isin(common_nace)].copy()
        
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        countries = ['Switzerland', 'EU27']
        x = np.arange(len(common_nace))
        width = 0.35
        
        for i, country in enumerate(countries):
            df_country = df_filtered[df_filtered['country_name'] == country].copy()
            df_country['nace_r2'] = pd.Categorical(df_country['nace_r2'], categories=common_nace, ordered=True)
            df_country = df_country.sort_values('nace_r2')
            values = df_country['value'].tolist()
            
            if len(values) != len(common_nace):
                # Pad missing values with NaN if needed
                values = values + [0] * (len(common_nace) - len(values))
            
            color = get_country_color(country)
            bars = ax.bar(x + i*width, values, width, label=country, color=color, edgecolor='white', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    if 'PPS' in unit_val:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        if 'PPS' in unit_val:
            ax.set_ylabel(f'BERD ({unit_label})', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel(f'BERD ({unit_label})', fontsize=12, fontweight='bold')
        
        ax.set_title(f'Business Enterprise R&D Expenditure ({latest_year})\nSwitzerland vs EU27 by NACE Sector - ({unit_label})', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x + width/2)
        short_labels = [l.replace('Total - all NACE activities', 'Total').replace('Agriculture, forestry and fishing', 'Agric. & forestry')
                       .replace('Mining and quarrying', 'Mining') for l in common_nace]
        ax.set_xticklabels(short_labels, fontsize=9, rotation=15, ha='right')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.2)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        file_suffix = 'pps' if 'PPS' in unit_val else 'pct_gdp'
        plt.savefig(os.path.join(OUTPUT_DIR, f'4_berd_switzerland_vs_eu27_nace_{file_suffix}.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] 4_berd_switzerland_vs_eu27_nace_{file_suffix}.png")

def create_under_occupied_dwellings_graphs():
    """Create visualizations for under-occupied dwellings - Switzerland vs EU27"""
    print("\n[5] Creating under-occupied dwellings graphs...")
    
    df = load_under_occupied_dwellings_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    latest_year = int(df['TIME_PERIOD'].max())
    
    # By age (latest year)
    age_order = ['Total', 'Less than 18 years', 'From 18 to 64 years', '65 years or over']
    df_age = df[(df['TIME_PERIOD'] == latest_year) & (df['age'].isin(age_order))].copy()
    
    # Find common age groups
    df_swiss = df_age[df_age['country_name'] == 'Switzerland'].copy()
    df_eu27_data = df_age[df_age['country_name'] == 'EU27'].copy()
    
    common_ages = list(set(df_swiss['age'].unique()) & set(df_eu27_data['age'].unique()))
    common_ages = [a for a in age_order if a in common_ages]
    
    if not common_ages:
        print("  No common age groups found between Switzerland and EU27")
        return
    
    df_age = df_age[df_age['age'].isin(common_ages)]
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    countries = ['Switzerland', 'EU27']
    x = np.arange(len(common_ages))
    width = 0.35
    
    for i, country in enumerate(countries):
        df_country = df_age[df_age['country_name'] == country].copy()
        df_country['age'] = pd.Categorical(df_country['age'], categories=common_ages, ordered=True)
        df_country = df_country.sort_values('age')
        # Group by age and take the mean if there are duplicates
        df_country = df_country.groupby('age', as_index=False)['value'].mean()
        values = df_country['value'].tolist()
        
        color = get_country_color(country)
        bars = ax.bar(x + i*width, values, width, label=country, color=color, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Share (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Share of People Living in Under-occupied Dwellings ({latest_year})\nSwitzerland vs EU27 by Age', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x + width/2)
    short_labels = ['Total', '<18', '18-64', '65+']
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_under_occupied_switzerland_vs_eu27_age.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 5_under_occupied_switzerland_vs_eu27_age.png")

def create_tenure_status_graphs():
    """Create visualizations for tenure status - Switzerland vs EU27"""
    print("\n[6] Creating tenure status graphs...")
    
    df = load_tenure_status_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    latest_year = int(df['TIME_PERIOD'].max())
    df_data = df[df['TIME_PERIOD'] == latest_year].copy()
    
    # By tenure type (Total income group, Total household type)
    df_tenure = df_data[(df_data['incgrp'] == 'Total') & (df_data['hhtyp'] == 'Total')].copy()
    
    tenure_order = ['Owner', 'Owner, with mortgage or loan', 'Owner, no outstanding mortgage or housing loan',
                   'Tenant', 'Tenant, rent at market price', 'Tenant, rent at reduced price or free']
    df_tenure = df_tenure[df_tenure['tenure'].isin(tenure_order)].copy()
    
    # Find common tenure types
    df_swiss = df_tenure[df_tenure['country_name'] == 'Switzerland'].copy()
    df_eu27_data = df_tenure[df_tenure['country_name'] == 'EU27'].copy()
    
    common_tenure = list(set(df_swiss['tenure'].unique()) & set(df_eu27_data['tenure'].unique()))
    common_tenure = [t for t in tenure_order if t in common_tenure]
    
    if not common_tenure:
        print("  No common tenure types found between Switzerland and EU27")
        return
    
    df_tenure = df_tenure[df_tenure['tenure'].isin(common_tenure)]
    
    # Create comparison
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    countries = ['Switzerland', 'EU27']
    x = np.arange(len(common_tenure))
    width = 0.35
    
    for i, country in enumerate(countries):
        df_country = df_tenure[df_tenure['country_name'] == country].copy()
        df_country['tenure'] = pd.Categorical(df_country['tenure'], categories=common_tenure, ordered=True)
        df_country = df_country.sort_values('tenure')
        # Group by tenure and take the mean if there are duplicates
        df_country = df_country.groupby('tenure', as_index=False)['value'].mean()
        values = df_country['value'].tolist()
        
        color = get_country_color(country)
        bars = ax.bar(x + i*width, values, width, label=country, color=color, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Share (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution of Population by Tenure Status ({latest_year})\nSwitzerland vs EU27', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x + width/2)
    short_labels = ['Owner', 'Owner\nw/ mortgage', 'Owner\nno mortgage', 'Tenant', 'Tenant\nmarket', 'Tenant\nreduced/free']
    ax.set_xticklabels(short_labels, fontsize=9, rotation=0, ha='center')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_tenure_status_switzerland_vs_eu27.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 6_tenure_status_switzerland_vs_eu27.png")

def create_real_estate_countries_total():
    """Create real estate by countries - Total only - highlighting Switzerland"""
    print("\n[7] Creating real estate countries comparison...")
    
    df = load_real_estate_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    # Graph 2: Countries - Total values only
    df_countries = df[(df['country_name'] != 'EU27') & (df['quant_inc'] == 'Total')].copy()
    if not df_countries.empty:
        df_countries = df_countries.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(11, 12))
        
        # Switzerland in yellow, others in blue
        colors_list = [SWITZERLAND_COLOR if c == 'Switzerland' else EU_AGGREGATE_COLOR for c in df_countries['country_name']]
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

def create_real_estate_countries_map():
    """Create map visualization for real estate ownership by country using plot_functions"""
    print("\n[7b] Creating real estate countries map...")
    
    if not PLOT_FUNCTIONS_AVAILABLE:
        print("  plot_functions not available, skipping map visualization")
        return
    
    df = load_real_estate_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    # Get all countries - Total values only
    df_countries = df[(df['country_name'] != 'EU27') & (df['quant_inc'] == 'Total')].copy()
    if df_countries.empty:
        print("  No country data available")
        return
    
    try:
        # Map country names to ISO 2-letter codes
        country_to_iso = {
            'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
            'Cyprus': 'CY', 'Czech Republic': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE',
            'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'GR',
            'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE', 'Italy': 'IT',
            'Latvia': 'LV', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT',
            'Netherlands': 'NL', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT',
            'Romania': 'RO', 'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES',
            'Sweden': 'SE', 'Switzerland': 'CH'
        }
        
        # Prepare data for mapping
        df_countries['geo'] = df_countries['country_name'].map(country_to_iso)
        df_countries['year'] = df_countries['TIME_PERIOD'].astype(int)
        df_countries['value'] = pd.to_numeric(df_countries['value'], errors='coerce')
        
        df_plot = df_countries[['geo', 'year', 'value']].dropna()
        
        if df_plot.empty:
            print("  No valid data for mapping")
            return
        
        # Get latest year
        latest_year = int(df_plot['year'].max())
        
        # Define study countries (EU + EFTA)
        study_countries = ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 
                          'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 
                          'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 
                          'NO', 'CH', 'IS']
        
        # Construct path to shapefile
        shapefile_path = os.path.join(BASE_DIR, 'external_data', '0_shapefile', 
                                     'ne_50m_admin_0_countries', 'ne_50m_admin_0_countries.shp')
        
        # Call the plot_europe_map function from plot_functions
        fig, ax = plot_europe_map(
            df_plot,
            year=latest_year,
            colormap='YlOrRd',
            value_title='Real Estate Ownership\nOther Than Main Residence (%)',
            figsize=(14, 12),
            shapefile_path=shapefile_path,
            k=6
        )
        
        # Increase legend font size
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(11)
            legend.get_title().set_fontsize(12)
        
        # Save the figure
        plt.savefig(os.path.join(OUTPUT_DIR, '2_real_estate_countries_map.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 2_real_estate_countries_map.png")
        
    except Exception as e:
        print(f"  Error creating map: {e}")
        import traceback
        traceback.print_exc()

def create_tenure_status_countries_map():
    """Create map visualization for tenure status ownership (Owner) by country"""
    print("\n[8c] Creating tenure status countries map...")
    
    if not PLOT_FUNCTIONS_AVAILABLE:
        print("  plot_functions not available, skipping map visualization")
        return
    
    df = load_tenure_status_countries_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    # Get all countries with "Owner" tenure only
    df_countries = df[(df['country_name'] != 'EU27') & (df['tenure'] == 'Owner')].copy()
    if df_countries.empty:
        print("  No Owner tenure data available")
        return
    
    try:
        # Map country names to ISO 2-letter codes
        country_to_iso = {
            'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
            'Cyprus': 'CY', 'Czech Republic': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE',
            'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'GR',
            'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE', 'Italy': 'IT',
            'Latvia': 'LV', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT',
            'Netherlands': 'NL', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT',
            'Romania': 'RO', 'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES',
            'Sweden': 'SE', 'Switzerland': 'CH'
        }
        
        # Prepare data for mapping
        df_countries['geo'] = df_countries['country_name'].map(country_to_iso)
        df_countries['year'] = df_countries['TIME_PERIOD'].astype(int)
        df_countries['value'] = pd.to_numeric(df_countries['value'], errors='coerce')
        
        df_plot = df_countries[['geo', 'year', 'value']].dropna()
        
        if df_plot.empty:
            print("  No valid data for mapping")
            return
        
        # Get latest year
        latest_year = int(df_plot['year'].max())
        print(f"  Tenure status map (6c) - Using latest year: {latest_year}")
        print(f"  Available years: {sorted(df_plot['year'].unique())}")
        
        # Define study countries (EU + EFTA)
        study_countries = ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 
                          'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 
                          'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 
                          'NO', 'CH', 'IS']
        
        # Construct path to shapefile
        shapefile_path = os.path.join(BASE_DIR, 'external_data', '0_shapefile', 
                                     'ne_50m_admin_0_countries', 'ne_50m_admin_0_countries.shp')
        
        # Call the plot_europe_map function from plot_functions
        fig, ax = plot_europe_map(
            df_plot,
            year=latest_year,
            colormap='YlOrRd',
            value_title='Owner-Occupied Dwellings\n(%)',
            figsize=(14, 12),
            shapefile_path=shapefile_path,
            k=6
        )
        
        # Increase legend font size
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(11)
            legend.get_title().set_fontsize(12)
        
        # Save the figure
        plt.savefig(os.path.join(OUTPUT_DIR, '6c_tenure_status_countries_map.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 6c_tenure_status_countries_map.png")
        
    except Exception as e:
        print(f"  Error creating map: {e}")
        import traceback
        traceback.print_exc()

def create_dwellings_vs_price_scatter():
    """Create scatter plot: dwelling variation vs price-to-income ratio change with Switzerland highlighted"""
    print("\n[9] Creating dwellings vs price-to-income scatter plot...")
    
    countries_data = load_dwellings_and_price_data()
    if not countries_data:
        print("  No data available")
        return
    
    try:
        # Collect all data points to determine axis limits
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
        
        # Calculate regression
        slope = np.sum(all_x * all_y) / np.sum(all_x ** 2) if np.sum(all_x ** 2) > 0 else 0
        y_pred = slope * all_x
        ss_res = np.sum((all_y - y_pred) ** 2)
        ss_tot = np.sum((all_y - np.mean(all_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        regression_text = f'Regression: y={slope:.3f}x (R²={r_squared:.3f})'
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Set axis limits (x-axis from 0, y-axis data-driven)
        padding = 0.1
        x_max_data = max(all_x) if len(all_x) > 0 else 1
        y_min_data, y_max_data = min(all_y) if len(all_y) > 0 else 0, max(all_y) if len(all_y) > 0 else 1
        x_range = x_max_data
        y_range = y_max_data - y_min_data if y_max_data != y_min_data else 1
        x_max = x_max_data + padding * x_range
        y_min = y_min_data - padding * y_range
        y_max = y_max_data + padding * y_range
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Draw regression line
        x_line = np.array([0, x_max])
        y_line = slope * x_line
        ax.plot(x_line, y_line, color='grey', linestyle='--', linewidth=2, alpha=0.6, zorder=1)
        
        # Plot all circles and squares with arrows - Switzerland in yellow, others in blue
        for country, periods in countries_data.items():
            color = SWITZERLAND_COLOR if country == 'Switzerland' else EU_AGGREGATE_COLOR
            
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
                # Highlight Switzerland label
                fontweight = 'bold' if country == 'Switzerland' else 'normal'
                fontsize = 10 if country == 'Switzerland' else 8
                ax.annotate(country, (x1, y1), xytext=(5, 5), textcoords='offset points', 
                           fontsize=fontsize, fontweight=fontweight)
            elif '1121' in periods and '0111' not in periods:
                x1 = periods['1121']['x']
                y1 = periods['1121']['y']
                ax.scatter(x1, y1, s=150, alpha=0.9, color=color, marker='s', zorder=5)
                fontweight = 'bold' if country == 'Switzerland' else 'normal'
                fontsize = 10 if country == 'Switzerland' else 8
                ax.annotate(country, (x1, y1), xytext=(5, 5), textcoords='offset points', 
                           fontsize=fontsize, fontweight=fontweight)
            elif '0111' in periods:
                fontweight = 'bold' if country == 'Switzerland' else 'normal'
                fontsize = 10 if country == 'Switzerland' else 8
                ax.annotate(country, (x0, y0), xytext=(5, 5), textcoords='offset points', 
                           fontsize=fontsize, fontweight=fontweight)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#666666', markersize=10, label='2001-2011 (circles)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#666666', markersize=10, label='2011-2021 (squares)'),
            Line2D([0], [0], color='grey', linestyle='--', linewidth=2, label=regression_text),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=SWITZERLAND_COLOR, markersize=8, label='Switzerland'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=EU_AGGREGATE_COLOR, markersize=8, label='Other countries')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
        
        ax.set_xlabel('Dwelling Stock variation (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price-to-Income variation (%)', fontsize=12, fontweight='bold')
        ax.set_title('Price-to-Income vs Dwelling Stock:\n2001-2011 (circles) → 2011-2021 (squares)', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '9_dwellings_vs_price_scatter.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 9_dwellings_vs_price_scatter.png")
        
    except Exception as e:
        print(f"  Error creating scatter plot: {e}")
        import traceback
        traceback.print_exc()

def create_energy_efficiency_countries_16plus():
    """Create energy efficiency by countries - 16+ only, Total population - highlighting Switzerland"""
    print("\n[8] Creating energy efficiency countries comparison (16 years+, Total population)...")
    
    df = load_energy_efficiency_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    # Get all countries for 16+ age and Total yn_arope (Total population)
    df_countries = df[(df['country_name'] != 'EU27') & (df['age'] == '16 years or over') & (df['yn_arope'] == 'Total')].copy()
    if df_countries.empty:
        print("  No country data available")
        return
    
    df_countries = df_countries.sort_values('value', ascending=True)
    
    fig, ax = plt.subplots(figsize=(11, 12))
    
    # Custom color mapping: Switzerland yellow, all others blue
    colors_list = [SWITZERLAND_COLOR if c == 'Switzerland' else EU_AGGREGATE_COLOR for c in df_countries['country_name']]
    bars = ax.barh(range(len(df_countries)), df_countries['value'], color=colors_list, edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(range(len(df_countries)))
    ax.set_yticklabels(df_countries['country_name'], fontsize=10)
    ax.set_xlabel('Persons living in improved dwellings (%)', fontsize=12, fontweight='bold')
    latest_year = int(df_countries['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df_countries.columns else '2023'
    ax.set_title(f'Energy Efficiency: Persons in Improved Dwellings (16 years+, Total) ({latest_year})\nby Country', fontsize=13, fontweight='bold', pad=15)
    
    # Add value labels
    for i, (idx, row) in enumerate(df_countries.iterrows()):
        ax.text(row['value'] + 1, i, f"{row['value']:.1f}%", va='center', fontsize=9)
    
    ax.grid(axis='x', alpha=0.2)
    ax.set_facecolor('white')
    ax.set_xlim(0, max(df_countries['value']) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_energy_efficiency_countries_16plus.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 6_energy_efficiency_countries_16plus.png")

def create_energy_efficiency_countries_by_poverty_risk():
    """Create energy efficiency by poverty risk - side-by-side comparison"""
    print("\n[8b] Creating energy efficiency by poverty risk comparison...")
    
    df = load_energy_efficiency_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    # Plot for each poverty risk category (excluding Total)
    poverty_risk_order = ['Not at risk of poverty or social exclusion', 'At risk of poverty or social exclusion']
    
    # First pass: find the maximum value across both categories for consistent x-axis
    max_value = 0
    for risk_category in poverty_risk_order:
        df_risk_temp = df[(df['country_name'] != 'EU27') & (df['age'] == '16 years or over') & (df['yn_arope'] == risk_category)].copy()
        if not df_risk_temp.empty:
            max_value = max(max_value, df_risk_temp['value'].max())
    
    x_axis_limit = max_value * 1.1  # +10%
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    for risk_idx, risk_category in enumerate(poverty_risk_order):
        df_risk = df[(df['country_name'] != 'EU27') & (df['age'] == '16 years or over') & (df['yn_arope'] == risk_category)].copy()
        df_risk = df_risk.dropna(subset=['value'])
        
        if not df_risk.empty:
            df_risk = df_risk.sort_values('value', ascending=True)
            
            ax = axes[risk_idx]
            # Switzerland in yellow, others in blue
            colors_list = [SWITZERLAND_COLOR if c == 'Switzerland' else EU_AGGREGATE_COLOR for c in df_risk['country_name']]
            bars = ax.barh(range(len(df_risk)), df_risk['value'], color=colors_list, edgecolor='white', linewidth=1.5)
            
            ax.set_yticks(range(len(df_risk)))
            ax.set_yticklabels([get_country_label(c) for c in df_risk['country_name']], fontsize=9)
            ax.set_xlabel('Persons living in improved dwellings (%)', fontsize=12, fontweight='bold')
            
            # Shorten the risk category label for display
            risk_display = 'Not at risk' if 'Not at risk' in risk_category else 'At risk'
            latest_year = int(df_risk['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df_risk.columns else '2023'
            ax.set_title(f'{risk_display} ({latest_year})', fontsize=12, fontweight='bold', pad=10)
            
            # Add value labels
            for i, (idx, row) in enumerate(df_risk.iterrows()):
                ax.text(row['value'] + 1, i, f"{row['value']:.1f}%", va='center', fontsize=8)
            
            ax.grid(axis='x', alpha=0.2)
            ax.set_facecolor('white')
            ax.set_xlim(0, x_axis_limit)  # Use consistent x-axis limit for both panels
    
    fig.suptitle(f'Energy Efficiency: Persons in Improved Dwellings (16 years+)\nby Poverty Risk Status', 
                fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6b_energy_efficiency_countries_poverty_risk.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 6b_energy_efficiency_countries_poverty_risk.png")

def create_berd_countries_total():
    """Create BERD by countries - Total NACE - highlighting Switzerland"""
    print("\n[9] Creating BERD countries comparison (Total NACE)...")
    
    df = load_berd_data()
    if df is None or df.empty:
        print("  No data available")
        return
    
    latest_year = int(df['TIME_PERIOD'].max()) if 'TIME_PERIOD' in df.columns else 'Latest'
    
    # Get total NACE data for each country
    df_countries = df[(df['country_name'] != 'EU27') & (df['nace_r2'] == 'Total - all NACE activities')].copy()
    
    if df_countries.empty:
        print("  No Total NACE data available for countries")
        return
    
    # Get only % of GDP unit
    df_countries = df_countries[df_countries['unit'].str.contains('GDP', na=False)].copy()
    
    if df_countries.empty:
        print("  No GDP percentage data available")
        return
    
    df_countries = df_countries.sort_values('value', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [SWITZERLAND_COLOR if c == 'Switzerland' else DEFAULT_COUNTRY_COLOR for c in df_countries['country_name']]
    bars = ax.barh(range(len(df_countries)), df_countries['value'], color=colors, 
                  edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(range(len(df_countries)))
    ax.set_yticklabels(df_countries['country_name'], fontsize=10)
    ax.set_xlabel('BERD (% of GDP)', fontsize=12, fontweight='bold')
    ax.set_title(f'Business Enterprise R&D Expenditure ({latest_year})\nTotal NACE - as % of GDP by Country', 
                fontsize=13, fontweight='bold', pad=15)
    
    # Add value labels
    for i, (idx, row) in enumerate(df_countries.iterrows()):
        ax.text(row['value'] + 0.05, i, f"{row['value']:.2f}%", va='center', fontsize=9, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.2)
    ax.set_facecolor('white')
    ax.set_xlim(0, max(df_countries['value']) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '8_berd_countries_total_pct_gdp.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 8_berd_countries_total_pct_gdp.png")

def create_under_occupied_countries_side_by_side():
    """Create visualizations for under-occupied dwellings by age"""
    print("\n[10] Creating under-occupied dwellings side-by-side comparison...")
    
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_under-occupied dwellings.csv')
    if not os.path.exists(file_path):
        print("  File not found:", file_path)
        return
    
    df = pd.read_csv(file_path)
    df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    eu27_label = 'European Union - 27 countries (from 2020)'
    
    # Countries by age groups
    year_for_countries = 2024
    df_2024 = df[df['TIME_PERIOD'] == year_for_countries].copy()
    df_countries = df_2024[df_2024['geo'] != eu27_label].copy()
    df_countries['country_name'] = df_countries['geo'].apply(
        lambda x: 'EU27' if x == eu27_label else standardize_country_name(x)
    )
    df_countries = df_countries[df_countries['country_name'].notna()]
    
    # Plot for each age group except Total
    age_order = ['Less than 18 years', 'From 18 to 64 years', '65 years or over']
    age_labels = ['Less than 18 years', 'From 18 to 64 years', '65 years or over']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 10))
    
    for age_idx, (age_group, age_label) in enumerate(zip(age_order, age_labels)):
        df_age = df_countries[df_countries['age'] == age_group].copy()
        df_age = df_age.dropna(subset=['value'])
        
        if not df_age.empty:
            df_age = df_age.sort_values('value', ascending=True)
            
            ax = axes[age_idx]
            # Switzerland in yellow, others in blue
            colors_list = [SWITZERLAND_COLOR if c == 'Switzerland' else EU_AGGREGATE_COLOR for c in df_age['country_name']]
            bars = ax.barh(range(len(df_age)), df_age['value'], color=colors_list, edgecolor='white', linewidth=1.5)
            
            ax.set_yticks(range(len(df_age)))
            ax.set_yticklabels([get_country_label(c) for c in df_age['country_name']], fontsize=9)
            ax.set_xlabel('Share (%)', fontsize=12, fontweight='bold')
            
            # Add value labels
            for i, (idx, row) in enumerate(df_age.iterrows()):
                ax.text(row['value'] + 1.5, i, f"{row['value']:.1f}%", va='center', fontsize=9, fontweight='bold')
            
            age_display = age_group.replace('Less than ', '<').replace('From ', '').replace(' years', '').replace('or over', '+')
            ax.set_title(f'Age {age_display} ({year_for_countries})', fontsize=12, fontweight='bold', pad=10)
            
            ax.grid(axis='x', alpha=0.2)
            ax.set_facecolor('white')
            ax.set_xlim(0, 100)
    
    fig.suptitle(f'Under-occupied Dwellings by Country and Age Group\nShare of Population', 
                fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '7_under_occupied_countries_all_ages_sbs.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 7_under_occupied_countries_all_ages_sbs.png")

def create_government_expenditure_housing():
    """Create government expenditure on housing visualization - area chart over time"""
    print("\n[11] Creating government expenditure on housing area chart...")
    
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
    
    # Colors for categories
    category_colors = {
        'Housing social protection': '#fb8072',
        'Housing development': '#80b1d3',
        'Community development': '#fdb462',
        'Water supply': '#bebada',
        'Street lighting': '#8dd3c7',
        'R&D Housing and community amenities': '#ffffb3',
        'Housing and community amenities n.e.c.': '#a6d854'
    }
    
    # Define category order for stacking
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
    
    # Helper function to reorder pivot table columns
    def reorder_columns(pivot_table, category_order):
        cols = [col for col in category_order if col in pivot_table.columns]
        return pivot_table[cols]
    
    # Switzerland Area graph - % of GDP across time
    print("\n  Creating area graph - Switzerland % of GDP...")
    
    # Get Switzerland data
    ch_data = df_filtered[(df_filtered['geo'] == 'Switzerland')].copy()
    
    if not ch_data.empty:
        # Pivot for area chart
        pivot_gdp = ch_data.pivot_table(
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
            ax.set_title('Switzerland: Government Expenditure on Housing and Community Amenities (% of GDP)', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(OUTPUT_DIR, '12_government_expenditure_housing.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  [SAVED] 12_government_expenditure_housing.png")
            plt.close()
    else:
        print("    No Switzerland data found")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 70)
    print("EUROSTAT Analysis for Switzerland vs EU27 - Generating graphs...")
    print("=" * 70)
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Data directory: {EXTERNAL_DATA_DIR}")
    
    # Create all visualizations
    create_rooms_graphs()
    create_real_estate_graphs()
    create_real_estate_countries_total()
    create_real_estate_countries_map()
    create_energy_efficiency_graphs()
    create_energy_efficiency_countries_16plus()
    create_energy_efficiency_countries_by_poverty_risk()
    create_berd_graphs()
    create_berd_countries_total()
    create_under_occupied_dwellings_graphs()
    create_under_occupied_countries_side_by_side()
    create_tenure_status_graphs()
    create_tenure_status_countries_map()
    create_dwellings_vs_price_scatter()
    create_government_expenditure_housing()
    
    print("\n" + "=" * 70)
    print("SUCCESS! All visualizations have been generated successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
