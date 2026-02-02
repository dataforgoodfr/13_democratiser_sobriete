"""
OECD Analysis - Generate graphs from OECD data sources for EU27 + EFTA countries
This script creates visualizations for 31 countries (27 EU27 + 4 EFTA)
Outputs saved as PNG files in outputs/graphs/oecd folder
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Color palette (matching hbs_test_lu_2020.py)
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'
SWITZERLAND_COLOR = '#ffd558'
COMPONENT_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3', 
                    '#80b1d3', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']

# Base paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, 'external_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'graphs', 'oecd')

# Target countries: EU27 + EFTA (31 total)
EU27_COUNTRIES = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
    'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
    'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
    'Spain', 'Sweden'
]
EFTA_COUNTRIES = ['Iceland', 'Liechtenstein', 'Norway', 'Switzerland']
TARGET_COUNTRIES = EU27_COUNTRIES + EFTA_COUNTRIES

# ISO3 mapping
ISO3_MAPPING = {
    'CHE': 'Switzerland', 'DEU': 'Germany', 'FRA': 'France', 'ITA': 'Italy',
    'ESP': 'Spain', 'NLD': 'Netherlands', 'BEL': 'Belgium', 'AUT': 'Austria',
    'SWE': 'Sweden', 'DNK': 'Denmark', 'FIN': 'Finland', 'NOR': 'Norway',
    'PRT': 'Portugal', 'GRC': 'Greece', 'IRL': 'Ireland', 'LUX': 'Luxembourg',
    'CZE': 'Czech Republic', 'HUN': 'Hungary', 'POL': 'Poland', 'SVK': 'Slovakia',
    'SVN': 'Slovenia', 'EST': 'Estonia', 'LVA': 'Latvia', 'LTU': 'Lithuania',
    'HRV': 'Croatia', 'ROU': 'Romania', 'BGR': 'Bulgaria', 'CYP': 'Cyprus',
    'MLT': 'Malta', 'ISL': 'Iceland', 'LIE': 'Liechtenstein', 'EU27': 'EU27'
}

# Create comprehensive color map for all countries (31 distinct colors, matching eurostat_analysis.py)
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
    """Get color for country (matching eurostat colors)"""
    if country in COUNTRY_COLOR_MAP:
        return COUNTRY_COLOR_MAP[country]
    elif country in ['EU27', 'EU-27']:
        return '#80b1d3'
    else:
        return DEFAULT_COUNTRY_COLOR

def filter_target_countries(df, country_col='country'):
    """Filter to target countries only"""
    return df[df[country_col].isin(TARGET_COUNTRIES)]

def load_excel_data(filename):
    """Load Excel file from external_data"""
    filepath = os.path.join(EXTERNAL_DATA_DIR, filename)
    try:
        xls = pd.ExcelFile(filepath)
        print(f"  {filename}: {xls.sheet_names}")
        df = pd.read_excel(filepath, sheet_name=0, header=0)
        return df
    except Exception as e:
        print(f"  ERROR loading {filename}: {e}")
        return None

def clean_country_name(iso3_code):
    """Convert ISO3 to country name"""
    if pd.isna(iso3_code):
        return None
    iso3_code = str(iso3_code).strip().upper()
    return ISO3_MAPPING.get(iso3_code, iso3_code)

def create_household_debt_graph():
    """Household debt to GDP ratio graph"""
    print("\n[1] Creating household debt graph...")
    df = load_excel_data('ocde_household debt to GDP.xlsx')
    if df is None:
        return
    
    df['country'] = df['iso3'].apply(clean_country_name)
    df = df[df['country'].notna()]
    df = filter_target_countries(df)
    
    df_2008 = df[df['year'] == 2008]
    df_2021 = df[df['year'] == 2021]
    
    merged = pd.merge(df_2021[['country', 'Household debt to GDP']], 
                     df_2008[['country', 'Household debt to GDP']], 
                     on='country', how='left', suffixes=('_2021', '_2008'))
    merged = merged.sort_values('Household debt to GDP_2021', ascending=True)
    merged = merged.dropna(subset=['Household debt to GDP_2021'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    countries = merged['country'].tolist()
    values_2021 = [v * 100 for v in merged['Household debt to GDP_2021']]
    values_2008 = [v * 100 if pd.notna(v) else None for v in merged['Household debt to GDP_2008']]
    
    colors = [get_country_color(c) for c in countries]
    bars = ax.barh(range(len(countries)), values_2021, color=colors, alpha=0.8)
    
    for i, val in enumerate(values_2008):
        if val is not None:
            ax.scatter(val, i, color='red' if countries[i]=='Switzerland' else 'black', 
                      s=100, zorder=5, marker='o', edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=10)
    ax.set_xlabel('Household Debt to GDP (%)', fontsize=12, fontweight='bold')
    ax.set_title('Household Debt to GDP Ratio (EU27 + EFTA)\n2021 (bars) vs 2008 (dots)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_household_debt_to_gdp_ratio.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 1_household_debt_to_gdp_ratio.png")

def create_house_price_graphs():
    """Real house price changes graphs"""
    print("\n[2] Creating house price graphs...")
    
    df_2002 = load_excel_data('ocde_real house price rise 2002-2007.xlsx')
    df_2017 = load_excel_data('ocde_real house price rise 2017-2022.xlsx')
    
    if df_2002 is None or df_2017 is None:
        return
    
    if 'variable' in df_2002.columns:
        df_2002['iso3'] = df_2002['variable']
    
    df_2002['country'] = df_2002['iso3'].apply(clean_country_name)
    df_2017['country'] = df_2017['iso3'].apply(clean_country_name)
    
    df_2002 = df_2002[df_2002['country'].notna()]
    df_2017 = df_2017[df_2017['country'].notna()]
    
    df_2002 = filter_target_countries(df_2002)
    df_2017 = filter_target_countries(df_2017)
    
    # Bar charts for each period
    for df, period, filename in [(df_2002, '2002-2007', '2_house_prices_2002_2007.png'),
                                 (df_2017, '2017-2022', '3_house_prices_2017_2022.png')]:
        df_sorted = df.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        countries = df_sorted['country'].tolist()
        values = [v * 100 for v in df_sorted['value']]
        colors = [get_country_color(c) for c in countries]
        
        ax.barh(range(len(countries)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries, fontsize=10)
        ax.set_xlabel('Real House Price Change (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Real House Prices {period} (EU27 + EFTA)', fontsize=14, fontweight='bold', pad=15)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [SAVED] {filename}")
    
    # Scatter plot comparing both periods
    merged = pd.merge(df_2002[['country', 'value']], df_2017[['country', 'value']], 
                     on='country', how='inner', suffixes=('_2002', '_2017'))
    merged = merged.dropna()
    
    if not merged.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for _, row in merged.iterrows():
            color = get_country_color(row['country'])
            ax.scatter(row['value_2002']*100, row['value_2017']*100, 
                      color=color, s=100, alpha=0.8, edgecolor='white', linewidth=1)
        
        ax.set_xlabel('2002-2007 (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('2017-2022 (%)', fontsize=12, fontweight='bold')
        ax.set_title('House Price Changes Comparison', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '4_house_prices_comparison.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 4_house_prices_comparison.png")

def create_social_rental_graph():
    """Social rental dwelling stock graph"""
    print("\n[3] Creating social rental dwelling graph...")
    df = load_excel_data('ocde_social rental dwelling stock.xlsx')
    if df is None:
        return
    
    df['country'] = df['iso3'].apply(clean_country_name)
    df = df[df['country'].notna()]
    df = filter_target_countries(df)
    
    df_sorted = df.sort_values('2020', ascending=True)
    df_sorted = df_sorted.dropna(subset=['2020'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    countries = df_sorted['country'].tolist()
    values_2020 = [v * 100 for v in df_sorted['2020']]
    values_2010 = [v * 100 if pd.notna(v) else None for v in df_sorted['2010']]
    colors = [get_country_color(c) for c in countries]
    
    ax.barh(range(len(countries)), values_2020, color=colors, alpha=0.8)
    
    for i, val in enumerate(values_2010):
        if val is not None:
            ax.scatter(val, i, color='black', s=80, zorder=5, marker='o', 
                      edgecolor='white', linewidth=1)
    
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=10)
    ax.set_xlabel('Social Rental Dwellings (% of total)', fontsize=12, fontweight='bold')
    ax.set_title('Social Rental Housing Stock (EU27 + EFTA)\n2020 (bars) vs 2010 (dots)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_social_rental_dwelling_stock.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 5_social_rental_dwelling_stock.png")

def create_homeownership_graph():
    """Homeownership rates graph"""
    print("\n[4] Creating homeownership rates graph...")
    df = load_excel_data('ocde_Homeownership rates.xlsx')
    if df is None:
        return
    
    df['country'] = df['iso3'].apply(clean_country_name)
    df = df[df['country'].notna()]
    df = filter_target_countries(df)
    
    outright = df[df['Type'] == 'Outright owners'].copy()
    mortgage = df[df['Type'] == 'Owners with mortgages'].copy()
    
    merged = pd.merge(outright[['country', 'Homeownership rate']], 
                     mortgage[['country', 'Homeownership rate']], 
                     on='country', how='outer', suffixes=('_outright', '_mortgage'))
    merged['total'] = merged['Homeownership rate_outright'].fillna(0) + merged['Homeownership rate_mortgage'].fillna(0)
    merged = merged.sort_values('total', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    countries = merged['country'].tolist()
    y_pos = range(len(countries))
    
    ax.barh(y_pos, merged['Homeownership rate_outright'].fillna(0)*100, 
           color='#8dd3c7', alpha=0.8, label='Outright owners')
    ax.barh(y_pos, merged['Homeownership rate_mortgage'].fillna(0)*100,
           left=merged['Homeownership rate_outright'].fillna(0)*100,
           color='#fb8072', alpha=0.8, label='Owners with mortgages')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(countries, fontsize=10)
    ax.set_xlabel('Homeownership Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Homeownership Rates by Type (EU27 + EFTA)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_homeownership_rates_by_type.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 6_homeownership_rates_by_type.png")

def create_rent_allowance_graph():
    """Rent allowance as share of household earnings"""
    print("\n[5] Creating rent allowance graph...")
    df = load_excel_data('ocde_Average of rent allowance.xlsx')
    if df is None:
        return
    
    # Handle country column
    if 'Unnamed: 0' in df.columns:
        df['country'] = df['Unnamed: 0']
    else:
        col_name = df.columns[0]
        df['country'] = df[col_name]
    
    df = df[df['country'].notna()]
    df = filter_target_countries(df)
    
    if df.empty:
        print("  No data for target countries")
        return
    
    # Find percentile columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("  Insufficient percentile data")
        return
    
    p10_col = numeric_cols[0]
    p50_col = numeric_cols[1]
    
    df_clean = df.dropna(subset=[p10_col, p50_col]).copy()
    df_clean = df_clean.sort_values(p10_col, ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    countries = df_clean['country'].tolist()
    x_pos = np.arange(len(countries))
    
    ax.scatter(x_pos, df_clean[p10_col].values*100, 
              color='#fb8072', s=150, alpha=0.8, label='10th percentile', zorder=5)
    ax.scatter(x_pos, df_clean[p50_col].values*100,
              color='white', s=100, alpha=0.8, edgecolor='black', 
              linewidth=1.5, label='50th percentile', zorder=6)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(countries, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Rent Allowance (% of gross wage)', fontsize=12, fontweight='bold')
    ax.set_title('Average Rent Allowance (EU27 + EFTA)\nBy income percentile, 2024 or latest', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '7_average_rent_allowance.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 7_average_rent_allowance.png")

def create_debt_change_vs_price_scatter():
    """Change in household debt vs change in house prices (2008-2021)"""
    print("\n[6] Creating debt change vs house price scatter...")
    df = load_excel_data('ocde_change in household debt to GDP.xlsx')
    if df is None:
        return
    
    if 'iso3' in df.columns:
        df['country'] = df['iso3'].apply(clean_country_name)
    elif 'variable' in df.columns:
        df['country'] = df['variable'].apply(clean_country_name)
    else:
        df['country'] = df.columns[0]
    
    df = df[df['country'].notna()]
    df = filter_target_countries(df)
    
    if df.empty:
        print("  No data for target countries")
        return
    
    # Look for the debt and price change columns
    debt_col = None
    price_col = None
    
    if 'd.FLH_GDP' in df.columns:
        debt_col = 'd.FLH_GDP'
    if 'd.HPI' in df.columns:
        price_col = 'd.HPI'
    
    if debt_col is None or price_col is None:
        print("  Missing required data columns")
        return
    
    df_clean = df.dropna(subset=[debt_col, price_col]).copy()
    
    if df_clean.empty:
        print("  No complete data pairs for target countries")
        return
    
    fig, ax = plt.subplots(figsize=(11, 8))
    
    for _, row in df_clean.iterrows():
        color = get_country_color(row['country'])
        ax.scatter(row[price_col]*100, row[debt_col]*100,
                  color=color, s=150, alpha=0.8, edgecolor='white', linewidth=1.5, zorder=5)
    
    # Add regression line
    x_vals = df_clean[price_col].values * 100
    y_vals = df_clean[debt_col].values * 100
    slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
    line = slope * x_vals + intercept
    ax.plot(x_vals, line, 'r--', alpha=0.4, linewidth=2.5, label=f'Trend (R2={r_value**2:.2f})')
    
    ax.set_xlabel('Change in House Prices (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Change in Household Debt to GDP (%)', fontsize=12, fontweight='bold')
    ax.set_title('Household Debt vs House Prices Change (2008-2021)\nEU27 + EFTA', 
                fontsize=14, fontweight='bold', pad=15)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.4, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.4, linewidth=1)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '8_debt_change_vs_price.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 8_debt_change_vs_price.png")

def create_elasticities_graph():
    """Housing supply elasticities by metropolitan area"""
    print("\n[7] Creating elasticities graph...")
    df = load_excel_data('ocde_elasticities.xlsx')
    if df is None:
        return
    
    if 'iso3' in df.columns:
        df['country'] = df['iso3'].apply(clean_country_name)
    elif 'Country' in df.columns:
        df['country'] = df['Country'].apply(clean_country_name)
    else:
        df['country'] = df.columns[0]
    
    df = df[df['country'].notna()]
    df = filter_target_countries(df)
    
    if df.empty:
        print("  No data for target countries")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 1:
        return
    
    elasticity_col = numeric_cols[0]
    df_clean = df.dropna(subset=[elasticity_col]).copy()
    
    countries_unique = sorted(df_clean['country'].unique())
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, country in enumerate(countries_unique):
        country_data = df_clean[df_clean['country'] == country]
        x_vals = [i + np.random.uniform(-0.15, 0.15) for _ in range(len(country_data))]
        
        color = get_country_color(country)
        ax.scatter(x_vals, country_data[elasticity_col]*100, 
                  color=color, s=100, alpha=0.7, edgecolor='white', linewidth=1)
        
        if len(country_data) > 1:
            avg = country_data[elasticity_col].mean()
            ax.plot([i-0.3, i+0.3], [avg*100, avg*100], 
                   color='red' if country=='Switzerland' else 'gray', 
                   linewidth=2, alpha=0.6)
    
    ax.set_xticks(range(len(countries_unique)))
    ax.set_xticklabels(countries_unique, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Housing Supply Elasticity', fontsize=12, fontweight='bold')
    ax.set_title('Housing Supply Elasticities by Metropolitan Area (EU27 + EFTA)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '9_housing_elasticities.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 9_housing_elasticities.png")

def create_elasticities_density_stacked_bar():
    """Housing supply elasticity density: stacked bar chart by elasticity range and country"""
    print("\n[9] Creating elasticity density stacked bar chart...")
    df = load_excel_data('ocde_elasticities.xlsx')
    if df is None:
        return
    
    # Clean country names
    if 'Country' in df.columns:
        df['country'] = df['Country'].apply(clean_country_name)
    else:
        df['country'] = df.columns[0]
    
    df = df[df['country'].notna()]
    df = filter_target_countries(df)
    
    if df.empty:
        print("  No data for target countries")
        return
    
    # Get elasticity column (usually 'Elast')
    elast_col = None
    for col in ['Elast', 'Elasticity', 'elasticity']:
        if col in df.columns:
            elast_col = col
            break
    
    if elast_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            elast_col = numeric_cols[0]
        else:
            print("  No elasticity column found")
            return
    
    df_clean = df.dropna(subset=[elast_col]).copy()
    
    # Create elasticity bins (density ranges)
    # Define elasticity ranges based on the data distribution
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-3.0', '3.0+']
    
    df_clean['elasticity_bin'] = pd.cut(df_clean[elast_col], bins=bins, labels=labels, right=False)
    
    # Count cities per elasticity bin and country
    pivot_data = df_clean.groupby(['elasticity_bin', 'country']).size().unstack(fill_value=0)
    
    # Create stacked bar chart with countries as colors
    # Size: 7 cm width (2.76 in) x 10 cm height (3.94 in)
    fig, ax = plt.subplots(figsize=(2.76, 3.94))
    
    # Get country colors
    countries = pivot_data.columns.tolist()
    colors = [get_country_color(country) for country in countries]
    
    pivot_data.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Elasticity Range', fontsize=8, fontweight='bold')
    ax.set_ylabel('Number of Cities', fontsize=8, fontweight='bold')
    ax.set_title('Housing supply elasticity per City 2003-2017', 
                fontsize=9, fontweight='bold', pad=8)
    ax.legend(title='Country', fontsize=7, title_fontsize=7, loc='upper right', framealpha=0.95, ncol=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('white')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '10_elasticity_density_stacked_bar.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 10_elasticity_density_stacked_bar.png")

def create_mobility_homeownership_scatter():
    """High-homeownership countries have low residential mobility"""
    print("\n[8] Creating mobility vs homeownership scatter...")
    
    mobility_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_homeownership residential mobility.xlsx')
    if not os.path.exists(mobility_file):
        print("  Missing data file")
        return
    
    try:
        df = pd.read_excel(mobility_file)
        
        if 'ISO3_Code' in df.columns:
            df['country'] = df['ISO3_Code'].apply(clean_country_name)
        elif 'iso3' in df.columns:
            df['country'] = df['iso3'].apply(clean_country_name)
        elif 'Country' in df.columns:
            df['country'] = df['Country']
        else:
            col_name = df.columns[0]
            df['country'] = df[col_name]
        
        df = df[df['country'].notna()]
        df = filter_target_countries(df)
        
        if df.empty:
            print("  No data for target countries")
            return
        
        # Get mobility and homeowner columns
        mobility_col = None
        homeowner_col = None
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            mobility_col = numeric_cols[0]
            homeowner_col = numeric_cols[1]
        elif 'mobility' in df.columns and 'homeowner' in df.columns:
            mobility_col = 'mobility'
            homeowner_col = 'homeowner'
        
        if mobility_col is None or homeowner_col is None:
            print("  Insufficient data columns")
            return
        
        df_clean = df.dropna(subset=[mobility_col, homeowner_col]).copy()
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for _, row in df_clean.iterrows():
            color = get_country_color(row['country'])
            ax.scatter(row[homeowner_col]*100, row[mobility_col]*100,
                      color=color, s=120, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add regression line
        x_vals = df_clean[homeowner_col].values * 100
        y_vals = df_clean[mobility_col].values * 100
        slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
        line = slope * x_vals + intercept
        ax.plot(x_vals, line, 'r--', alpha=0.5, linewidth=2, label=f'Trend (R2={r_value**2:.2f})')
        
        ax.set_xlabel('Homeownership Rate (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residential Mobility Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Residential Mobility vs Homeownership (EU27 + EFTA)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '10_mobility_vs_homeownership.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 10_mobility_vs_homeownership.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_housing_allowance_graph():
    """Share of households receiving housing allowance"""
    print("\n[9] Creating housing allowance graph...")
    df = load_excel_data('ocde_Share of households receiving a housing allowance.xlsx')
    if df is None:
        return
    
    if 'Unnamed: 0' in df.columns:
        df['country'] = df['Unnamed: 0']
    elif 'Country' in df.columns:
        df['country'] = df['Country']
    else:
        col_name = df.columns[0]
        df['country'] = df[col_name]
    
    df = df[df['country'].notna()]
    df = filter_target_countries(df)
    
    if df.empty:
        print("  No data for target countries")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 1:
        return
    
    value_col = numeric_cols[0]
    df_clean = df.dropna(subset=[value_col]).copy()
    df_clean = df_clean.sort_values(value_col, ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    countries = df_clean['country'].tolist()
    values = df_clean[value_col].values * 100
    colors = [get_country_color(c) for c in countries]
    
    ax.barh(range(len(countries)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=10)
    ax.set_xlabel('Share of Households (%)', fontsize=12, fontweight='bold')
    ax.set_title('Share Receiving Housing Allowance (EU27 + EFTA)\nBottom and third quintiles, 2024 or latest', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '11_housing_allowance_share.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] 11_housing_allowance_share.png")

def create_mortgage_debt_graph():
    """Mortgage debt is the largest part of household debt"""
    print("\n[10] Creating mortgage debt graph...")
    
    mortgage_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_Mortgage debt.xlsx')
    if not os.path.exists(mortgage_file):
        print("  Missing data file")
        return
    
    try:
        df = pd.read_excel(mortgage_file)
        
        # Clean country column
        df['country'] = df['Country'].apply(lambda x: clean_country_name(x) if pd.notna(x) else None)
        df = df[df['country'].notna()]
        df = filter_target_countries(df)
        
        if df.empty:
            print("  No data for target countries")
            return
        
        # Get mortgage share column
        mortgage_col = 'Share of total mortgage debt in total households debt'
        df_clean = df.dropna(subset=[mortgage_col]).copy()
        df_clean = df_clean.sort_values(mortgage_col, ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        countries = df_clean['country'].tolist()
        values = df_clean[mortgage_col].values * 100
        colors = [get_country_color(c) for c in countries]
        
        ax.barh(range(len(countries)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries, fontsize=10)
        ax.set_xlabel('Share of Mortgage in Total Household Debt (%)', fontsize=12, fontweight='bold')
        ax.set_title('Mortgage Debt Share (EU27 + EFTA)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, 100)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor('white')
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '12_mortgage_debt_share.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 12_mortgage_debt_share.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_variable_rate_mortgage_graph():
    """Share of variable-rate mortgage lending"""
    print("\n[11] Creating variable-rate mortgage graph...")
    
    var_rate_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_share of variable-rate mortgage.xlsx')
    if not os.path.exists(var_rate_file):
        print("  Missing data file")
        return
    
    try:
        # Read raw data - data starts at row 25 with iso3 at column 1
        df = pd.read_excel(var_rate_file, header=None, skiprows=25)
        
        # Extract iso3, var_type, value from columns 1, 2, 3
        df_clean = df[[1, 2, 3]].copy()
        df_clean.columns = ['iso3', 'var_type', 'value']
        df_clean = df_clean.dropna()
        
        # Skip header row if present
        df_clean = df_clean[df_clean['iso3'] != 'iso3']
        
        # Convert country code
        df_clean['country'] = df_clean['iso3'].apply(clean_country_name)
        df_clean = filter_target_countries(df_clean)
        
        if df_clean.empty:
            print("  No data for target countries")
            return
        
        # Filter for variable rate mortgages only
        df_var = df_clean[df_clean['var_type'].str.contains('Variable', case=False, na=False)].copy()
        if df_var.empty:
            print("  No variable rate data found")
            return
            
        df_var = df_var.groupby('country')['value'].mean().reset_index()
        df_var = df_var.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        countries = df_var['country'].tolist()
        values = df_var['value'].values * 100
        colors = [get_country_color(c) for c in countries]
        
        ax.barh(range(len(countries)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries, fontsize=10)
        ax.set_xlabel('Share of Variable-Rate Mortgages (%)', fontsize=12, fontweight='bold')
        ax.set_title('Variable-Rate Mortgage Share (EU27 + EFTA)\nNew Loans, 2022 or Latest', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, 100)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor('white')
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '13_variable_rate_mortgage.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 13_variable_rate_mortgage.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_building_stock_age_graph():
    """Share of building stock more than 50 years old"""
    print("\n[12] Creating building stock age graph...")
    
    age_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_building stock age.xlsx')
    if not os.path.exists(age_file):
        print("  Missing data file")
        return
    
    try:
        # Read raw data - data starts at row 25 with iso3 at column 1
        df = pd.read_excel(age_file, header=None, skiprows=25)
        
        # Extract iso3, construction_period, value from columns 1, 2, 3
        df_clean = df[[1, 2, 3]].copy()
        df_clean.columns = ['iso3', 'construction_period', 'value']
        df_clean = df_clean.dropna()
        
        # Skip header row if present
        df_clean = df_clean[df_clean['iso3'] != 'iso3']
        
        # Convert value to numeric
        df_clean['value'] = pd.to_numeric(df_clean['value'], errors='coerce')
        df_clean = df_clean.dropna(subset=['value'])
        
        # Convert country code
        df_clean['country'] = df_clean['iso3'].apply(clean_country_name)
        df_clean = filter_target_countries(df_clean)
        
        if df_clean.empty:
            print("  No data for target countries")
            return
        
        # Filter for "Before 1970" (50+ years old as of 2020)
        df_old = df_clean[df_clean['construction_period'].str.contains('Before 1970', case=False, na=False)].copy()
        if df_old.empty:
            print("  No building stock age data found")
            return
        
        # Group by country and average if multiple entries
        df_old = df_old.groupby('country')['value'].mean().reset_index()
        df_old = df_old.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        countries = df_old['country'].tolist()
        values = df_old['value'].values
        colors = [get_country_color(c) for c in countries]
        
        ax.barh(range(len(countries)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries, fontsize=10)
        ax.set_xlabel('Share of Building Stock (%)', fontsize=12, fontweight='bold')
        ax.set_title('Building Stock Built Before 1970 (>50 years old)\nEU27 + EFTA', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, 100)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor('white')
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '14_building_stock_age.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 14_building_stock_age.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_public_investment_graph():
    """Public direct investment vs incentives across time"""
    print("\n[13] Creating public investment graph...")
    
    invest_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_Public direct investment.xlsx')
    if not os.path.exists(invest_file):
        print("  Missing data file")
        return
    
    try:
        df = pd.read_excel(invest_file)
        
        # Extract year from datetime
        df['year'] = pd.to_datetime(df['year']).dt.year
        
        fig, ax = plt.subplots(figsize=(13, 7))
        
        # Plot each variable type as a line
        variables = df['variable'].unique()
        colors_map = {
            'Direct investment in housing development': '#e41a1c',
            'Public capital transfers for housing development': '#377eb8',
            'Housing allowances': '#4daf4a'
        }
        
        for var in variables:
            data = df[df['variable'] == var].sort_values('year')
            color = colors_map.get(var, '#999999')
            ax.plot(data['year'], data['value']*100, marker='o', linewidth=2.5, 
                   label=var, color=color, markersize=5, alpha=0.8)
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Public Investment as % GDP', fontsize=12, fontweight='bold')
        ax.set_title('Public Investment in Housing: OECD-30 Average\nDirect Investment, Capital Transfers, and Allowances', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '15_public_direct_investment.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 15_public_direct_investment.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_public_investment_housing_graph():
    """Public investment in housing declined since 2009"""
    print("\n[14] Creating public investment in housing graph...")
    
    housing_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_Public investment in housing.xlsx')
    if not os.path.exists(housing_file):
        print("  Missing data file")
        return
    
    try:
        # Read raw data with headers at row 24
        df = pd.read_excel(housing_file, header=None, skiprows=24)
        
        # Extract year, variable, value from columns 1, 2, 3
        df_clean = df[[1, 2, 3]].copy()
        df_clean.columns = ['year', 'variable', 'value']
        df_clean = df_clean.dropna()
        
        # Skip header row if present
        df_clean = df_clean[df_clean['year'] != 'year']
        
        # Convert to numeric
        df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
        df_clean['value'] = pd.to_numeric(df_clean['value'], errors='coerce')
        df_clean = df_clean.dropna()
        
        if df_clean.empty:
            print("  No valid data found")
            return
        
        fig, ax = plt.subplots(figsize=(13, 7))
        
        # Plot only total public investment line
        total_var = 'Total public investment in Housing\n\n and community amenities (%GDP)'
        data = df_clean[df_clean['variable'].str.contains('Total', case=False, na=False)].sort_values('year')
        
        if not data.empty:
            ax.plot(data['year'], data['value']*100, marker='o', linewidth=2.5, 
                   color='#377eb8', markersize=7, label='Total Public Investment', alpha=0.8)
            ax.fill_between(data['year'], data['value']*100, alpha=0.2, color='#377eb8')
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Public Investment as % GDP', fontsize=12, fontweight='bold')
        ax.set_title('Total Public Investment in Housing & Community Amenities\nOECD-30 Average', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '16_public_investment_housing_trend.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 16_public_investment_housing_trend.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_housing_chief_asset_graph():
    """Housing is the chief asset of the middle class"""
    print("\n[15] Creating housing as chief asset graph...")
    
    asset_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_Housing is the chief asset.xlsx')
    if not os.path.exists(asset_file):
        print("  Missing data file")
        return
    
    try:
        df = pd.read_excel(asset_file)
        
        # Convert country code
        df['country'] = df['Country'].apply(clean_country_name)
        
        # Filter for EU27 + EFTA countries + OECD average
        eu_efta_oecd = TARGET_COUNTRIES + ['OECD']
        df_filtered = df[df['country'].isin(eu_efta_oecd)].copy()
        
        if df_filtered.empty:
            print("  No data for target countries")
            return
        
        # Focus on housing (main residence) asset type
        df_housing = df_filtered[df_filtered['variable'] == 'Housing (main residence)'].copy()
        df_housing = df_housing.sort_values('value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 9))
        countries = df_housing['country'].tolist()
        values = df_housing['value'].values * 100
        colors = [get_country_color(c) if c != 'OECD' else '#80b1d3' for c in countries]
        
        ax.barh(range(len(countries)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries, fontsize=10)
        ax.set_xlabel('Share of Total Household Assets (%)', fontsize=12, fontweight='bold')
        ax.set_title('Housing (Main Residence) as Share of Household Assets\nEU27 + EFTA + OECD Average', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, 60)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor('white')
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '17_housing_chief_asset.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 17_housing_chief_asset.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_housing_price_supply_scatter():
    """Housing price against housing supply"""
    print("\n[16] Creating housing price-supply scatter...")
    
    supply_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_Housing price - supply.xlsx')
    if not os.path.exists(supply_file):
        print("  Missing data file")
        return
    
    try:
        # Read raw data - file has two side-by-side datasets
        df = pd.read_excel(supply_file, header=None, skiprows=26)
        
        # Left side: columns 1-2 (country codes, variable 1)
        left = df[[1, 2]].copy()
        left.columns = ['country', 'var1']
        left = left.dropna()
        left = left[left['country'] != 'variable']
        left['var1'] = pd.to_numeric(left['var1'], errors='coerce')
        left = left.dropna(subset=['var1'])
        
        # Right side: columns 11-12 (country codes, variable 2)
        right = df[[11, 12]].copy()
        right.columns = ['country', 'var2']
        right = right.dropna()
        right = right[right['country'] != 'iso3']
        right['var2'] = pd.to_numeric(right['var2'], errors='coerce')
        right = right.dropna(subset=['var2'])
        
        # Merge on country code
        df_merged = pd.merge(left, right, on='country', how='inner')
        
        # Convert country codes to names
        df_merged['country_name'] = df_merged['country'].apply(clean_country_name)
        df_merged = df_merged[df_merged['country_name'].notna()]
        
        # Filter to target countries
        df_merged = df_merged[df_merged['country_name'].isin(TARGET_COUNTRIES)]
        
        if df_merged.empty or len(df_merged) < 10:
            print("  Insufficient data")
            return
        
        fig, ax = plt.subplots(figsize=(11, 8))
        
        for _, row in df_merged.iterrows():
            country = row['country_name']
            color = get_country_color(country)
            ax.scatter(row['var1'], row['var2'],
                      color=color, s=150, alpha=0.8, edgecolor='white', linewidth=1.5, zorder=5)
        
        # Add regression line
        x_vals = df_merged['var1'].values
        y_vals = df_merged['var2'].values
        slope, intercept, r_value, _, _ = stats.linregress(x_vals, y_vals)
        line = slope * x_vals + intercept
        ax.plot(np.sort(x_vals), slope * np.sort(x_vals) + intercept, 'r--', alpha=0.4, linewidth=2.5, label=f'Trend (R²={r_value**2:.2f})')
        
        ax.set_xlabel('Housing Price Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Housing Supply', fontsize=12, fontweight='bold')
        ax.set_title('Housing Price vs Housing Supply (EU27 + EFTA)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('white')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '18_housing_price_supply_scatter.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 18_housing_price_supply_scatter.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_price_to_income_trend():
    """House price-to-income ratio over time"""
    print("\n[17] Creating price-to-income trend...")
    
    ratio_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_house price-to-income ratio.csv')
    if not os.path.exists(ratio_file):
        print("  Missing data file")
        return
    
    try:
        df = pd.read_csv(ratio_file)
        
        # Extract year from TIME_PERIOD (format: YYYY-QX)
        df['year'] = df['TIME_PERIOD'].str.extract(r'(\d{4})')[0].astype(int)
        df['country'] = df['REF_AREA'].apply(clean_country_name)
        df = df[df['country'].notna()]
        df = filter_target_countries(df)
        
        if df.empty:
            print("  No data for target countries")
            return
        
        # Use annual averages (group by year and country)
        df_clean = df.groupby(['year', 'country'])['OBS_VALUE'].mean().reset_index()
        df_clean.columns = ['year', 'country', 'ratio']
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot line for each country
        for country in sorted(df_clean['country'].unique()):
            data = df_clean[df_clean['country'] == country].sort_values('year')
            if len(data) > 1:
                color = get_country_color(country)
                ax.plot(data['year'], data['ratio'], marker='o', linewidth=1.5, 
                       label=country, color=color, markersize=4, alpha=0.7)
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price-to-Income Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Standardised House Price-to-Income Ratio Over Time\nEU27 + EFTA', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '19_price_to_income_trend.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 19_price_to_income_trend.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_price_to_income_latest():
    """House price-to-income ratio for latest year"""
    print("\n[18] Creating price-to-income latest year...")
    
    ratio_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_house price-to-income ratio.csv')
    if not os.path.exists(ratio_file):
        print("  Missing data file")
        return
    
    try:
        df = pd.read_csv(ratio_file)
        
        # Extract year from TIME_PERIOD
        df['year'] = df['TIME_PERIOD'].str.extract(r'(\d{4})')[0].astype(int)
        df['country'] = df['REF_AREA'].apply(clean_country_name)
        df = df[df['country'].notna()]
        df = filter_target_countries(df)
        
        if df.empty:
            print("  No data for target countries")
            return
        
        # Get latest year data
        latest_year = df['year'].max()
        df_latest = df[df['year'] == latest_year].copy()
        
        # Average quarterly data to annual
        df_latest = df_latest.groupby('country')['OBS_VALUE'].mean().reset_index()
        df_latest.columns = ['country', 'ratio']
        df_latest = df_latest.sort_values('ratio', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        countries = df_latest['country'].tolist()
        values = df_latest['ratio'].values
        colors = [get_country_color(c) for c in countries]
        
        bars = ax.barh(range(len(countries)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries, fontsize=10)
        ax.set_xlabel('Price-to-Income Ratio', fontsize=12, fontweight='bold')
        ax.set_title(f'Standardised House Price-to-Income Ratio ({latest_year})\nEU27 + EFTA', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor('white')
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.2, i, f'{v:.1f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '20_price_to_income_latest.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 20_price_to_income_latest.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_berd_total_stacked_bar():
    """BERD total by activity for 2022 as % of GDP"""
    print("\n[19] Creating BERD total stacked bar for 2022...")
    
    berd_file = os.path.join(EXTERNAL_DATA_DIR, 'oecd_BERD_total.csv')
    gdp_file = os.path.join(EXTERNAL_DATA_DIR, 'oecd_gdp.csv')
    
    if not os.path.exists(berd_file):
        print("  Missing BERD total file")
        return
    if not os.path.exists(gdp_file):
        print("  Missing GDP file")
        return
    
    try:
        # Translation mapping
        activity_translation = {
            'Activités de fabrication': 'Manufacturing',
            'Activités extractives': 'Mining and quarrying',
            'Agriculture, sylviculture et pêche': 'Agriculture, forestry and fishing',
            'Construction': 'Construction',
            'Électricité, gaz, eau et gestion des déchets': 'Electricity, gas, water and waste',
            'Services': 'Services'
        }
        
        # Load BERD data
        df_berd = pd.read_csv(berd_file)
        df_berd['year'] = pd.to_numeric(df_berd['TIME_PERIOD'], errors='coerce')
        df_berd = df_berd[df_berd['year'] == 2022].copy()
        
        # Load GDP data
        df_gdp = pd.read_csv(gdp_file)
        df_gdp['year'] = pd.to_numeric(df_gdp['TIME_PERIOD'], errors='coerce')
        df_gdp = df_gdp[df_gdp['year'] == 2022].copy()
        
        # Clean zone names
        df_berd['country'] = df_berd['Zone de référence'].str.strip()
        df_gdp['country'] = df_gdp['Zone de référence'].str.strip()
        
        # Pivot BERD by activity
        pivot_berd = df_berd.pivot_table(
            index='country',
            columns='Activité économique',
            values='OBS_VALUE',
            aggfunc='first'
        )
        # Normalize raw BERD data by dividing by 1e6
        pivot_berd = pivot_berd / 1e6
        
        # Get GDP by country
        df_gdp_clean = df_gdp[['country', 'OBS_VALUE']].drop_duplicates()
        df_gdp_clean.columns = ['country', 'gdp_value']
        df_gdp_clean = df_gdp_clean.set_index('country')
        
        # Calculate BERD as % of GDP
        pivot_pct = pivot_berd.div(df_gdp_clean['gdp_value'], axis=0) * 100
        pivot_pct = pivot_pct.fillna(0)
        
        # Filter to countries in the dataset
        pivot_pct = pivot_pct.dropna(how='all')
        
        # Remove countries where total BERD is 0 or NaN
        pivot_pct['total'] = pivot_pct.sum(axis=1)
        pivot_pct = pivot_pct[pivot_pct['total'] > 0]
        
        # Sort by total value (highest at top)
        pivot_pct = pivot_pct.sort_values('total', ascending=True)
        pivot_pct = pivot_pct.drop('total', axis=1)
        
        if pivot_pct.empty:
            print("  No valid data for 2022")
            return
        
        # Translate activity names
        pivot_pct.columns = [activity_translation.get(col, col) for col in pivot_pct.columns]
        
        # Reorder columns: Manufacturing first, then Services, then others
        col_order = ['Manufacturing', 'Services', 'Construction', 'Mining and quarrying', 
                     'Agriculture, forestry and fishing', 'Electricity, gas, water and waste']
        pivot_pct = pivot_pct[[col for col in col_order if col in pivot_pct.columns]]
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use component colors
        activity_colors = {
            'Manufacturing': '#ffd558',
            'Mining and quarrying': '#fb8072',
            'Agriculture, forestry and fishing': '#b3de69',
            'Construction': '#fdb462',
            'Electricity, gas, water and waste': '#bebada',
            'Services': '#8dd3c7'
        }
        
        colors = [activity_colors.get(col, '#cccccc') for col in pivot_pct.columns]
        pivot_pct.plot(kind='barh', stacked=True, ax=ax, color=colors, alpha=0.8)
        
        ax.set_xlabel('Business Enterprise R&D Expenditure (% of GDP)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Country', fontsize=12, fontweight='bold')
        ax.set_title('BERD by Economic Activity - 2022 (% of GDP)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(title='Economic Activity', fontsize=9, loc='lower right', framealpha=0.95)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '21_berd_total_stacked_2022.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 21_berd_total_stacked_2022.png")
    except Exception as e:
        print(f"  Error: {e}")
        
        ax.set_xlabel('Business Enterprise R&D Expenditure (% of GDP)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Country', fontsize=12, fontweight='bold')
        ax.set_title('BERD by Economic Activity - 2022 (% of GDP)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(title='Economic Activity', fontsize=9, loc='lower right', framealpha=0.95)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '21_berd_total_stacked_2022.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 21_berd_total_stacked_2022.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_berd_total_stacked_bar_2021():
    """BERD total by activity for 2021 as % of GDP"""
    print("\n[19b] Creating BERD total stacked bar for 2021...")
    
    berd_file = os.path.join(EXTERNAL_DATA_DIR, 'oecd_BERD_total.csv')
    gdp_file = os.path.join(EXTERNAL_DATA_DIR, 'oecd_gdp.csv')
    
    if not os.path.exists(berd_file):
        print("  Missing BERD total file")
        return
    if not os.path.exists(gdp_file):
        print("  Missing GDP file")
        return
    
    try:
        # Translation mapping
        activity_translation = {
            'Activités de fabrication': 'Manufacturing',
            'Activités extractives': 'Mining and quarrying',
            'Agriculture, sylviculture et pêche': 'Agriculture, forestry and fishing',
            'Construction': 'Construction',
            'Électricité, gaz, eau et gestion des déchets': 'Electricity, gas, water and waste',
            'Services': 'Services'
        }
        
        # Load BERD data
        df_berd = pd.read_csv(berd_file)
        df_berd['year'] = pd.to_numeric(df_berd['TIME_PERIOD'], errors='coerce')
        
        # Get 2021 data
        df_berd_2021 = df_berd[df_berd['year'] == 2021].copy()
        # Also load 2020 data for fallback
        df_berd_2020 = df_berd[df_berd['year'] == 2020].copy()
        
        # Load GDP data
        df_gdp = pd.read_csv(gdp_file)
        df_gdp['year'] = pd.to_numeric(df_gdp['TIME_PERIOD'], errors='coerce')
        df_gdp_2021 = df_gdp[df_gdp['year'] == 2021].copy()
        
        # Clean zone names
        df_berd_2021['country'] = df_berd_2021['Zone de référence'].str.strip()
        df_berd_2020['country'] = df_berd_2020['Zone de référence'].str.strip()
        df_gdp_2021['country'] = df_gdp_2021['Zone de référence'].str.strip()
        
        # Pivot BERD by activity for 2021
        pivot_berd = df_berd_2021.pivot_table(
            index='country',
            columns='Activité économique',
            values='OBS_VALUE',
            aggfunc='first'
        )
        # Normalize raw BERD data by dividing by 1e6
        pivot_berd = pivot_berd / 1e6
        
        # For USA Services, use 2020 data if missing
        if 'États-Unis' in pivot_berd.index and pd.isna(pivot_berd.loc['États-Unis', 'Services']):
            pivot_berd_2020 = df_berd_2020.pivot_table(
                index='country',
                columns='Activité économique',
                values='OBS_VALUE',
                aggfunc='first'
            )
            # Normalize 2020 fallback data by dividing by 1e6
            pivot_berd_2020 = pivot_berd_2020 / 1e6
            if 'États-Unis' in pivot_berd_2020.index and 'Services' in pivot_berd_2020.columns:
                pivot_berd.loc['États-Unis', 'Services'] = pivot_berd_2020.loc['États-Unis', 'Services']
        
        # Get GDP by country
        df_gdp_clean = df_gdp_2021[['country', 'OBS_VALUE']].drop_duplicates()
        df_gdp_clean.columns = ['country', 'gdp_value']
        df_gdp_clean = df_gdp_clean.set_index('country')
        
        # Calculate BERD as % of GDP
        pivot_pct = pivot_berd.div(df_gdp_clean['gdp_value'], axis=0) * 100
        pivot_pct = pivot_pct.fillna(0)
        
        # Filter to countries in the dataset
        pivot_pct = pivot_pct.dropna(how='all')
        
        # Remove countries where total BERD is 0 or NaN
        pivot_pct['total'] = pivot_pct.sum(axis=1)
        pivot_pct = pivot_pct[pivot_pct['total'] > 0]
        
        # Sort by total value (highest at top)
        pivot_pct = pivot_pct.sort_values('total', ascending=True)
        pivot_pct = pivot_pct.drop('total', axis=1)
        
        if pivot_pct.empty:
            print("  No valid data for 2021")
            return
        
        # Translate activity names
        pivot_pct.columns = [activity_translation.get(col, col) for col in pivot_pct.columns]
        
        # Reorder columns: Manufacturing first, then Services, then others
        col_order = ['Manufacturing', 'Services', 'Construction', 'Mining and quarrying', 
                     'Agriculture, forestry and fishing', 'Electricity, gas, water and waste']
        pivot_pct = pivot_pct[[col for col in col_order if col in pivot_pct.columns]]
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use component colors
        activity_colors = {
            'Manufacturing': '#ffd558',
            'Mining and quarrying': '#fb8072',
            'Agriculture, forestry and fishing': '#b3de69',
            'Construction': '#fdb462',
            'Electricity, gas, water and waste': '#bebada',
            'Services': '#8dd3c7'
        }
        
        colors = [activity_colors.get(col, '#cccccc') for col in pivot_pct.columns]
        pivot_pct.plot(kind='barh', stacked=True, ax=ax, color=colors, alpha=0.8)
        
        ax.set_xlabel('Business Enterprise R&D Expenditure (% of GDP)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Country', fontsize=12, fontweight='bold')
        ax.set_title('BERD by Economic Activity - 2021 (% of GDP)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(title='Economic Activity', fontsize=9, loc='lower right', framealpha=0.95)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor('white')
        ax.ticklabel_format(style='plain', axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '21_berd_total_stacked_2021.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 21_berd_total_stacked_2021.png")
    except Exception as e:
        print(f"  Error: {e}")
        
        ax.set_xlabel('Business Enterprise R&D Expenditure (% of GDP)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Country', fontsize=12, fontweight='bold')
        ax.set_title('BERD by Economic Activity - 2021 (% of GDP)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(title='Economic Activity', fontsize=9, loc='lower right', framealpha=0.95)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '21_berd_total_stacked_2021.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 21_berd_total_stacked_2021.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_berd_total_stacked_bar_2020():
    """BERD total by activity for 2020 as % of GDP"""
    print("\n[19c] Creating BERD total stacked bar for 2020...")
    
    berd_file = os.path.join(EXTERNAL_DATA_DIR, 'oecd_BERD_total.csv')
    gdp_file = os.path.join(EXTERNAL_DATA_DIR, 'oecd_gdp.csv')
    
    if not os.path.exists(berd_file):
        print("  Missing BERD total file")
        return
    if not os.path.exists(gdp_file):
        print("  Missing GDP file")
        return
    
    try:
        # Translation mapping
        activity_translation = {
            'Activités de fabrication': 'Manufacturing',
            'Activités extractives': 'Mining and quarrying',
            'Agriculture, sylviculture et pêche': 'Agriculture, forestry and fishing',
            'Construction': 'Construction',
            'Électricité, gaz, eau et gestion des déchets': 'Electricity, gas, water and waste',
            'Services': 'Services'
        }
        
        # Load BERD data
        df_berd = pd.read_csv(berd_file)
        df_berd['year'] = pd.to_numeric(df_berd['TIME_PERIOD'], errors='coerce')
        df_berd = df_berd[df_berd['year'] == 2020].copy()
        
        # Load GDP data
        df_gdp = pd.read_csv(gdp_file)
        df_gdp['year'] = pd.to_numeric(df_gdp['TIME_PERIOD'], errors='coerce')
        df_gdp = df_gdp[df_gdp['year'] == 2020].copy()
        
        # Clean zone names
        df_berd['country'] = df_berd['Zone de référence'].str.strip()
        df_gdp['country'] = df_gdp['Zone de référence'].str.strip()
        
        # Pivot BERD by activity
        pivot_berd = df_berd.pivot_table(
            index='country',
            columns='Activité économique',
            values='OBS_VALUE',
            aggfunc='first'
        )
        # Normalize raw BERD data by dividing by 1e6
        pivot_berd = pivot_berd / 1e6
        
        # Get GDP by country
        df_gdp_clean = df_gdp[['country', 'OBS_VALUE']].drop_duplicates()
        df_gdp_clean.columns = ['country', 'gdp_value']
        df_gdp_clean = df_gdp_clean.set_index('country')
        
        # Calculate BERD as % of GDP
        pivot_pct = pivot_berd.div(df_gdp_clean['gdp_value'], axis=0) * 100
        pivot_pct = pivot_pct.fillna(0)
        
        # Filter to countries in the dataset
        pivot_pct = pivot_pct.dropna(how='all')
        
        # Remove countries where total BERD is 0 or NaN
        pivot_pct['total'] = pivot_pct.sum(axis=1)
        pivot_pct = pivot_pct[pivot_pct['total'] > 0]
        
        # Sort by total value (highest at top)
        pivot_pct = pivot_pct.sort_values('total', ascending=True)
        pivot_pct = pivot_pct.drop('total', axis=1)
        
        if pivot_pct.empty:
            print("  No valid data for 2020")
            return
        
        # Translate activity names
        pivot_pct.columns = [activity_translation.get(col, col) for col in pivot_pct.columns]
        
        # Reorder columns: Manufacturing first, then Services, then others
        col_order = ['Manufacturing', 'Services', 'Construction', 'Mining and quarrying', 
                     'Agriculture, forestry and fishing', 'Electricity, gas, water and waste']
        pivot_pct = pivot_pct[[col for col in col_order if col in pivot_pct.columns]]
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use component colors
        activity_colors = {
            'Manufacturing': '#ffd558',
            'Mining and quarrying': '#fb8072',
            'Agriculture, forestry and fishing': '#b3de69',
            'Construction': '#fdb462',
            'Electricity, gas, water and waste': '#bebada',
            'Services': '#8dd3c7'
        }
        
        colors = [activity_colors.get(col, '#cccccc') for col in pivot_pct.columns]
        pivot_pct.plot(kind='barh', stacked=True, ax=ax, color=colors, alpha=0.8)
        
        ax.set_xlabel('Business Enterprise R&D Expenditure (% of GDP)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Country', fontsize=12, fontweight='bold')
        ax.set_title('BERD by Economic Activity - 2020 (% of GDP)', fontsize=14, fontweight='bold', pad=15)
        ax.legend(title='Economic Activity', fontsize=9, loc='lower right', framealpha=0.95)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '21_berd_total_stacked_2020.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  [SAVED] 21_berd_total_stacked_2020.png")
    except Exception as e:
        print(f"  Error: {e}")

def create_berd_construction_area():
    """BERD construction area graphs for USA and France"""
    print("\n[20] Creating BERD construction area graphs...")
    
    berd_file = os.path.join(EXTERNAL_DATA_DIR, 'oecd_BERD_construction.csv')
    gdp_file = os.path.join(EXTERNAL_DATA_DIR, 'oecd_gdp.csv')
    
    if not os.path.exists(berd_file):
        print("  Missing BERD construction file")
        return
    if not os.path.exists(gdp_file):
        print("  Missing GDP file")
        return
    
    try:
        # Load data
        df_berd = pd.read_csv(berd_file)
        df_gdp = pd.read_csv(gdp_file)
        
        # Clean data
        df_berd['year'] = pd.to_numeric(df_berd['TIME_PERIOD'], errors='coerce')
        df_gdp['year'] = pd.to_numeric(df_gdp['TIME_PERIOD'], errors='coerce')
        df_berd['value'] = pd.to_numeric(df_berd['OBS_VALUE'], errors='coerce')
        df_gdp['value'] = pd.to_numeric(df_gdp['OBS_VALUE'], errors='coerce')
        
        df_berd = df_berd.dropna(subset=['year', 'value'])
        df_gdp = df_gdp.dropna(subset=['year', 'value'])
        
        # Activities to include
        activities = ['Construction', 'Activités immobilières', 'Électricité, gaz, eau et gestion des déchets']
        
        # Create area graphs for each country
        for country_name in ['États-Unis', 'France']:
            print(f"    Creating {country_name} area graph...")
            
            # Filter BERD by country and activities
            df_country = df_berd[
                (df_berd['Zone de référence'] == country_name) & 
                (df_berd['Activité économique'].isin(activities))
            ].copy()
            
            # Filter GDP by country
            df_gdp_country = df_gdp[df_gdp['Zone de référence'] == country_name].copy()
            
            if df_country.empty or df_gdp_country.empty:
                print(f"      Missing data for {country_name}")
                continue
            
            # Pivot BERD data
            pivot_berd = df_country.pivot_table(
                index='year',
                columns='Activité économique',
                values='value',
                aggfunc='first'
            )
            # Normalize raw BERD data by dividing by 1e6
            pivot_berd = pivot_berd / 1e6
            
            # Get GDP by year
            gdp_by_year = df_gdp_country.groupby('year')['value'].mean()
            
            # Calculate BERD as % of GDP
            pct_gdp = pivot_berd.div(gdp_by_year, axis=0) * 100
            pct_gdp = pct_gdp.fillna(0)
            pct_gdp = pct_gdp.sort_index()
            
            if pct_gdp.empty:
                print(f"      No valid data for {country_name}")
                continue
            
            # Reorder columns
            col_order = [col for col in activities if col in pct_gdp.columns]
            pct_gdp = pct_gdp[col_order]
            
            # Create area chart
            fig, ax = plt.subplots(figsize=(13, 7))
            
            # Colors for construction activities
            colors_map = {
                'Construction': '#fdb462',
                'Activités immobilières': '#80b1d3',
                'Électricité, gaz, eau et gestion des déchets': '#bebada'
            }
            colors = [colors_map.get(col, '#cccccc') for col in pct_gdp.columns]
            
            ax.stackplot(pct_gdp.index, pct_gdp.T, labels=pct_gdp.columns, 
                        colors=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('% of GDP', fontsize=12, fontweight='bold')
            ax.set_title(f'BERD Construction Activities - {country_name} (% of GDP)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('white')
            
            plt.tight_layout()
            filename = f'22_berd_construction_{country_name.lower()}_area.png'
            plt.savefig(os.path.join(OUTPUT_DIR, filename), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    [SAVED] {filename}")
    except Exception as e:
        print(f"  Error: {e}")

def main():
    """Generate all visualizations"""
    print("\n" + "="*70)
    print("OECD Analysis - EU27 + EFTA Countries (31 total)")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nData location: {EXTERNAL_DATA_DIR}")
    print(f"Output location: {OUTPUT_DIR}")
    
    print("\nAvailable files:")
    if os.path.exists(EXTERNAL_DATA_DIR):
        files = [f for f in os.listdir(EXTERNAL_DATA_DIR) if f.startswith('ocde_')]
        for f in sorted(files):
            print(f"  - {f}")
    
    print("\nGenerating visualizations...")
    create_household_debt_graph()
    create_house_price_graphs()
    create_social_rental_graph()
    create_homeownership_graph()
    create_rent_allowance_graph()
    create_debt_change_vs_price_scatter()
    create_elasticities_graph()
    create_elasticities_density_stacked_bar()
    create_mobility_homeownership_scatter()
    create_housing_allowance_graph()
    create_mortgage_debt_graph()
    create_variable_rate_mortgage_graph()
    create_building_stock_age_graph()
    create_public_investment_graph()
    create_public_investment_housing_graph()
    create_housing_chief_asset_graph()
    create_housing_price_supply_scatter()
    create_price_to_income_trend()
    create_price_to_income_latest()
    create_berd_total_stacked_bar()
    create_berd_total_stacked_bar_2021()
    create_berd_total_stacked_bar_2020()
    create_berd_construction_area()
    
    print("\n" + "="*70)
    print("All 25 visualizations completed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
