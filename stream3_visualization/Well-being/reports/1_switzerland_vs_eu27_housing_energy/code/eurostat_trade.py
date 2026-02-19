"""
Eurostat Trade Visualization Script
Generates trade visualizations based on Eurostat trade data for Switzerland vs EU27 analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set font
plt.rcParams['font.family'] = 'Arial'

# Define paths
BASE_DIR = r'c:\Users\valentin.stuhlfauth\Desktop\Git D4G\13_democratiser_sobriete\stream3_visualization\Well-being\reports\1_switzerland_vs_eu27_housing_energy'
IMPORT_DATA_PATH = os.path.join(BASE_DIR, 'external_data', 'eurostat_trade_import.csv')
EXPORT_DATA_PATH = os.path.join(BASE_DIR, 'external_data', 'eurostat_trade_export.csv')
GDP_PATH = r'c:\Users\valentin.stuhlfauth\Desktop\Git D4G\13_democratiser_sobriete\stream3_visualization\Well-being\reports\3_eu_analysis_with_examples\external_data\eurostat_gdp_current_price.csv'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'graphs', 'EUROSTAT_trade')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define aggregate partners to exclude (only show individual countries)
AGGREGATE_PARTNERS = [
    'All countries of the world',
    'Extra-EU', 'Extra-EU27 (from 2020)', 'Extra-euro area',
    'Intra-EU', 'Intra-EU27 (from 2020)', 'Intra-euro area', 
    'Countries and territories not specified for commercial or military reasons in the framework of extra-Union trade',
    'Countries and territories not specified for commercial or military reasons in the framework of intra-Union trade',
    'Countries and territories not specified within the framework of extra-Union trade',
    'Countries and territories not specified within the framework of intra-Union trade',
    'Stores and provisions within the framework of extra-Union trade',
    'Stores and provisions within the framework of intra-Union trade',
    'Extra-euro area - 21 countries (from 2026)',
    'Intra-euro area - 21 countries (from 2026)',
    'United States Minor Outlying Islands'  # Exclude to avoid duplicate with main US
]

# Define constants
TARGET_YEAR = 2024  # Using 2024 as it has more complete data than 2025
N_PARTNERS = 15

# Define EU27 and EFTA countries
EU27_COUNTRIES = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 
    'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 
    'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 
    'Spain', 'Sweden'
]

EFTA_COUNTRIES = ['Switzerland', 'Norway', 'Iceland', 'Liechtenstein']

# Color palette - replacing light yellow with light brown for mineral fuels (22 unique colors)
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3', 
                  '#80b1d3', '#D2B48C', '#ffb3a7', '#a6d155', '#ffc274', '#d4c8e8', '#7fc5b8', '#fff5a0', 
                  '#6fa5c7', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']
EU_27_COLOR = '#80b1d3'
SWITZERLAND_COLOR = '#ffd558'
COLORS = COUNTRY_COLORS  # For backward compatibility
BALANCE_COLOR = '#fb8072'

# Product mapping for cleaner labels
PRODUCT_MAPPING = {
    'Food and live animals': 'Food & Live Animals',
    'Beverages and tobacco': 'Beverages & Tobacco',
    'Crude materials, inedible, except fuels': 'Raw Materials',
    'Mineral fuels, lubricants and related materials': 'Mineral Fuels',
    'Animal and vegetable oils, fats and waxes': 'Oils & Fats',
    'Chemicals and related products, n.e.s.': 'Chemicals',
    'Manufactured goods classified chiefly by material': 'Manufactured Goods',
    'Machinery and transport equipment': 'Machinery & Transport',
    'Miscellaneous manufactured articles': 'Misc. Manufactured',
    'Commodities and transactions not classified elsewhere in the SITC': 'Other Commodities'
}

# Country name to code mapping using ISO 3166-1 alpha-2 codes
COUNTRY_CODES = {
    'Albania': 'AL',
    'Austria': 'AT',
    'Belgium (incl. Luxembourg \'LU\' -> 1998)': 'BE',
    'Bosnia and Herzegovina': 'BA',
    'Bulgaria': 'BG',
    'Croatia': 'HR',
    'Cyprus': 'CY',
    'Czechia': 'CZ',
    'Denmark': 'DK',
    'Estonia': 'EE',
    'Finland': 'FI',
    'France (incl. Saint Barthélemy \'BL\' -> 2012; incl. French Guiana \'GF\', Guadeloupe \'GP\', Martinique \'MQ\', Réunion \'RE\' from 1997; incl. Mayotte \'YT\' from 2014)': 'FR',
    'Germany (incl. German Democratic Republic \'DD\' from 1991)': 'DE',
    'Greece': 'GR',
    'Hungary': 'HU',
    'Ireland (Eire)': 'IE',
    'Italy (incl. San Marino \'SM\' -> 1993)': 'IT',
    'Latvia': 'LV',
    'Lithuania': 'LT',
    'Luxembourg': 'LU',
    'Malta': 'MT',
    'Netherlands': 'NL',
    'North Macedonia': 'MK',
    'Poland': 'PL',
    'Portugal': 'PT',
    'Romania': 'RO',
    'Slovakia': 'SK',
    'Slovenia': 'SI',
    'Spain (incl. Canary Islands \'XB\' from 1997)': 'ES',
    'Sweden': 'SE',
    'Switzerland': 'CH',
    'Switzerland (incl. Liechtenstein \'LI\' -> 1994)': 'CH',
    'Norway (incl. Svalbard and Jan Mayen \'SJ\' -> 1994 and again from 1997)': 'NO',
    'Iceland': 'IS',
    'Liechtenstein': 'LI',
    'United States (incl. Navassa Island (part of \'UM\') from 1995 -> 2000)': 'US',
    'United Kingdom': 'GB',
    'United Kingdom (Northern Ireland)': 'GB',
    'Georgia': 'GE',
    'Moldova, Republic of': 'MD',
    'Montenegro': 'ME',
    'Serbia': 'RS',
    'Ukraine': 'UA',
    'Kosovo': 'XK',
    'T�rkiye': 'TR',
    'Türkiye': 'TR',
    'European Union - 27 countries (AT, BE, BG, CY, CZ, DE, DK, EE, ES, FI, FR, GR, HR, HU, IE, IT, LT, LU, LV, MT, NL, PL, PT, RO, SE, SI, SK)': 'EU',
    'Euro area (AT-01/1999, BE-01/1999, CY-01/2008, DE-01/1999, EE-01/2011, ES-01/1999, FI-01/1999, FR-01/1999, GR-01/2001, HR-01/2023, IE-01/1999, IT-01/1999, LT-01/2015, LU-01/1999, LV-01/2014, MT-01/2008, NL-01/1999, PT-01/1999, SI-01/2007, SK-01/2009)': 'EA',
    'Euro area - 21 countries (AT, BE, BG, CY, DE, EE, ES, FI, FR, GR, HR, IE, IT, LT, LU, LV, MT, NL, PT, SI, SK)': 'EA',
    # Additional common countries that might appear
    'China': 'CN',
    'Japan': 'JP',
    'India': 'IN',
    'Brazil': 'BR',
    'Canada': 'CA',
    'Australia': 'AU',
    'Russia': 'RU',
    'South Korea': 'KR',
    'Mexico': 'MX',
    'Argentina': 'AR',
    'Chile': 'CL',
    'Israel': 'IL',
    'New Zealand': 'NZ',
    'South Africa': 'ZA',
    'Egypt': 'EG',
    'Morocco': 'MA',
    'Tunisia': 'TN',
    'Algeria': 'DZ',
    'Nigeria': 'NG',
    'Thailand': 'TH',
    'Vietnam': 'VN',
    'United Arab Emirates': 'AE',
    'Indonesia': 'ID',
    'Malaysia': 'MY',
    'Philippines': 'PH',
    'Singapore': 'SG',
    'Taiwan': 'TW',
    'Hong Kong': 'HK'
}

# Country name mapping for GDP data matching
COUNTRY_NAME_MAPPING = {
    'Belgium (incl. Luxembourg \'LU\' -> 1998)': 'Belgium',
    'France (incl. Saint Barthélemy \'BL\' -> 2012; incl. French Guiana \'GF\', Guadeloupe \'GP\', Martinique \'MQ\', Réunion \'RE\' from 1997; incl. Mayotte \'YT\' from 2014)': 'France',
    'Germany (incl. German Democratic Republic \'DD\' from 1991)': 'Germany',
    'Ireland (Eire)': 'Ireland',
    'Italy (incl. San Marino \'SM\' -> 1993)': 'Italy',
    'Spain (incl. Canary Islands \'XB\' from 1997)': 'Spain',
    'Switzerland (incl. Liechtenstein \'LI\' -> 1994)': 'Switzerland',
    'Norway (incl. Svalbard and Jan Mayen \'SJ\' -> 1994 and again from 1997)': 'Norway',
    'United States (incl. Navassa Island (part of \'UM\') from 1995 -> 2000)': 'United States',
    'United Kingdom (Northern Ireland)': 'United Kingdom',
    'T�rkiye': 'T�rkiye',
    'Türkiye': 'T�rkiye'
}

def load_gdp_data(year=TARGET_YEAR):
    """
    Load GDP data for the specified year.
    """
    print(f"Loading GDP data for {year}...")
    gdp_df = pd.read_csv(GDP_PATH)
    
    # Filter for the target year and convert to standard units (EUR, not million EUR)
    gdp_filtered = gdp_df[gdp_df['TIME_PERIOD'] == year].copy()
    gdp_filtered['gdp_eur'] = gdp_filtered['OBS_VALUE'] * 1_000_000  # Convert million EUR to EUR
    
    return gdp_filtered[['geo', 'gdp_eur']].rename(columns={'geo': 'country'})


def match_country_names(trade_country_name):
    """
    Map trade data country names to GDP data country names.
    """
    # First check direct mapping
    if trade_country_name in COUNTRY_NAME_MAPPING:
        return COUNTRY_NAME_MAPPING[trade_country_name]
    
    # Otherwise, return as-is (for countries like Austria, Denmark, etc.)
    return trade_country_name


def get_country_code(country_name):
    """
    Get country code for labeling.
    """
    return COUNTRY_CODES.get(country_name, country_name[:3].upper())


def load_trade_data():
    """Load and preprocess trade data from import and export datasets."""
    print("Loading Eurostat trade data from import and export files...")
    
    # Load both import and export data
    import_df = pd.read_csv(IMPORT_DATA_PATH)
    export_df = pd.read_csv(EXPORT_DATA_PATH)
    
    # Combine both datasets
    df = pd.concat([import_df, export_df], ignore_index=True)
    
    # Rename columns for easier handling
    df = df.rename(columns={
        'TIME_PERIOD': 'year',
        'OBS_VALUE': 'value'
    })
    
    print(f"Available years: {sorted(df['year'].unique())}")
    print(f"Available reporters: {len(df['reporter'].unique())} countries")
    
    # Filter for VALUE_EUR indicator only
    df = df[df['indicators'] == 'VALUE_EUR'].copy()
    
    # Convert value to numeric, handling any non-numeric values
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    
    return df

def get_eu27_trade_by_product(df, year=TARGET_YEAR):
    """
    Get EU27 trade data aggregated by product type and partner.
    """
    print(f"Processing EU27 trade data for {year}...")
    
    # Filter for EU27 as aggregate reporter and target year
    eu27_df = df[
        (df['reporter'] == 'European Union - 27 countries (AT, BE, BG, CY, CZ, DE, DK, EE, ES, FI, FR, GR, HR, HU, IE, IT, LT, LU, LV, MT, NL, PL, PT, RO, SE, SI, SK)') & 
        (df['year'] == year) & 
        (df['product'] != 'TOTAL')  # Exclude total to get individual products
    ].copy()
    
    # Aggregate EU27 data by partner and product
    if eu27_df.empty:
        print("No EU27 data found for the specified year")
        return pd.DataFrame()
    
    # Filter out aggregate partners - only show individual countries
    eu27_df = eu27_df[~eu27_df['partner'].isin(AGGREGATE_PARTNERS)].copy()
    
    if eu27_df.empty:
        print("No individual country data found after filtering aggregates")
        return pd.DataFrame()
    
    eu27_agg = eu27_df.groupby(['partner', 'product', 'flow'])['value'].sum().reset_index()
    
    # Pivot to get imports and exports as columns
    trade_pivot = eu27_agg.pivot_table(
        index=['partner', 'product'], 
        columns='flow', 
        values='value', 
        fill_value=0
    ).reset_index()
    
    # Clean column names
    trade_pivot.columns.name = None
    if 'EXPORT' not in trade_pivot.columns:
        trade_pivot['EXPORT'] = 0
    if 'IMPORT' not in trade_pivot.columns:
        trade_pivot['IMPORT'] = 0
    
    return trade_pivot

def get_switzerland_trade_by_product(df, year=TARGET_YEAR, top_n=N_PARTNERS):
    """
    Get trade data with Switzerland as partner for top N countries.
    Note: Switzerland is now available as reporter in the new dataset.
    """
    print(f"Processing Switzerland trade data as reporter for {year}...")
    
    # Filter for Switzerland as reporter and target year
    ch_df = df[
        (df['reporter'] == "Switzerland (incl. Liechtenstein 'LI' -> 1994)") & 
        (df['year'] == year)
    ].copy()
    
    # First, get total trade volumes to identify top partner countries
    ch_total = ch_df[ch_df['product'] == 'TOTAL'].copy()
    
    # Filter out aggregate partners - only show individual countries
    ch_total = ch_total[~ch_total['partner'].isin(AGGREGATE_PARTNERS)].copy()
    
    if ch_total.empty:
        print("No individual country data found for Switzerland after filtering aggregates")
        return pd.DataFrame(), []
    
    ch_total_pivot = ch_total.pivot_table(
        index='partner',  # Partner countries for Switzerland's trade
        columns='flow', 
        values='value', 
        fill_value=0
    ).reset_index()
    
    # Clean column names and calculate total trade
    ch_total_pivot.columns.name = None
    if 'EXPORT' not in ch_total_pivot.columns:
        ch_total_pivot['EXPORT'] = 0
    if 'IMPORT' not in ch_total_pivot.columns:
        ch_total_pivot['IMPORT'] = 0
    
    ch_total_pivot['total_trade'] = ch_total_pivot['EXPORT'] + ch_total_pivot['IMPORT']
    
    # Get top N partner countries
    top_partners = ch_total_pivot.nlargest(top_n, 'total_trade')['partner'].tolist()
    
    # Now get detailed product breakdown for these top partners
    ch_products = ch_df[
        (ch_df['partner'].isin(top_partners)) & 
        (ch_df['product'] != 'TOTAL')
    ].copy()
    
    # Aggregate by partner and product
    ch_products_agg = ch_products.groupby(['partner', 'product', 'flow'])['value'].sum().reset_index()
    
    # Pivot to get imports and exports as columns
    trade_pivot = ch_products_agg.pivot_table(
        index=['partner', 'product'],  # Using partner again since Switzerland is reporter
        columns='flow', 
        values='value', 
        fill_value=0
    ).reset_index()
    
    # Clean column names
    trade_pivot.columns.name = None
    if 'EXPORT' not in trade_pivot.columns:
        trade_pivot['EXPORT'] = 0
    if 'IMPORT' not in trade_pivot.columns:
        trade_pivot['IMPORT'] = 0
    
    return trade_pivot, top_partners

def create_stacked_trade_chart(trade_data, partners_order, title, filename, reporter_name="EU27"):
    """
    Create stacked bar chart showing trade by product categories.
    """
    print(f"Creating stacked trade chart: {title}")
    
    # Get unique products
    products = trade_data['product'].unique()
    
    # Prepare data for plotting
    partners = partners_order
    n_partners = len(partners)
    
    # Initialize arrays for stacked bars
    exports_by_product = {}
    imports_by_product = {}
    
    for product in products:
        exports_by_product[product] = []
        imports_by_product[product] = []
        
        for partner in partners:
            partner_data = trade_data[
                (trade_data['partner'] == partner) & 
                (trade_data['product'] == product)
            ]
            
            if not partner_data.empty:
                exports_by_product[product].append(partner_data['EXPORT'].iloc[0])
                imports_by_product[product].append(partner_data['IMPORT'].iloc[0])
            else:
                exports_by_product[product].append(0)
                imports_by_product[product].append(0)
    
    # Calculate trade balance for each partner
    balance = []
    total_exports = []
    total_imports = []
    
    for partner in partners:
        partner_data = trade_data[trade_data['partner'] == partner]
        exp_total = partner_data['EXPORT'].sum()
        imp_total = partner_data['IMPORT'].sum()
        
        total_exports.append(exp_total)
        total_imports.append(imp_total)
        balance.append(exp_total - imp_total)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(n_partners)
    bar_width = 0.6
    
    # Plot stacked exports (positive)
    bottom_exp = np.zeros(n_partners)
    for i, product in enumerate(products):
        values = np.array(exports_by_product[product])
        label = PRODUCT_MAPPING.get(product, product)
        color = COLORS[i % len(COLORS)]
        ax.bar(x, values, bottom=bottom_exp, width=bar_width, 
               label=label, color=color)
        bottom_exp += values
    
    # Plot stacked imports (negative)
    bottom_imp = np.zeros(n_partners)
    for i, product in enumerate(products):
        values = -np.array(imports_by_product[product])  # Negative for visual distinction
        color = COLORS[i % len(COLORS)]
        ax.bar(x, values, bottom=bottom_imp, width=bar_width, color=color)
        bottom_imp += values
    
    # Plot trade balance line
    ax.plot(x, balance, color=BALANCE_COLOR, label='Trade Balance', 
            marker='o', linewidth=2, markersize=4)
    
    # Formatting
    ax.set_xticks(x)
    # Use country codes for x-axis labels
    partner_codes = [get_country_code(partner) for partner in partners]
    ax.set_xticklabels(partner_codes, rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Trade Value (EUR)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved chart to: {output_path}")
    
    plt.close()  # Close instead of show
    return fig

def create_stacked_trade_chart_gdp_pct(trade_data, partners_order, title, filename, reporter_gdp, reporter_name="EU27"):
    """
    Create stacked bar chart showing trade by product categories as % of GDP.
    """
    print(f"Creating stacked trade chart (% of GDP): {title}")
    
    # Get unique products
    products = trade_data['product'].unique()
    
    # Prepare data for plotting
    partners = partners_order
    n_partners = len(partners)
    
    # Initialize arrays for stacked bars
    exports_by_product = {}
    imports_by_product = {}
    
    for product in products:
        exports_by_product[product] = []
        imports_by_product[product] = []
        
        for partner in partners:
            partner_data = trade_data[
                (trade_data['partner'] == partner) & 
                (trade_data['product'] == product)
            ]
            
            if not partner_data.empty:
                # Convert to % of GDP
                exp_pct = (partner_data['EXPORT'].iloc[0] / reporter_gdp) * 100
                imp_pct = (partner_data['IMPORT'].iloc[0] / reporter_gdp) * 100
                exports_by_product[product].append(exp_pct)
                imports_by_product[product].append(imp_pct)
            else:
                exports_by_product[product].append(0)
                imports_by_product[product].append(0)
    
    # Calculate trade balance for each partner (% of GDP)
    balance = []
    total_exports = []
    total_imports = []
    
    for partner in partners:
        partner_data = trade_data[trade_data['partner'] == partner]
        exp_total = (partner_data['EXPORT'].sum() / reporter_gdp) * 100
        imp_total = (partner_data['IMPORT'].sum() / reporter_gdp) * 100
        
        total_exports.append(exp_total)
        total_imports.append(imp_total)
        balance.append(exp_total - imp_total)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(n_partners)
    bar_width = 0.6
    
    # Plot stacked exports (positive)
    bottom_exp = np.zeros(n_partners)
    for i, product in enumerate(products):
        values = np.array(exports_by_product[product])
        label = PRODUCT_MAPPING.get(product, product)
        color = COLORS[i % len(COLORS)]
        ax.bar(x, values, bottom=bottom_exp, width=bar_width, 
               label=label, color=color)
        bottom_exp += values
    
    # Plot stacked imports (negative)
    bottom_imp = np.zeros(n_partners)
    for i, product in enumerate(products):
        values = -np.array(imports_by_product[product])  # Negative for visual distinction
        color = COLORS[i % len(COLORS)]
        ax.bar(x, values, bottom=bottom_imp, width=bar_width, color=color)
        bottom_imp += values
    
    # Plot trade balance line
    ax.plot(x, balance, color=BALANCE_COLOR, label='Trade Balance', 
            marker='o', linewidth=2, markersize=4)
    
    # Formatting
    ax.set_xticks(x)
    # Use country codes for x-axis labels
    partner_codes = [get_country_code(partner) for partner in partners]
    ax.set_xticklabels(partner_codes, rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel(f'Trade Value (% of {reporter_name} GDP)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved chart to: {output_path}")
    
    plt.close()  # Close instead of show
    return fig

def get_total_trade_data(df, year=TARGET_YEAR):
    """
    Get total trade data for available EU and EFTA countries.
    """
    print(f"Processing total trade data for {year}...")
    
    # Get available reporters from the dataset
    available_reporters = df['reporter'].unique()
    
    # Filter for available countries and TOTAL product
    total_df = df[
        (df['year'] == year) & 
        (df['product'] == 'TOTAL')
    ].copy()
    
    if total_df.empty:
        print("No total trade data found for any country")
        return pd.DataFrame()
    
    # Aggregate by reporter and flow (sum across all partners)
    total_agg = total_df.groupby(['reporter', 'flow'])['value'].sum().reset_index()
    
    # Pivot to get imports and exports as columns
    trade_pivot = total_agg.pivot_table(
        index='reporter', 
        columns='flow', 
        values='value', 
        fill_value=0
    ).reset_index()
    
    # Clean column names
    trade_pivot.columns.name = None
    if 'EXPORT' not in trade_pivot.columns:
        trade_pivot['EXPORT'] = 0
    if 'IMPORT' not in trade_pivot.columns:
        trade_pivot['IMPORT'] = 0
    
    return trade_pivot

def create_import_export_scatter(trade_data, title, filename):
    """
    Create scatter plot with imports on x-axis and exports on y-axis.
    """
    print(f"Creating import-export scatter plot: {title}")
    
    # Get available reporters
    available_reporters = trade_data['reporter'].unique()
    print(f"Available countries for scatter plot: {len(available_reporters)}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Separate countries by type for different colors
    eu27_data = trade_data[trade_data['reporter'].str.contains('European Union')].copy()
    swiss_data = trade_data[trade_data['gdp_country'] == 'Switzerland'].copy()
    turkey_data = trade_data[trade_data['gdp_country'].isin(['Türkiye', 'T\ufffdrkiye'])].copy()
    other_data = trade_data[~trade_data['reporter'].str.contains('European Union') & 
                          (trade_data['gdp_country'] != 'Switzerland') &
                          (~trade_data['gdp_country'].isin(['Türkiye', 'T\ufffdrkiye']))].copy()
    
    # Plot EU countries (excluding non-EU countries)
    if not other_data.empty:
        ax.scatter(other_data['IMPORT_GDP_PCT'], other_data['EXPORT_GDP_PCT'], 
                  c=EU_27_COLOR, s=80, alpha=0.7, label='EU Countries')
    
    # Plot EU27 aggregate with special color
    if not eu27_data.empty:
        ax.scatter(eu27_data['IMPORT_GDP_PCT'], eu27_data['EXPORT_GDP_PCT'], 
                  c='#2E8B57', s=120, alpha=0.8, label='EU27 Aggregate', marker='s')
    
    # Plot Switzerland with special color
    if not swiss_data.empty:
        ax.scatter(swiss_data['IMPORT_GDP_PCT'], swiss_data['EXPORT_GDP_PCT'], 
                  c=SWITZERLAND_COLOR, s=140, alpha=0.9, label='Switzerland', marker='D')
    
    # Plot Turkey with different color to distinguish it
    if not turkey_data.empty:
        ax.scatter(turkey_data['IMPORT_GDP_PCT'], turkey_data['EXPORT_GDP_PCT'], 
                  c='#ff7f0e', s=100, alpha=0.8, label='Turkey (Non-EU)', marker='^')
    
    # Add country labels for ALL countries
    for _, row in trade_data.iterrows():
        # Use country codes for labels
        country_code = get_country_code(row['reporter'])
        
        # Adjust font size and positioning based on importance
        if row['gdp_country'] == 'Switzerland':
            fontsize, fontweight = 11, 'bold'
        elif 'European Union' in row['reporter']:
            fontsize, fontweight = 10, 'bold'
        else:
            fontsize, fontweight = 8, 'normal'
        
        ax.annotate(country_code, 
                   (row['IMPORT_GDP_PCT'], row['EXPORT_GDP_PCT']), 
                   xytext=(3, 3), textcoords='offset points', 
                   fontsize=fontsize, alpha=0.9, fontweight=fontweight)
    
    # Add diagonal line (trade balance = 0)
    max_val = max(trade_data['IMPORT_GDP_PCT'].max(), trade_data['EXPORT_GDP_PCT'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1, 
            label='Trade Balance = 0')
    
    # Set equal aspect ratio (1:1 scaling)
    ax.set_aspect('equal', adjustable='box')
    
    # Formatting
    ax.set_xlabel('Imports (% of GDP)', fontsize=12)
    ax.set_ylabel('Exports (% of GDP)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to: {output_path}")
    
    plt.close()  # Close instead of show
    return fig

def export_switzerland_trade_to_excel(ch_trade, ch_gdp, year, output_dir):
    """
    Export Switzerland trade data to Excel with specified column format.
    
    Args:
        ch_trade: DataFrame with Switzerland trade data by product and partner
        ch_gdp: Switzerland GDP value in EUR
        year: Year of the data
        output_dir: Directory to save the Excel file
        
    Returns:
        DataFrame: The formatted data that was exported
    """
    print(f"Formatting Switzerland trade data for Excel export...")
    
    # Create list to hold all rows
    excel_rows = []
    
    # Process each row in the trade data
    for _, row in ch_trade.iterrows():
        partner = row['partner']
        product = row['product']
        
        # Map product names using PRODUCT_MAPPING
        clean_product = PRODUCT_MAPPING.get(product, product)
        
        # Create export row
        export_row = {
            'visual_number': np.nan,
            'visual_name': f"Switzerland Trade - {clean_product}",
            'year': year,
            'filter_1': partner,
            'filter_2': 'Export',
            'filter_3': clean_product,
            'decile': np.nan,
            'value': (row['EXPORT'] / ch_gdp) * 100 if ch_gdp > 0 else 0,
            'unit': '% of GDP'
        }
        excel_rows.append(export_row)
        
        # Create import row
        import_row = {
            'visual_number': np.nan,
            'visual_name': f"Switzerland Trade - {clean_product}",
            'year': year,
            'filter_1': partner,
            'filter_2': 'Import',
            'filter_3': clean_product,
            'decile': np.nan,
            'value': (row['IMPORT'] / ch_gdp) * 100 if ch_gdp > 0 else 0,
            'unit': '% of GDP'
        }
        excel_rows.append(import_row)
    
    # Create DataFrame
    switzerland_trade_by_product_2024 = pd.DataFrame(excel_rows)
    
    # Sort by product (visual_name) and then by filters
    switzerland_trade_by_product_2024 = switzerland_trade_by_product_2024.sort_values(['visual_name', 'filter_1', 'filter_2'])
    
    # Export to Excel
    excel_filename = f"switzerland_trade_by_product_{year}.xlsx"
    excel_path = os.path.join(output_dir, excel_filename)
    
    switzerland_trade_by_product_2024.to_excel(excel_path, index=False)
    
    print(f"Excel file saved: {excel_path}")
    print(f"Total rows exported: {len(switzerland_trade_by_product_2024)}")
    
    return switzerland_trade_by_product_2024

def main():
    """Main function to generate all visualizations for both 2024 and 2025."""
    print("Starting Eurostat Trade Visualization Script")
    print("=" * 50)
    
    # Load data
    df = load_trade_data()
    print(f"Loaded {len(df)} trade records")
    
    # Generate charts for both years
    for year in [2024, 2025]:
        print(f"\n{'='*20} YEAR {year} {'='*20}")
        
        # Check if data exists for this year
        year_data = df[df['year'] == year]
        if year_data.empty:
            print(f"No data available for {year}")
            continue
        
        print(f"Processing {len(year_data):,} records for {year}")
        
        # 1. EU27 Per Product Types Chart
        print(f"\n1. Creating EU27 per product types chart for {year}...")
        eu27_trade = get_eu27_trade_by_product(df, year)
        
        if not eu27_trade.empty:
            # Load GDP data for EU27
            gdp_data = load_gdp_data(year)
            eu27_gdp_row = gdp_data[gdp_data['country'] == 'European Union - 27 countries (from 2020)']
            
            if not eu27_gdp_row.empty:
                eu27_gdp = eu27_gdp_row['gdp_eur'].iloc[0]
                
                # Get top partners by total trade volume
                partner_totals = eu27_trade.groupby('partner').agg({
                    'EXPORT': 'sum',
                    'IMPORT': 'sum'
                }).reset_index()
                partner_totals['total_trade'] = partner_totals['EXPORT'] + partner_totals['IMPORT']
                top_eu27_partners = partner_totals.nlargest(N_PARTNERS, 'total_trade')['partner'].tolist()
                
                # Filter trade data for top partners
                eu27_trade_filtered = eu27_trade[eu27_trade['partner'].isin(top_eu27_partners)]
                
                create_stacked_trade_chart_gdp_pct(
                    eu27_trade_filtered,
                    top_eu27_partners,
                    f"EU27 Trade by Product Type - Top {N_PARTNERS} Partners (% of GDP, {year})",
                    f"eu27_trade_by_product_{year}.png",
                    eu27_gdp,
                    "EU27"
                )
            else:
                print(f"No GDP data found for EU27 in {year}")
        else:
            print(f"No EU27 trade data found for {year}")
        
        # 2. Switzerland Trade Chart
        print(f"\n2. Creating Switzerland trade chart for {year}...")
        try:
            ch_trade, ch_top_partners = get_switzerland_trade_by_product(df, year, N_PARTNERS)
            
            if not ch_trade.empty:
                # Load GDP data for Switzerland
                gdp_data = load_gdp_data(year)
                ch_gdp_row = gdp_data[gdp_data['country'] == 'Switzerland']
                
                if not ch_gdp_row.empty:
                    ch_gdp = ch_gdp_row['gdp_eur'].iloc[0]
                    
                    create_stacked_trade_chart_gdp_pct(
                        ch_trade,
                        ch_top_partners,
                        f"Switzerland Trade by Product Type - Top {N_PARTNERS} Partners (% of GDP, {year})",
                        f"switzerland_trade_by_product_{year}.png",
                        ch_gdp,
                        "Switzerland"
                    )
                    
                    # Export Switzerland trade data to Excel for 2024
                    if year == 2024:
                        print("Exporting Switzerland trade data to Excel...")
                        switzerland_trade_by_product_2024 = export_switzerland_trade_to_excel(ch_trade, ch_gdp, year, OUTPUT_DIR)
                else:
                    print(f"No GDP data found for Switzerland in {year}")
            else:
                print(f"No Switzerland trade data found for {year}")
        except Exception as e:
            print(f"Error processing Switzerland data for {year}: {e}")
        
        # 3. Import-Export Scatter Plot with GDP percentages
        print(f"\n3. Creating import-export scatter plot with GDP percentages for {year}...")
        total_trade = get_total_trade_data(df, year)
        
        if not total_trade.empty:
            # Load GDP data for the same year only
            gdp_data = load_gdp_data(year)
            
            if not gdp_data.empty:
                # Map trade country names to GDP country names
                total_trade['gdp_country'] = total_trade['reporter'].apply(match_country_names)
                
                # Merge with GDP data
                trade_with_gdp = total_trade.merge(gdp_data, left_on='gdp_country', right_on='country', how='inner')
                
                if not trade_with_gdp.empty:
                    # Calculate trade as percentage of GDP
                    trade_with_gdp['IMPORT_GDP_PCT'] = (trade_with_gdp['IMPORT'] / trade_with_gdp['gdp_eur']) * 100
                    trade_with_gdp['EXPORT_GDP_PCT'] = (trade_with_gdp['EXPORT'] / trade_with_gdp['gdp_eur']) * 100
                    
                    create_import_export_scatter(
                        trade_with_gdp,
                        f"Total Trade as % of GDP: Imports vs Exports ({year})",
                        f"countries_import_export_scatter_{year}.png"
                    )
                else:
                    print(f"No country matches found between trade and GDP data for {year}")
            else:
                print(f"No GDP data found for {year}. Skipping scatter plot generation.")
        else:
            print(f"No total trade data found for {year}")
    
    print("\n" + "=" * 50)
    print("All visualizations completed successfully!")
    print(f"Charts saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
