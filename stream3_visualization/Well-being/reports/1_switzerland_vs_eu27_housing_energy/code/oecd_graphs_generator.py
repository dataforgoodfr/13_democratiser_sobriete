"""
oecd_graphs_generator.py - Generate graphs from OECD and EUROSTAT data sources

This script creates graphs from various external data sources located in the external_data folder,
with consistent styling and formatting. Each graph is saved as PNG files in the outputs/graphs/OECD folder.

Color scheme and styling follow the same patterns as ewbi_treatment.py for consistency.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Color palette consistent with existing code
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'
SWITZERLAND_COLOR = '#ffd558'  # Use yellow for Switzerland as requested

# Base paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, 'external_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'graphs', 'OECD')

# Global ISO3 to country name mapping
ISO3_MAPPING = {
    'CHE': 'Switzerland',
    'DEU': 'Germany', 
    'FRA': 'France',
    'ITA': 'Italy',
    'ESP': 'Spain',
    'NLD': 'Netherlands',
    'BEL': 'Belgium',
    'AUT': 'Austria',
    'SWE': 'Sweden',
    'DNK': 'Denmark',
    'FIN': 'Finland',
    'NOR': 'Norway',
    'GBR': 'United Kingdom',
    'PRT': 'Portugal',
    'GRC': 'Greece',
    'IRL': 'Ireland',
    'LUX': 'Luxembourg',
    'CZE': 'Czech Republic',
    'HUN': 'Hungary',
    'POL': 'Poland',
    'SVK': 'Slovakia',
    'SVN': 'Slovenia',
    'EST': 'Estonia',
    'LVA': 'Latvia',
    'LTU': 'Lithuania',
    'HRV': 'Croatia',
    'ROU': 'Romania',
    'BGR': 'Bulgaria',
    'CYP': 'Cyprus',
    'MLT': 'Malta',
    'USA': 'United States',
    'CAN': 'Canada',
    'JPN': 'Japan',
    'KOR': 'South Korea',
    'AUS': 'Australia',
    'NZL': 'New Zealand',
    'ISL': 'Iceland',
    'TUR': 'Turkey',
    'MEX': 'Mexico',
    'CHL': 'Chile',
    'ISR': 'Israel',
    'EU27': 'EU27',
    'EU28': 'EU-28',
    'OECD': 'OECD Average'
}

def load_excel_data(filename):
    """Load Excel data from external_data folder"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, filename)
    try:
        # Try to read the Excel file, handle different possible sheet structures
        excel_file = pd.ExcelFile(file_path)
        
        # Print available sheets for debugging
        print(f"Available sheets in {filename}: {excel_file.sheet_names}")
        
        # Try to read the first sheet
        df = pd.read_excel(file_path, sheet_name=0, header=0)
        print(f"Shape of data: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def clean_country_name(iso3_code):
    """Convert ISO3 codes to clean country names"""
    if pd.isna(iso3_code):
        return None
    
    iso3_code = str(iso3_code).strip().upper()
    
    # ISO3 to country name mapping
    iso3_mapping = {
        'CHE': 'Switzerland',
        'DEU': 'Germany', 
        'FRA': 'France',
        'ITA': 'Italy',
        'ESP': 'Spain',
        'NLD': 'Netherlands',
        'BEL': 'Belgium',
        'AUT': 'Austria',
        'SWE': 'Sweden',
        'DNK': 'Denmark',
        'FIN': 'Finland',
        'NOR': 'Norway',
        'GBR': 'United Kingdom',
        'PRT': 'Portugal',
        'GRC': 'Greece',
        'IRL': 'Ireland',
        'LUX': 'Luxembourg',
        'CZE': 'Czech Republic',
        'HUN': 'Hungary',
        'POL': 'Poland',
        'SVK': 'Slovakia',
        'SVN': 'Slovenia',
        'EST': 'Estonia',
        'LVA': 'Latvia',
        'LTU': 'Lithuania',
        'HRV': 'Croatia',
        'ROU': 'Romania',
        'BGR': 'Bulgaria',
        'CYP': 'Cyprus',
        'MLT': 'Malta',
        'USA': 'United States',
        'CAN': 'Canada',
        'JPN': 'Japan',
        'KOR': 'South Korea',
        'AUS': 'Australia',
        'NZL': 'New Zealand',
        'ISL': 'Iceland',
        'TUR': 'Turkey',
        'MEX': 'Mexico',
        'CHL': 'Chile',
        'ISR': 'Israel',
        'EU27': 'EU-27',
        'EU28': 'EU-28',
        'OECD': 'OECD Average'
    }
    
    return iso3_mapping.get(iso3_code, iso3_code)

def get_country_color(country):
    """Get consistent color for each country"""
    if country == 'Switzerland':
        return '#ffd558'  # Yellow for Switzerland
    elif country in ['EU-27', 'EU27', 'EU-28', 'OECD Average', 'OECD', 'European Union', 'Euro area', 'World']:
        return '#80b1d3'  # Blue for aggregates
    else:
        # Light teal for all other countries
        return '#8dd3c7'

def create_household_debt_graph():
    """
    Create a graph showing Household Debt to GDP Ratio by Country for 2021 and 2008
    
    Regular bar chart of debt classified by overall household debt in 2021/2022
    with dots for the first date (2008)
    """
    print("Creating Household Debt to GDP Ratio graph...")
    
    # Load the data
    df = load_excel_data('ocde_household debt to GDP.xlsx')
    if df is None:
        print("Could not load household debt data")
        return
    
    # Print data structure for debugging
    print("Data structure analysis:")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Available years: {sorted(df['year'].unique())}")
    print(f"Available countries: {sorted(df['iso3'].unique())}")
    
    # Filter for years closest to 2008 and 2021/2022
    available_years = sorted(df['year'].unique())
    early_year = 2008  # Should be available
    
    # Check if Switzerland has data for different years
    swiss_data = df[df['iso3'] == 'CHE']
    print(f"Switzerland data years: {sorted(swiss_data['year'].unique())}")
    
    # Prefer 2021 if available, otherwise use latest
    if 2021 in available_years:
        late_year = 2021
    else:
        late_year = max([y for y in available_years if y >= 2021])  # Latest year >= 2021
    
    print(f"Using {early_year} as early year")
    print(f"Using {late_year} as late year")
    
    # Filter data for these years
    df_early = df[df['year'] == early_year].copy()
    df_late = df[df['year'] == late_year].copy()
    
    print(f"Countries with {early_year} data: {len(df_early)}")
    print(f"Countries with {late_year} data: {len(df_late)}")
    
    # Convert ISO3 codes to country names
    df_early['country'] = df_early['iso3'].apply(clean_country_name)
    df_late['country'] = df_late['iso3'].apply(clean_country_name)
    
    # Remove entries where country name couldn't be resolved
    df_early = df_early[df_early['country'].notna()]
    df_late = df_late[df_late['country'].notna()]
    
    # Focus on European countries + some key comparisons
    european_countries = [
        'Switzerland', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 
        'Belgium', 'Austria', 'Sweden', 'Denmark', 'Finland', 'Norway',
        'United Kingdom', 'Portugal', 'Greece', 'Ireland', 'Luxembourg',
        'Czech Republic', 'Hungary', 'Poland', 'Slovakia', 'Slovenia',
        'Estonia', 'Latvia', 'Lithuania', 'Croatia', 'Romania', 'Bulgaria',
        'Cyprus', 'Malta', 'EU-27', 'EU27', 'United States', 'Canada', 'Japan', 'Australia'
    ]
    
    # Filter for relevant countries
    df_early = df_early[df_early['country'].isin(european_countries)]
    df_late = df_late[df_late['country'].isin(european_countries)]
    
    # Merge data on country (include both household and mortgage debt)
    merged_data = pd.merge(df_late[['country', 'Household debt to GDP', 'Household mortgage to GDP']], 
                          df_early[['country', 'Household debt to GDP', 'Household mortgage to GDP']], 
                          on='country', 
                          how='left', 
                          suffixes=(f'_{late_year}', f'_{early_year}'))
    
    # Sort by latest year value (ascending - biggest values at top for horizontal bars)
    merged_data = merged_data.sort_values(f'Household debt to GDP_{late_year}', ascending=True)
    
    # Remove any rows with missing data in the latest year
    merged_data = merged_data.dropna(subset=[f'Household debt to GDP_{late_year}'])
    
    print(f"Final dataset shape: {merged_data.shape}")
    print(f"Countries included: {merged_data['country'].tolist()}")
    
    # Check if Switzerland is included
    if 'Switzerland' in merged_data['country'].values:
        print("✓ Switzerland is included in the data")
    else:
        print("✗ Switzerland is missing from the final data")
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    countries = merged_data['country'].tolist()
    # Prepare household debt data
    household_late_values = merged_data[f'Household debt to GDP_{late_year}'].tolist()
    household_early_values = merged_data[f'Household debt to GDP_{early_year}'].tolist()
    # Prepare mortgage debt data
    mortgage_late_values = merged_data[f'Household mortgage to GDP_{late_year}'].tolist()
    mortgage_early_values = merged_data[f'Household mortgage to GDP_{early_year}'].tolist()
    
    # Convert ratios to percentages
    household_late_pct = [v * 100 for v in household_late_values]
    household_early_pct = [v * 100 if pd.notna(v) else np.nan for v in household_early_values]
    mortgage_late_pct = [v * 100 if pd.notna(v) else 0 for v in mortgage_late_values]
    mortgage_early_pct = [v * 100 if pd.notna(v) else np.nan for v in mortgage_early_values]
    
    # Create horizontal bar chart - household debt first (background), then mortgage debt (foreground)
    colors = [get_country_color(country) for country in countries]
    
    # Plot household debt (background bars)
    household_bars = plt.barh(range(len(countries)), household_late_pct, 
                             color=colors, alpha=0.4, height=0.6, label='Household Debt')
    
    # Plot mortgage debt (foreground bars)
    mortgage_bars = plt.barh(range(len(countries)), mortgage_late_pct, 
                            color=colors, alpha=1.0, height=0.6, label='Mortgage Debt')
    
    # Highlight Switzerland bars (no black border)
    for i, country in enumerate(countries):
        if country == 'Switzerland':
            household_bars[i].set_color('#ffd558')  # Yellow for Switzerland household
            household_bars[i].set_alpha(0.4)
            mortgage_bars[i].set_color('#ffd558')   # Yellow for Switzerland mortgage
            mortgage_bars[i].set_alpha(1.0)
    
    # Add dots for early year household debt
    for i, (early_val, country) in enumerate(zip(household_early_pct, countries)):
        if pd.notna(early_val):
            color = 'red' if country == 'Switzerland' else 'black'
            plt.scatter(early_val, i, color=color, s=120, zorder=5, marker='o', 
                       edgecolor='white', linewidth=1.5)
    
    # Add triangles for early year mortgage debt
    for i, (early_val, country) in enumerate(zip(mortgage_early_pct, countries)):
        if pd.notna(early_val):
            color = 'red' if country == 'Switzerland' else 'darkblue'
            plt.scatter(early_val, i, color=color, s=100, zorder=6, marker='^', 
                       edgecolor='white', linewidth=1.5)
    
    # Customize the plot
    plt.yticks(range(len(countries)), countries, fontsize=12)
    plt.xlabel('Household Debt to GDP Ratio (%)', fontsize=14, fontweight='bold')
    
    # Rephrased title as requested
    plt.title('Household Financial Leverage Across Countries\n' + 
              f'Total Debt as Share of GDP: {late_year} (bars) vs {early_year} (dots)', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Add grid
    plt.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.xlim(0, max(household_late_pct) * 1.15)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=10, alpha=1.0,
               label=f'Mortgage Debt {late_year}'),
        Line2D([0], [0], color='gray', linewidth=10, alpha=0.4,
               label=f'Total Household Debt {late_year}'),
        Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=10, 
               label=f'Household Debt {early_year}'),
        Line2D([0], [0], color='darkblue', marker='^', linestyle='None', markersize=10, 
               label=f'Mortgage Debt {early_year}')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    # Set white background
    plt.gca().set_facecolor('white')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'household_debt_to_gdp_ratio.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved graph: {output_path}")
    
    # Print some summary statistics
    swiss_data = merged_data[merged_data['country'] == 'Switzerland']
    if not swiss_data.empty:
        swiss_household_latest = swiss_data[f'Household debt to GDP_{late_year}'].iloc[0] * 100
        swiss_mortgage_latest = swiss_data[f'Household mortgage to GDP_{late_year}'].iloc[0] * 100 if pd.notna(swiss_data[f'Household mortgage to GDP_{late_year}'].iloc[0]) else 0
        swiss_household_early = swiss_data[f'Household debt to GDP_{early_year}'].iloc[0] * 100 if pd.notna(swiss_data[f'Household debt to GDP_{early_year}'].iloc[0]) else None
        swiss_mortgage_early = swiss_data[f'Household mortgage to GDP_{early_year}'].iloc[0] * 100 if pd.notna(swiss_data[f'Household mortgage to GDP_{early_year}'].iloc[0]) else None
        
        print(f"Switzerland {late_year}: Household {swiss_household_latest:.1f}%, Mortgage {swiss_mortgage_latest:.1f}%")
        if swiss_household_early and swiss_mortgage_early:
            print(f"Switzerland {early_year}: Household {swiss_household_early:.1f}%, Mortgage {swiss_mortgage_early:.1f}%")

def create_debt_vs_house_price_scatter():
    """
    Create a scatter plot showing the relationship between change in household debt to GDP 
    and change in nominal housing prices (2008-2021)
    
    X-axis: % change in household debt to GDP (2008-2021)
    Y-axis: % change in nominal housing prices (2008-2021)
    """
    print("Creating debt vs house price change scatter plot...")
    
    # Load the data
    df = load_excel_data('ocde_change in household debt to GDP.xlsx')
    if df is None:
        print("Could not load change in household debt data")
        return
    
    # Print data structure for debugging
    print("Data structure analysis:")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Clean country names
    df['country'] = df['iso3'].apply(clean_country_name)
    df = df[df['country'].notna()]
    
    # Focus on European countries + some key comparisons
    european_countries = [
        'Switzerland', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 
        'Belgium', 'Austria', 'Sweden', 'Denmark', 'Finland', 'Norway',
        'United Kingdom', 'Portugal', 'Greece', 'Ireland', 'Luxembourg',
        'Czech Republic', 'Hungary', 'Poland', 'Slovakia', 'Slovenia',
        'Estonia', 'Latvia', 'Lithuania', 'Croatia', 'Romania', 'Bulgaria',
        'Cyprus', 'Malta', 'United States', 'Canada', 'Japan', 'Australia'
    ]
    
    # Filter for relevant countries
    df_filtered = df[df['country'].isin(european_countries)].copy()
    
    # Remove rows with missing data (using actual column names)
    df_filtered = df_filtered.dropna(subset=['d.FLH_GDP', 'd.HPI'])
    
    print(f"Countries with complete data: {len(df_filtered)}")
    print(f"Countries included: {sorted(df_filtered['country'].tolist())}")
    
    if df_filtered.empty:
        print("No data available for the scatter plot")
        return
    
    # Check if Switzerland is included
    if 'Switzerland' in df_filtered['country'].values:
        print("✓ Switzerland is included in the scatter plot")
    else:
        print("✗ Switzerland is missing from the scatter plot")
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    
    # Get data for plotting (convert to percentages)
    x_values = [x * 100 for x in df_filtered['d.FLH_GDP'].tolist()]  # Debt change as percentage
    y_values = [y * 100 for y in df_filtered['d.HPI'].tolist()]      # House price change as percentage
    countries = df_filtered['country'].tolist()
    
    # Create scatter plot with country-specific colors
    colors = [get_country_color(country) for country in countries]
    
    # Plot all countries
    plt.scatter(x_values, y_values, c=colors, alpha=0.8, s=100, edgecolor='white', linewidth=1.5)
    
    # Highlight Switzerland if present
    swiss_data = df_filtered[df_filtered['country'] == 'Switzerland']
    if not swiss_data.empty:
        swiss_x = swiss_data['d.FLH_GDP'].iloc[0] * 100
        swiss_y = swiss_data['d.HPI'].iloc[0] * 100
        plt.scatter(swiss_x, swiss_y, color='#ffd558', s=200, edgecolor='black', linewidth=2, zorder=5)
        
        # Add label for Switzerland
        plt.annotate('Switzerland', (swiss_x, swiss_y), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add labels for a few other interesting countries
    for _, row in df_filtered.iterrows():
        country = row['country']
        if country in ['Germany', 'France', 'Netherlands', 'United Kingdom', 'United States'] and country != 'Switzerland':
            plt.annotate(country, (row['d.FLH_GDP'] * 100, row['d.HPI'] * 100), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Change in Household Debt to GDP Ratio (2008-2021, %)', fontsize=14, fontweight='bold')
    plt.ylabel('Change in Nominal Housing Prices (2008-2021, %)', fontsize=14, fontweight='bold')
    
    # Rephrased title
    plt.title('Financial Leverage vs Housing Market Dynamics\n' + 
              'Relationship between Debt Growth and House Price Appreciation (2008-2021)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add reference lines at zero
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Set background
    plt.gca().set_facecolor('white')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'debt_vs_house_price_changes.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved scatter plot: {output_path}")
    
    # Print some summary statistics
    if not swiss_data.empty:
        swiss_debt_change = swiss_data['d.FLH_GDP'].iloc[0] * 100
        swiss_price_change = swiss_data['d.HPI'].iloc[0] * 100
        print(f"Switzerland: Debt change {swiss_debt_change:.1f}%, Price change {swiss_price_change:.1f}%")

def create_real_house_price_graphs():
    """
    Create graphs showing real house price changes for two periods:
    1. Bar chart for 2002-2007 period
    2. Bar chart for 2017-2022 period
    3. Scatter plot showing relationship between the two periods
    """
    print("Creating real house price change graphs...")
    
    # Load both datasets
    df_2002_2007 = load_excel_data('ocde_real house price rise 2002-2007.xlsx')
    df_2017_2022 = load_excel_data('ocde_real house price rise 2017-2022.xlsx')
    
    if df_2002_2007 is None or df_2017_2022 is None:
        print("Could not load real house price data")
        return
    
    # Handle different column structures
    # 2002-2007 has 'variable' column with country codes
    # 2017-2022 has 'iso3' column with country codes
    if 'variable' in df_2002_2007.columns:
        df_2002_2007['iso3'] = df_2002_2007['variable']
    
    # Clean country names for both datasets
    df_2002_2007['country'] = df_2002_2007['iso3'].apply(clean_country_name)
    df_2017_2022['country'] = df_2017_2022['iso3'].apply(clean_country_name)
    
    # Remove entries where country name couldn't be resolved
    df_2002_2007 = df_2002_2007[df_2002_2007['country'].notna()]
    df_2017_2022 = df_2017_2022[df_2017_2022['country'].notna()]
    
    # Focus on European countries + some key comparisons
    european_countries = [
        'Switzerland', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 
        'Belgium', 'Austria', 'Sweden', 'Denmark', 'Finland', 'Norway',
        'United Kingdom', 'Portugal', 'Greece', 'Ireland', 'Luxembourg',
        'Czech Republic', 'Hungary', 'Poland', 'Slovakia', 'Slovenia',
        'Estonia', 'Latvia', 'Lithuania', 'Croatia', 'Romania', 'Bulgaria',
        'Cyprus', 'Malta', 'United States', 'Canada', 'Japan', 'Australia'
    ]
    
    # Filter for relevant countries
    df_2002_2007_filtered = df_2002_2007[df_2002_2007['country'].isin(european_countries)].copy()
    df_2017_2022_filtered = df_2017_2022[df_2017_2022['country'].isin(european_countries)].copy()
    
    # Identify the price change column (assume it's the numeric column that's not iso3 or country)
    price_col_2002 = [col for col in df_2002_2007_filtered.columns if col not in ['iso3', 'country'] and df_2002_2007_filtered[col].dtype in ['float64', 'int64']][0]
    price_col_2017 = [col for col in df_2017_2022_filtered.columns if col not in ['iso3', 'country'] and df_2017_2022_filtered[col].dtype in ['float64', 'int64']][0]
    
    # Rename the value columns to avoid conflicts in the scatter plot
    df_2002_2007_filtered = df_2002_2007_filtered.rename(columns={price_col_2002: 'price_2002_2007'})
    df_2017_2022_filtered = df_2017_2022_filtered.rename(columns={price_col_2017: 'price_2017_2022'})
    
    print(f"Using columns: price_2002_2007 (2002-2007), price_2017_2022 (2017-2022)")
    
    # Create Graph 1: Bar chart for 2002-2007
    create_house_price_bar_chart(df_2002_2007_filtered, 'price_2002_2007', '2002-2007', 'real_house_prices_2002_2007.png')
    
    # Create Graph 2: Bar chart for 2017-2022
    create_house_price_bar_chart(df_2017_2022_filtered, 'price_2017_2022', '2017-2022', 'real_house_prices_2017_2022.png')
    
    # Create Graph 3: Scatter plot comparing both periods
    create_house_price_comparison_scatter(df_2002_2007_filtered, df_2017_2022_filtered, 'price_2002_2007', 'price_2017_2022')

def create_house_price_bar_chart(df, price_column, period, filename):
    """
    Create a horizontal bar chart for house price changes in a specific period
    """
    # Remove missing data
    df_clean = df.dropna(subset=[price_column]).copy()
    
    # Convert to percentages if needed (assume data is in decimal format)
    df_clean['price_change_pct'] = df_clean[price_column] * 100
    
    # Sort by price change (ascending for horizontal bars - highest at top)
    df_clean = df_clean.sort_values('price_change_pct', ascending=True)
    
    print(f"Creating bar chart for {period}: {len(df_clean)} countries")
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    countries = df_clean['country'].tolist()
    price_changes = df_clean['price_change_pct'].tolist()
    
    # Create horizontal bar chart
    colors = [get_country_color(country) for country in countries]
    bars = plt.barh(range(len(countries)), price_changes, color=colors, alpha=0.8, height=0.6)
    
    # Highlight Switzerland if present
    for i, country in enumerate(countries):
        if country == 'Switzerland':
            bars[i].set_color('#ffd558')  # Yellow for Switzerland
            bars[i].set_alpha(1.0)
    
    # Customize the plot
    plt.yticks(range(len(countries)), countries, fontsize=12)
    plt.xlabel('Real House Price Change (%)', fontsize=14, fontweight='bold')
    
    # Dynamic title based on period
    plt.title(f'Real House Price Dynamics {period}\n' + 
              f'Inflation-Adjusted Property Value Changes', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Add grid and reference line at zero
    plt.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Set background
    plt.gca().set_facecolor('white')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved bar chart: {output_path}")
    
    # Print Switzerland stats if present
    swiss_data = df_clean[df_clean['country'] == 'Switzerland']
    if not swiss_data.empty:
        swiss_change = swiss_data['price_change_pct'].iloc[0]
        print(f"Switzerland {period}: {swiss_change:.1f}% real price change")

def create_house_price_comparison_scatter(df_2002, df_2017, price_col_2002, price_col_2017):
    """
    Create a scatter plot comparing house price changes between the two periods
    """
    # Merge the datasets on country
    merged = pd.merge(df_2002[['country', price_col_2002]], 
                     df_2017[['country', price_col_2017]], 
                     on='country', how='inner')
    
    # Remove rows with missing data
    merged = merged.dropna(subset=[price_col_2002, price_col_2017])
    
    print(f"Creating comparison scatter plot: {len(merged)} countries with data for both periods")
    
    if merged.empty:
        print("No countries with data for both periods")
        return
    
    # Convert to percentages
    x_values = [x * 100 for x in merged[price_col_2002].tolist()]
    y_values = [y * 100 for y in merged[price_col_2017].tolist()]
    countries = merged['country'].tolist()
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with country-specific colors
    colors = [get_country_color(country) for country in countries]
    
    # Plot all countries
    plt.scatter(x_values, y_values, c=colors, alpha=0.8, s=100, edgecolor='white', linewidth=1.5)
    
    # Highlight Switzerland if present
    swiss_data = merged[merged['country'] == 'Switzerland']
    if not swiss_data.empty:
        swiss_x = swiss_data[price_col_2002].iloc[0] * 100
        swiss_y = swiss_data[price_col_2017].iloc[0] * 100
        plt.scatter(swiss_x, swiss_y, color='#ffd558', s=200, edgecolor='black', linewidth=2, zorder=5)
        
        # Add label for Switzerland
        plt.annotate('Switzerland', (swiss_x, swiss_y), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add labels for a few other interesting countries
    for _, row in merged.iterrows():
        country = row['country']
        if country in ['Germany', 'France', 'Netherlands', 'United Kingdom', 'United States', 'Spain'] and country != 'Switzerland':
            plt.annotate(country, (row[price_col_2002] * 100, row[price_col_2017] * 100), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Real House Price Change 2002-2007 (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Real House Price Change 2017-2022 (%)', fontsize=14, fontweight='bold')
    
    plt.title('Housing Market Cycles Comparison\n' + 
              'Pre-Crisis vs Post-Crisis Real Price Dynamics', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add grid and reference lines
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add diagonal reference line (equal changes in both periods)
    min_val = min(min(x_values), min(y_values))
    max_val = max(max(x_values), max(y_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.3, linewidth=1, label='Equal change line')
    
    plt.legend()
    
    # Set background
    plt.gca().set_facecolor('white')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'real_house_prices_comparison_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved comparison scatter plot: {output_path}")
    
    # Print Switzerland comparison if present
    if not swiss_data.empty:
        swiss_2002 = swiss_data[price_col_2002].iloc[0] * 100
        swiss_2017 = swiss_data[price_col_2017].iloc[0] * 100
        print(f"Switzerland comparison: 2002-2007: {swiss_2002:.1f}%, 2017-2022: {swiss_2017:.1f}%")

def create_social_rental_dwelling_graph():
    """
    Create a bar chart showing social rental dwellings as a share of total housing stock
    Bar chart for 2020 with dots for 2010
    """
    print("Creating social rental dwelling stock graph...")
    
    # Load the dataset
    df = load_excel_data('ocde_social rental dwelling stock.xlsx')
    
    if df is None:
        print("Could not load social rental dwelling stock data")
        return
    
    print(f"Available sheets in ocde_social rental dwelling stock.xlsx: {df.columns.tolist()}")
    print(f"Shape of data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("First few rows:")
    print(df.head())
    
    # Clean country names
    if 'iso3' in df.columns:
        df['country'] = df['iso3'].apply(clean_country_name)
    elif 'variable' in df.columns:
        df['iso3'] = df['variable']
        df['country'] = df['iso3'].apply(clean_country_name)
    else:
        print("Could not find country identifier column")
        return
    
    # Remove entries where country name couldn't be resolved
    df = df[df['country'].notna()]
    
    # Focus on European countries + some key comparisons
    european_countries = [
        'Switzerland', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 
        'Belgium', 'Austria', 'Sweden', 'Denmark', 'Finland', 'Norway',
        'United Kingdom', 'Portugal', 'Greece', 'Ireland', 'Luxembourg',
        'Czech Republic', 'Hungary', 'Poland', 'Slovakia', 'Slovenia',
        'Estonia', 'Latvia', 'Lithuania', 'Croatia', 'Romania', 'Bulgaria',
        'Cyprus', 'Malta', 'United States', 'Canada', 'Japan', 'Australia'
    ]
    
    # Filter for relevant countries
    df_filtered = df[df['country'].isin(european_countries)].copy()
    
    print(f"Data structure analysis:")
    print(f"DataFrame shape: {df_filtered.shape}")
    print(f"Columns: {df_filtered.columns.tolist()}")
    
    # Identify year and value columns
    if 'year' in df_filtered.columns:
        year_col = 'year'
        # Find the numeric value column
        value_cols = [col for col in df_filtered.columns if col not in ['iso3', 'country', 'year'] and df_filtered[col].dtype in ['float64', 'int64']]
        if value_cols:
            value_col = value_cols[0]
        else:
            print("Could not find value column")
            return
    else:
        # Assume columns represent years and values
        year_cols = [col for col in df_filtered.columns if col not in ['iso3', 'country'] and str(col).isdigit()]
        if len(year_cols) >= 2:
            # Look for 2010 and 2020 data
            year_2010 = '2010' if '2010' in year_cols else None
            year_2020 = '2020' if '2020' in year_cols else (year_cols[-1] if year_cols else None)
            
            if year_2010 and year_2020:
                create_social_rental_bar_chart_with_dots(df_filtered, year_2010, year_2020)
                return
        
        print("Could not identify year columns for 2010 and 2020")
        print(f"Available columns: {df_filtered.columns.tolist()}")
        return
    
    # Get unique years
    available_years = sorted(df_filtered[year_col].unique())
    print(f"Available years: {available_years}")
    
    # Find closest years to 2010 and 2020
    target_2010 = 2010
    target_2020 = 2020
    
    year_2010 = min(available_years, key=lambda x: abs(x - target_2010)) if available_years else None
    year_2020 = min(available_years, key=lambda x: abs(x - target_2020)) if available_years else None
    
    if year_2010 is None or year_2020 is None:
        print(f"Could not find suitable years. Available: {available_years}")
        return
    
    print(f"Using years: {year_2010} (for 2010) and {year_2020} (for 2020)")
    
    # Create datasets for both years
    df_2010 = df_filtered[df_filtered[year_col] == year_2010].copy()
    df_2020 = df_filtered[df_filtered[year_col] == year_2020].copy()
    
    print(f"Countries with {year_2010} data: {len(df_2010)}")
    print(f"Countries with {year_2020} data: {len(df_2020)}")
    
    # Merge the datasets - use 'outer' join to keep all countries from both years
    merged = pd.merge(df_2010[['country', value_col]], 
                     df_2020[['country', value_col]], 
                     on='country', how='outer', suffixes=('_2010', '_2020'))
    
    # Keep all countries, even if they only have data for one year
    # Fill missing data with NaN but keep the country in the dataset
    
    print(f"Dataset after merge: {len(merged)} countries total")
    print(f"Countries with 2020 data: {merged[f'{value_col}_2020'].notna().sum()}")
    print(f"Countries with 2010 data: {merged[f'{value_col}_2010'].notna().sum()}")
    print(f"Countries with both years: {(merged[f'{value_col}_2020'].notna() & merged[f'{value_col}_2010'].notna()).sum()}")
    print(f"Countries included: {sorted(merged['country'].tolist())}")
    
    # Debug: Check for countries with no data at all
    countries_with_no_data = merged[(merged[f'{value_col}_2020'].isna()) & (merged[f'{value_col}_2010'].isna())]
    if not countries_with_no_data.empty:
        print(f"⚠ Countries with no data at all: {countries_with_no_data['country'].tolist()}")
    
    if 'Switzerland' in merged['country'].values:
        print("✓ Switzerland is included in the data")
        swiss_row = merged[merged['country'] == 'Switzerland'].iloc[0]
        print(f"  Switzerland 2010: {swiss_row[f'{value_col}_2010']}")
        print(f"  Switzerland 2020: {swiss_row[f'{value_col}_2020']}")
    else:
        print("⚠ Switzerland is not in the data")
    
    # Remove countries that have no data for either year
    merged_clean = merged[(merged[f'{value_col}_2020'].notna()) | (merged[f'{value_col}_2010'].notna())]
    print(f"After removing countries with no data: {len(merged_clean)} countries remain")
    
    # Create the visualization
    create_social_rental_bar_chart(merged_clean, f'{value_col}_2020', f'{value_col}_2010', year_2020, year_2010)

def create_social_rental_bar_chart_with_dots(df, col_2010, col_2020):
    """
    Create bar chart when data is in separate year columns
    """
    print(f"Debug: Processing {len(df)} countries from column-based data")
    print(f"Countries in dataset: {sorted(df['country'].tolist())}")
    
    # Check for countries with no data at all
    countries_with_no_data = df[(df[col_2010].isna()) & (df[col_2020].isna())]
    if not countries_with_no_data.empty:
        print(f"⚠ Countries with no data at all: {countries_with_no_data['country'].tolist()}")
    
    # Remove countries that have no data for either year
    df_clean = df[(df[col_2010].notna()) | (df[col_2020].notna())].copy()
    print(f"After removing countries with no data: {len(df_clean)} countries remain")
    
    # Convert to percentages if needed and merge all countries
    df_clean['value_2020'] = df_clean[col_2020] * 100  # May have NaN
    df_clean['value_2010'] = df_clean[col_2010] * 100  # May have NaN
    
    # Debug: Show data for Czech Republic and Switzerland
    if 'Czech Republic' in df_clean['country'].values:
        czech_row = df_clean[df_clean['country'] == 'Czech Republic'].iloc[0]
        print(f"Czech Republic: 2010={czech_row['value_2010']}, 2020={czech_row['value_2020']}")
    if 'Switzerland' in df_clean['country'].values:
        swiss_row = df_clean[df_clean['country'] == 'Switzerland'].iloc[0]
        print(f"Switzerland: 2010={swiss_row['value_2010']}, 2020={swiss_row['value_2020']}")
    
    create_social_rental_bar_chart(df_clean, 'value_2020', 'value_2010', '2020', '2010')

def create_social_rental_bar_chart(df, col_2020, col_2010, year_2020, year_2010):
    """
    Create the actual bar chart with dots visualization
    """
    # Create a copy and handle missing data
    df_plot = df.copy()
    
    # For countries with only 2010 data, we'll show just dots (no bars)
    # For countries with only 2020 data, we'll show just bars (no dots)
    # For countries with both, we'll show bars + dots
    
    # Replace NaN values with 0 for plotting bars (but track which are actually missing)
    has_2020_data = df_plot[col_2020].notna()
    has_2010_data = df_plot[col_2010].notna()
    
    # Fill NaN values with 0 for plotting purposes
    df_plot[col_2020] = df_plot[col_2020].fillna(0)
    df_plot[col_2010] = df_plot[col_2010].fillna(0)
    
    # Calculate mean values for sorting (ignoring zero values that were originally NaN)
    def calculate_mean_for_sorting(row):
        values = []
        if has_2020_data.loc[row.name]:
            values.append(row[col_2020])
        if has_2010_data.loc[row.name]:
            values.append(row[col_2010])
        return np.mean(values) if values else 0
    
    df_plot['sort_mean'] = df_plot.apply(calculate_mean_for_sorting, axis=1)
    
    # Sort by mean values in descending order (highest at top)
    df_sorted = df_plot.sort_values('sort_mean', ascending=True)  # ascending=True puts highest at top for horizontal bars
    
    # Update the data availability tracking to match the sorted order
    has_2020_sorted = has_2020_data[df_sorted.index]
    has_2010_sorted = has_2010_data[df_sorted.index]
    
    print(f"Creating social rental dwelling chart: {len(df_sorted)} countries total")
    print(f"  - {has_2020_sorted.sum()} countries with {year_2020} data")
    print(f"  - {has_2010_sorted.sum()} countries with {year_2010} data")
    
    # Create the plot
    plt.figure(figsize=(14, 12))  # Increased height for more countries
    
    countries = df_sorted['country'].tolist()
    values_2020 = df_sorted[col_2020].tolist()
    values_2010 = df_sorted[col_2010].tolist()
    
    # Create horizontal bar chart for all countries
    colors = [get_country_color(country) for country in countries]
    bars = plt.barh(range(len(countries)), values_2020, color=colors, alpha=0.8, height=0.6, label=f'{year_2020}')
    
    # Make bars transparent for countries with no 2020 data
    for i, (country, has_data) in enumerate(zip(countries, has_2020_sorted)):
        if not has_data:
            bars[i].set_alpha(0.2)  # Very transparent
            bars[i].set_color('#e0e0e0')  # Light gray
        elif country == 'Switzerland':
            bars[i].set_color('#ffd558')  # Yellow for Switzerland
            bars[i].set_alpha(1.0)
    
    # Add dots for countries with 2010 data
    dot_added_to_legend = False
    for i, (country, val_2010, has_data) in enumerate(zip(countries, values_2010, has_2010_sorted)):
        if has_data and val_2010 > 0:  # Only plot if we have real 2010 data
            color = '#ffd558' if country == 'Switzerland' else '#2c3e50'
            size = 120 if country == 'Switzerland' else 80
            label = f'{year_2010}' if not dot_added_to_legend else ''
            plt.scatter(val_2010, i, color=color, s=size, zorder=5, 
                       edgecolor='white', linewidth=1.5, label=label)
            dot_added_to_legend = True
    
    # Customize the plot
    plt.yticks(range(len(countries)), countries, fontsize=11)
    plt.xlabel('Social Rental Dwellings (% of total housing stock)', fontsize=14, fontweight='bold')
    
    plt.title('Social Housing Provision Across Countries\n' + 
              f'Public Rental Stock as Share of Total Dwellings ({year_2020} bars, {year_2010} dots)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add grid and reference lines
    plt.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add legend - larger size, no 'No 2020 data' entry
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate labels
    by_label = dict(zip(labels, handles))
    legend_handles = list(by_label.values())
    legend_labels = list(by_label.keys())
    
    plt.legend(legend_handles, legend_labels, loc='lower right', fontsize=12, frameon=True, 
               fancybox=True, shadow=True, framealpha=0.9)
    
    # Set background
    plt.gca().set_facecolor('white')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'social_rental_dwelling_stock.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved social rental dwelling chart: {output_path}")
    
    # Print Switzerland stats if present
    swiss_data = df_sorted[df_sorted['country'] == 'Switzerland']
    if not swiss_data.empty:
        idx = swiss_data.index[0]
        if has_2020_sorted.iloc[idx]:
            swiss_2020 = swiss_data[col_2020].iloc[0]
            print(f"Switzerland: {year_2020}: {swiss_2020:.1f}%", end="")
        else:
            print(f"Switzerland: no {year_2020} data", end="")
            
        if has_2010_sorted.iloc[idx]:
            swiss_2010 = swiss_data[col_2010].iloc[0]
            print(f", {year_2010}: {swiss_2010:.1f}%")
        else:
            print(f", no {year_2010} data")

def create_elasticities_scatter_plot():
    """
    Create a scatter plot showing elasticities of different metropolitan areas per country
    Each country appears on x-axis with different metropolitan values scattered vertically
    """
    print("Creating elasticities scatter plot...")
    
    # Load the dataset
    df = load_excel_data('ocde_elasticities.xlsx')
    
    if df is None:
        print("Could not load elasticities data")
        return
    
    print(f"Available sheets in ocde_elasticities.xlsx: {df.columns.tolist()}")
    print(f"Shape of data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("First few rows:")
    print(df.head())
    
    # Clean country names
    if 'iso3' in df.columns:
        df['country'] = df['iso3'].apply(clean_country_name)
    elif 'variable' in df.columns:
        df['iso3'] = df['variable']
        df['country'] = df['iso3'].apply(clean_country_name)
    elif 'Country' in df.columns:
        # If there's already a country column with codes, map them to full names
        df['country'] = df['Country'].apply(clean_country_name)
    else:
        print("Could not find country identifier column")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Remove entries where country name couldn't be resolved
    df_before_filter = len(df)
    df = df[df['country'].notna()]
    print(f"Filtered from {df_before_filter} to {len(df)} rows after country name mapping")
    
    # Focus on European countries + some key comparisons
    european_countries = [
        'Switzerland', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 
        'Belgium', 'Austria', 'Sweden', 'Denmark', 'Finland', 'Norway',
        'United Kingdom', 'Portugal', 'Greece', 'Ireland', 'Luxembourg',
        'Czech Republic', 'Hungary', 'Poland', 'Slovakia', 'Slovenia',
        'Estonia', 'Latvia', 'Lithuania', 'Croatia', 'Romania', 'Bulgaria',
        'Cyprus', 'Malta', 'United States', 'Canada', 'Japan', 'Australia'
    ]
    
    # Filter for relevant countries
    df_filtered = df[df['country'].isin(european_countries)].copy()
    
    print(f"Data structure analysis:")
    print(f"DataFrame shape: {df_filtered.shape}")
    print(f"Columns: {df_filtered.columns.tolist()}")
    print(f"Countries: {sorted(df_filtered['country'].unique().tolist())}")
    
    # Identify the elasticity value column
    numeric_cols = [col for col in df_filtered.columns if col not in ['iso3', 'country', 'Country', 'variable'] and df_filtered[col].dtype in ['float64', 'int64']]
    
    # Look for metropolitan area identifier
    string_cols = [col for col in df_filtered.columns if col not in ['iso3', 'country', 'Country'] and df_filtered[col].dtype == 'object']
    
    print(f"Numeric columns (potential elasticity values): {numeric_cols}")
    print(f"String columns (potential metro area names): {string_cols}")
    
    if not numeric_cols:
        print("Could not find numeric elasticity column")
        return
    
    elasticity_col = numeric_cols[0]  # Use first numeric column
    metro_col = string_cols[0] if string_cols else None  # Use first string column for metro areas
    
    print(f"Using elasticity column: {elasticity_col}")
    print(f"Using metro area column: {metro_col}")
    
    # Remove rows with missing elasticity values
    df_clean = df_filtered.dropna(subset=[elasticity_col]).copy()
    
    print(f"Final dataset: {len(df_clean)} metropolitan areas from {df_clean['country'].nunique()} countries")
    
    if 'Switzerland' in df_clean['country'].values:
        print("✓ Switzerland is included in the data")
        swiss_data = df_clean[df_clean['country'] == 'Switzerland']
        print(f"  Switzerland has {len(swiss_data)} metropolitan areas")
        for _, row in swiss_data.iterrows():
            metro_name = row[metro_col] if metro_col else 'Metro Area'
            elasticity = row[elasticity_col]
            print(f"    {metro_name}: {elasticity:.2f}")
    else:
        print("⚠ Switzerland is not in the data")
    
    # Create the scatter plot
    create_elasticity_scatter_chart(df_clean, elasticity_col, metro_col)

def create_elasticity_scatter_chart(df, elasticity_col, metro_col):
    """
    Create the actual scatter plot for elasticities by country
    """
    # Get unique countries and sort them
    countries = sorted(df['country'].unique())
    
    print(f"Creating elasticities scatter plot: {len(countries)} countries, {len(df)} metropolitan areas")
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # Create x-axis positions for countries
    x_positions = {country: i for i, country in enumerate(countries)}
    
    # Plot points for each metropolitan area
    for _, row in df.iterrows():
        country = row['country']
        elasticity = row[elasticity_col]
        x_pos = x_positions[country]
        
        # Add some jitter to x-position to avoid overlapping points
        jitter = np.random.uniform(-0.2, 0.2)
        x_plot = x_pos + jitter
        
        # Color coding
        if country == 'Switzerland':
            color = '#ffd558'  # Yellow for Switzerland
            size = 120
            alpha = 1.0
            zorder = 5
        else:
            color = get_country_color(country)
            size = 80
            alpha = 0.7
            zorder = 3
        
        plt.scatter(x_plot, elasticity, color=color, s=size, alpha=alpha, 
                   edgecolor='white', linewidth=1, zorder=zorder)
    
    # Add country average lines
    for country in countries:
        country_data = df[df['country'] == country]
        if len(country_data) > 1:  # Only if multiple metro areas
            avg_elasticity = country_data[elasticity_col].mean()
            x_pos = x_positions[country]
            
            # Draw a horizontal line for the average
            plt.plot([x_pos-0.3, x_pos+0.3], [avg_elasticity, avg_elasticity], 
                    color='red' if country == 'Switzerland' else 'gray', 
                    linewidth=2, alpha=0.6, zorder=4)
    
    # Customize the plot
    plt.xticks(range(len(countries)), countries, rotation=45, ha='right', fontsize=10)
    plt.ylabel('Housing Supply Elasticity', fontsize=14, fontweight='bold')
    
    plt.title('Metropolitan Housing Supply Elasticities by Country\n' + 
              'Individual Metro Areas and Country Averages', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    plt.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add reference lines for elasticity interpretation
    plt.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Elastic threshold (1.0)')
    
    # Set background
    plt.gca().set_facecolor('white')
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], color='#ffd558', s=120, label='Switzerland'),
        plt.scatter([], [], color='#8dd3c7', s=80, alpha=0.7, label='Other countries'),
        plt.Line2D([0], [0], color='red', linewidth=2, alpha=0.6, label='Switzerland average'),
        plt.Line2D([0], [0], color='gray', linewidth=2, alpha=0.6, label='Country averages'),
        plt.Line2D([0], [0], color='orange', linestyle='--', alpha=0.5, label='Elastic threshold (1.0)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True, 
               fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'metropolitan_elasticities.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved elasticities scatter plot: {output_path}")
    
    # Print summary statistics
    print("\nSummary by country:")
    for country in countries:
        country_data = df[df['country'] == country]
        avg_elasticity = country_data[elasticity_col].mean()
        metro_count = len(country_data)
        elasticity_range = f"{country_data[elasticity_col].min():.2f}-{country_data[elasticity_col].max():.2f}" if metro_count > 1 else f"{avg_elasticity:.2f}"
        print(f"  {country}: {metro_count} metro areas, avg elasticity: {avg_elasticity:.2f}, range: {elasticity_range}")

def create_average_rent_allowance_graph():
    """
    Create a dot chart showing average rent allowance as share of gross wage
    10th percentile as dots with 50th percentile dots overlay
    """
    print("Creating average rent allowance graph...")
    
    # Load the dataset
    df = load_excel_data('ocde_Average of rent allowance.xlsx')
    
    if df is None:
        print("Could not load average rent allowance data")
        return
    
    print(f"Shape of data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("First few rows:")
    print(df.head())
    
    # Clean country names - handle the 'Unnamed: 0' column containing country names
    if 'Unnamed: 0' in df.columns:
        df['country'] = df['Unnamed: 0']
    else:
        print("Could not find country identifier column")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Remove entries where country name couldn't be resolved
    df_before_filter = len(df)
    df = df[df['country'].notna()]
    print(f"Filtered from {df_before_filter} to {len(df)} rows after country name mapping")
    
    # Focus on European countries + some key comparisons
    european_countries = [
        'Switzerland', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 
        'Belgium', 'Austria', 'Sweden', 'Denmark', 'Finland', 'Norway',
        'United Kingdom', 'Portugal', 'Greece', 'Ireland', 'Luxembourg',
        'Czechia', 'Czech Republic', 'Hungary', 'Poland', 'Slovakia', 'Slovenia',
        'Estonia', 'Latvia', 'Lithuania', 'Croatia', 'Romania', 'Bulgaria',
        'Cyprus', 'Malta', 'EU27', 'EU-27', 'United States', 'Canada', 'Japan', 'Australia'
    ]
    
    # Filter for relevant countries
    df_filtered = df[df['country'].isin(european_countries)].copy()
    
    print(f"Data structure analysis:")
    print(f"DataFrame shape: {df_filtered.shape}")
    print(f"Columns: {df_filtered.columns.tolist()}")
    print(f"Countries: {sorted(df_filtered['country'].unique().tolist())}")
    
    # Identify the percentile columns
    p10_col = 'Average rent allowance as share of gross wage at 10th percentile'
    p50_col = 'Average rent allowance as share of gross wage at 50th percentile'
    
    if p10_col not in df_filtered.columns or p50_col not in df_filtered.columns:
        print(f"Could not find expected columns: {p10_col}, {p50_col}")
        return
    
    print(f"Using 10th percentile column: {p10_col}")
    print(f"Using 50th percentile column: {p50_col}")
    
    # Convert to percentages and handle missing values
    df_filtered[p10_col] = pd.to_numeric(df_filtered[p10_col], errors='coerce') * 100
    df_filtered[p50_col] = pd.to_numeric(df_filtered[p50_col], errors='coerce') * 100
    
    # Remove rows with missing data in both columns
    df_final = df_filtered.dropna(subset=[p10_col, p50_col])
    
    print(f"Final dataset: {len(df_final)} countries with complete data")
    
    # Check if Switzerland is in the data
    switzerland_data = df_final[df_final['country'] == 'Switzerland']
    if not switzerland_data.empty:
        print("✓ Switzerland is included in the data")
        swiss_10th = switzerland_data[p10_col].iloc[0]
        swiss_50th = switzerland_data[p50_col].iloc[0]
        print(f"  Switzerland 10th percentile: {swiss_10th:.1f}%")
        print(f"  Switzerland 50th percentile: {swiss_50th:.1f}%")
    else:
        print("✗ Switzerland is not in the final dataset")
    
    # Sort by 10th percentile values for better visualization
    df_final_sorted = df_final.sort_values(p10_col, ascending=True)
    
    # Create the horizontal bar chart
    fig, ax = plt.subplots(figsize=(14, len(df_final_sorted) * 0.4 + 2))
    
    countries = df_final_sorted['country'].tolist()
    values_10th = df_final_sorted[p10_col].tolist()
    values_50th = df_final_sorted[p50_col].tolist()
    
    y_pos = range(len(countries))
    
    # Plot 10th percentile as dots (primary data)
    for i, (country, val_10th) in enumerate(zip(countries, values_10th)):
        color = get_country_color(country)
        if country == 'Switzerland':
            size = 150
            marker = 's'  # Square for Switzerland
            alpha = 1.0
            edgecolor = 'black'
            linewidth = 2
            zorder = 5
        elif country in ['EU27', 'EU-27']:
            size = 130
            marker = 'D'  # Diamond for EU-27
            alpha = 0.9
            edgecolor = 'white'
            linewidth = 1.5
            zorder = 4
        else:
            size = 100
            marker = 'o'  # Circle for other countries
            alpha = 0.8
            edgecolor = 'white'
            linewidth = 1
            zorder = 3
        
        # For 0 values, place dots on the y-axis (x=0)
        x_position = max(val_10th, 0.01) if val_10th == 0 else val_10th
        ax.scatter(x_position, i, color=color, s=size, marker=marker, alpha=alpha,
                  edgecolors=edgecolor, linewidth=linewidth, zorder=zorder)
    
    # Add 50th percentile as smaller overlay dots
    for i, (country, val_50th) in enumerate(zip(countries, values_50th)):
        if val_50th > 0:  # Only show dots if there's a value > 0
            ax.scatter(val_50th, i, color='white', s=60, marker='o', 
                      edgecolors='black', linewidth=1, zorder=6)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(countries, fontsize=12)
    ax.set_xlabel('Average rent allowance as share of gross wage (%)', fontsize=14, weight='bold')
    ax.set_title('Average Rent Allowance for Different Family Types\n(Share of gross wage at 10th and 50th percentiles, 2024 or latest available)', 
                fontsize=16, weight='bold', pad=20)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_facecolor('#f8f9fa')
    
    # Set x-axis to start from 0
    ax.set_xlim(0, max(max(values_10th), max(values_50th)) * 1.05)
    
    # Create custom legend
    legend_elements = [
        plt.scatter([], [], color='#ffd558', s=150, marker='s', alpha=1.0, 
                   edgecolors='black', linewidth=2, label='Switzerland (10th percentile)'),
        plt.scatter([], [], color='#80b1d3', s=130, marker='D', alpha=0.9, 
                   edgecolors='white', linewidth=1.5, label='EU-27 (10th percentile)'),
        plt.scatter([], [], color='#8dd3c7', s=100, marker='o', alpha=0.8, 
                   edgecolors='white', linewidth=1, label='Other countries (10th percentile)'),
        plt.scatter([], [], color='white', s=60, marker='o', 
                   edgecolors='black', linewidth=1, label='50th percentile')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'average_rent_allowance_by_percentile.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Creating average rent allowance dot chart: {len(df_final)} countries")
    print(f"Saved average rent allowance scatter plot: {output_path}")
    
    # Print summary sorted by 10th percentile values
    print(f"\nSummary by country (10th percentile):")
    df_summary = df_final.sort_values(p10_col)
    for _, row in df_summary.iterrows():
        country = row['country']
        val_10th = row[p10_col]
        val_50th = row[p50_col]
        print(f"  {country}: {val_10th:.1f}% (10th), {val_50th:.1f}% (50th)")


def create_housing_allowance_graph():
    """
    Create a scatter plot showing share of households receiving housing allowance
    Bottom and third quintiles of income distribution, dots classified by bottom quintile
    """
    print("Creating housing allowance share graph...")
    
    # Load the dataset
    df = load_excel_data('ocde_Share of households receiving a housing allowance.xlsx')
    
    if df is None:
        print("Could not load housing allowance data")
        return
    
    print(f"Available sheets in ocde_Share of households receiving a housing allowance.xlsx: {df.columns.tolist()}")
    print(f"Shape of data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("First few rows:")
    print(df.head())
    
    # Clean country names - handle the 'Unnamed: 0' column containing country names
    if 'iso3' in df.columns:
        df['country'] = df['iso3'].apply(clean_country_name)
    elif 'variable' in df.columns:
        df['iso3'] = df['variable']
        df['country'] = df['iso3'].apply(clean_country_name)
    elif 'Country' in df.columns:
        # If there's already a country column with codes, map them to full names
        df['country'] = df['Country'].apply(clean_country_name)
    elif 'Unnamed: 0' in df.columns:
        # For housing allowance data, the country names are directly in 'Unnamed: 0'
        df['country'] = df['Unnamed: 0']
    elif 'Country' in df.columns:
        # If there's already a country column with codes, map them to full names
        df['country'] = df['Country'].apply(clean_country_name)
    else:
        print("Could not find country identifier column")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Remove entries where country name couldn't be resolved
    df_before_filter = len(df)
    df = df[df['country'].notna()]
    print(f"Filtered from {df_before_filter} to {len(df)} rows after country name mapping")
    
    # Focus on European countries + some key comparisons
    european_countries = [
        'Switzerland', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 
        'Belgium', 'Austria', 'Sweden', 'Denmark', 'Finland', 'Norway',
        'United Kingdom', 'Portugal', 'Greece', 'Ireland', 'Luxembourg',
        'Czech Republic', 'Hungary', 'Poland', 'Slovakia', 'Slovenia',
        'Estonia', 'Latvia', 'Lithuania', 'Croatia', 'Romania', 'Bulgaria',
        'Cyprus', 'Malta', 'United States', 'Canada', 'Japan', 'Australia'
    ]
    
    # Filter for relevant countries
    df_filtered = df[df['country'].isin(european_countries)].copy()
    
    print(f"Data structure analysis:")
    print(f"DataFrame shape: {df_filtered.shape}")
    print(f"Columns: {df_filtered.columns.tolist()}")
    print(f"Countries: {sorted(df_filtered['country'].unique().tolist())}")
    
    # Identify quintile columns
    numeric_cols = [col for col in df_filtered.columns if col not in ['iso3', 'country', 'Country', 'variable'] and df_filtered[col].dtype in ['float64', 'int64']]
    print(f"Numeric columns (potential quintile values): {numeric_cols}")
    
    # Look for bottom and third quintile columns
    bottom_quintile_col = None
    third_quintile_col = None
    
    # Try to identify columns by name patterns
    for col in numeric_cols:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['bottom', 'first', '1st', 'q1']):
            bottom_quintile_col = col
        elif any(keyword in col_lower for keyword in ['third', '3rd', 'q3']):
            third_quintile_col = col
    
    # If not found by name, use positional logic (first two numeric columns)
    if not bottom_quintile_col and len(numeric_cols) >= 1:
        bottom_quintile_col = numeric_cols[0]
    if not third_quintile_col and len(numeric_cols) >= 2:
        third_quintile_col = numeric_cols[1]
    
    print(f"Using bottom quintile column: {bottom_quintile_col}")
    print(f"Using third quintile column: {third_quintile_col}")
    
    if not bottom_quintile_col:
        print("Could not find bottom quintile column")
        return
    
    # Remove rows with missing data for bottom quintile (required for classification)
    df_clean = df_filtered.dropna(subset=[bottom_quintile_col]).copy()
    
    print(f"Final dataset: {len(df_clean)} countries with bottom quintile data")
    
    if 'Switzerland' in df_clean['country'].values:
        print("✓ Switzerland is included in the data")
        swiss_data = df_clean[df_clean['country'] == 'Switzerland']
        swiss_bottom = swiss_data[bottom_quintile_col].iloc[0] if not swiss_data.empty else None
        swiss_third = swiss_data[third_quintile_col].iloc[0] if third_quintile_col and not swiss_data.empty else None
        print(f"  Switzerland bottom quintile: {swiss_bottom:.1f}%")
        if swiss_third is not None:
            print(f"  Switzerland third quintile: {swiss_third:.1f}%")
    else:
        print("⚠ Switzerland is not in the data")
    
    # Create the scatter plot
    create_housing_allowance_scatter_chart(df_clean, bottom_quintile_col, third_quintile_col)

def create_housing_allowance_scatter_chart(df, bottom_col, third_col):
    """
    Create the actual scatter plot for housing allowance shares
    """
    # Sort countries by bottom quintile values for better visualization
    df_sorted = df.sort_values(bottom_col, ascending=True)
    
    print(f"Creating housing allowance scatter plot: {len(df_sorted)} countries")
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    countries = df_sorted['country'].tolist()
    bottom_values = df_sorted[bottom_col].tolist()
    third_values = df_sorted[third_col].tolist() if third_col else [None] * len(countries)
    
    # Create x-axis positions for countries
    x_positions = list(range(len(countries)))
    
    # Plot bottom quintile dots (primary classification)
    for i, (country, bottom_val) in enumerate(zip(countries, bottom_values)):
        if pd.notna(bottom_val):
            if country == 'Switzerland':
                color = '#ffd558'  # Yellow for Switzerland
                size = 150
                alpha = 1.0
                zorder = 5
                edgecolor = 'black'
                linewidth = 2
            else:
                color = get_country_color(country)
                size = 100
                alpha = 0.8
                zorder = 3
                edgecolor = 'white'
                linewidth = 1
            
            plt.scatter(i, bottom_val, color=color, s=size, alpha=alpha, 
                       edgecolor=edgecolor, linewidth=linewidth, zorder=zorder, 
                       label='Bottom quintile' if i == 0 else '')
    
    # Plot third quintile dots (if available)
    if third_col:
        for i, (country, third_val) in enumerate(zip(countries, third_values)):
            if pd.notna(third_val):
                if country == 'Switzerland':
                    color = '#ffd558'  # Yellow for Switzerland
                    size = 120
                    alpha = 0.7
                    zorder = 4
                    edgecolor = 'black'
                    linewidth = 1.5
                    marker = 's'  # Square for third quintile
                else:
                    color = get_country_color(country)
                    size = 80
                    alpha = 0.6
                    zorder = 2
                    edgecolor = 'white'
                    linewidth = 1
                    marker = 's'  # Square for third quintile
                
                plt.scatter(i, third_val, color=color, s=size, alpha=alpha, 
                           edgecolor=edgecolor, linewidth=linewidth, zorder=zorder,
                           marker=marker, label='Third quintile' if i == 0 else '')
    
    # Customize the plot
    plt.xticks(x_positions, countries, rotation=45, ha='right', fontsize=10)
    plt.ylabel('Share of Households Receiving Housing Allowance (%)', fontsize=14, fontweight='bold')
    
    title_text = 'Housing Allowance Coverage by Income Quintile\n'
    if third_col:
        title_text += 'Bottom Quintile (circles) vs Third Quintile (squares) - 2024 or Latest Available'
    else:
        title_text += 'Bottom Quintile Coverage - 2024 or Latest Available'
    
    plt.title(title_text, fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    plt.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Set background
    plt.gca().set_facecolor('white')
    
    # Add legend
    legend_elements = []
    legend_elements.append(plt.scatter([], [], color='#ffd558', s=150, edgecolor='black', linewidth=2, label='Switzerland'))
    legend_elements.append(plt.scatter([], [], color='#8dd3c7', s=100, alpha=0.8, label='Other countries'))
    legend_elements.append(plt.scatter([], [], color='gray', s=100, alpha=0.8, label='Bottom quintile'))
    
    if third_col:
        legend_elements.append(plt.scatter([], [], color='gray', s=80, alpha=0.6, marker='s', label='Third quintile'))
    
    plt.legend(handles=legend_elements, loc='upper left', fontsize=11, frameon=True, 
               fancybox=True, shadow=True, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, 'housing_allowance_by_quintile.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved housing allowance scatter plot: {output_path}")
    
    # Print summary statistics
    print("\nSummary by country (Bottom quintile):")
    for _, row in df_sorted.iterrows():
        country = row['country']
        bottom_val = row[bottom_col]
        third_val = row[third_col] if third_col and pd.notna(row[third_col]) else None
        
        summary = f"  {country}: {bottom_val:.1f}%"
        if third_val is not None:
            summary += f" (bottom), {third_val:.1f}% (third)"
        print(summary)

def create_residential_mobility_homeownership_graph():
    """
    Create scatter plot of residential mobility vs homeownership rates with regression line
    
    Data needed:
    - % of individuals that changed residence within the last 5 years (Y-axis)
    - Rate of homeownership (X-axis)
    """
    print("Creating residential mobility vs homeownership graph...")
    
    # File paths
    mobility_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_homeownership residential mobility.xlsx')
    homeownership_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_Homeownership rates.xlsx')
    
    # Check if files exist
    if not os.path.exists(mobility_file):
        print(f"⚠️  Missing data file: {mobility_file}")
        print("   Please provide the residential mobility data file with the following structure:")
        print("   - Column 1: Country names or ISO3 codes")
        print("   - Column 2: % of individuals that changed residence within the last 5 years")
        return
    
    if not os.path.exists(homeownership_file):
        print(f"⚠️  Missing data file: {homeownership_file}")
        return
    
    try:
        # Load residential mobility data
        mobility_df = load_excel_data(mobility_file)
        print(f"Mobility data shape: {mobility_df.shape}")
        print(f"Mobility data columns: {mobility_df.columns.tolist()}")
        
        # Load homeownership data
        homeownership_df = load_excel_data(homeownership_file)
        print(f"Homeownership data shape: {homeownership_df.shape}")
        print(f"Homeownership data columns: {homeownership_df.columns.tolist()}")
        
        # Process homeownership data (sum outright owners + owners with mortgages)
        homeownership_processed = homeownership_df.groupby('iso3')['Homeownership rate'].sum().reset_index()
        homeownership_processed.columns = ['iso3', 'homeownership_rate']
        
        # Add country names to homeownership data
        homeownership_processed['country'] = homeownership_processed['iso3'].map(ISO3_MAPPING)
        homeownership_final = homeownership_processed[homeownership_processed['country'].notna()].copy()
        
        # Detect mobility data columns
        mobility_cols = [col for col in mobility_df.columns if 'mobility' in col.lower() or 'residence' in col.lower() or 'moved' in col.lower()]
        country_col = 'Unnamed: 0' if 'Unnamed: 0' in mobility_df.columns else mobility_df.columns[0]
        
        if not mobility_cols:
            mobility_cols = [col for col in mobility_df.columns if col != country_col]
            print(f"Using column for mobility data: {mobility_cols[0] if mobility_cols else 'None found'}")
        
        # Process mobility data
        mobility_processed = mobility_df[[country_col] + mobility_cols[:1]].copy()
        mobility_processed.columns = ['country_raw', 'mobility_rate']
        mobility_processed['country'] = mobility_processed['country_raw'].map(lambda x: ISO3_MAPPING.get(x, x))
        mobility_final = mobility_processed[mobility_processed['country'].notna()].copy()
        
        # Merge datasets
        merged_df = pd.merge(homeownership_final[['country', 'homeownership_rate']], 
                            mobility_final[['country', 'mobility_rate']], 
                            on='country', how='inner')
        
        print(f"Merged dataset: {len(merged_df)} countries with both metrics")
        
        if len(merged_df) == 0:
            print("❌ No countries with both homeownership and mobility data found")
            return
        
        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot points
        for _, row in merged_df.iterrows():
            country = row['country']
            color = get_country_color(country)
            
            if country == 'Switzerland':
                marker = 's'
                size = 150
                alpha = 1.0
                edgecolor = 'black'
                linewidth = 2
                zorder = 5
            elif country in ['EU27', 'EU-27']:
                marker = 'D'
                size = 130
                alpha = 0.9
                edgecolor = 'white' 
                linewidth = 1.5
                zorder = 4
            else:
                marker = 'o'
                size = 100
                alpha = 0.8
                edgecolor = 'white'
                linewidth = 1
                zorder = 3
            
            ax.scatter(row['homeownership_rate'], row['mobility_rate'], 
                      color=color, s=size, marker=marker, alpha=alpha,
                      edgecolors=edgecolor, linewidth=linewidth, zorder=zorder)
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['homeownership_rate'], merged_df['mobility_rate'])
        
        x_range = np.linspace(merged_df['homeownership_rate'].min(), merged_df['homeownership_rate'].max(), 100)
        y_predicted = slope * x_range + intercept
        
        ax.plot(x_range, y_predicted, color='#fb8072', linestyle='--', alpha=0.8, linewidth=2, 
                label=f'Linear fit (R² = {r_value**2:.3f})')
        
        # Formatting
        ax.set_xlabel('Homeownership Rate (%)', fontsize=14, weight='bold')
        ax.set_ylabel('Residential Mobility Rate (%)\n(Changed residence within last 5 years)', fontsize=14, weight='bold')
        ax.set_title('Residential Mobility vs Homeownership Rate\n(Higher homeownership may reduce residential mobility)', 
                    fontsize=16, weight='bold', pad=20)
        
        # Convert to percentages if needed
        if merged_df['homeownership_rate'].max() <= 1:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        if merged_df['mobility_rate'].max() <= 1:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))
        
        # Add country labels for key countries
        for _, row in merged_df.iterrows():
            if row['country'] in ['Switzerland', 'EU27']:
                ax.annotate(row['country'], 
                           (row['homeownership_rate'], row['mobility_rate']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Grid and styling
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        # Legend
        legend_elements = [
            plt.scatter([], [], color='#ffd558', s=150, marker='s', alpha=1.0, 
                       edgecolors='black', linewidth=2, label='Switzerland'),
            plt.scatter([], [], color='#80b1d3', s=130, marker='D', alpha=0.9, 
                       edgecolors='white', linewidth=1.5, label='EU-27'),
            plt.scatter([], [], color='#8dd3c7', s=100, marker='o', alpha=0.8, 
                       edgecolors='white', linewidth=1, label='Other countries'),
            plt.Line2D([0], [0], color='#fb8072', linestyle='--', alpha=0.8, linewidth=2, 
                      label=f'Linear fit (R² = {r_value**2:.3f})')
        ]
        ax.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(OUTPUT_DIR, 'residential_mobility_vs_homeownership.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved residential mobility vs homeownership scatter plot: {output_path}")
        
        # Print summary statistics
        print(f"\nRegression Analysis:")
        print(f"  Slope: {slope:.4f}")
        print(f"  R-squared: {r_value**2:.3f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Countries in analysis: {len(merged_df)}")
        
        if 'Switzerland' in merged_df['country'].values:
            swiss_data = merged_df[merged_df['country'] == 'Switzerland'].iloc[0]
            print(f"\nSwitzerland data:")
            print(f"  Homeownership rate: {swiss_data['homeownership_rate']:.1%}")
            print(f"  Mobility rate: {swiss_data['mobility_rate']:.1%}")
            
    except Exception as e:
        print(f"❌ Error creating residential mobility graph: {str(e)}")
        print("   Please check the data file format and structure")

def create_homeownership_rates_bar_chart():
    """
    Create bar chart showing homeownership rates with outright owners and owners with mortgages
    Data from: ocde_Homeownership rates.xlsx
    """
    print("Creating homeownership rates bar chart...")
    
    # File path
    homeownership_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_Homeownership rates.xlsx')
    
    if not os.path.exists(homeownership_file):
        print(f"⚠️  Missing data file: {homeownership_file}")
        return
    
    try:
        # Load homeownership data
        df = load_excel_data(homeownership_file)
        print(f"Homeownership data shape: {df.shape}")
        print(f"Homeownership data columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
        
        # Check unique types
        print(f"Available ownership types: {df['Type'].unique()}")
        
        # Process the data - get outright owners and mortgage owners
        outright_owners = df[df['Type'] == 'Outright owners'].copy()
        mortgage_owners = df[df['Type'] == 'Owners with mortgages'].copy()
        
        # Add country names
        outright_owners['country'] = outright_owners['iso3'].map(ISO3_MAPPING)
        mortgage_owners['country'] = mortgage_owners['iso3'].map(ISO3_MAPPING)
        
        # Filter for countries with valid names
        outright_owners = outright_owners[outright_owners['country'].notna()].copy()
        mortgage_owners = mortgage_owners[mortgage_owners['country'].notna()].copy()
        
        # Merge the datasets
        merged_df = pd.merge(outright_owners[['country', 'Homeownership rate']], 
                            mortgage_owners[['country', 'Homeownership rate']], 
                            on='country', how='outer', suffixes=('_outright', '_mortgage'))
        
        # Fill NaN values with 0
        merged_df['Homeownership rate_outright'] = merged_df['Homeownership rate_outright'].fillna(0)
        merged_df['Homeownership rate_mortgage'] = merged_df['Homeownership rate_mortgage'].fillna(0)
        
        # Calculate total homeownership
        merged_df['total_homeownership'] = merged_df['Homeownership rate_outright'] + merged_df['Homeownership rate_mortgage']
        
        # Sort by total homeownership rate (descending)
        merged_df = merged_df.sort_values('total_homeownership', ascending=False).reset_index(drop=True)
        
        print(f"Final dataset: {len(merged_df)} countries with homeownership data")
        
        if len(merged_df) == 0:
            print("❌ No countries with homeownership data found")
            return
        
        # Create the bar chart
        fig, ax = plt.subplots(figsize=(16, 10))
        
        countries = merged_df['country'].tolist()
        outright_rates = merged_df['Homeownership rate_outright'].tolist()
        mortgage_rates = merged_df['Homeownership rate_mortgage'].tolist()
        
        # Create positions for bars
        y_positions = range(len(countries))
        
        # Create stacked horizontal bars
        # First, plot outright owners (base of stack)
        bars1 = ax.barh(y_positions, outright_rates, height=0.6, 
                       color='#8dd3c7', alpha=0.8, label='Outright owners',
                       edgecolor='white', linewidth=0.5)
        
        # Then, plot mortgage owners stacked on top of outright owners
        bars2 = ax.barh(y_positions, mortgage_rates, height=0.6, 
                       left=outright_rates, color='#fb8072', alpha=0.9, 
                       label='Owners with mortgages', edgecolor='white', linewidth=0.5)
        
        # Highlight Switzerland and EU27 if present
        for i, country in enumerate(countries):
            if country == 'Switzerland':
                # Add yellow border for Switzerland (stacked bars)
                ax.barh(i, outright_rates[i], height=0.6, 
                       fill=False, edgecolor='#ffd558', linewidth=3)
                ax.barh(i, mortgage_rates[i], height=0.6, left=outright_rates[i],
                       fill=False, edgecolor='#ffd558', linewidth=3)
            elif country in ['EU27', 'EU-27']:
                # Add blue border for EU27 (stacked bars)
                ax.barh(i, outright_rates[i], height=0.6, 
                       fill=False, edgecolor='#80b1d3', linewidth=3)
                ax.barh(i, mortgage_rates[i], height=0.6, left=outright_rates[i],
                       fill=False, edgecolor='#80b1d3', linewidth=3)
        
        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(countries, fontsize=12)
        ax.set_xlabel('Homeownership Rate (%)', fontsize=14, weight='bold')
        ax.set_title('Homeownership Rates by Type\n(Stacked: Outright owners + Owners with mortgages, 2020 or latest available)', 
                    fontsize=16, weight='bold', pad=20)
        
        # Convert to percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        
        # Set x-axis limits to 100%
        ax.set_xlim(0, 1.0)
        
        # Grid and styling
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor('#f8f9fa')
        
        # Add total homeownership labels at the end of bars
        for i, (country, total_rate) in enumerate(zip(countries, merged_df['total_homeownership'])):
            if total_rate > 0:
                ax.text(total_rate + 0.01, i, f'{total_rate:.0%}', 
                       va='center', ha='left', fontsize=10, weight='bold')
        
        # Legend
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        # Invert y-axis to show highest rates at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(OUTPUT_DIR, 'homeownership_rates_by_type.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved homeownership rates bar chart: {output_path}")
        
        # Print summary statistics
        print(f"\nHomeownership Summary:")
        print(f"  Countries analyzed: {len(merged_df)}")
        
        if 'Switzerland' in countries:
            swiss_idx = countries.index('Switzerland')
            print(f"\nSwitzerland data:")
            print(f"  Outright owners: {outright_rates[swiss_idx]:.1%}")
            print(f"  Owners with mortgages: {mortgage_rates[swiss_idx]:.1%}")
            print(f"  Total homeownership: {merged_df.loc[swiss_idx, 'total_homeownership']:.1%}")
        
        # Show top 5 countries
        print(f"\nTop 5 countries by total homeownership:")
        for i in range(min(5, len(countries))):
            total = merged_df.iloc[i]['total_homeownership']
            outright = outright_rates[i] 
            mortgage = mortgage_rates[i]
            print(f"  {countries[i]}: {total:.1%} (Outright: {outright:.1%}, Mortgage: {mortgage:.1%})")
            
    except Exception as e:
        print(f"❌ Error creating homeownership rates chart: {str(e)}")
        print("   Please check the data file format and structure")

def create_construction_rd_investment_chart():
    """
    Create bar chart showing R&D investment in Construction (NACE F) by country
    Data from: eurostat_BERD by NACE Rev. 2 activity.xlsx, Sheet 7 (Construction)
    """
    print("Creating construction R&D investment chart...")
    
    # File path
    rd_file = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_BERD by NACE Rev. 2 activity.xlsx')
    
    if not os.path.exists(rd_file):
        print(f"⚠️  Missing data file: {rd_file}")
        return
    
    try:
        # Load construction R&D data (Sheet 7)
        df = pd.read_excel(rd_file, sheet_name='Sheet 7', skiprows=7)
        print(f"Construction R&D data shape: {df.shape}")
        
        # Clean the data structure
        # First row contains years, first column contains countries
        years_row = df.iloc[0, 1:].values  # Skip first column (country names)
        countries_col = df.iloc[1:, 0].values  # Skip first row (years)
        
        # Get the latest year with data (rightmost non-empty column)
        latest_year_idx = None
        for i in range(len(years_row)-1, -1, -1):
            if pd.notna(years_row[i]) and str(years_row[i]).strip() != '':
                latest_year_idx = i + 1  # +1 because we skipped first column
                latest_year = years_row[i]
                break
        
        if latest_year_idx is None:
            print("❌ No valid year columns found")
            return
            
        print(f"Using latest year: {latest_year}")
        
        # Extract data for the latest year
        latest_data = df.iloc[1:, latest_year_idx].values  # Skip header row
        
        # Create dataframe with countries and latest year data
        rd_df = pd.DataFrame({
            'country_raw': countries_col,
            'rd_investment': latest_data
        })
        
        # Clean country names and convert to numeric
        rd_df = rd_df[rd_df['country_raw'].notna()].copy()
        rd_df['country_raw'] = rd_df['country_raw'].astype(str).str.strip()
        
        # Convert R&D investment to numeric, handling special symbols
        rd_df['rd_investment'] = pd.to_numeric(rd_df['rd_investment'], errors='coerce')
        
        # Remove rows with no data
        rd_df = rd_df[rd_df['rd_investment'].notna()].copy()
        rd_df = rd_df[rd_df['rd_investment'] > 0].copy()  # Remove zero/negative values
        
        # Map country names
        country_name_mapping = {
            'Belgium': 'Belgium',
            'Bulgaria': 'Bulgaria', 
            'Czechia': 'Czech Republic',
            'Denmark': 'Denmark',
            'Germany': 'Germany',
            'Estonia': 'Estonia',
            'Ireland': 'Ireland',
            'Greece': 'Greece',
            'Spain': 'Spain',
            'France': 'France',
            'Croatia': 'Croatia',
            'Italy': 'Italy',
            'Cyprus': 'Cyprus',
            'Latvia': 'Latvia',
            'Lithuania': 'Lithuania',
            'Luxembourg': 'Luxembourg',
            'Hungary': 'Hungary',
            'Malta': 'Malta',
            'Netherlands': 'Netherlands',
            'Austria': 'Austria',
            'Poland': 'Poland',
            'Portugal': 'Portugal',
            'Romania': 'Romania',
            'Slovenia': 'Slovenia',
            'Slovakia': 'Slovakia',
            'Finland': 'Finland',
            'Sweden': 'Sweden',
            'Switzerland': 'Switzerland',
            'United Kingdom': 'United Kingdom',
            'Norway': 'Norway',
            'European Union - 27 countries (from 2020)': 'EU27'
        }
        
        # Apply country name mapping
        rd_df['country'] = rd_df['country_raw'].map(country_name_mapping).fillna(rd_df['country_raw'])
        
        # Filter for valid countries
        rd_df = rd_df[~rd_df['country'].str.contains('Euro area|countries|GEO', na=False)].copy()
        
        # Sort by R&D investment (descending)
        rd_df = rd_df.sort_values('rd_investment', ascending=False).reset_index(drop=True)
        
        print(f"Final dataset: {len(rd_df)} countries with construction R&D data")
        
        if len(rd_df) == 0:
            print("❌ No countries with valid construction R&D data found")
            return
        
        # Create the bar chart
        fig, ax = plt.subplots(figsize=(14, 10))
        
        countries = rd_df['country'].tolist()
        investments = rd_df['rd_investment'].tolist()
        
        # Create horizontal bars with colors
        colors = []
        for country in countries:
            if country == 'Switzerland':
                colors.append('#ffd558')  # Yellow for Switzerland
            elif country in ['EU27']:
                colors.append('#80b1d3')  # Blue for EU27
            else:
                colors.append('#8dd3c7')  # Teal for others
        
        y_positions = range(len(countries))
        bars = ax.barh(y_positions, investments, height=0.7, color=colors, 
                      alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add value labels at the end of bars
        for i, (country, investment) in enumerate(zip(countries, investments)):
            ax.text(investment + max(investments) * 0.01, i, f'{investment:.1f}', 
                   va='center', ha='left', fontsize=10, weight='bold')
        
        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(countries, fontsize=12)
        ax.set_xlabel('R&D Investment in Construction (Million PPS)', fontsize=14, weight='bold')
        ax.set_title(f'R&D Investment in Construction Sector by Country\n(NACE F - {latest_year} or latest available)', 
                    fontsize=16, weight='bold', pad=20)
        
        # Set x-axis limits
        ax.set_xlim(0, max(investments) * 1.15)
        
        # Grid and styling
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor('#f8f9fa')
        
        # Invert y-axis to show highest investments at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(OUTPUT_DIR, 'construction_rd_investment_by_country.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved construction R&D investment chart: {output_path}")
        
        # Print summary statistics
        print(f"\nConstruction R&D Investment Summary ({latest_year}):")
        print(f"  Countries analyzed: {len(rd_df)}")
        print(f"  Total investment: {sum(investments):.1f} Million PPS")
        
        if 'Switzerland' in countries:
            swiss_idx = countries.index('Switzerland')
            swiss_investment = investments[swiss_idx]
            print(f"\nSwitzerland:")
            print(f"  Investment: {swiss_investment:.1f} Million PPS")
            print(f"  Rank: {swiss_idx + 1} out of {len(countries)}")
        
        # Show top 5
        print(f"\nTop 5 countries by construction R&D investment:")
        for i in range(min(5, len(countries))):
            print(f"  {i+1}. {countries[i]}: {investments[i]:.1f} Million PPS")
            
    except Exception as e:
        print(f"❌ Error creating construction R&D investment chart: {str(e)}")
        print("   Please check the data file format and structure")

def create_construction_rd_ratio_timeline():
    """
    Create timeline chart showing Construction R&D as ratio of total R&D over time
    Uses Sheet 7 (Construction) and Sheet 1 (Total) from eurostat_BERD data
    """
    print("Creating construction R&D ratio timeline...")
    
    # File path
    rd_file = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_BERD by NACE Rev. 2 activity.xlsx')
    
    if not os.path.exists(rd_file):
        print(f"⚠️  Missing data file: {rd_file}")
        return
    
    try:
        # Load construction R&D data (Sheet 7) and total R&D data (Sheet 1)
        construction_df = pd.read_excel(rd_file, sheet_name='Sheet 7', skiprows=7)
        total_df = pd.read_excel(rd_file, sheet_name='Sheet 1', skiprows=7)
        
        print(f"Construction data shape: {construction_df.shape}")
        print(f"Total R&D data shape: {total_df.shape}")
        
        # Extract years from first row
        years = construction_df.iloc[0, 1:].values
        years = [y for y in years if pd.notna(y) and str(y).strip() != '']
        print(f"Available years: {years}")
        
        # Function to extract country data over time
        def extract_time_series(df, country_name):
            # Find the row for the country
            country_row_idx = None
            for i, country in enumerate(df.iloc[1:, 0]):
                if pd.notna(country) and str(country).strip() == country_name:
                    country_row_idx = i + 1
                    break
            
            if country_row_idx is None:
                return None
                
            # Extract values for each year
            values = []
            for year_idx in range(1, len(years) + 1):
                if year_idx < df.shape[1]:
                    val = df.iloc[country_row_idx, year_idx]
                    values.append(pd.to_numeric(val, errors='coerce'))
                else:
                    values.append(None)
            return values
        
        # Countries to analyze
        target_countries = ['Switzerland', 'Germany', 'France', 'European Union - 27 countries (from 2020)']
        
        print(f"Searching for countries in data...")
        # Debug: Check which countries are actually in the data
        available_countries = []
        for i, country in enumerate(construction_df.iloc[1:, 0]):
            if pd.notna(country):
                country_str = str(country).strip()
                available_countries.append(country_str)
                if any(target in country_str for target in ['Switzerland', 'Germany', 'France', 'European Union']):
                    print(f"Found target country: {country_str}")
        
        print(f"All available countries: {available_countries[:10]}...")  # Show first 10
        
        # Create timeline data
        timeline_data = {}
        
        for country in target_countries:
            construction_values = extract_time_series(construction_df, country)
            total_values = extract_time_series(total_df, country)
            
            if construction_values and total_values:
                # Calculate ratios
                ratios = []
                for c_val, t_val in zip(construction_values, total_values):
                    if pd.notna(c_val) and pd.notna(t_val) and t_val > 0:
                        ratios.append(c_val / t_val * 100)  # Convert to percentage
                    else:
                        ratios.append(None)
                
                # Clean country name for display
                display_name = country.replace('European Union - 27 countries (from 2020)', 'EU27')
                timeline_data[display_name] = ratios
        
        if not timeline_data:
            print("❌ No valid timeline data found")
            return
        
        # Create the timeline chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot lines for each country
        for country, ratios in timeline_data.items():
            # Filter out None values
            valid_data = [(year, ratio) for year, ratio in zip(years, ratios) if ratio is not None]
            if not valid_data:
                continue
                
            x_vals, y_vals = zip(*valid_data)
            
            # Color and style based on country
            if country == 'Switzerland':
                color = '#ffd558'
                linewidth = 3
                marker = 's'
                markersize = 8
                alpha = 1.0
            elif country == 'EU27':
                color = '#80b1d3'
                linewidth = 2.5
                marker = 'D'
                markersize = 7
                alpha = 0.9
            else:
                color = '#8dd3c7'
                linewidth = 2
                marker = 'o'
                markersize = 6
                alpha = 0.8
            
            ax.plot(x_vals, y_vals, color=color, linewidth=linewidth, 
                   marker=marker, markersize=markersize, alpha=alpha, 
                   label=country, markeredgecolor='white', markeredgewidth=1)
        
        # Formatting
        ax.set_xlabel('Year', fontsize=14, weight='bold')
        ax.set_ylabel('Construction R&D Share (% of Total R&D)', fontsize=14, weight='bold')
        ax.set_title('Construction R&D Investment as Share of Total R&D Over Time\n(NACE F as percentage of all NACE activities)', 
                    fontsize=16, weight='bold', pad=20)
        
        # Grid and styling
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        # Legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(OUTPUT_DIR, 'construction_rd_ratio_timeline.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved construction R&D ratio timeline: {output_path}")
        
        # Print summary
        print(f"\nConstruction R&D Ratio Analysis:")
        for country, ratios in timeline_data.items():
            valid_ratios = [r for r in ratios if r is not None]
            if valid_ratios:
                print(f"  {country}: {min(valid_ratios):.2f}% - {max(valid_ratios):.2f}% (range)")
                
    except Exception as e:
        print(f"❌ Error creating construction R&D ratio timeline: {str(e)}")
        print("   Please check the data file format and structure")

def create_social_housing_vs_prices_scatter():
    """
    Create scatter plot showing relationship between social rental housing share and house price changes
    Data from: ocde_social rental dwelling stock.xlsx & ocde_real house price rise 2017-2022.xlsx
    """
    print("Creating social housing vs house prices scatter plot...")
    
    # File paths
    social_housing_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_social rental dwelling stock.xlsx')
    house_prices_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_real house price rise 2017-2022.xlsx')
    
    if not os.path.exists(social_housing_file):
        print(f"⚠️  Missing social housing data file: {social_housing_file}")
        return
    
    if not os.path.exists(house_prices_file):
        print(f"⚠️  Missing house prices data file: {house_prices_file}")
        return
    
    try:
        # Load social housing data
        social_df = load_excel_data(social_housing_file)
        print(f"Social housing data shape: {social_df.shape}")
        
        # Load house price data
        prices_df = load_excel_data(house_prices_file)
        print(f"House prices data shape: {prices_df.shape}")
        
        # Process social housing data (use 2020 data, fallback to 2010)
        social_processed = social_df.copy()
        social_processed['country'] = social_processed['iso3'].map(ISO3_MAPPING)
        social_processed = social_processed[social_processed['country'].notna()].copy()
        
        # Use 2020 data where available, otherwise 2010
        social_processed['social_housing_share'] = social_processed['2020'].fillna(social_processed['2010'])
        social_final = social_processed[social_processed['social_housing_share'].notna()].copy()
        
        # Process house price data
        prices_processed = prices_df.copy()
        prices_processed['country'] = prices_processed['iso3'].map(ISO3_MAPPING)
        prices_final = prices_processed[prices_processed['country'].notna()].copy()
        prices_final = prices_final[prices_final['value'].notna()].copy()
        
        # Merge datasets
        merged_df = pd.merge(social_final[['country', 'social_housing_share']], 
                            prices_final[['country', 'value']], 
                            on='country', how='inner')
        
        print(f"Merged dataset: {len(merged_df)} countries with both metrics")
        
        if len(merged_df) == 0:
            print("❌ No countries with both social housing and price data found")
            return
        
        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot points
        for _, row in merged_df.iterrows():
            country = row['country']
            color = get_country_color(country)
            
            if country == 'Switzerland':
                marker = 's'
                size = 150
                alpha = 1.0
                edgecolor = 'black'
                linewidth = 2
                zorder = 5
            elif country in ['EU27', 'EU-27']:
                marker = 'D'
                size = 130
                alpha = 0.9
                edgecolor = 'white' 
                linewidth = 1.5
                zorder = 4
            else:
                marker = 'o'
                size = 100
                alpha = 0.8
                edgecolor = 'white'
                linewidth = 1
                zorder = 3
            
            ax.scatter(row['social_housing_share'], row['value'], 
                      color=color, s=size, marker=marker, alpha=alpha,
                      edgecolors=edgecolor, linewidth=linewidth, zorder=zorder)
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['social_housing_share'], merged_df['value'])
        
        x_range = np.linspace(merged_df['social_housing_share'].min(), merged_df['social_housing_share'].max(), 100)
        y_predicted = slope * x_range + intercept
        
        ax.plot(x_range, y_predicted, color='#fb8072', linestyle='--', alpha=0.8, linewidth=2, 
                label=f'Linear fit (R² = {r_value**2:.3f})')
        
        # Formatting
        ax.set_xlabel('Social Rental Dwelling Share (%)', fontsize=14, weight='bold')
        ax.set_ylabel('Real House Price Change 2017-2022 (%)', fontsize=14, weight='bold')
        ax.set_title('Social Housing Share vs House Price Changes\n(Higher public housing may moderate price increases)', 
                    fontsize=16, weight='bold', pad=20)
        
        # Convert to percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))
        
        # Add country labels for key countries
        for _, row in merged_df.iterrows():
            if row['country'] in ['Switzerland', 'EU27']:
                ax.annotate(row['country'], 
                           (row['social_housing_share'], row['value']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Grid and styling
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        # Legend
        legend_elements = [
            plt.scatter([], [], color='#ffd558', s=150, marker='s', alpha=1.0, 
                       edgecolors='black', linewidth=2, label='Switzerland'),
            plt.scatter([], [], color='#80b1d3', s=130, marker='D', alpha=0.9, 
                       edgecolors='white', linewidth=1.5, label='EU-27'),
            plt.scatter([], [], color='#8dd3c7', s=100, marker='o', alpha=0.8, 
                       edgecolors='white', linewidth=1, label='Other countries'),
            plt.Line2D([0], [0], color='#fb8072', linestyle='--', alpha=0.8, linewidth=2, 
                      label=f'Linear fit (R² = {r_value**2:.3f})')
        ]
        ax.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(OUTPUT_DIR, 'social_housing_vs_house_prices.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved social housing vs house prices scatter plot: {output_path}")
        
        # Print summary statistics
        print(f"\nSocial Housing vs House Prices Analysis:")
        print(f"  Slope: {slope:.4f}")
        print(f"  R-squared: {r_value**2:.3f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Countries in analysis: {len(merged_df)}")
        
        if 'Switzerland' in merged_df['country'].values:
            swiss_data = merged_df[merged_df['country'] == 'Switzerland'].iloc[0]
            print(f"\nSwitzerland data:")
            print(f"  Social housing share: {swiss_data['social_housing_share']:.1%}")
            print(f"  House price change: {swiss_data['value']:.1%}")
            
    except Exception as e:
        print(f"❌ Error creating social housing vs prices scatter: {str(e)}")
        print("   Please check the data file format and structure")

def create_homeownership_vs_prices_scatter():
    """
    Create scatter plot showing relationship between homeownership rates and house price changes
    Data from: ocde_Homeownership rates.xlsx & ocde_real house price rise 2017-2022.xlsx
    """
    print("Creating homeownership vs house prices scatter plot...")
    
    # File paths
    homeownership_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_Homeownership rates.xlsx')
    house_prices_file = os.path.join(EXTERNAL_DATA_DIR, 'ocde_real house price rise 2017-2022.xlsx')
    
    if not os.path.exists(homeownership_file):
        print(f"⚠️  Missing homeownership data file: {homeownership_file}")
        return
    
    if not os.path.exists(house_prices_file):
        print(f"⚠️  Missing house prices data file: {house_prices_file}")
        return
    
    try:
        # Load homeownership data
        homeownership_df = load_excel_data(homeownership_file)
        print(f"Homeownership data shape: {homeownership_df.shape}")
        
        # Load house price data
        prices_df = load_excel_data(house_prices_file)
        print(f"House prices data shape: {prices_df.shape}")
        
        # Process homeownership data (sum outright owners + owners with mortgages)
        homeownership_processed = homeownership_df.groupby('iso3')['Homeownership rate'].sum().reset_index()
        homeownership_processed.columns = ['iso3', 'total_homeownership']
        
        # Add country names
        homeownership_processed['country'] = homeownership_processed['iso3'].map(ISO3_MAPPING)
        homeownership_final = homeownership_processed[homeownership_processed['country'].notna()].copy()
        
        # Process house price data
        prices_processed = prices_df.copy()
        prices_processed['country'] = prices_processed['iso3'].map(ISO3_MAPPING)
        prices_final = prices_processed[prices_processed['country'].notna()].copy()
        prices_final = prices_final[prices_final['value'].notna()].copy()
        
        # Merge datasets
        merged_df = pd.merge(homeownership_final[['country', 'total_homeownership']], 
                            prices_final[['country', 'value']], 
                            on='country', how='inner')
        
        print(f"Merged dataset: {len(merged_df)} countries with both metrics")
        
        if len(merged_df) == 0:
            print("❌ No countries with both homeownership and price data found")
            return
        
        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot points
        for _, row in merged_df.iterrows():
            country = row['country']
            color = get_country_color(country)
            
            if country == 'Switzerland':
                marker = 's'
                size = 150
                alpha = 1.0
                edgecolor = 'black'
                linewidth = 2
                zorder = 5
            elif country in ['EU27', 'EU-27']:
                marker = 'D'
                size = 130
                alpha = 0.9
                edgecolor = 'white' 
                linewidth = 1.5
                zorder = 4
            else:
                marker = 'o'
                size = 100
                alpha = 0.8
                edgecolor = 'white'
                linewidth = 1
                zorder = 3
            
            ax.scatter(row['total_homeownership'], row['value'], 
                      color=color, s=size, marker=marker, alpha=alpha,
                      edgecolors=edgecolor, linewidth=linewidth, zorder=zorder)
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['total_homeownership'], merged_df['value'])
        
        x_range = np.linspace(merged_df['total_homeownership'].min(), merged_df['total_homeownership'].max(), 100)
        y_predicted = slope * x_range + intercept
        
        ax.plot(x_range, y_predicted, color='#fb8072', linestyle='--', alpha=0.8, linewidth=2, 
                label=f'Linear fit (R² = {r_value**2:.3f})')
        
        # Formatting
        ax.set_xlabel('Total Homeownership Rate (%)', fontsize=14, weight='bold')
        ax.set_ylabel('Real House Price Change 2017-2022 (%)', fontsize=14, weight='bold')
        ax.set_title('Homeownership Rate vs House Price Changes\n(Higher ownership may amplify price sensitivity)', 
                    fontsize=16, weight='bold', pad=20)
        
        # Convert to percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))
        
        # Add country labels for key countries
        for _, row in merged_df.iterrows():
            if row['country'] in ['Switzerland', 'EU27']:
                ax.annotate(row['country'], 
                           (row['total_homeownership'], row['value']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Grid and styling
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        # Legend
        legend_elements = [
            plt.scatter([], [], color='#ffd558', s=150, marker='s', alpha=1.0, 
                       edgecolors='black', linewidth=2, label='Switzerland'),
            plt.scatter([], [], color='#80b1d3', s=130, marker='D', alpha=0.9, 
                       edgecolors='white', linewidth=1.5, label='EU-27'),
            plt.scatter([], [], color='#8dd3c7', s=100, marker='o', alpha=0.8, 
                       edgecolors='white', linewidth=1, label='Other countries'),
            plt.Line2D([0], [0], color='#fb8072', linestyle='--', alpha=0.8, linewidth=2, 
                      label=f'Linear fit (R² = {r_value**2:.3f})')
        ]
        ax.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(OUTPUT_DIR, 'homeownership_vs_house_prices.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved homeownership vs house prices scatter plot: {output_path}")
        
        # Print summary statistics
        print(f"\nHomeownership vs House Prices Analysis:")
        print(f"  Slope: {slope:.4f}")
        print(f"  R-squared: {r_value**2:.3f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Countries in analysis: {len(merged_df)}")
        
        if 'Switzerland' in merged_df['country'].values:
            swiss_data = merged_df[merged_df['country'] == 'Switzerland'].iloc[0]
            print(f"\nSwitzerland data:")
            print(f"  Total homeownership rate: {swiss_data['total_homeownership']:.1%}")
            print(f"  House price change: {swiss_data['value']:.1%}")
            
    except Exception as e:
        print(f"❌ Error creating homeownership vs prices scatter: {str(e)}")
        print("   Please check the data file format and structure")

def main():
    """Main function to generate all graphs"""
    print("OECD Graphs Generator")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # List available data files
    print(f"Looking for data files in: {EXTERNAL_DATA_DIR}")
    if os.path.exists(EXTERNAL_DATA_DIR):
        data_files = [f for f in os.listdir(EXTERNAL_DATA_DIR) if f.endswith('.xlsx')]
        print(f"Available Excel files: {data_files}")
    else:
        print("External data directory not found!")
        return
    
    # Generate all graphs
    create_household_debt_graph()
    create_debt_vs_house_price_scatter()
    create_real_house_price_graphs()
    create_social_rental_dwelling_graph()
    create_elasticities_scatter_plot()
    create_housing_allowance_graph()
    create_average_rent_allowance_graph()
    create_residential_mobility_homeownership_graph()
    create_homeownership_rates_bar_chart()
    create_construction_rd_investment_chart()
    create_construction_rd_ratio_timeline()
    create_social_housing_vs_prices_scatter()
    create_homeownership_vs_prices_scatter()
    
    print("\nAll graphs generated successfully!")
    print(f"Graphs saved to: {OUTPUT_DIR}")

# Template function for future graphs
def create_graph_template(data_filename, graph_title, output_filename):
    """
    Template function for creating new graphs from OECD/EUROSTAT data
    
    Parameters:
    - data_filename: Excel file name in external_data folder
    - graph_title: Title for the graph
    - output_filename: Output PNG filename (without extension)
    """
    print(f"Creating {graph_title}...")
    
    # Load the data
    df = load_excel_data(data_filename)
    if df is None:
        print(f"Could not load {data_filename}")
        return
    
    print(f"Data loaded: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # TODO: Implement specific graph logic here
    # This is where you would add the specific data processing and plotting code
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, f'{output_filename}.png')
    # plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"Graph template ready for: {output_path}")

if __name__ == "__main__":
    main()