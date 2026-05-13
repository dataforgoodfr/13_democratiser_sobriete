"""
Energy vs GDP Analysis - Generate graphs of GDP PPS (log) vs Final Energy Consumption (log)
This script creates visualizations for years: 1994, 2004, 2014, 2024
Outputs saved as PNG files in outputs/graphs/energy folder
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# Base paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, 'external_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'graphs', 'energy')
EWBI_MASTER_PATH = os.path.abspath(
    os.path.join(BASE_DIR, '..', '..', 'output', 'ewbi_master_aggregated.csv')
)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set font
plt.rcParams['font.family'] = 'Arial'

# Exchange rate for converting constant 2021 international $ to constant 2021 €
USD_TO_EUR_2021 = 1.0 / 1.1827

# List of countries to include
study_countries = ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI',  
                   'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 
                   'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 
                   'UK', 'NO', 'CH', 'IS', 'LI']

# Mapping Eurostat country names to country codes
EUROSTAT_TO_CODE = {
    'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
    'Cyprus': 'CY', 'Czechia': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE',
    'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'GR',
    'Hungary': 'HU', 'Ireland': 'IE', 'Italy': 'IT', 'Latvia': 'LV',
    'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT', 'Netherlands': 'NL',
    'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO', 'Slovakia': 'SK',
    'Slovenia': 'SI', 'Spain': 'ES', 'Sweden': 'SE', 'Norway': 'NO',
    'Switzerland': 'CH', 'Iceland': 'IS', 'Liechtenstein': 'LI',
    'United Kingdom': 'UK'
}

# Years to analyze
years_to_plot = [1996, 2004, 2014, 2024]

# Mapping ISO2 study country codes to World Bank ISO3 country codes
STUDY_ISO2_TO_ISO3 = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'HR': 'HRV', 'CY': 'CYP', 'CZ': 'CZE',
    'DK': 'DNK', 'EE': 'EST', 'FI': 'FIN', 'FR': 'FRA', 'DE': 'DEU', 'GR': 'GRC',
    'HU': 'HUN', 'IE': 'IRL', 'IT': 'ITA', 'LV': 'LVA', 'LT': 'LTU', 'LU': 'LUX',
    'MT': 'MLT', 'NL': 'NLD', 'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'SK': 'SVK',
    'SI': 'SVN', 'ES': 'ESP', 'SE': 'SWE', 'UK': 'GBR', 'NO': 'NOR', 'CH': 'CHE',
    'IS': 'ISL', 'LI': 'LIE'
}
STUDY_ISO3_TO_ISO2 = {v: k for k, v in STUDY_ISO2_TO_ISO3.items()}

def prepare_worldbank_gdp_ppp(df_worldbank_raw, selected_years=None):
    """Prepare World Bank GDP PPP (constant 2021 €) in long format with study country ISO2 codes."""
    if selected_years is None:
        year_columns = [col for col in df_worldbank_raw.columns if str(col).isdigit()]
    else:
        year_columns = [str(year) for year in selected_years]

    available_year_columns = [col for col in year_columns if col in df_worldbank_raw.columns]
    if not available_year_columns:
        return pd.DataFrame(columns=['geo_code', 'year', 'gdp_ppp_2021'])

    df_wb_gdp = df_worldbank_raw[['Country Code', 'Indicator Code'] + available_year_columns].copy()
    df_wb_gdp = df_wb_gdp[df_wb_gdp['Indicator Code'] == 'NY.GDP.MKTP.PP.KD'].copy()

    df_wb_gdp = df_wb_gdp.melt(
        id_vars=['Country Code'],
        value_vars=available_year_columns,
        var_name='year',
        value_name='gdp_ppp_2021'
    )

    df_wb_gdp['year'] = pd.to_numeric(df_wb_gdp['year'], errors='coerce').astype('Int64')
    df_wb_gdp['gdp_ppp_2021'] = pd.to_numeric(df_wb_gdp['gdp_ppp_2021'], errors='coerce')
    df_wb_gdp['geo_code'] = df_wb_gdp['Country Code'].map(STUDY_ISO3_TO_ISO2)
    df_wb_gdp = df_wb_gdp[df_wb_gdp['geo_code'].notnull()].copy()
    df_wb_gdp = df_wb_gdp[df_wb_gdp['geo_code'].isin(study_countries)].copy()
    df_wb_gdp = df_wb_gdp.dropna(subset=['year', 'gdp_ppp_2021']).copy()
    df_wb_gdp['year'] = df_wb_gdp['year'].astype(int)

    # Convert from constant 2021 international $ to constant 2021 €
    df_wb_gdp['gdp_ppp_2021'] = df_wb_gdp['gdp_ppp_2021'] * USD_TO_EUR_2021

    return df_wb_gdp[['geo_code', 'year', 'gdp_ppp_2021']].copy()

def main():
    print("=" * 80)
    print("ENERGY VS GDP ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df_gdp_pps = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_gdp_pps.csv'))
    df_gdp_eur = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_gdp_eur-2005.csv'))
    df_energy_raw = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_final_energy.csv'))
    df_population_raw = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_population.csv'))
    df_hicp = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_HICP.csv'))
    df_worldbank_gdp_ppp = pd.read_csv(
        os.path.join(EXTERNAL_DATA_DIR, 'worldbank_gdp_ppp_$2021.csv'),
        skiprows=4
    )
    df_ewbi_master = pd.read_csv(EWBI_MASTER_PATH, low_memory=False)
    df_penn_raw = pd.read_csv(
        os.path.join(EXTERNAL_DATA_DIR, 'penn_capital_stock.csv'), sep=';'
    )
    df_hdi_raw = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'human-development-index.csv'))
    df_eurostat_nfa_raw = pd.read_csv(
        os.path.join(EXTERNAL_DATA_DIR, 'eurostat_nama_10_nfa_bs__custom_20803373_linear.csv')
    )
    df_wb_gcf_raw = pd.read_csv(
        os.path.join(EXTERNAL_DATA_DIR, 'world_bank_API_NE.GDI.TOTL.KD_DS2_en_csv_v2_14082.csv'),
        skiprows=4
    )
    df_wb_gni_raw = pd.read_excel(
        os.path.join(EXTERNAL_DATA_DIR, 'world_bank_API_NY.GNP.PCAP.PP.CD_DS2_en_excel_v2_715.xls')
    )
    
    print(f"  GDP PPS data: {len(df_gdp_pps)} rows")
    print(f"  GDP EUR 2005 data: {len(df_gdp_eur)} rows")
    print(f"  Energy data: {len(df_energy_raw)} rows")
    print(f"  Population data: {len(df_population_raw)} rows")
    print(f"  HICP data: {len(df_hicp)} rows")
    print(f"  World Bank GDP PPP (2021 €) data: {len(df_worldbank_gdp_ppp)} rows")
    print(f"  EWBI master aggregated data: {len(df_ewbi_master)} rows")
    print(f"  Penn World capital stock data: {len(df_penn_raw)} rows")
    print(f"  Human Development Index data: {len(df_hdi_raw)} rows")
    print(f"  Eurostat capital stocks (NFA) data: {len(df_eurostat_nfa_raw)} rows")
    print(f"  World Bank Gross capital formation data: {len(df_wb_gcf_raw)} rows")
    print(f"  World Bank GNI per capita PPP data: {len(df_wb_gni_raw)} rows")
    
    # === PART 1: LOG-LOG ANALYSIS ===
    print("\n" + "=" * 80)
    print("PART 1: LOG-LOG SCATTER PLOTS")
    print("=" * 80)
    
    # Process GDP PPS data (Current prices, million purchasing power standards, PPS EU27 from 2020)
    df_gdp = df_gdp_pps[['geo', 'TIME_PERIOD', 'OBS_VALUE']].copy()
    df_gdp.columns = ['geo', 'year', 'gdp_pps']
    df_gdp['geo_code'] = df_gdp['geo'].map(EUROSTAT_TO_CODE)
    df_gdp = df_gdp[df_gdp['geo_code'].notnull()].copy()
    df_gdp = df_gdp[df_gdp['geo_code'].isin(study_countries)].copy()
    
    # Process energy data (Final consumption - energy use in Gigawatt-hour)
    df_energy = df_energy_raw[['geo', 'TIME_PERIOD', 'OBS_VALUE']].copy()
    df_energy.columns = ['geo', 'year', 'energy']
    df_energy['geo_code'] = df_energy['geo'].map(EUROSTAT_TO_CODE)
    df_energy = df_energy[df_energy['geo_code'].notnull()].copy()
    df_energy = df_energy[df_energy['geo_code'].isin(study_countries)].copy()
    
    # Merge datasets
    df_merged = df_gdp.merge(df_energy[['geo_code', 'year', 'energy']], 
                              on=['geo_code', 'year'], how='inner')
    
    # Filter for years of interest
    df_merged = df_merged[df_merged['year'].isin(years_to_plot)].copy()
    
    # Remove missing or zero values
    df_merged = df_merged[(df_merged['gdp_pps'] > 0) & (df_merged['energy'] > 0)].copy()
    
    print(f"\nMerged data: {len(df_merged)} rows")
    print(f"Countries: {df_merged['geo_code'].nunique()}")
    print(f"Years available: {sorted(df_merged['year'].unique())}")
    
    # Create the plot
    create_energy_gdp_plot(df_merged)
    
    # Create simplified version
    create_energy_gdp_plot_simple(df_merged)
    
    # Export data to Excel
    export_path = os.path.join(OUTPUT_DIR, 'energy_gdp_data.xlsx')
    df_export = df_merged[['geo_code', 'year', 'energy', 'gdp_pps']].copy()
    df_export.columns = ['Country', 'Year', 'Final Energy (GWh)', 'GDP PPS (Million, EU27 from 2020)']
    df_export.to_excel(export_path, index=False)
    print(f"\n✓ Data exported to: {export_path}")
    
    # === PART 1B: LOG-LOG ANALYSIS WITH INFLATION CORRECTION ===
    print("\n" + "=" * 80)
    print("PART 1B: LOG-LOG SCATTER PLOTS (INFLATION-CORRECTED GDP)")
    print("=" * 80)
    
    # Process HICP data
    df_hicp_proc = df_hicp[['geo', 'TIME_PERIOD', 'OBS_VALUE']].copy()
    df_hicp_proc.columns = ['geo', 'year', 'hicp']
    df_hicp_proc['geo_code'] = df_hicp_proc['geo'].map(EUROSTAT_TO_CODE)
    df_hicp_proc = df_hicp_proc[df_hicp_proc['geo_code'].notnull()].copy()
    df_hicp_proc = df_hicp_proc[df_hicp_proc['geo_code'].isin(study_countries)].copy()
    
    # Merge GDP PPS with HICP
    df_gdp_hicp = df_gdp.merge(df_hicp_proc[['geo_code', 'year', 'hicp']], 
                                on=['geo_code', 'year'], how='inner')
    
    # Correct GDP for inflation (divide by HICP index and multiply by 100 to normalize)
    df_gdp_hicp['gdp_pps_real'] = (df_gdp_hicp['gdp_pps'] / df_gdp_hicp['hicp']) * 100
    
    # Identify and remove aberrant HICP values (check for extreme outliers in 1996)
    df_1996 = df_gdp_hicp[df_gdp_hicp['year'] == 1996].copy()
    if len(df_1996) > 0:
        # Calculate z-score for gdp_pps_real to identify outliers
        mean_gdp = df_1996['gdp_pps_real'].mean()
        std_gdp = df_1996['gdp_pps_real'].std()
        df_1996['z_score'] = (df_1996['gdp_pps_real'] - mean_gdp) / std_gdp
        
        # Find countries with z-score > 3 (statistical outliers)
        outliers = df_1996[df_1996['z_score'].abs() > 3]
        if len(outliers) > 0:
            print(f"\n  Identified aberrant HICP values (outliers) in 1996:")
            for _, row in outliers.iterrows():
                print(f"    {row['geo_code']}: HICP={row['hicp']:.2f}, GDP_real={row['gdp_pps_real']:.2f}")
            
            # Remove outliers from the dataset
            outlier_countries = outliers['geo_code'].tolist()
            df_gdp_hicp = df_gdp_hicp[~((df_gdp_hicp['year'] == 1996) & 
                                        (df_gdp_hicp['geo_code'].isin(outlier_countries)))].copy()
            print(f"    Removed {len(outliers)} outlier(s) from 1996 data")
    
    # Merge with energy data
    df_merged_real = df_gdp_hicp.merge(df_energy[['geo_code', 'year', 'energy']], 
                                        on=['geo_code', 'year'], how='inner')
    
    # Filter for years of interest
    df_merged_real = df_merged_real[df_merged_real['year'].isin(years_to_plot)].copy()
    
    # Remove missing or zero values
    df_merged_real = df_merged_real[(df_merged_real['gdp_pps_real'] > 0) & 
                                     (df_merged_real['energy'] > 0)].copy()
    
    print(f"\nMerged data (inflation-corrected): {len(df_merged_real)} rows")
    print(f"Countries: {df_merged_real['geo_code'].nunique()}")
    print(f"Years available: {sorted(df_merged_real['year'].unique())}")
    
    # Create the inflation-corrected plot
    create_energy_gdp_plot_real(df_merged_real)
    
    # Create simplified version
    create_energy_gdp_plot_real_simple(df_merged_real)
    
    # Export data to Excel
    export_path_real = os.path.join(OUTPUT_DIR, 'energy_gdp_data_inflation_corrected.xlsx')
    df_export_real = df_merged_real[['geo_code', 'year', 'energy', 'gdp_pps', 'hicp', 'gdp_pps_real']].copy()
    df_export_real.columns = ['Country', 'Year', 'Final Energy (GWh)', 
                              'GDP PPS (Million, EU27 from 2020)', 'HICP Index', 
                              'GDP PPS Real (Inflation-Corrected)']
    df_export_real.to_excel(export_path_real, index=False)
    print(f"\n✓ Inflation-corrected data exported to: {export_path_real}")

    # === PART 1C: LOG-LOG ANALYSIS WITH WORLD BANK GDP PPP (CONSTANT 2021 €) ===
    print("\n" + "=" * 80)
    print("PART 1C: LOG-LOG SCATTER PLOTS (WORLD BANK GDP PPP, CONSTANT 2021 €)")
    print("=" * 80)

    # Process World Bank GDP PPP data (selected years only)
    df_wb_gdp = prepare_worldbank_gdp_ppp(df_worldbank_gdp_ppp, selected_years=years_to_plot)

    # Merge with energy data
    df_merged_wb = df_wb_gdp.merge(
        df_energy[['geo_code', 'year', 'energy']],
        on=['geo_code', 'year'],
        how='inner'
    )

    # Remove missing or zero values
    df_merged_wb = df_merged_wb[(df_merged_wb['gdp_ppp_2021'] > 0) & (df_merged_wb['energy'] > 0)].copy()

    print(f"\nMerged data (World Bank PPP 2021): {len(df_merged_wb)} rows")
    print(f"Countries: {df_merged_wb['geo_code'].nunique()}")
    print(f"Years available: {sorted(df_merged_wb['year'].unique())}")

    # Create simplified World Bank PPP plot
    create_energy_gdp_plot_worldbank_simple(df_merged_wb)

    # Export data to Excel
    export_path_wb = os.path.join(OUTPUT_DIR, 'energy_gdp_data_worldbank_ppp2021.xlsx')
    df_export_wb = df_merged_wb[['geo_code', 'year', 'energy', 'gdp_ppp_2021']].copy()
    df_export_wb.columns = [
        'Country',
        'Year',
        'Final Energy (GWh)',
        'GDP PPP (Constant 2021 €)'
    ]
    df_export_wb.to_excel(export_path_wb, index=False)
    print(f"\n✓ World Bank PPP data exported to: {export_path_wb}")

    # === PART 1D: EWBI AGAINST ENERGY/GDP PPP PER CAPITA ===
    print("\n" + "=" * 80)
    print("PART 1D: EWBI VS ENERGY AND GDP PPP PER CAPITA")
    print("=" * 80)

    # Process population data (explicit source: eurostat_population.csv)
    df_population = df_population_raw[['geo', 'TIME_PERIOD', 'OBS_VALUE']].copy()
    df_population.columns = ['geo', 'year', 'population']
    df_population['geo_code'] = df_population['geo'].map(EUROSTAT_TO_CODE)
    df_population['year'] = pd.to_numeric(df_population['year'], errors='coerce')
    df_population['population'] = pd.to_numeric(df_population['population'], errors='coerce')
    df_population = df_population[df_population['geo_code'].notnull()].copy()
    df_population = df_population[df_population['geo_code'].isin(study_countries)].copy()
    df_population = df_population.dropna(subset=['year', 'population']).copy()
    df_population['year'] = df_population['year'].astype(int)
    df_population = df_population[df_population['population'] > 0].copy()

    # Build energy per capita (MWh/person)
    df_energy_pc = df_energy.merge(
        df_population[['geo_code', 'year', 'population']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_energy_pc['energy_per_capita_mwh'] = (df_energy_pc['energy'] * 1000.0) / df_energy_pc['population']
    df_energy_pc = df_energy_pc[df_energy_pc['energy_per_capita_mwh'] > 0].copy()

    # Build GDP PPP per capita (constant 2021 € per person)
    df_wb_all_years = prepare_worldbank_gdp_ppp(df_worldbank_gdp_ppp)
    df_gdp_pc = df_wb_all_years.merge(
        df_population[['geo_code', 'year', 'population']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_gdp_pc['gdp_ppp_2021_per_capita'] = df_gdp_pc['gdp_ppp_2021'] / df_gdp_pc['population']
    df_gdp_pc = df_gdp_pc[df_gdp_pc['gdp_ppp_2021_per_capita'] > 0].copy()

    # Process EWBI country-level data using app-consistent filtering logic
    ewbi_columns = ['Year', 'Country', 'Decile', 'Level', 'Primary and raw data', 'Value']
    df_ewbi = df_ewbi_master[ewbi_columns].copy()
    df_ewbi = df_ewbi[
        (df_ewbi['Level'] == 1) &
        (df_ewbi['Primary and raw data'] == 'EWBI') &
        (df_ewbi['Decile'] == 'All Deciles')
    ].copy()

    # Match dashboard logic for excluding aggregate/non-country rows
    df_ewbi = df_ewbi[
        df_ewbi['Country'].notna() &
        (df_ewbi['Country'] != 'EU-27') &
        (df_ewbi['Country'] != 'All Countries') &
        (~df_ewbi['Country'].astype(str).str.contains('Average|Median', na=False))
    ].copy()

    # Normalize Greece code for alignment with Eurostat energy mapping
    df_ewbi['geo_code'] = df_ewbi['Country'].replace({'EL': 'GR'})
    df_ewbi['year'] = pd.to_numeric(df_ewbi['Year'], errors='coerce')
    df_ewbi['ewbi'] = pd.to_numeric(df_ewbi['Value'], errors='coerce')
    df_ewbi = df_ewbi[df_ewbi['geo_code'].isin(study_countries)].copy()
    df_ewbi = df_ewbi.dropna(subset=['year', 'ewbi']).copy()
    df_ewbi['year'] = df_ewbi['year'].astype(int)

    # Merge EWBI with energy per capita
    df_ewbi_energy = df_ewbi[['geo_code', 'year', 'ewbi']].merge(
        df_energy_pc[['geo_code', 'year', 'energy_per_capita_mwh']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_ewbi_energy = df_ewbi_energy[(df_ewbi_energy['ewbi'] > 0) & (df_ewbi_energy['energy_per_capita_mwh'] > 0)].copy()

    print(f"\nEWBI + energy per capita merged rows: {len(df_ewbi_energy)}")
    print(f"Countries: {df_ewbi_energy['geo_code'].nunique()}")
    print(f"Years available: {sorted(df_ewbi_energy['year'].unique())}")

    create_ewbi_trajectory_plot(
        df_ewbi_energy,
        x_col='energy_per_capita_mwh',
        y_col='ewbi',
        x_label='Final Energy per Capita (MWh/person)',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs Final Energy per Capita Over Time',
        output_filename='ewbi_vs_energy_per_capita_trajectory.png',
        save_svg=True
    )
    create_ewbi_last_year_plot(
        df_ewbi_energy,
        x_col='energy_per_capita_mwh',
        y_col='ewbi',
        x_label='Final Energy per Capita (MWh/person)',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs Final Energy per Capita',
        output_filename='ewbi_vs_energy_per_capita_last_year.png'
    )

    # Merge EWBI with GDP PPP per capita
    df_ewbi_gdp_pc = df_ewbi[['geo_code', 'year', 'ewbi']].merge(
        df_gdp_pc[['geo_code', 'year', 'gdp_ppp_2021_per_capita']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_ewbi_gdp_pc = df_ewbi_gdp_pc[(df_ewbi_gdp_pc['ewbi'] > 0) & (df_ewbi_gdp_pc['gdp_ppp_2021_per_capita'] > 0)].copy()

    print(f"\nEWBI + GDP PPP per capita merged rows: {len(df_ewbi_gdp_pc)}")
    print(f"Countries: {df_ewbi_gdp_pc['geo_code'].nunique()}")
    print(f"Years available: {sorted(df_ewbi_gdp_pc['year'].unique())}")

    create_ewbi_trajectory_plot(
        df_ewbi_gdp_pc,
        x_col='gdp_ppp_2021_per_capita',
        y_col='ewbi',
        x_label='GDP PPP per Capita (Constant 2021 €)',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs GDP PPP per Capita Over Time',
        output_filename='ewbi_vs_gdp_ppp2021_per_capita_trajectory.png',
        save_svg=True
    )
    create_ewbi_last_year_plot(
        df_ewbi_gdp_pc,
        x_col='gdp_ppp_2021_per_capita',
        y_col='ewbi',
        x_label='GDP PPP per Capita (Constant 2021 €)',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs GDP PPP per Capita',
        output_filename='ewbi_vs_gdp_ppp2021_per_capita_last_year.png'
    )

    # Export merged EWBI analysis tables
    export_ewbi_energy_path = os.path.join(OUTPUT_DIR, 'ewbi_energy_per_capita_data.xlsx')
    df_export_ewbi_energy = df_ewbi_energy[['geo_code', 'year', 'ewbi', 'energy_per_capita_mwh']].copy()
    df_export_ewbi_energy.columns = ['Country', 'Year', 'EWBI', 'Final Energy per Capita (MWh/person)']
    df_export_ewbi_energy.to_excel(export_ewbi_energy_path, index=False)
    print(f"  ✓ EWBI-energy per capita data exported to: {export_ewbi_energy_path}")

    export_ewbi_gdp_path = os.path.join(OUTPUT_DIR, 'ewbi_gdp_ppp2021_per_capita_data.xlsx')
    df_export_ewbi_gdp = df_ewbi_gdp_pc[['geo_code', 'year', 'ewbi', 'gdp_ppp_2021_per_capita']].copy()
    df_export_ewbi_gdp.columns = ['Country', 'Year', 'EWBI', 'GDP PPP per Capita (Constant 2021 €)']
    df_export_ewbi_gdp.to_excel(export_ewbi_gdp_path, index=False)
    print(f"  ✓ EWBI-GDP per capita data exported to: {export_ewbi_gdp_path}")

    # === PART 1E: EWBI VS PENN WORLD CAPITAL STOCK PER CAPITA ===
    print("\n" + "=" * 80)
    print("PART 1E: EWBI VS PENN WORLD CAPITAL STOCK PER CAPITA")
    print("=" * 80)

    # Process Penn World Table capital stock data
    df_penn = df_penn_raw.copy()
    df_penn['year'] = pd.to_numeric(df_penn['year'], errors='coerce')
    df_penn['value'] = pd.to_numeric(df_penn['value'], errors='coerce')
    df_penn['geo_code'] = df_penn['countrycode'].map(STUDY_ISO3_TO_ISO2)
    df_penn = df_penn[df_penn['geo_code'].notnull()].copy()
    df_penn = df_penn[df_penn['geo_code'].isin(study_countries)].copy()
    df_penn = df_penn.dropna(subset=['year', 'value']).copy()
    df_penn['year'] = df_penn['year'].astype(int)
    df_penn = df_penn[df_penn['value'] > 0].copy()

    # Capital stock per capita (mil. 2017US$ / population -> €/person: multiply by 1e6 and convert)
    df_penn_pc = df_penn[['geo_code', 'year', 'value']].merge(
        df_population[['geo_code', 'year', 'population']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_penn_pc['capital_stock_per_capita'] = (df_penn_pc['value'] * 1e6 * USD_TO_EUR_2021) / df_penn_pc['population']
    df_penn_pc = df_penn_pc[df_penn_pc['capital_stock_per_capita'] > 0].copy()

    # Merge EWBI with Penn capital stock per capita
    df_ewbi_penn = df_ewbi[['geo_code', 'year', 'ewbi']].merge(
        df_penn_pc[['geo_code', 'year', 'capital_stock_per_capita']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_ewbi_penn = df_ewbi_penn[(df_ewbi_penn['ewbi'] > 0) & (df_ewbi_penn['capital_stock_per_capita'] > 0)].copy()

    print(f"\nEWBI + Penn capital stock per capita merged rows: {len(df_ewbi_penn)}")
    print(f"Countries: {df_ewbi_penn['geo_code'].nunique()}")
    print(f"Years available: {sorted(df_ewbi_penn['year'].unique())}")

    create_ewbi_trajectory_plot(
        df_ewbi_penn,
        x_col='capital_stock_per_capita',
        y_col='ewbi',
        x_label='Capital Stock per Capita (2017 € PPP)',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs Capital Stock per Capita (Penn World Table)',
        output_filename='ewbi_vs_penn_capital_stock_per_capita_trajectory.png'
    )
    create_ewbi_last_year_plot(
        df_ewbi_penn,
        x_col='capital_stock_per_capita',
        y_col='ewbi',
        x_label='Capital Stock per Capita (2017 € PPP)',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs Capital Stock per Capita (Penn World Table)',
        output_filename='ewbi_vs_penn_capital_stock_per_capita_last_year.png'
    )

    export_ewbi_penn_path = os.path.join(OUTPUT_DIR, 'ewbi_penn_capital_stock_per_capita_data.xlsx')
    df_export_penn = df_ewbi_penn[['geo_code', 'year', 'ewbi', 'capital_stock_per_capita']].copy()
    df_export_penn.columns = ['Country', 'Year', 'EWBI', 'Capital Stock per Capita (2017 € PPP)']
    df_export_penn.to_excel(export_ewbi_penn_path, index=False)
    print(f"  ✓ EWBI-Penn capital stock data exported to: {export_ewbi_penn_path}")

    # === PART 1F: EWBI VS HUMAN DEVELOPMENT INDEX ===
    print("\n" + "=" * 80)
    print("PART 1F: EWBI VS HUMAN DEVELOPMENT INDEX")
    print("=" * 80)

    # Process HDI data (Code is ISO3)
    df_hdi = df_hdi_raw.copy()
    df_hdi['year'] = pd.to_numeric(df_hdi['Year'], errors='coerce')
    df_hdi['hdi'] = pd.to_numeric(df_hdi['Human Development Index'], errors='coerce')
    df_hdi['geo_code'] = df_hdi['Code'].map(STUDY_ISO3_TO_ISO2)
    df_hdi = df_hdi[df_hdi['geo_code'].notnull()].copy()
    df_hdi = df_hdi[df_hdi['geo_code'].isin(study_countries)].copy()
    df_hdi = df_hdi.dropna(subset=['year', 'hdi']).copy()
    df_hdi['year'] = df_hdi['year'].astype(int)
    df_hdi = df_hdi[df_hdi['hdi'] > 0].copy()

    # Merge EWBI with HDI (no per capita needed, HDI is already an index)
    df_ewbi_hdi = df_ewbi[['geo_code', 'year', 'ewbi']].merge(
        df_hdi[['geo_code', 'year', 'hdi']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_ewbi_hdi = df_ewbi_hdi[(df_ewbi_hdi['ewbi'] > 0) & (df_ewbi_hdi['hdi'] > 0)].copy()

    print(f"\nEWBI + HDI merged rows: {len(df_ewbi_hdi)}")
    print(f"Countries: {df_ewbi_hdi['geo_code'].nunique()}")
    print(f"Years available: {sorted(df_ewbi_hdi['year'].unique())}")

    create_ewbi_trajectory_plot(
        df_ewbi_hdi,
        x_col='hdi',
        y_col='ewbi',
        x_label='Human Development Index',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs Human Development Index Over Time',
        output_filename='ewbi_vs_hdi_trajectory.png',
        regression_by_year=True,
        save_svg=True
    )
    create_ewbi_last_year_plot(
        df_ewbi_hdi,
        x_col='hdi',
        y_col='ewbi',
        x_label='Human Development Index',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs Human Development Index',
        output_filename='ewbi_vs_hdi_last_year.png'
    )

    # --- HDI without GNI component ---
    # Process World Bank GNI per capita PPP data into long format
    gni_year_cols = [c for c in df_wb_gni_raw.columns if str(c).isdigit()]
    df_gni = df_wb_gni_raw[['Country Code'] + gni_year_cols].copy()
    df_gni = df_gni.melt(
        id_vars=['Country Code'],
        value_vars=gni_year_cols,
        var_name='year',
        value_name='gni_pc_ppp'
    )
    df_gni['year'] = pd.to_numeric(df_gni['year'], errors='coerce').astype('Int64')
    df_gni['gni_pc_ppp'] = pd.to_numeric(df_gni['gni_pc_ppp'], errors='coerce')
    df_gni['geo_code'] = df_gni['Country Code'].map(STUDY_ISO3_TO_ISO2)
    df_gni = df_gni[df_gni['geo_code'].notnull()].copy()
    df_gni = df_gni[df_gni['geo_code'].isin(study_countries)].copy()
    df_gni = df_gni.dropna(subset=['year', 'gni_pc_ppp']).copy()
    df_gni['year'] = df_gni['year'].astype(int)
    df_gni = df_gni[df_gni['gni_pc_ppp'] > 100].copy()  # ln(GNI)-ln(100) must be > 0

    # Merge EWBI-HDI with GNI and compute HDI without GNI
    df_ewbi_hdi_gni = df_ewbi_hdi.merge(
        df_gni[['geo_code', 'year', 'gni_pc_ppp']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_ewbi_hdi_gni['hdi_no_gni'] = (
        df_ewbi_hdi_gni['hdi']
        / (np.log(df_ewbi_hdi_gni['gni_pc_ppp']) - np.log(100)) ** (1.0 / 3.0)
    )
    df_ewbi_hdi_gni = df_ewbi_hdi_gni[df_ewbi_hdi_gni['hdi_no_gni'].notnull()].copy()

    print(f"\nEWBI + HDI + GNI merged rows: {len(df_ewbi_hdi_gni)}")
    print(f"Countries: {df_ewbi_hdi_gni['geo_code'].nunique()}")
    print(f"Years available: {sorted(df_ewbi_hdi_gni['year'].unique())}")

    # Side-by-side plot: HDI (left) vs HDI without GNI (right)
    create_ewbi_hdi_side_by_side(
        df_ewbi_hdi_gni,
        output_filename='ewbi_vs_hdi_with_and_without_gni.png'
    )

    export_ewbi_hdi_path = os.path.join(OUTPUT_DIR, 'ewbi_hdi_data.xlsx')
    df_export_hdi = df_ewbi_hdi_gni[['geo_code', 'year', 'ewbi', 'hdi', 'gni_pc_ppp', 'hdi_no_gni']].copy()
    df_export_hdi.columns = ['Country', 'Year', 'EWBI', 'HDI', 'GNI per capita PPP (current int$)', 'HDI without GNI']
    df_export_hdi.to_excel(export_ewbi_hdi_path, index=False)
    print(f"  ✓ EWBI-HDI data exported to: {export_ewbi_hdi_path}")

    # Rank comparison slope chart: HDI rank vs EWBI rank
    df_rank = create_ewbi_hdi_rank_slope_chart(
        df_ewbi_hdi,
        output_filename='ewbi_vs_hdi_rank_slope_chart.png'
    )
    if df_rank is not None:
        export_rank_path = os.path.join(OUTPUT_DIR, 'ewbi_hdi_rank_comparison.xlsx')
        df_rank_export = df_rank.copy()
        df_rank_export.columns = ['Country', 'Year', 'HDI', 'EWBI', 'HDI Rank', 'EWBI Rank', 'Rank Change (HDI-EWBI)']
        df_rank_export.to_excel(export_rank_path, index=False)
        print(f"  ✓ Rank comparison exported to: {export_rank_path}")

    # Rank heatmap: EWBI & HDI ranks for first and last year
    create_ewbi_hdi_rank_heatmap(
        df_ewbi_hdi,
        output_filename='ewbi_hdi_rank_heatmap.png'
    )

    # === PART 1G: EWBI VS EUROSTAT CAPITAL STOCKS PER CAPITA (N1N + N2N, S1) ===
    print("\n" + "=" * 80)
    print("PART 1G: EWBI VS EUROSTAT CAPITAL STOCKS PER CAPITA")
    print("=" * 80)

    # Filter for sector = Total economy (S1) and asset10 = N1N + N2N
    df_nfa = df_eurostat_nfa_raw.copy()
    df_nfa = df_nfa[df_nfa['sector'] == 'Total economy'].copy()
    # Sum produced (N1N) + non-produced (N2N) non-financial assets
    df_nfa['year'] = pd.to_numeric(df_nfa['TIME_PERIOD'], errors='coerce')
    df_nfa['obs_value'] = pd.to_numeric(df_nfa['OBS_VALUE'], errors='coerce')
    df_nfa['geo_code'] = df_nfa['geo'].map(EUROSTAT_TO_CODE)
    df_nfa = df_nfa[df_nfa['geo_code'].notnull()].copy()
    df_nfa = df_nfa[df_nfa['geo_code'].isin(study_countries)].copy()
    df_nfa = df_nfa.dropna(subset=['year', 'obs_value']).copy()
    df_nfa['year'] = df_nfa['year'].astype(int)

    # Sum both asset types per country-year
    df_nfa_sum = df_nfa.groupby(['geo_code', 'year'], as_index=False)['obs_value'].sum()
    df_nfa_sum = df_nfa_sum[df_nfa_sum['obs_value'] > 0].copy()

    # Capital stocks per capita (million EUR / population -> EUR/person: multiply by 1e6)
    df_nfa_pc = df_nfa_sum.merge(
        df_population[['geo_code', 'year', 'population']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_nfa_pc['capital_stocks_per_capita'] = (df_nfa_pc['obs_value'] * 1e6) / df_nfa_pc['population']
    df_nfa_pc = df_nfa_pc[df_nfa_pc['capital_stocks_per_capita'] > 0].copy()

    # Merge EWBI with Eurostat capital stocks per capita
    df_ewbi_nfa = df_ewbi[['geo_code', 'year', 'ewbi']].merge(
        df_nfa_pc[['geo_code', 'year', 'capital_stocks_per_capita']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_ewbi_nfa = df_ewbi_nfa[(df_ewbi_nfa['ewbi'] > 0) & (df_ewbi_nfa['capital_stocks_per_capita'] > 0)].copy()

    print(f"\nEWBI + Eurostat capital stocks per capita merged rows: {len(df_ewbi_nfa)}")
    print(f"Countries: {df_ewbi_nfa['geo_code'].nunique()}")
    print(f"Years available: {sorted(df_ewbi_nfa['year'].unique())}")

    create_ewbi_trajectory_plot(
        df_ewbi_nfa,
        x_col='capital_stocks_per_capita',
        y_col='ewbi',
        x_label='Capital Stocks per Capita (EUR, N1N+N2N, Total Economy)',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs Eurostat Capital Stocks per Capita Over Time',
        output_filename='ewbi_vs_eurostat_capital_stocks_per_capita_trajectory.png'
    )
    create_ewbi_last_year_plot(
        df_ewbi_nfa,
        x_col='capital_stocks_per_capita',
        y_col='ewbi',
        x_label='Capital Stocks per Capita (EUR, N1N+N2N, Total Economy)',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs Eurostat Capital Stocks per Capita',
        output_filename='ewbi_vs_eurostat_capital_stocks_per_capita_last_year.png'
    )

    export_ewbi_nfa_path = os.path.join(OUTPUT_DIR, 'ewbi_eurostat_capital_stocks_per_capita_data.xlsx')
    df_export_nfa = df_ewbi_nfa[['geo_code', 'year', 'ewbi', 'capital_stocks_per_capita']].copy()
    df_export_nfa.columns = ['Country', 'Year', 'EWBI', 'Capital Stocks per Capita (EUR)']
    df_export_nfa.to_excel(export_ewbi_nfa_path, index=False)
    print(f"  ✓ EWBI-Eurostat capital stocks data exported to: {export_ewbi_nfa_path}")

    # === PART 1H: EWBI VS WORLD BANK GROSS CAPITAL FORMATION PER CAPITA ===
    print("\n" + "=" * 80)
    print("PART 1H: EWBI VS WORLD BANK GROSS CAPITAL FORMATION PER CAPITA")
    print("=" * 80)

    # Process World Bank GCF data (same format as worldbank_gdp_ppp)
    year_columns_gcf = [col for col in df_wb_gcf_raw.columns if str(col).replace('"', '').strip().isdigit()]
    df_wb_gcf = df_wb_gcf_raw[['Country Code'] + year_columns_gcf].copy()
    df_wb_gcf = df_wb_gcf.melt(
        id_vars=['Country Code'],
        value_vars=year_columns_gcf,
        var_name='year',
        value_name='gcf_2015usd'
    )
    df_wb_gcf['year'] = pd.to_numeric(df_wb_gcf['year'], errors='coerce').astype('Int64')
    df_wb_gcf['gcf_2015usd'] = pd.to_numeric(df_wb_gcf['gcf_2015usd'], errors='coerce')
    df_wb_gcf['geo_code'] = df_wb_gcf['Country Code'].map(STUDY_ISO3_TO_ISO2)
    df_wb_gcf = df_wb_gcf[df_wb_gcf['geo_code'].notnull()].copy()
    df_wb_gcf = df_wb_gcf[df_wb_gcf['geo_code'].isin(study_countries)].copy()
    df_wb_gcf = df_wb_gcf.dropna(subset=['year', 'gcf_2015usd']).copy()
    df_wb_gcf['year'] = df_wb_gcf['year'].astype(int)
    df_wb_gcf = df_wb_gcf[df_wb_gcf['gcf_2015usd'] > 0].copy()

    # GCF per capita (constant 2015 € / population)
    df_gcf_pc = df_wb_gcf[['geo_code', 'year', 'gcf_2015usd']].merge(
        df_population[['geo_code', 'year', 'population']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_gcf_pc['gcf_per_capita'] = (df_gcf_pc['gcf_2015usd'] * USD_TO_EUR_2021) / df_gcf_pc['population']
    df_gcf_pc = df_gcf_pc[df_gcf_pc['gcf_per_capita'] > 0].copy()

    # Merge EWBI with GCF per capita
    df_ewbi_gcf = df_ewbi[['geo_code', 'year', 'ewbi']].merge(
        df_gcf_pc[['geo_code', 'year', 'gcf_per_capita']],
        on=['geo_code', 'year'],
        how='inner'
    )
    df_ewbi_gcf = df_ewbi_gcf[(df_ewbi_gcf['ewbi'] > 0) & (df_ewbi_gcf['gcf_per_capita'] > 0)].copy()

    print(f"\nEWBI + WB Gross capital formation per capita merged rows: {len(df_ewbi_gcf)}")
    print(f"Countries: {df_ewbi_gcf['geo_code'].nunique()}")
    print(f"Years available: {sorted(df_ewbi_gcf['year'].unique())}")

    create_ewbi_trajectory_plot(
        df_ewbi_gcf,
        x_col='gcf_per_capita',
        y_col='ewbi',
        x_label='Gross Capital Formation per Capita (Constant 2015 €)',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs Gross Capital Formation per Capita Over Time',
        output_filename='ewbi_vs_wb_gcf_per_capita_trajectory.png'
    )
    create_ewbi_last_year_plot(
        df_ewbi_gcf,
        x_col='gcf_per_capita',
        y_col='ewbi',
        x_label='Gross Capital Formation per Capita (Constant 2015 €)',
        y_label='EWBI (Country level, All Deciles)',
        title='EWBI vs Gross Capital Formation per Capita',
        output_filename='ewbi_vs_wb_gcf_per_capita_last_year.png'
    )

    export_ewbi_gcf_path = os.path.join(OUTPUT_DIR, 'ewbi_wb_gcf_per_capita_data.xlsx')
    df_export_gcf = df_ewbi_gcf[['geo_code', 'year', 'ewbi', 'gcf_per_capita']].copy()
    df_export_gcf.columns = ['Country', 'Year', 'EWBI', 'Gross Capital Formation per Capita (Constant 2015 €)']
    df_export_gcf.to_excel(export_ewbi_gcf_path, index=False)
    print(f"  ✓ EWBI-WB GCF per capita data exported to: {export_ewbi_gcf_path}")

    # === PART 2: VARIATION ANALYSIS ===
    print("\n" + "=" * 80)
    print("PART 2: VARIATION ANALYSIS (PERCENTAGE CHANGE)")
    print("=" * 80)
    
    # Process GDP EUR 2005 data (Chain linked volumes 2005, million euro)
    df_gdp_eur_proc = df_gdp_eur[['geo', 'TIME_PERIOD', 'OBS_VALUE']].copy()
    df_gdp_eur_proc.columns = ['geo', 'year', 'gdp_eur']
    df_gdp_eur_proc['geo_code'] = df_gdp_eur_proc['geo'].map(EUROSTAT_TO_CODE)
    df_gdp_eur_proc = df_gdp_eur_proc[df_gdp_eur_proc['geo_code'].notnull()].copy()
    df_gdp_eur_proc = df_gdp_eur_proc[df_gdp_eur_proc['geo_code'].isin(study_countries)].copy()
    
    # Process energy data for variation analysis
    df_energy_proc = df_energy_raw[['geo', 'TIME_PERIOD', 'OBS_VALUE']].copy()
    df_energy_proc.columns = ['geo', 'year', 'energy']
    df_energy_proc['geo_code'] = df_energy_proc['geo'].map(EUROSTAT_TO_CODE)
    df_energy_proc = df_energy_proc[df_energy_proc['geo_code'].notnull()].copy()
    df_energy_proc = df_energy_proc[df_energy_proc['geo_code'].isin(study_countries)].copy()
    
    # Create variation analysis
    create_variation_plot(df_gdp_eur_proc, df_energy_proc)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

def create_energy_gdp_plot(df):
    """Create log-log scatter plot of GDP PPS vs Final Energy Consumption"""
    print("\nCreating energy vs GDP plot...")
    
    # Define colors for each year
    year_colors = {
        1996: '#95b0e8',
        2004: '#ffd558',
        2014: '#fb8072',
        2024: '#b3de69'
    }
    
    plt.figure(figsize=(10, 10))
    
    # Store regression data for legend
    regression_data = []
    
    # Plot each year
    for year in years_to_plot:
        df_year = df[df['year'] == year].copy()
        
        if len(df_year) == 0:
            print(f"  Warning: No data for year {year}")
            continue
        
        # Log-transform for regression
        x_log = np.log(df_year['energy'])
        y_log = np.log(df_year['gdp_pps'])
        
        # Perform regression
        if len(df_year) >= 2:
            slope, intercept, r_value, _, _ = linregress(x_log, y_log)
            
            # Generate regression line
            x_vals = np.logspace(
                np.log10(df_year['energy'].min() * 0.9),
                np.log10(df_year['energy'].max() * 1.1),
                100
            )
            y_fit = np.exp(intercept) * x_vals ** slope
            
            # Plot regression line
            label = f'{year}: y = {np.exp(intercept):.2e} × x^{slope:.2f}  (R² = {r_value**2:.2f})'
            plt.plot(x_vals, y_fit, '--', color=year_colors.get(year, '#cccccc'), linewidth=1.5, 
                    label=label, zorder=1, alpha=0.7)
            
            regression_data.append({
                'year': year,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2
            })
        
        # Plot data points
        plt.scatter(df_year['energy'], df_year['gdp_pps'], 
                   c=year_colors.get(year, '#cccccc'), 
                   marker='o',
                   s=100, label=f'{year} values', zorder=2, alpha=0.7)
    
    # Draw connection lines between consecutive years for each country
    all_countries = df['geo_code'].unique()
    for country in all_countries:
        df_country = df[df['geo_code'] == country].sort_values('year')
        if len(df_country) >= 2:
            for i in range(len(df_country) - 1):
                row_start = df_country.iloc[i]
                row_end = df_country.iloc[i + 1]
                plt.plot(
                    [row_start['energy'], row_end['energy']],
                    [row_start['gdp_pps'], row_end['gdp_pps']],
                    color='grey', alpha=0.3, linewidth=0.8, zorder=0
                )
    
    # Annotate countries (use most recent year position for each country)
    for country in all_countries:
        df_country = df[df['geo_code'] == country].sort_values('year')
        last_row = df_country.iloc[-1]
        plt.text(last_row['energy'], last_row['gdp_pps'], country, 
                fontsize=12, ha='right', va='bottom', zorder=3, alpha=0.8)
    
    # Axis scales and labels
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Final Energy Consumption (Gigawatt-hour, log scale)', fontsize=16)
    plt.ylabel('GDP PPS (Million PPS, EU27 from 2020, log scale)', fontsize=16)
    plt.title('GDP PPS vs Final Energy Consumption (1996-2024)', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, which='both', ls='--', linewidth=0.4, alpha=0.5)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'energy_gdp_loglog.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved to: {output_path}")
    plt.close()
    
    # Print regression statistics
    print("\n  Regression statistics:")
    for data in regression_data:
        print(f"    {data['year']}: slope = {data['slope']:.3f}, R² = {data['r_squared']:.3f}")

def create_energy_gdp_plot_simple(df):
    """Create simplified log-log scatter plot without country labels"""
    print("\nCreating simplified energy vs GDP plot...")
    
    # Define colors for each year
    year_colors = {
        1996: '#95b0e8',
        2004: '#ffd558',
        2014: '#fb8072',
        2024: '#b3de69'
    }
    
    plt.figure(figsize=(10, 10))
    
    # Store regression data for legend
    regression_data = []
    
    # Plot each year
    for year in years_to_plot:
        df_year = df[df['year'] == year].copy()
        
        if len(df_year) == 0:
            print(f"  Warning: No data for year {year}")
            continue
        
        # Log-transform for regression
        x_log = np.log(df_year['energy'])
        y_log = np.log(df_year['gdp_pps'])
        
        # Perform regression
        if len(df_year) >= 2:
            slope, intercept, r_value, _, _ = linregress(x_log, y_log)
            
            # Generate regression line
            x_vals = np.logspace(
                np.log10(df_year['energy'].min() * 0.9),
                np.log10(df_year['energy'].max() * 1.1),
                100
            )
            y_fit = np.exp(intercept) * x_vals ** slope
            
            # Plot regression line in same color as points
            label = f'{year}: y = {np.exp(intercept):.2e} × x^{slope:.2f}  (R² = {r_value**2:.2f})'
            plt.plot(x_vals, y_fit, '--', color=year_colors.get(year, '#cccccc'), 
                    linewidth=1.5, label=label, zorder=1, alpha=0.7)
            
            regression_data.append({
                'year': year,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2
            })
        
        # Plot data points - circles only, no edge
        plt.scatter(df_year['energy'], df_year['gdp_pps'], 
                   c=year_colors.get(year, '#cccccc'), 
                   marker='o',
                   s=100, label=f'{year} values', zorder=2, alpha=0.7)
    
    # Draw connection lines between consecutive years for each country
    all_countries = df['geo_code'].unique()
    for country in all_countries:
        df_country = df[df['geo_code'] == country].sort_values('year')
        if len(df_country) >= 2:
            for i in range(len(df_country) - 1):
                row_start = df_country.iloc[i]
                row_end = df_country.iloc[i + 1]
                plt.plot(
                    [row_start['energy'], row_end['energy']],
                    [row_start['gdp_pps'], row_end['gdp_pps']],
                    color='grey', alpha=0.3, linewidth=0.8, zorder=0
                )
    
    # Axis scales and labels
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Final Energy Consumption (Gigawatt-hour, log scale)', fontsize=16)
    plt.ylabel('GDP PPS (Million PPS, EU27 from 2020, log scale)', fontsize=16)
    plt.title('GDP PPS vs Final Energy Consumption (1996-2024)', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, which='both', ls='--', linewidth=0.4, alpha=0.5)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'energy_gdp_loglog_simple.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Simplified plot saved to: {output_path}")
    plt.close()
    
    # Print regression statistics
    print("\n  Regression statistics (simplified):")
    for data in regression_data:
        print(f"    {data['year']}: slope = {data['slope']:.3f}, R² = {data['r_squared']:.3f}")

def create_energy_gdp_plot_real(df):
    """Create log-log scatter plot of GDP PPS (inflation-corrected) vs Final Energy Consumption"""
    print("\nCreating energy vs GDP plot (inflation-corrected)...")
    
    # Define colors for each year
    year_colors = {
        1996: '#95b0e8',
        2004: '#ffd558',
        2014: '#fb8072',
        2024: '#b3de69'
    }
    
    plt.figure(figsize=(10, 10))
    
    # Store regression data for legend
    regression_data = []
    
    # Plot each year
    for year in years_to_plot:
        df_year = df[df['year'] == year].copy()
        
        if len(df_year) == 0:
            print(f"  Warning: No data for year {year}")
            continue
        
        # Log-transform for regression
        x_log = np.log(df_year['energy'])
        y_log = np.log(df_year['gdp_pps_real'])
        
        # Perform regression
        if len(df_year) >= 2:
            slope, intercept, r_value, _, _ = linregress(x_log, y_log)
            
            # Generate regression line
            x_vals = np.logspace(
                np.log10(df_year['energy'].min() * 0.9),
                np.log10(df_year['energy'].max() * 1.1),
                100
            )
            y_fit = np.exp(intercept) * x_vals ** slope
            
            # Plot regression line
            label = f'{year}: y = {np.exp(intercept):.2e} × x^{slope:.2f}  (R² = {r_value**2:.2f})'
            plt.plot(x_vals, y_fit, '--', color=year_colors.get(year, '#cccccc'), linewidth=1.5, 
                    label=label, zorder=1, alpha=0.7)
            
            regression_data.append({
                'year': year,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2
            })
        
        # Plot data points
        plt.scatter(df_year['energy'], df_year['gdp_pps_real'], 
                   c=year_colors.get(year, '#cccccc'), 
                   marker='o',
                   s=100, label=f'{year} values', zorder=2, alpha=0.7)
    
    # Draw connection lines between consecutive years for each country
    all_countries = df['geo_code'].unique()
    for country in all_countries:
        df_country = df[df['geo_code'] == country].sort_values('year')
        if len(df_country) >= 2:
            for i in range(len(df_country) - 1):
                row_start = df_country.iloc[i]
                row_end = df_country.iloc[i + 1]
                plt.plot(
                    [row_start['energy'], row_end['energy']],
                    [row_start['gdp_pps_real'], row_end['gdp_pps_real']],
                    color='grey', alpha=0.3, linewidth=0.8, zorder=0
                )
    
    # Annotate countries (use most recent year position for each country)
    for country in all_countries:
        df_country = df[df['geo_code'] == country].sort_values('year')
        last_row = df_country.iloc[-1]
        plt.text(last_row['energy'], last_row['gdp_pps_real'], country, 
                fontsize=12, ha='right', va='bottom', zorder=3, alpha=0.8)
    
    # Axis scales and labels
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Final Energy Consumption (Gigawatt-hour, log scale)', fontsize=16)
    plt.ylabel('GDP PPS (Real, HICP-Corrected, log scale)', fontsize=16)
    plt.title('GDP PPS (Inflation-Corrected) vs Final Energy Consumption (1996-2024)', 
             fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, which='both', ls='--', linewidth=0.4, alpha=0.5)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'energy_gdp_loglog_inflation_corrected.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved to: {output_path}")
    plt.close()
    
    # Print regression statistics
    print("\n  Regression statistics (inflation-corrected):")
    for data in regression_data:
        print(f"    {data['year']}: slope = {data['slope']:.3f}, R² = {data['r_squared']:.3f}")
def create_energy_gdp_plot_real_simple(df):
    """Create simplified log-log scatter plot of inflation-corrected GDP without country labels"""
    print("\nCreating simplified energy vs GDP plot (inflation-corrected)...")
    
    # Define colors for each year
    year_colors = {
        1996: '#95b0e8',
        2004: '#ffd558',
        2014: '#fb8072',
        2024: '#b3de69'
    }
    
    plt.figure(figsize=(10, 10))
    
    # Store regression data for legend
    regression_data = []
    
    # Plot each year
    for year in years_to_plot:
        df_year = df[df['year'] == year].copy()
        
        if len(df_year) == 0:
            print(f"  Warning: No data for year {year}")
            continue
        
        # Log-transform for regression
        x_log = np.log(df_year['energy'])
        y_log = np.log(df_year['gdp_pps_real'])
        
        # Perform regression
        if len(df_year) >= 2:
            slope, intercept, r_value, _, _ = linregress(x_log, y_log)
            
            # Generate regression line
            x_vals = np.logspace(
                np.log10(df_year['energy'].min() * 0.9),
                np.log10(df_year['energy'].max() * 1.1),
                100
            )
            y_fit = np.exp(intercept) * x_vals ** slope
            
            # Plot regression line in same color as points
            label = f'{year}: y = {np.exp(intercept):.2e} × x^{slope:.2f}  (R² = {r_value**2:.2f})'
            plt.plot(x_vals, y_fit, '--', color=year_colors.get(year, '#cccccc'), 
                    linewidth=1.5, label=label, zorder=1, alpha=0.7)
            
            regression_data.append({
                'year': year,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2
            })
        
        # Plot data points - circles only, no edge
        plt.scatter(df_year['energy'], df_year['gdp_pps_real'], 
                   c=year_colors.get(year, '#cccccc'), 
                   marker='o',
                   s=100, label=f'{year} values', zorder=2, alpha=0.7)
    
    # Draw connection lines between consecutive years for each country
    all_countries = df['geo_code'].unique()
    for country in all_countries:
        df_country = df[df['geo_code'] == country].sort_values('year')
        if len(df_country) >= 2:
            for i in range(len(df_country) - 1):
                row_start = df_country.iloc[i]
                row_end = df_country.iloc[i + 1]
                plt.plot(
                    [row_start['energy'], row_end['energy']],
                    [row_start['gdp_pps_real'], row_end['gdp_pps_real']],
                    color='grey', alpha=0.3, linewidth=0.8, zorder=0
                )
    
    # Axis scales and labels
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Final Energy Consumption (Gigawatt-hour, log scale)', fontsize=16)
    plt.ylabel('GDP PPS (Real, HICP-Corrected, log scale)', fontsize=16)
    plt.title('GDP PPS (Inflation-Corrected) vs Final Energy Consumption (1996-2024)', 
             fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, which='both', ls='--', linewidth=0.4, alpha=0.5)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'energy_gdp_loglog_inflation_corrected_simple.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Simplified plot saved to: {output_path}")
    plt.close()
    
    # Print regression statistics
    print("\n  Regression statistics (inflation-corrected, simplified):")
    for data in regression_data:
        print(f"    {data['year']}: slope = {data['slope']:.3f}, R² = {data['r_squared']:.3f}")

def create_energy_gdp_plot_worldbank_simple(df):
    """Create simplified log-log scatter plot using World Bank GDP PPP (constant 2021 €)."""
    print("\nCreating simplified energy vs GDP plot (World Bank PPP 2021)...")

    # Define colors for each year
    year_colors = {
        1996: '#95b0e8',
        2004: '#ffd558',
        2014: '#fb8072',
        2024: '#b3de69'
    }

    plt.figure(figsize=(10, 10))

    # Store regression data for legend
    regression_data = []

    # Plot each year
    for year in years_to_plot:
        df_year = df[df['year'] == year].copy()

        if len(df_year) == 0:
            print(f"  Warning: No data for year {year}")
            continue

        # Log-transform for regression
        x_log = np.log(df_year['energy'])
        y_log = np.log(df_year['gdp_ppp_2021'])

        # Perform regression
        if len(df_year) >= 2:
            slope, intercept, r_value, _, _ = linregress(x_log, y_log)

            # Generate regression line
            x_vals = np.logspace(
                np.log10(df_year['energy'].min() * 0.9),
                np.log10(df_year['energy'].max() * 1.1),
                100
            )
            y_fit = np.exp(intercept) * x_vals ** slope

            # Plot regression line in same color as points
            label = f'{year}: y = {np.exp(intercept):.2e} × x^{slope:.2f}  (R² = {r_value**2:.2f})'
            plt.plot(x_vals, y_fit, '--', color=year_colors.get(year, '#cccccc'),
                    linewidth=1.5, label=label, zorder=1, alpha=0.7)

            regression_data.append({
                'year': year,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2
            })

        # Plot data points - circles only, no edge
        plt.scatter(df_year['energy'], df_year['gdp_ppp_2021'],
                   c=year_colors.get(year, '#cccccc'),
                   marker='o',
                   s=100, label=f'{year} values', zorder=2, alpha=0.7)

    # Draw connection lines between consecutive years for each country
    all_countries = df['geo_code'].unique()
    for country in all_countries:
        df_country = df[df['geo_code'] == country].sort_values('year')
        if len(df_country) >= 2:
            for i in range(len(df_country) - 1):
                row_start = df_country.iloc[i]
                row_end = df_country.iloc[i + 1]
                plt.plot(
                    [row_start['energy'], row_end['energy']],
                    [row_start['gdp_ppp_2021'], row_end['gdp_ppp_2021']],
                    color='grey', alpha=0.3, linewidth=0.8, zorder=0
                )

    # Axis scales and labels
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Final Energy Consumption (Gigawatt-hour, log scale)', fontsize=16)
    plt.ylabel('GDP PPP (Constant 2021 €, log scale)', fontsize=16)
    plt.title('GDP PPP (Constant 2021 €) vs Final Energy Consumption (1996-2024)',
             fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, which='both', ls='--', linewidth=0.4, alpha=0.5)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'energy_gdp_loglog_worldbank_ppp2021_simple.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Simplified World Bank PPP plot saved to: {output_path}")
    plt.close()

    # Print regression statistics
    print("\n  Regression statistics (World Bank PPP 2021, simplified):")
    for data in regression_data:
        print(f"    {data['year']}: slope = {data['slope']:.3f}, R² = {data['r_squared']:.3f}")

def create_ewbi_trajectory_plot(df, x_col, y_col, x_label, y_label, title, output_filename,
                                regression_by_year=False, save_svg=False):
    """Create EWBI trajectory scatter with year colors and country paths over time."""
    print(f"\nCreating EWBI trajectory plot: {output_filename}...")

    if df.empty:
        print("  Warning: No data available for EWBI trajectory plot")
        return

    plt.figure(figsize=(10, 10))

    # Draw country trajectories
    all_countries = sorted(df['geo_code'].unique())
    for country in all_countries:
        df_country = df[df['geo_code'] == country].sort_values('year')
        if len(df_country) >= 2:
            plt.plot(
                df_country[x_col],
                df_country[y_col],
                color='grey',
                alpha=0.35,
                linewidth=0.9,
                zorder=1
            )

        # Label at most recent point
        last_row = df_country.iloc[-1]
        plt.text(
            last_row[x_col],
            last_row[y_col],
            country,
            fontsize=10,
            ha='right',
            va='bottom',
            alpha=0.8,
            zorder=3
        )

    # Scatter points colored by year (each year gets a distinct color shade)
    scatter = plt.scatter(
        df[x_col],
        df[y_col],
        c=df['year'],
        cmap='viridis',
        s=75,
        alpha=0.8,
        zorder=2
    )

    # Add color bar for years
    cbar = plt.colorbar(scatter)
    cbar.set_label('Year', fontsize=12)

    # Per-year regression lines
    if regression_by_year:
        year_colors_map = plt.cm.viridis
        all_years = sorted(df['year'].unique())
        y_min, y_max = df['year'].min(), df['year'].max()
        for year in all_years:
            df_yr = df[df['year'] == year].dropna(subset=[x_col, y_col])
            if len(df_yr) >= 3:
                slope, intercept, r_value, _, _ = linregress(df_yr[x_col], df_yr[y_col])
                x_range = np.linspace(df_yr[x_col].min(), df_yr[x_col].max(), 100)
                y_fit = intercept + slope * x_range
                norm_val = (year - y_min) / max(y_max - y_min, 1)
                color = year_colors_map(norm_val)
                plt.plot(x_range, y_fit, '--', color=color, linewidth=1.8, alpha=0.8, zorder=4,
                         label=f'{year}: slope={slope:.3f}, R²={r_value**2:.2f}')
        plt.legend(loc='best', fontsize=10, framealpha=0.9)

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, which='both', ls='--', linewidth=0.4, alpha=0.5)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ EWBI trajectory plot saved to: {output_path}")

    if save_svg:
        svg_path = os.path.join(OUTPUT_DIR, output_filename.replace('.png', '.svg'))
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"  ✓ EWBI trajectory plot saved to: {svg_path}")

    plt.close()

def _draw_ewbi_trajectory_panel(ax, df, x_col, x_label, title):
    """Draw one panel of the EWBI trajectory plot with per-year regression lines."""
    all_countries = sorted(df['geo_code'].unique())
    for country in all_countries:
        df_country = df[df['geo_code'] == country].sort_values('year')
        if len(df_country) >= 2:
            ax.plot(df_country[x_col], df_country['ewbi'],
                    color='grey', alpha=0.35, linewidth=0.9, zorder=1)
        last_row = df_country.iloc[-1]
        ax.text(last_row[x_col], last_row['ewbi'], country,
                fontsize=8, ha='right', va='bottom', alpha=0.8, zorder=3)

    scatter = ax.scatter(df[x_col], df['ewbi'],
                         c=df['year'], cmap='viridis', s=55, alpha=0.8, zorder=2)

    # Per-year regression lines
    year_colors_map = plt.cm.viridis
    all_years = sorted(df['year'].unique())
    y_min, y_max = df['year'].min(), df['year'].max()
    for year in all_years:
        df_yr = df[df['year'] == year].dropna(subset=[x_col, 'ewbi'])
        if len(df_yr) >= 3:
            slope, intercept, r_value, _, _ = linregress(df_yr[x_col], df_yr['ewbi'])
            x_range = np.linspace(df_yr[x_col].min(), df_yr[x_col].max(), 100)
            y_fit = intercept + slope * x_range
            norm_val = (year - y_min) / max(y_max - y_min, 1)
            color = year_colors_map(norm_val)
            ax.plot(x_range, y_fit, '--', color=color, linewidth=1.8, alpha=0.8, zorder=4,
                    label=f'{year}: slope={slope:.3f}, R²={r_value**2:.2f}')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)

    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel('EWBI (Country level, All Deciles)', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(True, which='both', ls='--', linewidth=0.4, alpha=0.5)
    return scatter

def create_ewbi_hdi_side_by_side(df, output_filename):
    """Create side-by-side: EWBI vs HDI (left) and EWBI vs HDI without GNI (right)."""
    print(f"\nCreating side-by-side HDI plot: {output_filename}...")

    if df.empty:
        print("  Warning: No data available for HDI side-by-side plot")
        return

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 10))

    # Left panel: EWBI vs HDI
    scatter = _draw_ewbi_trajectory_panel(
        ax_left, df, x_col='hdi',
        x_label='Human Development Index',
        title='EWBI vs HDI'
    )

    # Right panel: EWBI vs HDI without GNI
    _draw_ewbi_trajectory_panel(
        ax_right, df, x_col='hdi_no_gni',
        x_label='HDI / (ln(GNI) − ln(100))^{1/3}',
        title='EWBI vs HDI without GNI'
    )

    # Shared colorbar
    cbar = fig.colorbar(scatter, ax=[ax_left, ax_right], shrink=0.6, pad=0.02)
    cbar.set_label('Year', fontsize=12)

    fig.suptitle('EWBI vs Human Development Index — with and without GNI component',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Side-by-side HDI plot saved to: {output_path}")

    svg_path = os.path.join(OUTPUT_DIR, output_filename.replace('.png', '.svg'))
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"  ✓ Side-by-side HDI plot saved to: {svg_path}")

    plt.close(fig)

def create_ewbi_hdi_rank_slope_chart(df, output_filename):
    """Slope chart comparing HDI rank (left) vs EWBI rank (right) for the last available year."""
    print(f"\nCreating HDI vs EWBI rank slope chart: {output_filename}...")

    if df.empty:
        print("  Warning: No data available for rank slope chart")
        return

    # Use last available year per country
    idx = df.groupby('geo_code')['year'].idxmax()
    df_last = df.loc[idx].copy()
    last_year = int(df_last['year'].max())

    # Compute ranks (1 = best = highest value) — use method='min' for ties
    df_last['hdi_rank'] = df_last['hdi'].rank(ascending=False, method='min').astype(int)
    df_last['ewbi_rank'] = df_last['ewbi'].rank(ascending=False, method='min').astype(int)
    df_last['rank_change'] = df_last['hdi_rank'] - df_last['ewbi_rank']  # positive = climbed

    n = len(df_last)
    fig, ax = plt.subplots(figsize=(8, max(10, n * 0.45)))

    x_left, x_right = 0, 1

    # Offset overlapping labels: group countries sharing the same rank position
    def _jitter_positions(ranks, offset_step=0.35):
        """Return y-positions with vertical offsets when ranks collide."""
        from collections import Counter
        rank_counts = Counter(ranks.values())
        rank_cursors = {r: 0 for r in rank_counts}
        positions = {}
        for idx_r, (country, rank) in enumerate(ranks.items()):
            count = rank_counts[rank]
            if count == 1:
                positions[country] = rank
            else:
                cursor = rank_cursors[rank]
                positions[country] = rank + (cursor - (count - 1) / 2.0) * offset_step
                rank_cursors[rank] = cursor + 1
        return positions

    hdi_ranks = {row['geo_code']: row['hdi_rank'] for _, row in df_last.iterrows()}
    ewbi_ranks = {row['geo_code']: row['ewbi_rank'] for _, row in df_last.iterrows()}
    hdi_pos = _jitter_positions(hdi_ranks)
    ewbi_pos = _jitter_positions(ewbi_ranks)

    for _, row in df_last.iterrows():
        cc = row['geo_code']
        hdi_y = hdi_pos[cc]
        ewbi_y = ewbi_pos[cc]
        change = row['rank_change']

        if change > 0:
            color = '#4daf4a'    # green: country climbs (better EWBI rank)
        elif change < 0:
            color = '#e41a1c'    # red: country drops
        else:
            color = '#999999'    # grey: same rank

        linewidth = 1.2 + min(abs(change) * 0.15, 1.5)
        ax.plot([x_left, x_right], [hdi_y, ewbi_y],
                color=color, linewidth=linewidth, alpha=0.7, zorder=1)

        # Country label on the left (HDI side)
        ax.text(x_left - 0.03, hdi_y, f"{cc}  ({int(row['hdi_rank'])})",
                fontsize=9, ha='right', va='center', fontweight='bold')
        # Country label on the right (EWBI side)
        ax.text(x_right + 0.03, ewbi_y, f"({int(row['ewbi_rank'])})  {cc}",
                fontsize=9, ha='left', va='center', fontweight='bold')

    # Dots at each position
    ax.scatter([x_left] * n, [hdi_pos[c] for c in df_last['geo_code']],
               c='#377eb8', s=60, zorder=2)
    ax.scatter([x_right] * n, [ewbi_pos[c] for c in df_last['geo_code']],
               c='#ff7f00', s=60, zorder=2)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(n + 1.5, -0.5)  # rank 1 on top
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(['HDI Rank', 'EWBI Rank'], fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.set_title(f'Country Ranking: HDI vs EWBI ({last_year})',
                 fontsize=16, fontweight='bold', pad=15)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#4daf4a', lw=2, label='Better EWBI rank than HDI'),
        Line2D([0], [0], color='#e41a1c', lw=2, label='Worse EWBI rank than HDI'),
        Line2D([0], [0], color='#999999', lw=2, label='Same rank'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=10,
             framealpha=0.9, ncol=3, bbox_to_anchor=(0.5, -0.04))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='y', ls='--', linewidth=0.3, alpha=0.4)

    fig.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  \u2713 Rank slope chart saved to: {output_path}")

    svg_path = os.path.join(OUTPUT_DIR, output_filename.replace('.png', '.svg'))
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"  \u2713 Rank slope chart saved to: {svg_path}")

    plt.close(fig)

    return df_last[['geo_code', 'year', 'hdi', 'ewbi', 'hdi_rank', 'ewbi_rank', 'rank_change']]

def create_ewbi_hdi_rank_heatmap(df, output_filename):
    """Heatmap of EWBI and HDI ranks for first and last year, sorted by EWBI rank (last year)."""
    print(f"\nCreating HDI/EWBI rank heatmap: {output_filename}...")

    if df.empty:
        print("  Warning: No data for rank heatmap")
        return

    years = sorted(df['year'].unique())
    last_year = years[-1]

    # Find first year where all last-year countries are present
    last_year_countries = set(df[df['year'] == last_year]['geo_code'].unique())
    first_year = None
    for yr in years:
        yr_countries = set(df[df['year'] == yr]['geo_code'].unique())
        if last_year_countries.issubset(yr_countries):
            first_year = yr
            break
    if first_year is None:
        # Fallback: use the earliest year with the most countries
        first_year = years[0]
        print(f"  Warning: no single early year has all {len(last_year_countries)} countries; using {first_year}")

    rows = []
    for yr in [first_year, last_year]:
        df_yr = df[df['year'] == yr].copy()
        if df_yr.empty:
            continue
        df_yr['hdi_rank'] = df_yr['hdi'].rank(ascending=False, method='min').astype(int)
        df_yr['ewbi_rank'] = df_yr['ewbi'].rank(ascending=False, method='min').astype(int)
        for _, r in df_yr.iterrows():
            rows.append({'geo_code': r['geo_code'], 'year': yr,
                         'hdi_rank': r['hdi_rank'], 'ewbi_rank': r['ewbi_rank']})

    df_ranks = pd.DataFrame(rows)
    if df_ranks.empty:
        print("  Warning: No rank data for heatmap")
        return

    # Build matrix: rows = countries sorted by EWBI rank (last year), columns = EWBI first / HDI first / EWBI last / HDI last
    df_first = df_ranks[df_ranks['year'] == first_year].set_index('geo_code')
    df_last_yr = df_ranks[df_ranks['year'] == last_year].set_index('geo_code')

    # Use all countries from the last year; first-year cells left blank when missing
    all_countries = sorted(df_last_yr.index, key=lambda c: df_last_yr.loc[c, 'ewbi_rank'])

    col_labels = [f'EWBI {first_year}', f'HDI {first_year}',
                  f'EWBI {last_year}', f'HDI {last_year}']
    matrix = np.full((len(all_countries), 4), np.nan)
    for i, cc in enumerate(all_countries):
        if cc in df_first.index:
            matrix[i, 0] = df_first.loc[cc, 'ewbi_rank']
            matrix[i, 1] = df_first.loc[cc, 'hdi_rank']
        matrix[i, 2] = df_last_yr.loc[cc, 'ewbi_rank']
        matrix[i, 3] = df_last_yr.loc[cc, 'hdi_rank']

    n_countries = len(all_countries)
    fig, ax = plt.subplots(figsize=(7, max(8, n_countries * 0.38)))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('rank_cmap', ['#1a9641', '#ffffbf', '#d7191c'],
                                              N=n_countries)

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=1, vmax=n_countries)

    ax.set_xticks(range(4))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight='bold')
    ax.set_yticks(range(n_countries))
    ax.set_yticklabels(all_countries, fontsize=10, fontweight='bold')

    # Annotate cells with rank number
    for i in range(n_countries):
        for j in range(4):
            val = int(matrix[i, j]) if not np.isnan(matrix[i, j]) else ''
            text_color = 'white' if matrix[i, j] <= n_countries * 0.25 or matrix[i, j] >= n_countries * 0.75 else 'black'
            ax.text(j, i, val, ha='center', va='center', fontsize=9, fontweight='bold',
                    color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Rank (1 = best)', fontsize=11)

    ax.set_title(f'EWBI & HDI Rankings — {first_year} vs {last_year}',
                 fontsize=14, fontweight='bold', pad=12)
    fig.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  \u2713 Rank heatmap saved to: {output_path}")

    svg_path = os.path.join(OUTPUT_DIR, output_filename.replace('.png', '.svg'))
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"  \u2713 Rank heatmap saved to: {svg_path}")

    plt.close(fig)

def create_ewbi_last_year_plot(df, x_col, y_col, x_label, y_label, title, output_filename):
    """Create EWBI scatter plot showing only the last available year per country."""
    print(f"\nCreating EWBI last-year plot: {output_filename}...")

    if df.empty:
        print("  Warning: No data available for EWBI last-year plot")
        return

    # Keep only the last available year per country
    idx = df.groupby('geo_code')['year'].idxmax()
    df_last = df.loc[idx].copy()
    last_year = int(df_last['year'].max())

    plt.figure(figsize=(10, 10))

    plt.scatter(
        df_last[x_col],
        df_last[y_col],
        c='#4c72b0',
        s=100,
        alpha=0.8,
        zorder=2
    )

    # Label each country
    for _, row in df_last.iterrows():
        plt.text(
            row[x_col],
            row[y_col],
            row['geo_code'],
            fontsize=10,
            ha='right',
            va='bottom',
            alpha=0.8,
            zorder=3
        )

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.title(f'{title} ({last_year})', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, which='both', ls='--', linewidth=0.4, alpha=0.5)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ EWBI last-year plot saved to: {output_path}")
    plt.close()

def create_variation_plot(df_gdp, df_energy):
    """Create scatter plot of % variation in GDP vs % variation in Energy"""
    print("\nCreating variation analysis plot...")
    
    # Periods to analyze
    periods = [
        {'start': 2004, 'end': 2014, 'label': '2004-2014', 'color': '#ffd558', 'marker': 'o'},
        {'start': 2014, 'end': 2024, 'label': '2014-2024', 'color': '#fb8072', 'marker': '^'}
    ]
    
    all_variations = []
    
    for period in periods:
        year_start = period['start']
        year_end = period['end']
        
        # Get data for start and end years
        df_gdp_start = df_gdp[df_gdp['year'] == year_start][['geo_code', 'gdp_eur']].copy()
        df_gdp_end = df_gdp[df_gdp['year'] == year_end][['geo_code', 'gdp_eur']].copy()
        df_energy_start = df_energy[df_energy['year'] == year_start][['geo_code', 'energy']].copy()
        df_energy_end = df_energy[df_energy['year'] == year_end][['geo_code', 'energy']].copy()
        
        # Merge start and end data
        df_period = df_gdp_start.merge(df_gdp_end, on='geo_code', suffixes=('_start', '_end'))
        df_period = df_period.merge(df_energy_start, on='geo_code')
        df_period = df_period.merge(df_energy_end, on='geo_code', suffixes=('_start', '_end'))
        
        # Calculate percentage variations
        df_period['gdp_variation'] = ((df_period['gdp_eur_end'] - df_period['gdp_eur_start']) / 
                                      df_period['gdp_eur_start']) * 100
        df_period['energy_variation'] = ((df_period['energy_end'] - df_period['energy_start']) / 
                                         df_period['energy_start']) * 100
        
        # Calculate annual percentage variations
        years_diff = year_end - year_start
        df_period['gdp_variation_annual'] = df_period['gdp_variation'] / years_diff
        df_period['energy_variation_annual'] = df_period['energy_variation'] / years_diff
        
        df_period['period'] = period['label']
        all_variations.append(df_period)
        
        print(f"  {period['label']}: {len(df_period)} countries")
    
    # Combine all periods
    df_all = pd.concat(all_variations, ignore_index=True)
    
    # Create plot
    plt.figure(figsize=(10, 10))
    
    # Plot each period
    for period in periods:
        df_period = df_all[df_all['period'] == period['label']].copy()
        
        if len(df_period) == 0:
            continue
        
        # Perform linear regression
        if len(df_period) >= 2:
            slope, intercept, r_value, _, _ = linregress(
                df_period['energy_variation_annual'], 
                df_period['gdp_variation_annual']
            )
            
            # Generate regression line
            x_min = df_period['energy_variation_annual'].min()
            x_max = df_period['energy_variation_annual'].max()
            x_vals = np.linspace(x_min - 0.5, x_max + 0.5, 100)
            y_fit = intercept + slope * x_vals
            
            # Plot regression line
            label_reg = f'{period["label"]}: y = {slope:.2f}x + {intercept:.2f}  (R² = {r_value**2:.2f})'
            plt.plot(x_vals, y_fit, '--', color='grey', linewidth=1.5, 
                    label=label_reg, zorder=1, alpha=0.7)
        
        # Plot data points
        plt.scatter(df_period['energy_variation_annual'], 
                   df_period['gdp_variation_annual'], 
                   c=period['color'], 
                   marker=period['marker'],
                   s=100, label=f'{period["label"]} values', 
                   zorder=2, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    # Draw connection lines between periods for each country
    countries_both_periods = set(all_variations[0]['geo_code']) & set(all_variations[1]['geo_code'])
    for country in countries_both_periods:
        row_first = all_variations[0][all_variations[0]['geo_code'] == country].iloc[0]
        row_second = all_variations[1][all_variations[1]['geo_code'] == country].iloc[0]
        plt.plot(
            [row_first['energy_variation_annual'], row_second['energy_variation_annual']],
            [row_first['gdp_variation_annual'], row_second['gdp_variation_annual']],
            color='grey', alpha=0.3, linewidth=0.8, zorder=0
        )
    
    # Annotate countries
    for country in df_all['geo_code'].unique():
        df_country = df_all[df_all['geo_code'] == country]
        last_row = df_country.iloc[-1]
        plt.text(last_row['energy_variation_annual'], 
                last_row['gdp_variation_annual'], 
                country, 
                fontsize=12, ha='right', va='bottom', zorder=3, alpha=0.8)
    
    # Add zero reference lines
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    # Labels and title
    plt.xlabel('Annual Variation in Final Energy Consumption (%/year)', fontsize=16)
    plt.ylabel('Annual Variation in GDP (Constant EUR 2005) (%/year)', fontsize=16)
    plt.title('GDP vs Energy Consumption Annual Variation (2004-2014, 2014-2024)', 
             fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, which='both', ls='--', linewidth=0.4, alpha=0.5)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'energy_gdp_variation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved to: {output_path}")
    plt.close()
    
    # Export data to Excel
    export_path = os.path.join(OUTPUT_DIR, 'energy_gdp_variation_data.xlsx')
    df_export = df_all[['geo_code', 'period', 'gdp_variation_annual', 'energy_variation_annual']].copy()
    df_export.columns = ['Country', 'Period', 'GDP Variation (%/year)', 'Energy Variation (%/year)']
    df_export.to_excel(export_path, index=False)
    print(f"  ✓ Data exported to: {export_path}")

if __name__ == "__main__":
    main()
