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

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set font
plt.rcParams['font.family'] = 'Arial'

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

def main():
    print("=" * 80)
    print("ENERGY VS GDP ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df_gdp_pps = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_gdp_pps.csv'))
    df_gdp_eur = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_gdp_eur-2005.csv'))
    df_energy_raw = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_final_energy.csv'))
    df_hicp = pd.read_csv(os.path.join(EXTERNAL_DATA_DIR, 'eurostat_HICP.csv'))
    
    print(f"  GDP PPS data: {len(df_gdp_pps)} rows")
    print(f"  GDP EUR 2005 data: {len(df_gdp_eur)} rows")
    print(f"  Energy data: {len(df_energy_raw)} rows")
    print(f"  HICP data: {len(df_hicp)} rows")
    
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
