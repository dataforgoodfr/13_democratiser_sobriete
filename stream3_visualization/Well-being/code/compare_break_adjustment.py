#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Script: Compare Raw Data vs Break-Adjusted Data

This script creates comparison visualizations showing the impact of structural break
adjustments made in Stage 1 (1_missing_data.py).

For 10 random indicators, creates side-by-side plots for France and Germany showing:
- Raw data (before break adjustment)
- Break-adjusted data (after Stage 1 processing)

Output: PNG files saved to stream3_visualization/Well-being/output/test/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ===============================
# CONFIGURATION
# ===============================
OUTPUT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')))
TEST_OUTPUT_DIR = OUTPUT_DIR / 'test'
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ['FR', 'DE']  # Country codes: FR=France, DE=Germany
COUNTRY_NAMES = {'FR': 'France', 'DE': 'Germany'}
NUM_INDICATORS = 10
RANDOM_SEED = 42

# ===============================
# DATA LOADING
# ===============================

def load_raw_data():
    """Load raw input data from Stage 1 inputs"""
    print("[LOAD] Loading raw input data...")
    
    # Load EU-SILC final summary data (household + personal)
    silc_final_merged_dir = OUTPUT_DIR / '0_raw_data_EUROSTAT' / '0_EU-SILC' / '3_final_merged_df'
    
    raw_data = pd.DataFrame()
    
    # Load household data
    silc_household_file = silc_final_merged_dir / 'EU_SILC_household_final_summary.csv'
    if silc_household_file.exists():
        silc_household = pd.read_csv(silc_household_file)
        raw_data = pd.concat([raw_data, silc_household], ignore_index=True)
        print(f"   [OK] EU-SILC Household: {len(silc_household):,} rows")
    
    # Load personal data
    silc_personal_file = silc_final_merged_dir / 'EU_SILC_personal_final_summary.csv'
    if silc_personal_file.exists():
        silc_personal = pd.read_csv(silc_personal_file)
        raw_data = pd.concat([raw_data, silc_personal], ignore_index=True)
        print(f"   [OK] EU-SILC Personal: {len(silc_personal):,} rows")
    
    # Load LFS data
    lfs_file = OUTPUT_DIR / '0_raw_data_EUROSTAT' / '0_LFS' / 'LFS_household_final_summary.csv'
    if lfs_file.exists():
        lfs_data = pd.read_csv(lfs_file)
        raw_data = pd.concat([raw_data, lfs_data], ignore_index=True)
        print(f"   [OK] LFS: {len(lfs_data):,} rows")
    
    print(f"   Total raw data: {len(raw_data):,} rows\n")
    return raw_data


def load_break_adjusted_data():
    """Load output data from Stage 1 (break-adjusted)"""
    print("[LOAD] Loading break-adjusted data from Stage 1...")
    
    adjusted_file = OUTPUT_DIR / '1_missing_data_output' / 'raw_data_break_adjusted.csv'
    
    if not adjusted_file.exists():
        print(f"[ERROR] File not found: {adjusted_file}")
        return None
    
    adjusted_data = pd.read_csv(adjusted_file)
    print(f"   [OK] Loaded {len(adjusted_data):,} rows from {adjusted_file.name}\n")
    
    return adjusted_data


# ===============================
# DATA PROCESSING
# ===============================

def filter_data_for_country_indicator(df, country, indicator):
    """Filter data for specific country and indicator across all deciles"""
    filtered = df[
        (df['Country'] == country) &
        (df['Primary and raw data'] == indicator)
    ].copy()
    
    filtered = filtered.sort_values('Year')
    return filtered


def extract_decile_data(df):
    """Extract data for a specific decile from filtered data"""
    if df.empty:
        return None
    
    # Get all unique deciles
    deciles = sorted([int(d) for d in df['Decile'].unique() if pd.notna(d)])
    
    if not deciles:
        return None
    
    # Use first decile (Decile 1)
    decile = deciles[0]
    decile_data = df[df['Decile'] == decile].copy()
    
    if decile_data.empty:
        return None
    
    return decile_data


# ===============================
# VISUALIZATION
# ===============================

def create_comparison_plot(raw_data, adjusted_data, indicator, countries=['FR', 'DE'], country_names={'FR': 'France', 'DE': 'Germany'}):
    """
    Create overlaid comparison plots for a specific indicator with all deciles in distinct colors.
    
    Layout:
    - Row 1: France (Raw data as dotted lines, Break-adjusted as solid lines)
    - Row 2: Germany (Raw data as dotted lines, Break-adjusted as solid lines)
    
    Each decile has a unique color for easy identification.
    """
    
    fig, axes = plt.subplots(len(countries), 1, figsize=(16, 10))
    if len(countries) == 1:
        axes = [axes]
    
    fig.suptitle(f'Indicator: {indicator}\nRaw Data (dotted) vs Break-Adjusted Data (solid)', 
                 fontsize=14, fontweight='bold')
    
    plot_success = False
    
    # Define distinct colors for each decile (10 deciles maximum)
    decile_colors = {
        1: '#e41a1c',    # Red
        2: '#377eb8',    # Blue
        3: '#4daf4a',    # Green
        4: '#984ea3',    # Purple
        5: '#ff7f00',    # Orange
        6: '#a65628',    # Brown
        7: '#f781bf',    # Pink
        8: '#999999',    # Grey
        9: '#66c2a5',    # Teal
        10: '#fc8d62'    # Salmon
    }
    
    for row, country_code in enumerate(countries):
        country_name = country_names.get(country_code, country_code)
        
        # Get raw data for this country/indicator
        raw_country = raw_data[
            (raw_data['Country'] == country_code) &
            (raw_data['Primary and raw data'] == indicator)
        ].copy()
        
        # Get adjusted data for this country/indicator
        adj_country = adjusted_data[
            (adjusted_data['Country'] == country_code) &
            (adjusted_data['Primary and raw data'] == indicator)
        ].copy()
        
        ax = axes[row]
        
        if raw_country.empty or adj_country.empty:
            ax.text(0.5, 0.5, f'No data for {country_name}', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{country_name}: No data available')
            continue
        
        # Get all deciles for visualization
        deciles = sorted(set(raw_country['Decile'].values) | set(adj_country['Decile'].values))
        # Remove NaN deciles
        deciles = [d for d in deciles if pd.notna(d)]
        # Filter to only include deciles 1 and 7
        deciles = [d for d in deciles if int(float(d)) in [1, 7]]
        
        # Plot both raw and adjusted for deciles 1 and 7 with distinct colors
        for decile_idx, decile in enumerate(deciles):
            decile_int = int(float(decile))  # Convert safely to int
            color = decile_colors.get(decile_int, '#000000')  # Default to black if decile > 10
            
            # Raw data (dotted line)
            raw_subset = raw_country[raw_country['Decile'] == decile].sort_values('Year')
            if not raw_subset.empty:
                ax.plot(raw_subset['Year'], raw_subset['Value'], 
                       marker='o', linestyle=':', linewidth=2, markersize=5,
                       color=color,
                       label=f'Decile {decile_int} (Raw)', alpha=0.6)
            
            # Break-adjusted data (solid line)
            adj_subset = adj_country[adj_country['Decile'] == decile].sort_values('Year')
            if not adj_subset.empty:
                ax.plot(adj_subset['Year'], adj_subset['Value'], 
                       marker='s', linestyle='-', linewidth=2.5, markersize=5,
                       color=color,
                       label=f'Decile {decile_int} (Adjusted)', alpha=0.9)
        
        ax.set_title(f'{country_name}: Raw Data (dotted) vs Break-Adjusted Data (solid)', fontsize=12)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.legend(loc='best', fontsize=8, ncol=3, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        plot_success = True
    
    plt.tight_layout()
    return fig, plot_success


# ===============================
# MAIN EXECUTION
# ===============================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("COMPARISON: Raw Data vs Break-Adjusted Data")
    print("="*70 + "\n")
    
    # Load data
    raw_input_data = load_raw_data()
    break_adjusted_data = load_break_adjusted_data()
    
    if raw_input_data is None or break_adjusted_data is None:
        print("[ERROR] Failed to load data")
        return
    
    # Standardize column names in raw data
    if 'Country' not in raw_input_data.columns and 'country' in raw_input_data.columns:
        raw_input_data = raw_input_data.rename(columns={
            'year': 'Year',
            'country': 'Country',
            'decile': 'Decile',
            'primary_index': 'Primary and raw data',
            'value': 'Value'
        })
    
    # Standardize data types
    raw_input_data['Year'] = pd.to_numeric(raw_input_data['Year'], errors='coerce').astype('Int64')
    raw_input_data['Decile'] = pd.to_numeric(raw_input_data['Decile'], errors='coerce')
    raw_input_data['Value'] = pd.to_numeric(raw_input_data['Value'], errors='coerce')
    
    break_adjusted_data['Year'] = pd.to_numeric(break_adjusted_data['Year'], errors='coerce').astype('Int64')
    break_adjusted_data['Decile'] = pd.to_numeric(break_adjusted_data['Decile'], errors='coerce')
    break_adjusted_data['Value'] = pd.to_numeric(break_adjusted_data['Value'], errors='coerce')
    
    # Get list of indicators present in both datasets
    raw_indicators = set(raw_input_data['Primary and raw data'].unique())
    adjusted_indicators = set(break_adjusted_data['Primary and raw data'].unique())
    common_indicators = list(raw_indicators & adjusted_indicators)
    
    print(f"[INFO] Common indicators in both datasets: {len(common_indicators)}")
    
    if len(common_indicators) == 0:
        print("[ERROR] No common indicators found between raw and adjusted data")
        return
    
    # Select random indicators
    random.seed(RANDOM_SEED)
    selected_indicators = random.sample(common_indicators, min(NUM_INDICATORS, len(common_indicators)))
    selected_indicators.sort()
    
    print(f"[INFO] Selected {len(selected_indicators)} indicators for visualization:")
    for ind in selected_indicators:
        print(f"   - {ind}")
    
    # Create visualizations
    print(f"\n[PLOT] Creating visualizations...")
    successful_plots = 0
    
    for idx, indicator in enumerate(selected_indicators, 1):
        try:
            fig, success = create_comparison_plot(raw_input_data, break_adjusted_data, indicator, COUNTRIES, COUNTRY_NAMES)
            
            if success:
                # Save figure
                filename = f"{idx:02d}_{indicator.replace('/', '_').replace(' ', '_')}_comparison.png"
                filepath = TEST_OUTPUT_DIR / filename
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   [OK] {idx}/{len(selected_indicators)}: Saved {filename}")
                successful_plots += 1
            else:
                print(f"   [SKIP] {idx}/{len(selected_indicators)}: {indicator} - No valid data")
                plt.close(fig)
        
        except Exception as e:
            print(f"   [ERROR] {idx}/{len(selected_indicators)}: {indicator} - {str(e)}")
            plt.close('all')
    
    print(f"\n[COMPLETE] Visualization complete")
    print(f"   Total plots created: {successful_plots}/{len(selected_indicators)}")
    print(f"   Output directory: {TEST_OUTPUT_DIR}")
    
    return successful_plots


if __name__ == '__main__':
    result = main()
    print("\n[OK] compare_break_adjustment.py execution complete")
