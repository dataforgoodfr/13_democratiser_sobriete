"""
ECB Analysis - Generate graphs from European Central Bank data
This script creates visualizations for Cost of Borrowing (Housing)
Outputs saved as PNG files in outputs/graphs/ECB folder
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Base paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, 'external_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'graphs', 'ECB')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette (matching eurostat_analysis.py)
COMPONENT_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3', 
                    '#80b1d3', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']
ECB_PRIMARY_COLOR = '#80b1d3'  # EU aggregate blue
ECB_HIGHLIGHT_COLOR = '#fb8072'  # Highlight red


def load_ecb_cost_of_borrowing():
    """Load ECB Cost of Borrowing data"""
    file_path = os.path.join(EXTERNAL_DATA_DIR, 'ECB_Cost of borrowing.csv')
    
    if not os.path.exists(file_path):
        print(f"  ERROR: Data file not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Extract year and month for grouping
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR_MONTH'] = df['DATE'].dt.strftime('%Y-%m')
    
    # Convert cost to numeric
    df['cost'] = pd.to_numeric(df['Cost of borrowing for households for house purchase  (MIR.M.U2.B.A2C.AM.R.A.2250.EUR.N)'], errors='coerce')
    df = df.dropna(subset=['cost'])
    
    return df.sort_values('DATE')


def create_cost_of_borrowing_timeseries():
    """Create time series of cost of borrowing"""
    df = load_ecb_cost_of_borrowing()
    
    if df is None or df.empty:
        print("  ERROR: No data to plot for cost of borrowing")
        return
    
    # Full time series
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(df['DATE'], df['cost'], 
           color=ECB_PRIMARY_COLOR, linewidth=2.5, alpha=0.8,
           label='Cost of Borrowing')
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost of Borrowing (% per annum)', fontsize=12, fontweight='bold')
    ax.set_title('Cost of Borrowing for Households - House Purchase\n(Euro Area, Monthly, % per annum)', 
                fontsize=13, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('white')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_cost_of_borrowing_timeseries.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVED] 1_cost_of_borrowing_timeseries.png")


def create_annual_averages():
    """Create annual average cost of borrowing"""
    df = load_ecb_cost_of_borrowing()
    
    if df is None or df.empty:
        print("  ERROR: No data to plot for annual averages")
        return
    
    # Calculate annual averages
    df_annual = df.groupby('YEAR')['cost'].mean().reset_index()
    df_annual = df_annual.sort_values('YEAR')
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create bar chart
    colors = [ECB_HIGHLIGHT_COLOR if i == len(df_annual) - 1 else ECB_PRIMARY_COLOR 
             for i in range(len(df_annual))]
    
    bars = ax.bar(df_annual['YEAR'].astype(str), df_annual['cost'], 
                 color=colors, edgecolor='white', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Cost of Borrowing (% per annum)', fontsize=12, fontweight='bold')
    ax.set_title('Average Annual Cost of Borrowing for Households - House Purchase\n(Euro Area, % per annum)', 
                fontsize=13, fontweight='bold', pad=15)
    
    ax.grid(axis='y', alpha=0.2)
    ax.set_facecolor('white')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_cost_of_borrowing_annual_avg.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVED] 2_cost_of_borrowing_annual_avg.png")


def create_monthly_heatmap():
    """Create heatmap showing seasonality (month vs year)"""
    df = load_ecb_cost_of_borrowing()
    
    if df is None or df.empty:
        print("  ERROR: No data to plot for monthly heatmap")
        return
    
    # Create pivot table for heatmap
    pivot_data = df.pivot_table(values='cost', index='MONTH', columns='YEAR', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create heatmap
    im = ax.imshow(pivot_data.values, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    
    ax.set_xticklabels(pivot_data.columns.astype(int), fontsize=9)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_yticklabels([month_names[i-1] for i in pivot_data.index], fontsize=10)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Month', fontsize=12, fontweight='bold')
    ax.set_title('Cost of Borrowing Seasonality Heatmap\n(Darker = Higher Rate, % per annum)', 
                fontsize=13, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cost of Borrowing (% per annum)', fontsize=11, fontweight='bold')
    
    # Add values to cells (for recent years)
    recent_years = 5
    for i in range(len(pivot_data.index)):
        for j in range(max(0, len(pivot_data.columns) - recent_years), len(pivot_data.columns)):
            value = pivot_data.iloc[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.1f}',
                       ha='center', va='center', color='black', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_cost_of_borrowing_seasonality.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVED] 3_cost_of_borrowing_seasonality.png")


def create_recent_trends():
    """Create detailed view of recent trends (last 5 years)"""
    df = load_ecb_cost_of_borrowing()
    
    if df is None or df.empty:
        print("  ERROR: No data to plot for recent trends")
        return
    
    # Last 5 years
    max_year = df['YEAR'].max()
    min_year = max_year - 4
    df_recent = df[df['YEAR'] >= min_year].copy()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot by year with different colors
    years = sorted(df_recent['YEAR'].unique())
    colors_years = COMPONENT_COLORS[:len(years)]
    
    for idx, year in enumerate(years):
        df_year = df_recent[df_recent['YEAR'] == year].sort_values('DATE')
        ax.plot(df_year['MONTH'], df_year['cost'],
               marker='o', linewidth=2, markersize=5, label=str(year),
               color=colors_years[idx], alpha=0.8)
    
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost of Borrowing (% per annum)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cost of Borrowing Monthly Trends - Recent Years ({min_year}-{max_year}, % per annum)',
                fontsize=13, fontweight='bold', pad=15)
    
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names[1:], fontsize=10)
    
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('white')
    ax.legend(fontsize=11, loc='best', framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_cost_of_borrowing_recent_trends.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVED] 4_cost_of_borrowing_recent_trends.png")


def create_discount_factor_vs_house_price():
    """Plot discount factors 1/(1+r)^N for N=[1,10,15,20,30,40] vs house prices"""
    print("\n[3] Creating discount factor vs house price graph...")
    
    # Load cost of borrowing data
    cob_file = os.path.join(EXTERNAL_DATA_DIR, 'ECB_Cost of borrowing.csv')
    if not os.path.exists(cob_file):
        print("  Missing cost of borrowing file")
        return
    
    df_cob = pd.read_csv(cob_file)
    df_cob['DATE'] = pd.to_datetime(df_cob['DATE'])
    cob_col = df_cob.columns[2]
    df_cob['cob'] = pd.to_numeric(df_cob[cob_col], errors='coerce') / 100  # Convert to decimal
    df_cob = df_cob[['DATE', 'cob']].dropna().sort_values('DATE')
    
    # Load house price data
    price_file = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_house price_EU.csv')
    if not os.path.exists(price_file):
        print("  Missing house price file")
        return
    
    df_price = pd.read_csv(price_file)
    df_price['price_value'] = pd.to_numeric(df_price['OBS_VALUE'], errors='coerce')
    df_price = df_price[['TIME_PERIOD', 'price_value']].dropna().sort_values('TIME_PERIOD')
    
    if df_cob.empty or df_price.empty:
        print("  No valid data found")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate discount factor for N=1
    n = 1
    discount_factors = 1 / (1 + df_cob['cob']) ** n
    
    # Map house prices to discount factors
    x_values = df_price['price_value'].values[:len(discount_factors)]
    y_values = discount_factors.values[:len(x_values)]
    
    # Plot the curve
    ax.plot(x_values, y_values, color='#377eb8', linewidth=2.5, marker='o', 
           markersize=6, alpha=0.8)
    
    ax.set_xlabel('House Price Index (Eurostat)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Discount Factor: 1/(1+r)', fontsize=12, fontweight='bold')
    ax.set_title('Discount Factor vs House Price Index\n(where r = Cost of Borrowing)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'discount_factor_vs_house_price.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] discount_factor_vs_house_price.png")


def create_timeseries_dual_axes():
    """Plot house price and discount factor over time with dual y-axes"""
    print("\n[4] Creating timeseries with dual axes...")
    
    # Colors from ewbi_visuals_fr.py
    HOUSE_PRICE_COLOR = '#ffd558'  # Yellow for main metric
    DISCOUNT_FACTOR_COLOR = '#80b1d3'  # Blue for secondary metric
    
    # Load cost of borrowing data
    cob_file = os.path.join(EXTERNAL_DATA_DIR, 'ECB_Cost of borrowing.csv')
    if not os.path.exists(cob_file):
        print("  Missing cost of borrowing file")
        return
    
    df_cob = pd.read_csv(cob_file)
    df_cob['DATE'] = pd.to_datetime(df_cob['DATE'])
    cob_col = df_cob.columns[2]
    df_cob['cob'] = pd.to_numeric(df_cob[cob_col], errors='coerce') / 100
    df_cob = df_cob[['DATE', 'cob']].dropna().sort_values('DATE')
    
    # Load house price data
    price_file = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_house price_EU.csv')
    if not os.path.exists(price_file):
        print("  Missing house price file")
        return
    
    df_price = pd.read_csv(price_file)
    df_price['TIME_PERIOD'] = pd.to_numeric(df_price['TIME_PERIOD'], errors='coerce')
    df_price['price_value'] = pd.to_numeric(df_price['OBS_VALUE'], errors='coerce')
    df_price = df_price[['TIME_PERIOD', 'price_value']].dropna().sort_values('TIME_PERIOD')
    
    # Rescale from 2015=100 to 2005=100
    price_2005 = df_price[df_price['TIME_PERIOD'] == 2005]['price_value'].values
    if len(price_2005) > 0:
        df_price['price_value'] = (df_price['price_value'] / price_2005[0]) * 100
    
    # Convert year to datetime for proper alignment
    df_price['DATE'] = pd.to_datetime(df_price['TIME_PERIOD'].astype(int).astype(str) + '-01-01')
    
    if df_cob.empty or df_price.empty:
        print("  No valid data found")
        return
    
    # Calculate discount factor for each date
    discount_factor = 1 / (1 + df_cob['cob'].values)
    
    # Rescale discount factor to 2005=100
    # Find 2005 discount factor value
    df_2005_mask = df_cob['DATE'].dt.year == 2005
    if df_2005_mask.any():
        df_2005_value = discount_factor[df_2005_mask][0]
        discount_factor_indexed = (discount_factor / df_2005_value) * 100
    else:
        # If 2005 not in data, use the first available year as base
        discount_factor_indexed = (discount_factor / discount_factor[0]) * 100
    
    # Get price range for setting y-axis limits
    price_min = df_price['price_value'].min()
    price_max = df_price['price_value'].max()
    price_range = price_max - price_min
    
    # Get discount factor indexed range
    df_min = discount_factor_indexed.min()
    df_max = discount_factor_indexed.max()
    df_range = df_max - df_min
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot house price on left axis
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('House Price Index (2005=100)', fontsize=12, fontweight='bold', color=HOUSE_PRICE_COLOR)
    ax1.plot(df_price['DATE'], df_price['price_value'], 
            color=HOUSE_PRICE_COLOR, linewidth=2.5, marker='o', markersize=6, label='House Price Index', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=HOUSE_PRICE_COLOR)
    ax1.grid(True, alpha=0.3)
    
    # Extend y-axis with padding
    padding = price_range * 0.1
    ax1.set_ylim(price_min - padding, price_max + padding)
    
    # Create right axis for discount factor
    ax2 = ax1.twinx()
    ax2.set_ylabel('Discount Factor Index (2005=100)', fontsize=12, fontweight='bold', color=DISCOUNT_FACTOR_COLOR)
    ax2.tick_params(axis='y', labelcolor=DISCOUNT_FACTOR_COLOR)
    
    ax2.plot(df_cob['DATE'], discount_factor_indexed, 
            color=DISCOUNT_FACTOR_COLOR, linewidth=1.8, marker='s', markersize=3, 
            label='Discount Factor (2005=100)', alpha=0.8)
    
    # Set right axis limits with padding
    df_padding = df_range * 0.1
    ax2.set_ylim(df_min - df_padding, df_max + df_padding)
    
    # Set title
    fig.suptitle('House Price Index vs Discount Factor Over Time', 
                fontsize=13, fontweight='bold', y=0.98)
    
    # Add combined legend
    line1, = ax1.plot([], [], color=HOUSE_PRICE_COLOR, linewidth=2.5, marker='o', markersize=6, label='House Price Index')
    line2, = ax2.plot([], [], color=DISCOUNT_FACTOR_COLOR, linewidth=1.8, marker='s', markersize=3, label='Discount Factor (2005=100)')
    ax1.legend([line1, line2], ['House Price Index (2005=100)', 'Discount Factor (2005=100)'], loc='upper left', fontsize=11, framealpha=0.95)
    
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'timeseries_dual_axes.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] timeseries_dual_axes.png")


def create_cob_and_price_simple():
    """Plot raw Cost of Borrowing and Price Index together (blue and yellow)"""
    print("\n[Simple] Creating Cost of Borrowing and Price Index graph...")
    
    # Colors
    COB_COLOR = '#80b1d3'  # Blue
    PRICE_COLOR = '#ffd558'  # Yellow
    
    # Load cost of borrowing data
    cob_file = os.path.join(EXTERNAL_DATA_DIR, 'ECB_Cost of borrowing.csv')
    if not os.path.exists(cob_file):
        print("  Missing cost of borrowing file")
        return
    
    df_cob = pd.read_csv(cob_file)
    df_cob['DATE'] = pd.to_datetime(df_cob['DATE'])
    cob_col = df_cob.columns[2]
    df_cob['cob'] = pd.to_numeric(df_cob[cob_col], errors='coerce')
    df_cob = df_cob[['DATE', 'cob']].dropna().sort_values('DATE')
    
    # Load house price data
    price_file = os.path.join(EXTERNAL_DATA_DIR, 'eurostat_house price_EU.csv')
    if not os.path.exists(price_file):
        print("  Missing house price file")
        return
    
    df_price = pd.read_csv(price_file)
    df_price['TIME_PERIOD'] = pd.to_numeric(df_price['TIME_PERIOD'], errors='coerce')
    df_price['price_value'] = pd.to_numeric(df_price['OBS_VALUE'], errors='coerce')
    df_price = df_price[['TIME_PERIOD', 'price_value']].dropna().sort_values('TIME_PERIOD')
    df_price['DATE'] = pd.to_datetime(df_price['TIME_PERIOD'].astype(int).astype(str) + '-01-01')
    
    if df_cob.empty or df_price.empty:
        print("  No valid data found")
        return
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot Cost of Borrowing on left axis
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cost of Borrowing (% per annum)', fontsize=12, fontweight='bold', color=COB_COLOR)
    ax1.plot(df_cob['DATE'], df_cob['cob'], 
            color=COB_COLOR, linewidth=2.5, marker='o', markersize=5, label='Cost of Borrowing', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=COB_COLOR)
    ax1.grid(True, alpha=0.3)
    
    # Create right axis for Price Index
    ax2 = ax1.twinx()
    ax2.set_ylabel('House Price Index', fontsize=12, fontweight='bold', color=PRICE_COLOR)
    ax2.tick_params(axis='y', labelcolor=PRICE_COLOR)
    ax2.plot(df_price['DATE'], df_price['price_value'], 
            color=PRICE_COLOR, linewidth=2.5, marker='s', markersize=5, 
            label='House Price Index', alpha=0.8)
    
    # Set title
    fig.suptitle('Cost of Borrowing (Blue) vs House Price Index (Yellow)', 
                fontsize=13, fontweight='bold', y=0.98)
    
    # Add combined legend
    line1, = ax1.plot([], [], color=COB_COLOR, linewidth=2.5, marker='o', markersize=5, label='Cost of Borrowing')
    line2, = ax2.plot([], [], color=PRICE_COLOR, linewidth=2.5, marker='s', markersize=5, label='House Price Index')
    ax1.legend([line1, line2], ['Cost of Borrowing (% per annum)', 'House Price Index'], 
              loc='upper left', fontsize=11, framealpha=0.95)
    
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'simple_cob_and_price.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  [SAVED] simple_cob_and_price.png")


def main():
    """Main function"""
    print("=" * 70)
    print("ECB Analysis - Cost of Borrowing Visualizations")
    print("=" * 70)
    
    print(f"\nData location: {EXTERNAL_DATA_DIR}")
    print(f"Output location: {OUTPUT_DIR}")
    
    print("\nGenerating visualizations...\n")
    
    print("[1] Creating time series graph...")
    create_cost_of_borrowing_timeseries()
    
    print("\n[2] Creating Cost of Borrowing and Price Index graph...")
    create_cob_and_price_simple()
    
    print("\n[3] Creating discount factor vs house price graph...")
    create_discount_factor_vs_house_price()
    
    print("\n[4] Creating dual-axis timeseries...")
    create_timeseries_dual_axes()
    
    print("\n" + "=" * 70)
    print("All visualizations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
