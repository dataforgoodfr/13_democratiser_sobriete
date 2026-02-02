"""
Compute Compensation for Atmospheric Appropriation

This script calculates the overshoot debt for countries that have exceeded their
carbon budget, based on:
1. Cumulative overshoot emissions from 2023 to 2050
2. Carbon price interpolation using cubic splines
3. Exponential emission trajectory towards 0.1 tCO2/capita by 2050

Output:
- Interpolated carbon price visualization
- World maps showing debt per country for each scenario
- CSV files with compensation data
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ==============================================================================
# Configuration
# ==============================================================================

# Get the absolute path to the data directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'Output'))
INPUT_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'Data'))
OUTPUT_DIR = os.path.join(DATA_DIR, 'compensation')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carbon price data (US$ per tonne of CO2 in 2010 prices)
# Source: Provided carbon price projections
CARBON_PRICE_DATA = {
    'Year': [2025, 2030, 2035, 2040, 2045, 2050],
    'Median': [135, 198, 267, 351, 465, 547],
    'Q1': [80, 158, 220, 276, 340, 394],
    'Q3': [197, 242, 438, 569, 715, 887]
}

# Time range for analysis
HISTORICAL_START_YEAR = 1970  # Start of historical period
OVERSHOOT_START_YEAR = 2023   # Year from which overshoot is computed
END_YEAR = 2050

# Historical EU ETS price file
ETS_PRICE_FILE = 'carbon_price_ETS.xlsx'
EUR_USD_RATE_FILE = 'eur_usd_annual_average_ecb.csv'


# ==============================================================================
# Carbon Price Interpolation
# ==============================================================================

def load_eur_usd_rates():
    """
    Load EUR/USD exchange rates from ECB data.
    
    Returns:
        DataFrame with Year and Rate columns (EUR to USD)
    """
    rate_file = os.path.join(INPUT_DATA_DIR, EUR_USD_RATE_FILE)
    rate_df = pd.read_csv(rate_file)
    rate_df = rate_df[['Year', 'Average']].copy()
    rate_df.columns = ['Year', 'EUR_USD_Rate']
    return rate_df


def load_historical_ets_prices():
    """
    Load historical EU ETS carbon prices from Excel file and convert to USD.
    
    Returns:
        DataFrame with Year, Price_EUR, Price_USD columns
    """
    # Load ETS prices in EUR
    ets_file = os.path.join(INPUT_DATA_DIR, ETS_PRICE_FILE)
    ets_df = pd.read_excel(ets_file)
    ets_df.columns = ['Year', 'Price_EUR']  # Rename for clarity
    
    # Load EUR/USD exchange rates
    eur_usd_df = load_eur_usd_rates()
    
    # Merge to get exchange rate for each year
    ets_df = ets_df.merge(eur_usd_df, on='Year', how='left')
    
    # Convert EUR to USD
    ets_df['Price_USD'] = ets_df['Price_EUR'] * ets_df['EUR_USD_Rate']
    
    print(f"   Converted EU ETS prices from EUR to USD using ECB exchange rates")
    
    return ets_df


def interpolate_carbon_prices_full():
    """
    Create complete carbon price series from 1970 to 2050 in USD.
    
    Data sources:
    - 1970-2004: Reconstituted using 5-year average of EU ETS prices 2005-2009 (converted to USD)
    - 2005-2021: Historical EU ETS prices (converted to USD using ECB exchange rates)
    - 2022-2024: Interpolated between ETS (2021) and projections (2025)
    - 2025-2050: Projected prices (Median, Q1, Q3) in USD with cubic spline interpolation
    
    Returns:
        DataFrame with Year, Price, Q1, Q3, Data_type columns (all in USD)
    """
    # Load historical ETS prices (already converted to USD)
    ets_df = load_historical_ets_prices()
    
    # Calculate 5-year average of 2005-2009 prices in USD for reconstituted past data
    ets_2005_2009 = ets_df[(ets_df['Year'] >= 2005) & (ets_df['Year'] <= 2009)]
    mean_2005_2009_usd = ets_2005_2009['Price_USD'].mean()
    mean_2005_2009_eur = ets_2005_2009['Price_EUR'].mean()
    print(f"   5-year average EU ETS price 2005-2009: €{mean_2005_2009_eur:.2f} = ${mean_2005_2009_usd:.2f}/tCO2")
    
    # Get 2021 ETS price in USD for interpolation bridge
    price_2021_usd = ets_df[ets_df['Year'] == 2021]['Price_USD'].values[0]
    price_2021_eur = ets_df[ets_df['Year'] == 2021]['Price_EUR'].values[0]
    print(f"   2021 EU ETS price: €{price_2021_eur:.2f} = ${price_2021_usd:.2f}/tCO2")
    
    # =========================================================================
    # Build the full price series (all in USD)
    # =========================================================================
    all_years = np.arange(HISTORICAL_START_YEAR, END_YEAR + 1)
    result = pd.DataFrame({'Year': all_years})
    
    # Initialize columns
    result['Median'] = np.nan
    result['Q1'] = np.nan
    result['Q3'] = np.nan
    result['Data_type'] = ''
    result['Is_original_data'] = False
    
    # -------------------------------------------------------------------------
    # 1. Reconstituted past (1970-2004): Use 5-year average of 2005-2009 in USD
    # -------------------------------------------------------------------------
    mask_reconstituted = (result['Year'] >= HISTORICAL_START_YEAR) & (result['Year'] <= 2004)
    result.loc[mask_reconstituted, 'Median'] = mean_2005_2009_usd
    result.loc[mask_reconstituted, 'Q1'] = mean_2005_2009_usd  # No uncertainty for reconstituted
    result.loc[mask_reconstituted, 'Q3'] = mean_2005_2009_usd
    result.loc[mask_reconstituted, 'Data_type'] = 'Reconstituted (5-year avg 2005-2009)'
    
    # -------------------------------------------------------------------------
    # 2. Historical EU ETS (2005-2021): Actual data converted to USD
    # -------------------------------------------------------------------------
    for _, row in ets_df.iterrows():
        year = int(row['Year'])
        price_usd = row['Price_USD']
        mask = result['Year'] == year
        result.loc[mask, 'Median'] = price_usd
        result.loc[mask, 'Q1'] = price_usd  # No uncertainty for historical
        result.loc[mask, 'Q3'] = price_usd
        result.loc[mask, 'Data_type'] = 'Historical (EU ETS)'
        result.loc[mask, 'Is_original_data'] = True
    
    # -------------------------------------------------------------------------
    # 3. Projections (2025-2050): Use IPCC data with cubic spline interpolation
    # -------------------------------------------------------------------------
    years_proj = np.array(CARBON_PRICE_DATA['Year'])
    
    for price_type in ['Median', 'Q1', 'Q3']:
        prices_proj = np.array(CARBON_PRICE_DATA[price_type])
        
        # Create cubic spline for projection years
        cs = CubicSpline(years_proj, prices_proj, bc_type='natural')
        
        # Apply to 2025-2050
        mask_proj = result['Year'] >= 2025
        result.loc[mask_proj, price_type] = cs(result.loc[mask_proj, 'Year'])
    
    # Mark projection data types
    mask_proj_original = result['Year'].isin(CARBON_PRICE_DATA['Year'])
    result.loc[mask_proj_original, 'Data_type'] = 'Projection (original data)'
    result.loc[mask_proj_original, 'Is_original_data'] = True
    
    mask_proj_interp = (result['Year'] >= 2025) & (~result['Year'].isin(CARBON_PRICE_DATA['Year']))
    result.loc[mask_proj_interp, 'Data_type'] = 'Projection (interpolated)'
    
    # -------------------------------------------------------------------------
    # 4. Bridge period (2022-2024): Linear interpolation between ETS (USD) and projections (USD)
    # -------------------------------------------------------------------------
    price_2025_median = CARBON_PRICE_DATA['Median'][0]
    price_2025_q1 = CARBON_PRICE_DATA['Q1'][0]
    price_2025_q3 = CARBON_PRICE_DATA['Q3'][0]
    
    bridge_years = [2022, 2023, 2024]
    for year in bridge_years:
        # Linear interpolation factor: 0 at 2021, 1 at 2025
        factor = (year - 2021) / (2025 - 2021)
        
        mask = result['Year'] == year
        result.loc[mask, 'Median'] = price_2021_usd + factor * (price_2025_median - price_2021_usd)
        result.loc[mask, 'Q1'] = price_2021_usd + factor * (price_2025_q1 - price_2021_usd)
        result.loc[mask, 'Q3'] = price_2021_usd + factor * (price_2025_q3 - price_2021_usd)
        result.loc[mask, 'Data_type'] = 'Interpolated (bridge)'
    
    return result


def interpolate_carbon_prices():
    """
    Interpolate carbon prices from 2023 to 2050 only (for backward compatibility).
    
    Returns:
        DataFrame with interpolated prices for each year (Median, Q1, Q3)
    """
    full_prices = interpolate_carbon_prices_full()
    return full_prices[full_prices['Year'] >= OVERSHOOT_START_YEAR][['Year', 'Median', 'Q1', 'Q3']].copy()


def plot_carbon_prices(carbon_prices_full_df):
    """
    Create a visualization of carbon prices from 1970 to 2050.
    
    Shows different periods with distinct colors:
    - Reconstituted (1970-2004): Orange - based on mean of 2005-2009
    - Historical EU ETS (2005-2021): Green - actual data
    - Interpolated bridge (2022-2024): Purple - linear interpolation
    - Projections (2025-2050): Blue - IPCC projections
    
    Args:
        carbon_prices_full_df: DataFrame with Year, Median, Q1, Q3, Data_type columns
    """
    fig = go.Figure()
    
    # Define colors for each data type
    colors = {
        'Reconstituted (5-year avg 2005-2009)': '#FF8C00',  # Dark Orange
        'Historical (EU ETS)': '#228B22',             # Forest Green
        'Interpolated (bridge)': '#9370DB',           # Medium Purple
        'Projection (original data)': '#2E86AB',      # Blue
        'Projection (interpolated)': '#87CEEB'        # Light Blue
    }
    
    # =========================================================================
    # Add Q1-Q3 range for projections only (2025-2050)
    # =========================================================================
    proj_data = carbon_prices_full_df[carbon_prices_full_df['Year'] >= 2025]
    fig.add_trace(go.Scatter(
        x=list(proj_data['Year']) + list(proj_data['Year'][::-1]),
        y=list(proj_data['Q3']) + list(proj_data['Q1'][::-1]),
        fill='toself',
        fillcolor='rgba(135, 206, 235, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Projection uncertainty (Q1-Q3)',
        showlegend=True,
        legendgroup='projection'
    ))
    
    # =========================================================================
    # Plot each period with different colors
    # =========================================================================
    
    # 1. Reconstituted past (1970-2004)
    reconst_data = carbon_prices_full_df[carbon_prices_full_df['Data_type'] == 'Reconstituted (5-year avg 2005-2009)']
    if len(reconst_data) > 0:
        fig.add_trace(go.Scatter(
            x=reconst_data['Year'],
            y=reconst_data['Median'],
            mode='lines',
            name='Reconstituted (1970-2004)',
            line=dict(color=colors['Reconstituted (5-year avg 2005-2009)'], width=2, dash='dot'),
            legendgroup='reconstituted'
        ))
    
    # 2. Historical EU ETS (2005-2021) - with markers for real data points
    hist_data = carbon_prices_full_df[carbon_prices_full_df['Data_type'] == 'Historical (EU ETS)']
    if len(hist_data) > 0:
        fig.add_trace(go.Scatter(
            x=hist_data['Year'],
            y=hist_data['Median'],
            mode='lines+markers',
            name='Historical EU ETS (2005-2021)',
            line=dict(color=colors['Historical (EU ETS)'], width=3),
            marker=dict(size=8, symbol='circle', line=dict(width=1, color='white')),
            legendgroup='historical'
        ))
    
    # 3. Bridge interpolation (2022-2024)
    bridge_data = carbon_prices_full_df[carbon_prices_full_df['Data_type'] == 'Interpolated (bridge)']
    if len(bridge_data) > 0:
        fig.add_trace(go.Scatter(
            x=bridge_data['Year'],
            y=bridge_data['Median'],
            mode='lines+markers',
            name='Interpolated (2022-2024)',
            line=dict(color=colors['Interpolated (bridge)'], width=2, dash='dash'),
            marker=dict(size=6, symbol='square'),
            legendgroup='bridge'
        ))
    
    # 4. Projections - interpolated (2025-2050)
    proj_interp = carbon_prices_full_df[carbon_prices_full_df['Data_type'] == 'Projection (interpolated)']
    if len(proj_interp) > 0:
        fig.add_trace(go.Scatter(
            x=proj_interp['Year'],
            y=proj_interp['Median'],
            mode='lines',
            name='Projection interpolated',
            line=dict(color=colors['Projection (interpolated)'], width=2),
            legendgroup='projection'
        ))
    
    # 5. Projections - original data points (diamonds)
    proj_orig = carbon_prices_full_df[carbon_prices_full_df['Data_type'] == 'Projection (original data)']
    if len(proj_orig) > 0:
        fig.add_trace(go.Scatter(
            x=proj_orig['Year'],
            y=proj_orig['Median'],
            mode='markers',
            name='Projection original data',
            marker=dict(size=12, color=colors['Projection (original data)'], 
                       symbol='diamond', line=dict(width=2, color='white')),
            legendgroup='projection'
        ))
    
    # =========================================================================
    # Add vertical lines to separate periods
    # =========================================================================
    for year, label in [(2005, 'EU ETS starts'), (2022, 'Projections bridge'), (2025, 'Projections')]:
        fig.add_vline(x=year, line_dash='dot', line_color='gray', opacity=0.5)
        fig.add_annotation(
            x=year, y=1.02, yref='paper',
            text=label, showarrow=False,
            font=dict(size=10, color='gray'),
            textangle=-45
        )
    
    fig.update_layout(
        title=dict(
            text='Carbon Price Evolution (1970-2050)<br><sub>Historical EU ETS (converted to USD) + IPCC Projections with Cubic Spline Interpolation</sub>',
            font=dict(size=18)
        ),
        xaxis_title='Year',
        yaxis_title='Carbon Price (USD/tCO₂)',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        hovermode='x unified',
        width=1200,
        height=600
    )
    
    # Save the figure
    fig.write_html(os.path.join(OUTPUT_DIR, 'carbon_price_interpolation.html'))
    fig.write_image(os.path.join(OUTPUT_DIR, 'carbon_price_interpolation.png'), scale=2)
    print(f"Saved carbon price visualization to {OUTPUT_DIR}")
    
    return fig


# ==============================================================================
# Emission Trajectory Computation
# ==============================================================================

def compute_emission_trajectory(co2_per_capita_2023, population_2023):
    """
    Compute the emission trajectory from 2023 to 2050 using linear decline.
    
    Linear interpolation from current emissions (2023) to 0 in 2050.
    Formula: Emissions(t) = E_2023 * Pop_2023 * (2050 - t) / (2050 - 2023)
    
    Args:
        co2_per_capita_2023: Initial per capita emissions (tonnes)
        population_2023: Population in 2023
    
    Returns:
        DataFrame with Year and Emissions_Mt columns
    """
    years = np.arange(OVERSHOOT_START_YEAR, END_YEAR + 1)
    
    # Initial total emissions in 2023 (in tonnes)
    initial_emissions = co2_per_capita_2023 * population_2023
    
    # Linear decline: from initial_emissions in 2023 to 0 in 2050
    # Factor goes from 1 (at 2023) to 0 (at 2050)
    time_span = END_YEAR - OVERSHOOT_START_YEAR  # 27 years
    linear_factor = (END_YEAR - years) / time_span
    
    emissions = initial_emissions * linear_factor
    
    # Convert to Mt (emissions are in tonnes, need to divide by 1e6)
    emissions_mt = emissions / 1e6
    
    return pd.DataFrame({
        'Year': years,
        'Emissions_Mt': emissions_mt
    })


# ==============================================================================
# Overshoot Debt Computation (Full Historical + Future)
# ==============================================================================

def load_historical_emissions():
    """
    Load historical emissions data for all countries.
    
    Returns:
        DataFrame with ISO2, Year, Emissions_scope, Annual_CO2_emissions_Mt
    """
    combined_data = pd.read_csv(os.path.join(DATA_DIR, 'combined_data.csv'))
    return combined_data


def compute_full_debt_trajectory(neutrality_year, iso2, emissions_scope, 
                                  historical_emissions_df, carbon_prices_full_df,
                                  co2_per_capita_2023, population_2023):
    """
    Compute the full debt from neutrality year to 2050.
    
    For countries with neutrality_year <= 2023:
    - Use historical emissions from neutrality_year to 2023
    - Project emissions linearly from 2024 to 0 in 2050
    
    For countries with neutrality_year > 2023:
    - Project emissions linearly from neutrality_year to 0 in 2050
    
    Periods:
    - Pre-2005: Historical emissions × reconstituted carbon price
    - ETS period (2005-2024): Historical emissions × EU ETS price (USD)
    - Future (2025-2050): Projected emissions (linear decline) × projected carbon price
    
    Args:
        neutrality_year: Year when country exhausts their budget
        iso2: Country ISO2 code
        emissions_scope: 'Territory' or 'Consumption'
        historical_emissions_df: DataFrame with historical emissions
        carbon_prices_full_df: Full carbon price series (1970-2050)
        co2_per_capita_2023: Per capita emissions in 2023
        population_2023: Population in 2023
    
    Returns:
        Dictionary with total debt, debt by period, and emissions by period
    """
    neutrality_year = int(neutrality_year)
    
    # Get carbon prices
    prices = carbon_prices_full_df[['Year', 'Median', 'Q1', 'Q3']].copy()
    
    # Get 2023 total emissions as reference for projections
    emissions_2023 = co2_per_capita_2023 * population_2023 / 1e6  # Convert to Mt
    
    # Initialize accumulators
    debt_pre_2005 = 0
    emissions_pre_2005 = 0
    debt_ets_period = 0
    emissions_ets_period = 0
    debt_future = 0
    emissions_future = 0
    total_debt_q1 = 0
    total_debt_q3 = 0
    
    # =========================================================================
    # Case 1: Neutrality year is in the past (<= 2023)
    # Use historical emissions, then project linearly to 0 in 2050
    # =========================================================================
    if neutrality_year <= OVERSHOOT_START_YEAR:
        start_year = max(neutrality_year, HISTORICAL_START_YEAR)
        
        # Get historical emissions for this country and scope
        country_hist = historical_emissions_df[
            (historical_emissions_df['ISO2'] == iso2) &
            (historical_emissions_df['Emissions_scope'] == emissions_scope) &
            (historical_emissions_df['Year'] >= start_year) &
            (historical_emissions_df['Year'] <= 2023)
        ][['Year', 'Annual_CO2_emissions_Mt']].copy()
        
        # Period 1: Pre-2005 (neutrality_year to 2004) - Historical emissions
        pre_2005_years = country_hist[country_hist['Year'] < 2005]
        pre_2005_merged = pre_2005_years.merge(prices, on='Year', how='left')
        
        if len(pre_2005_merged) > 0:
            pre_2005_merged['Debt'] = pre_2005_merged['Annual_CO2_emissions_Mt'] * pre_2005_merged['Median'] * 1e6
            debt_pre_2005 = pre_2005_merged['Debt'].sum()
            emissions_pre_2005 = pre_2005_merged['Annual_CO2_emissions_Mt'].sum()
            total_debt_q1 += (pre_2005_merged['Annual_CO2_emissions_Mt'] * pre_2005_merged['Q1'] * 1e6).sum()
            total_debt_q3 += (pre_2005_merged['Annual_CO2_emissions_Mt'] * pre_2005_merged['Q3'] * 1e6).sum()
        
        # Period 2: ETS period (2005-2024) - Historical emissions
        ets_period_years = country_hist[(country_hist['Year'] >= 2005) & (country_hist['Year'] <= 2024)]
        ets_merged = ets_period_years.merge(prices, on='Year', how='left')
        
        if len(ets_merged) > 0:
            ets_merged['Debt'] = ets_merged['Annual_CO2_emissions_Mt'] * ets_merged['Median'] * 1e6
            debt_ets_period = ets_merged['Debt'].sum()
            emissions_ets_period = ets_merged['Annual_CO2_emissions_Mt'].sum()
            total_debt_q1 += (ets_merged['Annual_CO2_emissions_Mt'] * ets_merged['Q1'] * 1e6).sum()
            total_debt_q3 += (ets_merged['Annual_CO2_emissions_Mt'] * ets_merged['Q3'] * 1e6).sum()
        
        # Period 3: Future (2025-2050) - Linear decline from 2023 emissions to 0 in 2050
        future_years = np.arange(2025, END_YEAR + 1)
        time_span = END_YEAR - OVERSHOOT_START_YEAR  # 27 years from 2023 to 2050
        
        future_emissions_list = []
        for year in future_years:
            factor = (END_YEAR - year) / time_span
            future_emissions_list.append({
                'Year': year,
                'Emissions_Mt': emissions_2023 * max(factor, 0)
            })
        
        future_df = pd.DataFrame(future_emissions_list)
        future_merged = future_df.merge(prices, on='Year', how='left')
        
        if len(future_merged) > 0:
            future_merged['Debt'] = future_merged['Emissions_Mt'] * future_merged['Median'] * 1e6
            debt_future = future_merged['Debt'].sum()
            emissions_future = future_merged['Emissions_Mt'].sum()
            total_debt_q1 += (future_merged['Emissions_Mt'] * future_merged['Q1'] * 1e6).sum()
            total_debt_q3 += (future_merged['Emissions_Mt'] * future_merged['Q3'] * 1e6).sum()
    
    # =========================================================================
    # Case 2: Neutrality year is in the future (> 2023)
    # Project emissions linearly from neutrality year to 0 in 2050
    # =========================================================================
    else:
        # Linear decline from emissions_2023 at neutrality_year to 0 in 2050
        time_span = END_YEAR - neutrality_year
        
        if time_span > 0:
            # Check if neutrality year falls in ETS period (2024 only possible here)
            if neutrality_year <= 2024:
                ets_years = np.arange(neutrality_year, 2025)
                for year in ets_years:
                    factor = (END_YEAR - year) / time_span
                    year_emissions = emissions_2023 * max(factor, 0)
                    year_price = prices[prices['Year'] == year]['Median'].values
                    year_q1 = prices[prices['Year'] == year]['Q1'].values
                    year_q3 = prices[prices['Year'] == year]['Q3'].values
                    if len(year_price) > 0:
                        debt_ets_period += year_emissions * year_price[0] * 1e6
                        emissions_ets_period += year_emissions
                        total_debt_q1 += year_emissions * year_q1[0] * 1e6
                        total_debt_q3 += year_emissions * year_q3[0] * 1e6
            
            # Future period (2025-2050 or from neutrality_year if > 2024)
            future_start = max(neutrality_year, 2025)
            future_years = np.arange(future_start, END_YEAR + 1)
            
            future_emissions_list = []
            for year in future_years:
                factor = (END_YEAR - year) / time_span
                future_emissions_list.append({
                    'Year': year,
                    'Emissions_Mt': emissions_2023 * max(factor, 0)
                })
            
            if future_emissions_list:
                future_df = pd.DataFrame(future_emissions_list)
                future_merged = future_df.merge(prices, on='Year', how='left')
                
                if len(future_merged) > 0:
                    future_merged['Debt'] = future_merged['Emissions_Mt'] * future_merged['Median'] * 1e6
                    debt_future = future_merged['Debt'].sum()
                    emissions_future = future_merged['Emissions_Mt'].sum()
                    total_debt_q1 += (future_merged['Emissions_Mt'] * future_merged['Q1'] * 1e6).sum()
                    total_debt_q3 += (future_merged['Emissions_Mt'] * future_merged['Q3'] * 1e6).sum()
    
    return {
        'Debt_Pre2005_USD': debt_pre_2005,
        'Debt_ETS_2005_2024_USD': debt_ets_period,
        'Debt_Future_2025_2050_USD': debt_future,
        'Debt_Total_Median_USD': debt_pre_2005 + debt_ets_period + debt_future,
        'Debt_Total_Q1_USD': total_debt_q1,
        'Debt_Total_Q3_USD': total_debt_q3,
        'Emissions_Pre2005_Mt': emissions_pre_2005,
        'Emissions_ETS_2005_2024_Mt': emissions_ets_period,
        'Emissions_Future_2025_2050_Mt': emissions_future,
        'Emissions_Total_Mt': emissions_pre_2005 + emissions_ets_period + emissions_future
    }


# ==============================================================================
# Fair Share Creditor/Debtor Computation
# ==============================================================================

def compute_remaining_budget_2050(co2_per_capita_2023, population_2023, 
                                   latest_cumulative_emissions, country_carbon_budget):
    """
    Compute the remaining carbon budget in 2050.
    
    Remaining_Budget_2050 = Country_carbon_budget - Projected_emissions_2024_to_2050
    
    This represents how much of the country's fair share they will NOT use
    if they decline linearly to 0 emissions by 2050.
    
    Args:
        co2_per_capita_2023: Per capita emissions in 2023 (tonnes)
        population_2023: Population in 2023
        latest_cumulative_emissions: Cumulative emissions up to 2023 (Mt) - NOT USED in new formula
        country_carbon_budget: Total carbon budget for this country (Mt)
    
    Returns:
        Remaining budget in 2050 (Mt), can be negative (debtor) or positive (creditor)
    """
    # Get 2023 total emissions (Mt)
    emissions_2023 = co2_per_capita_2023 * population_2023 / 1e6
    
    # Compute projected cumulative emissions from 2024 to 2050
    # Linear decline from emissions_2023 (at 2023) to 0 in 2050
    # Sum = E_2023 × (n_years / 2) where we include partial years
    # From 2024 to 2050 = 27 years, but emissions start declining from 2023
    # Sum = E_2023 × (26+25+...+1+0)/27 = E_2023 × (26×27/2)/27 = E_2023 × 13
    
    time_span = END_YEAR - OVERSHOOT_START_YEAR  # 27 years (2023 to 2050)
    
    # Sum of emissions from 2024 to 2050
    projected_cumulative = 0
    for year in range(2024, END_YEAR + 1):
        factor = (END_YEAR - year) / time_span
        projected_cumulative += emissions_2023 * max(factor, 0)
    
    # Remaining budget = Fair share budget - What they will emit
    remaining_budget = country_carbon_budget - projected_cumulative
    
    return remaining_budget, projected_cumulative


def identify_all_countries(scenario_parameters):
    """
    Identify all countries (for creditor/debtor computation).
    
    Args:
        scenario_parameters: DataFrame with scenario data
    
    Returns:
        DataFrame with all countries
    """
    # Filter for countries (not aggregates like WLD, EU, G20)
    countries = scenario_parameters[
        (~scenario_parameters['ISO2'].isin(['WLD', 'EU', 'G20'])) &
        (scenario_parameters['Country'] != 'All')
    ].copy()
    
    return countries


def identify_countries_with_debt(scenario_parameters):
    """
    Identify all countries that will have carbon debt (neutrality year before 2050).
    
    All countries with a valid neutrality year will accumulate debt from that year onwards.
    
    Args:
        scenario_parameters: DataFrame with scenario data
    
    Returns:
        DataFrame with countries and their data
    """
    # Filter for countries (not aggregates like WLD, EU, G20)
    countries = scenario_parameters[
        (~scenario_parameters['ISO2'].isin(['WLD', 'EU', 'G20'])) &
        (scenario_parameters['Country'] != 'All')
    ].copy()
    
    # Convert Neutrality_year to numeric
    countries['Neutrality_year_numeric'] = pd.to_numeric(
        countries['Neutrality_year'], errors='coerce'
    )
    
    # All countries with a valid neutrality year before 2050
    countries_with_debt = countries[
        (countries['Neutrality_year_numeric'].notna()) &
        (countries['Neutrality_year_numeric'] <= END_YEAR)
    ].copy()
    
    return countries_with_debt


# ==============================================================================
# Main Computation Pipeline
# ==============================================================================

def compute_compensation_for_scenario(scenario_params, carbon_prices_full_df, historical_emissions_df):
    """
    Compute compensation for all countries with carbon debt in a given scenario.
    
    Uses full debt computation from neutrality year to 2050.
    - Countries with past neutrality (<=2023): Use historical emissions, then project to 2050
    - Countries with future neutrality (>2023): Project emissions linearly from neutrality to 2050
    
    Args:
        scenario_params: DataFrame filtered for one specific scenario
        carbon_prices_full_df: Full carbon price series (1970-2050)
        historical_emissions_df: Historical emissions data
    
    Returns:
        DataFrame with compensation data for each country
    """
    results = []
    
    for _, row in scenario_params.iterrows():
        country = row['Country']
        iso2 = row['ISO2']
        iso3 = row.get('ISO3', '')
        neutrality_year = row['Neutrality_year_numeric']
        emissions_scope = row['Emissions_scope']
        
        # Get 2023 values
        co2_per_capita = row['Latest_emissions_per_capita_t']
        population = row['Latest_population']
        
        # Skip if missing data
        if pd.isna(co2_per_capita) or pd.isna(population) or co2_per_capita <= 0:
            continue
        
        if pd.isna(neutrality_year):
            continue
        
        # Compute full debt trajectory from neutrality year to 2050
        debt_result = compute_full_debt_trajectory(
            neutrality_year=neutrality_year,
            iso2=iso2,
            emissions_scope=emissions_scope,
            historical_emissions_df=historical_emissions_df,
            carbon_prices_full_df=carbon_prices_full_df,
            co2_per_capita_2023=co2_per_capita,
            population_2023=population
        )
        
        results.append({
            'Country': country,
            'ISO2': iso2,
            'ISO3': iso3,
            'Neutrality_year': int(neutrality_year),
            'CO2_per_capita_2023': co2_per_capita,
            'Population_2023': population,
            # Debt by period
            'Debt_Pre2005_USD': debt_result['Debt_Pre2005_USD'],
            'Debt_ETS_2005_2024_USD': debt_result['Debt_ETS_2005_2024_USD'],
            'Debt_Future_2025_2050_USD': debt_result['Debt_Future_2025_2050_USD'],
            'Debt_Total_Median_USD': debt_result['Debt_Total_Median_USD'],
            'Debt_Total_Q1_USD': debt_result['Debt_Total_Q1_USD'],
            'Debt_Total_Q3_USD': debt_result['Debt_Total_Q3_USD'],
            # Emissions by period
            'Emissions_Pre2005_Mt': debt_result['Emissions_Pre2005_Mt'],
            'Emissions_ETS_2005_2024_Mt': debt_result['Emissions_ETS_2005_2024_Mt'],
            'Emissions_Future_2025_2050_Mt': debt_result['Emissions_Future_2025_2050_Mt'],
            'Emissions_Total_Mt': debt_result['Emissions_Total_Mt'],
            # Scenario info
            'Budget_distribution_scenario': row['Budget_distribution_scenario'],
            'Warming_scenario': row['Warming_scenario'],
            'Probability_of_reach': row['Probability_of_reach'],
            'Emissions_scope': row['Emissions_scope']
        })
    
    return pd.DataFrame(results)


def compute_creditor_shares_for_scenario(scenario_params, debtors_df):
    """
    Compute creditor shares for countries with remaining carbon budget in 2050.
    
    Method:
    1. For each country, compute: Remaining_Budget_2050 = Country_carbon_budget - Latest_cumulative_CO2_emissions_Mt - Projected_emissions_to_2050
    2. Countries with Remaining_Budget_2050 > 0 are creditors
    3. Total debt from debtors is distributed to creditors proportionally to their Remaining_Budget_2050
    
    Args:
        scenario_params: DataFrame filtered for one specific scenario (all countries)
        debtors_df: DataFrame with debt data for debtor countries (same scenario)
    
    Returns:
        DataFrame with creditor data for each country
    """
    # Get total debt from debtors for this scenario
    total_debt = debtors_df['Debt_Total_Median_USD'].sum() if len(debtors_df) > 0 else 0
    total_debt_q1 = debtors_df['Debt_Total_Q1_USD'].sum() if len(debtors_df) > 0 else 0
    total_debt_q3 = debtors_df['Debt_Total_Q3_USD'].sum() if len(debtors_df) > 0 else 0
    
    creditor_data = []
    
    for _, row in scenario_params.iterrows():
        country = row['Country']
        iso2 = row['ISO2']
        iso3 = row.get('ISO3', '')
        
        # Get required values
        co2_per_capita = row['Latest_emissions_per_capita_t']
        population = row['Latest_population']
        country_carbon_budget = row['Country_carbon_budget']
        latest_cumulative = row.get('Latest_cumulative_CO2_emissions_Mt', 0)
        
        # Skip if missing data
        if pd.isna(co2_per_capita) or pd.isna(population) or co2_per_capita <= 0:
            continue
        if pd.isna(country_carbon_budget) or pd.isna(latest_cumulative):
            continue
        
        # Compute remaining budget in 2050
        remaining_budget, projected_emissions = compute_remaining_budget_2050(
            co2_per_capita_2023=co2_per_capita,
            population_2023=population,
            latest_cumulative_emissions=latest_cumulative,
            country_carbon_budget=country_carbon_budget
        )
        
        creditor_data.append({
            'Country': country,
            'ISO2': iso2,
            'ISO3': iso3,
            'Country_Carbon_Budget_Mt': country_carbon_budget,
            'Latest_Cumulative_Emissions_Mt': latest_cumulative,
            'Projected_Emissions_2024_2050_Mt': projected_emissions,
            'Remaining_Budget_2050_Mt': remaining_budget,
            'CO2_per_capita_2023': co2_per_capita,
            'Population_2023': population,
            # Scenario info
            'Budget_distribution_scenario': row['Budget_distribution_scenario'],
            'Warming_scenario': row['Warming_scenario'],
            'Probability_of_reach': row['Probability_of_reach'],
            'Emissions_scope': row['Emissions_scope']
        })
    
    creditor_df = pd.DataFrame(creditor_data)
    
    if len(creditor_df) == 0:
        return creditor_df
    
    # Identify creditors (positive remaining budget)
    creditor_df['Is_Creditor'] = creditor_df['Remaining_Budget_2050_Mt'] > 0
    
    # Calculate total remaining budget for all creditors
    total_remaining_budget = creditor_df[creditor_df['Is_Creditor']]['Remaining_Budget_2050_Mt'].sum()
    
    # Distribute debt proportionally to remaining budget
    if total_remaining_budget > 0:
        creditor_df['Credit_Share'] = creditor_df.apply(
            lambda x: x['Remaining_Budget_2050_Mt'] / total_remaining_budget if x['Is_Creditor'] else 0, 
            axis=1
        )
        creditor_df['Credit_Received_USD'] = creditor_df['Credit_Share'] * total_debt
        creditor_df['Credit_Received_Q1_USD'] = creditor_df['Credit_Share'] * total_debt_q1
        creditor_df['Credit_Received_Q3_USD'] = creditor_df['Credit_Share'] * total_debt_q3
    else:
        creditor_df['Credit_Share'] = 0
        creditor_df['Credit_Received_USD'] = 0
        creditor_df['Credit_Received_Q1_USD'] = 0
        creditor_df['Credit_Received_Q3_USD'] = 0
    
    # Add total debt info for reference
    creditor_df['Total_Debt_Pool_USD'] = total_debt
    
    return creditor_df


def create_debt_map(compensation_df, scenario_name):
    """
    Create a world map showing the total compensation debt per country.
    
    Args:
        compensation_df: DataFrame with compensation data
        scenario_name: Name for the chart title
    
    Returns:
        Plotly figure
    """
    # Convert debt to billions for better readability
    plot_df = compensation_df.copy()
    plot_df['Debt_Billion_USD'] = plot_df['Debt_Total_Median_USD'] / 1e9
    
    fig = px.choropleth(
        plot_df,
        locations='ISO3',
        locationmode='ISO-3',
        color='Debt_Billion_USD',
        hover_name='Country',
        hover_data={
            'ISO3': False,
            'Debt_Billion_USD': ':.2f',
            'Neutrality_year': True,
            'Emissions_Total_Mt': ':.1f',
            'CO2_per_capita_2023': ':.2f',
            'Population_2023': ':,.0f'
        },
        color_continuous_scale='Reds',
        labels={
            'Debt_Billion_USD': 'Total Debt (Billion USD)',
            'Neutrality_year': 'Neutrality Year',
            'Emissions_Total_Mt': 'Total Overshoot Emissions (Mt)',
            'CO2_per_capita_2023': 'CO2/capita 2023 (t)',
            'Population_2023': 'Population 2023'
        }
    )
    
    fig.update_layout(
        title=dict(
            text=f'Compensation for Atmospheric Appropriation<br><sub>{scenario_name} - Total Debt (Neutrality Year to 2050)</sub>',
            font=dict(size=16)
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            bgcolor='rgba(0,0,0,0)'
        ),
        template='plotly_white',
        width=1200,
        height=700,
        coloraxis_colorbar=dict(
            title='Total Debt<br>(Billion USD)',
            tickformat=',.0f'
        )
    )
    
    return fig


def create_top10_stacked_bar_chart(compensation_df, scenario_name):
    """
    Create a stacked bar chart showing debt breakdown for top 10 countries.
    
    Args:
        compensation_df: DataFrame with compensation data
        scenario_name: Name for the chart title
    
    Returns:
        Plotly figure
    """
    # Get top 10 countries by total debt
    top10 = compensation_df.nlargest(10, 'Debt_Total_Median_USD').copy()
    
    # Convert to billions
    top10['Debt_Pre2005_Billion'] = top10['Debt_Pre2005_USD'] / 1e9
    top10['Debt_ETS_Billion'] = top10['Debt_ETS_2005_2024_USD'] / 1e9
    top10['Debt_Future_Billion'] = top10['Debt_Future_2025_2050_USD'] / 1e9
    top10['Debt_Total_Billion'] = top10['Debt_Total_Median_USD'] / 1e9
    
    # Sort by total debt for better visualization
    top10 = top10.sort_values('Debt_Total_Median_USD', ascending=True)
    
    fig = go.Figure()
    
    # Add stacked bars
    fig.add_trace(go.Bar(
        name='Pre-2005 (Reconstituted price)',
        y=top10['Country'],
        x=top10['Debt_Pre2005_Billion'],
        orientation='h',
        marker_color='#FF8C00',  # Orange - matches reconstituted in price chart
        text=[f'${x:.1f}B' if x > 0 else '' for x in top10['Debt_Pre2005_Billion']],
        textposition='inside',
        hovertemplate='%{y}<br>Pre-2005: $%{x:.2f}B<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='2005-2024 (EU ETS price)',
        y=top10['Country'],
        x=top10['Debt_ETS_Billion'],
        orientation='h',
        marker_color='#228B22',  # Green - matches historical ETS in price chart
        text=[f'${x:.1f}B' if x > 0 else '' for x in top10['Debt_ETS_Billion']],
        textposition='inside',
        hovertemplate='%{y}<br>2005-2024: $%{x:.2f}B<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='2025-2050 (Projected price)',
        y=top10['Country'],
        x=top10['Debt_Future_Billion'],
        orientation='h',
        marker_color='#2E86AB',  # Blue - matches projections in price chart
        text=[f'${x:.1f}B' if x > 0 else '' for x in top10['Debt_Future_Billion']],
        textposition='inside',
        hovertemplate='%{y}<br>2025-2050: $%{x:.2f}B<extra></extra>'
    ))
    
    # Add total debt annotation on the right
    for i, row in top10.iterrows():
        fig.add_annotation(
            x=row['Debt_Total_Billion'],
            y=row['Country'],
            text=f"${row['Debt_Total_Billion']:.1f}B",
            showarrow=False,
            xanchor='left',
            xshift=10,
            font=dict(size=10, color='black')
        )
    
    fig.update_layout(
        barmode='stack',
        title=dict(
            text=f'Top 10 Countries by Compensation Debt<br><sub>{scenario_name} - Debt breakdown by period</sub>',
            font=dict(size=16),
            y=0.95  # Position title within visible area
        ),
        xaxis_title='Debt (Billion USD)',
        yaxis_title='',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.08,  # Move legend higher to avoid overlap with subtitle
            xanchor='center',
            x=0.5
        ),
        width=1000,
        height=600,
        margin=dict(r=100, t=120)  # More top margin for both title and legend
    )
    
    return fig


def create_creditor_debtor_map(creditor_df, debtors_df, scenario_name):
    """
    Create a world map showing creditors (green) and debtors (red).
    
    Args:
        creditor_df: DataFrame with creditor data (all countries with remaining budget info)
        debtors_df: DataFrame with debtor data
        scenario_name: Name for the chart title
    
    Returns:
        Plotly figure
    """
    # Merge creditor and debtor data
    # Creditors have Credit_Received_USD > 0
    # Debtors have debt (negative balance)
    
    plot_df = creditor_df.copy()
    
    # Merge with debtors to get debt info
    debtor_info = debtors_df[['ISO2', 'Debt_Total_Median_USD']].copy()
    debtor_info = debtor_info.rename(columns={'Debt_Total_Median_USD': 'Debt_USD'})
    
    plot_df = plot_df.merge(debtor_info, on='ISO2', how='left')
    plot_df['Debt_USD'] = plot_df['Debt_USD'].fillna(0)
    
    # Create balance: positive for creditors, negative for debtors
    # Creditors receive credit, debtors owe debt
    plot_df['Net_Balance_USD'] = plot_df['Credit_Received_USD'] - plot_df['Debt_USD']
    plot_df['Balance_Billion_USD'] = plot_df['Net_Balance_USD'] / 1e9
    
    # Create diverging color scale
    max_abs_balance = max(
        abs(plot_df['Balance_Billion_USD'].min()) if len(plot_df) > 0 else 1, 
        abs(plot_df['Balance_Billion_USD'].max()) if len(plot_df) > 0 else 1
    )
    
    fig = px.choropleth(
        plot_df,
        locations='ISO3',
        locationmode='ISO-3',
        color='Balance_Billion_USD',
        hover_name='Country',
        hover_data={
            'ISO3': False,
            'Balance_Billion_USD': ':.2f',
            'Remaining_Budget_2050_Mt': ':.1f',
            'Credit_Share': ':.2%',
            'Is_Creditor': True
        },
        color_continuous_scale='RdYlGn',  # Red-Yellow-Green diverging scale
        range_color=[-max_abs_balance, max_abs_balance],  # Center at 0
        labels={
            'Balance_Billion_USD': 'Net Balance (Billion USD)',
            'Remaining_Budget_2050_Mt': 'Remaining Budget 2050 (Mt)',
            'Credit_Share': 'Share of Credit Pool',
            'Is_Creditor': 'Creditor'
        }
    )
    
    fig.update_layout(
        title=dict(
            text=f'Carbon Debt Redistribution<br><sub>{scenario_name} - Green = Creditor (receives), Red = Debtor (owes)</sub>',
            font=dict(size=16)
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth',
            bgcolor='rgba(0,0,0,0)'
        ),
        template='plotly_white',
        width=1200,
        height=700,
        coloraxis_colorbar=dict(
            title='Net Balance<br>(Billion USD)',
            tickformat=',.0f'
        )
    )
    
    return fig


def create_creditor_debtor_bar_chart(creditor_df, debtors_df, scenario_name):
    """
    Create a bar chart showing top debtors and creditors.
    
    Args:
        creditor_df: DataFrame with creditor data
        debtors_df: DataFrame with debtor data  
        scenario_name: Name for the chart title
    
    Returns:
        Plotly figure
    """
    # Get top 10 creditors
    top_creditors = creditor_df[creditor_df['Is_Creditor']].nlargest(10, 'Credit_Received_USD').copy()
    top_creditors['Balance_Billion_USD'] = top_creditors['Credit_Received_USD'] / 1e9
    top_creditors['Type'] = 'Creditor'
    
    # Get top 10 debtors
    top_debtors = debtors_df.nlargest(10, 'Debt_Total_Median_USD').copy()
    top_debtors['Balance_Billion_USD'] = -top_debtors['Debt_Total_Median_USD'] / 1e9  # Negative for debtors
    top_debtors['Type'] = 'Debtor'
    
    # Combine
    combined = pd.concat([
        top_debtors[['Country', 'ISO2', 'Balance_Billion_USD', 'Type']],
        top_creditors[['Country', 'ISO2', 'Balance_Billion_USD', 'Type']]
    ], ignore_index=True)
    
    combined = combined.sort_values('Balance_Billion_USD', ascending=True)
    
    # Create colors based on type
    colors = ['#d62728' if t == 'Debtor' else '#2ca02c' for t in combined['Type']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=combined['Country'],
        x=combined['Balance_Billion_USD'],
        orientation='h',
        marker_color=colors,
        text=[f'${abs(x):.1f}B' for x in combined['Balance_Billion_USD']],
        textposition='inside',  # Put text inside bars to avoid overlap
        insidetextanchor='middle',
        textfont=dict(color='white', size=11),
        hovertemplate='%{y}<br>Balance: $%{x:.2f}B<extra></extra>'
    ))
    
    # Add vertical line at 0
    fig.add_vline(x=0, line_width=2, line_dash="solid", line_color="black")
    
    # Calculate x-axis range with padding for labels
    x_min = combined['Balance_Billion_USD'].min()
    x_max = combined['Balance_Billion_USD'].max()
    x_padding = max(abs(x_min), abs(x_max)) * 0.15  # 15% padding
    
    # Add annotations for DEBTOR and CREDITOR zones
    fig.add_annotation(
        x=(x_min - x_padding) * 0.6, y=1.08, text="← DEBTORS (owe)", showarrow=False,
        xref='x', yref='paper', font=dict(size=14, color='#d62728', weight='bold')
    )
    fig.add_annotation(
        x=(x_max + x_padding) * 0.6, y=1.08, text="CREDITORS (receive) →", showarrow=False,
        xref='x', yref='paper', font=dict(size=14, color='#2ca02c', weight='bold')
    )
    
    fig.update_layout(
        title=dict(
            text=f'Top 10 Debtors & Creditors<br><sub>{scenario_name} - Debt redistributed proportionally to remaining carbon budget</sub>',
            font=dict(size=16),
            y=0.95
        ),
        xaxis_title='Net Balance (Billion USD)',
        xaxis=dict(range=[x_min - x_padding, x_max + x_padding]),  # Add padding to x-axis
        yaxis_title='',
        template='plotly_white',
        width=1200,
        height=700,
        margin=dict(r=50, l=180, t=140),  # More left margin for country names, less right
        showlegend=False
    )
    
    return fig


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("COMPENSATION FOR ATMOSPHERIC APPROPRIATION COMPUTATION")
    print("=" * 70)
    
    # Step 1: Interpolate carbon prices (full series 1970-2050)
    print("\n1. Creating full carbon price series (1970-2050)...")
    carbon_prices_full_df = interpolate_carbon_prices_full()
    print(f"   Total years: {len(carbon_prices_full_df)}")
    print(f"   Data types: {carbon_prices_full_df['Data_type'].value_counts().to_dict()}")
    
    # Save full carbon prices
    carbon_prices_full_df.to_csv(os.path.join(OUTPUT_DIR, 'carbon_prices_full_1970_2050.csv'), index=False)
    
    # Plot full carbon price history
    print("\n2. Creating carbon price visualization (1970-2050)...")
    plot_carbon_prices(carbon_prices_full_df)
    
    # Step 2: Load historical emissions
    print("\n3. Loading historical emissions data...")
    historical_emissions_df = load_historical_emissions()
    print(f"   Countries: {historical_emissions_df['ISO2'].nunique()}")
    print(f"   Years: {historical_emissions_df['Year'].min()} - {historical_emissions_df['Year'].max()}")
    
    # Step 3: Load scenario parameters
    print("\n4. Loading scenario parameters...")
    scenario_parameters = pd.read_csv(os.path.join(DATA_DIR, 'scenario_parameters.csv'))
    
    # Add ISO3 if missing
    if 'ISO3' not in scenario_parameters.columns:
        combined_data = pd.read_csv(os.path.join(DATA_DIR, 'combined_data.csv'))
        if 'ISO2' in combined_data.columns and 'ISO3' in combined_data.columns:
            iso_mapping = combined_data[['ISO2', 'ISO3']].drop_duplicates()
            iso2_to_iso3 = dict(zip(iso_mapping['ISO2'], iso_mapping['ISO3']))
            scenario_parameters['ISO3'] = scenario_parameters['ISO2'].map(iso2_to_iso3)
    
    # Filter out 'Population' budget distribution scenario (as done in app.py)
    scenario_parameters = scenario_parameters[
        scenario_parameters['Budget_distribution_scenario'] != 'Population'
    ].copy()
    
    # Step 4: Identify countries with debt
    print("\n5. Identifying countries with carbon debt...")
    countries_with_debt = identify_countries_with_debt(scenario_parameters)
    print(f"   Found {countries_with_debt['ISO2'].nunique()} unique countries with debt in at least one scenario")
    
    # Count past vs future overshoot
    past_overshoot = countries_with_debt[countries_with_debt['Neutrality_year_numeric'] <= OVERSHOOT_START_YEAR]['ISO2'].nunique()
    future_overshoot = countries_with_debt[countries_with_debt['Neutrality_year_numeric'] > OVERSHOOT_START_YEAR]['ISO2'].nunique()
    print(f"     - Already overshot (neutrality <= 2023): {past_overshoot} countries")
    print(f"     - Will overshoot (neutrality > 2023): {future_overshoot} countries")
    
    # Step 5: Compute compensation for each scenario
    print("\n6. Computing compensation for each scenario...")
    
    # Define scenarios (matching app.py)
    budget_scenarios = ['Responsibility', 'Capability']
    probability_scenarios = scenario_parameters['Probability_of_reach'].unique()
    warming_scenarios = scenario_parameters['Warming_scenario'].unique()
    emissions_scopes = scenario_parameters['Emissions_scope'].unique()
    
    all_compensation_results = []
    
    for budget_dist in budget_scenarios:
        for probability in probability_scenarios:
            for warming in warming_scenarios:
                for scope in emissions_scopes:
                    # Filter for this specific scenario
                    scenario_mask = (
                        (countries_with_debt['Budget_distribution_scenario'] == budget_dist) &
                        (countries_with_debt['Probability_of_reach'] == probability) &
                        (countries_with_debt['Warming_scenario'] == warming) &
                        (countries_with_debt['Emissions_scope'] == scope)
                    )
                    
                    scenario_data = countries_with_debt[scenario_mask]
                    
                    if len(scenario_data) == 0:
                        continue
                    
                    scenario_name = f"{budget_dist}_{probability}_{warming}_{scope}"
                    print(f"   Processing: {scenario_name} ({len(scenario_data)} countries)")
                    
                    # Compute compensation with full historical + future debt
                    compensation = compute_compensation_for_scenario(
                        scenario_data, 
                        carbon_prices_full_df,
                        historical_emissions_df
                    )
                    
                    if len(compensation) > 0:
                        all_compensation_results.append(compensation)
                        
                        # Create visualizations for this scenario
                        safe_name = scenario_name.replace('°', '').replace('%', 'pct')
                        
                        # Create map for this scenario
                        fig_map = create_debt_map(compensation, scenario_name)
                        fig_map.write_html(os.path.join(OUTPUT_DIR, f'debt_map_{safe_name}.html'))
                        fig_map.write_image(os.path.join(OUTPUT_DIR, f'debt_map_{safe_name}.png'), scale=2)
                        
                        # Create stacked bar chart for top 10 countries
                        fig_bar = create_top10_stacked_bar_chart(compensation, scenario_name)
                        fig_bar.write_html(os.path.join(OUTPUT_DIR, f'debt_top10_{safe_name}.html'))
                        fig_bar.write_image(os.path.join(OUTPUT_DIR, f'debt_top10_{safe_name}.png'), scale=2)
    
    # Step 6: Combine all results and save
    print("\n7. Saving results...")
    if all_compensation_results:
        combined_compensation = pd.concat(all_compensation_results, ignore_index=True)
        combined_compensation.to_csv(os.path.join(OUTPUT_DIR, 'compensation_all_scenarios.csv'), index=False)
        
        # Create summary statistics
        summary = combined_compensation.groupby(
            ['Budget_distribution_scenario', 'Warming_scenario', 'Probability_of_reach', 'Emissions_scope']
        ).agg({
            'Country': 'count',
            'Debt_Total_Median_USD': ['sum', 'mean', 'max'],
            'Emissions_Total_Mt': ['sum', 'mean']
        }).round(2)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary.to_csv(os.path.join(OUTPUT_DIR, 'compensation_summary.csv'))
        
        print(f"\n   Total compensation records: {len(combined_compensation)}")
        print(f"   Unique countries with debt: {combined_compensation['ISO2'].nunique()}")
        print(f"   Total debt (Median, all scenarios): ${combined_compensation['Debt_Total_Median_USD'].sum()/1e12:.2f} trillion USD")
        
        # Print debt breakdown summary
        print("\n   Debt breakdown by period (across all scenarios):")
        print(f"     Pre-2005:   ${combined_compensation['Debt_Pre2005_USD'].sum()/1e12:.2f} trillion USD")
        print(f"     2005-2024:  ${combined_compensation['Debt_ETS_2005_2024_USD'].sum()/1e12:.2f} trillion USD")
        print(f"     2025-2050:  ${combined_compensation['Debt_Future_2025_2050_USD'].sum()/1e12:.2f} trillion USD")
    
    # ==========================================================================
    # Step 7: Compute Creditor Shares - Redistribute debt proportionally
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CREDITOR COMPUTATION - DEBT REDISTRIBUTION")
    print("=" * 70)
    
    print("\n8. Computing creditor shares for all countries...")
    all_countries = identify_all_countries(scenario_parameters)
    print(f"   Found {all_countries['ISO2'].nunique()} unique countries")
    
    all_creditor_results = []
    
    for budget_dist in budget_scenarios:
        for probability in probability_scenarios:
            for warming in warming_scenarios:
                for scope in emissions_scopes:
                    # Filter for this specific scenario
                    scenario_mask = (
                        (all_countries['Budget_distribution_scenario'] == budget_dist) &
                        (all_countries['Probability_of_reach'] == probability) &
                        (all_countries['Warming_scenario'] == warming) &
                        (all_countries['Emissions_scope'] == scope)
                    )
                    
                    scenario_data = all_countries[scenario_mask]
                    
                    if len(scenario_data) == 0:
                        continue
                    
                    scenario_name = f"{budget_dist}_{probability}_{warming}_{scope}"
                    
                    # Get corresponding debtor data for this scenario
                    debtor_mask = (
                        (combined_compensation['Budget_distribution_scenario'] == budget_dist) &
                        (combined_compensation['Probability_of_reach'] == probability) &
                        (combined_compensation['Warming_scenario'] == warming) &
                        (combined_compensation['Emissions_scope'] == scope)
                    ) if combined_compensation is not None else pd.Series([False])
                    
                    debtors_scenario = combined_compensation[debtor_mask] if combined_compensation is not None else pd.DataFrame()
                    
                    print(f"   Processing: {scenario_name} ({len(scenario_data)} countries, {len(debtors_scenario)} debtors)")
                    
                    # Compute creditor shares based on remaining budget
                    creditor_df = compute_creditor_shares_for_scenario(scenario_data, debtors_scenario)
                    
                    if len(creditor_df) > 0:
                        all_creditor_results.append(creditor_df)
                        
                        # Count creditors
                        n_creditors = creditor_df['Is_Creditor'].sum()
                        total_debt = debtors_scenario['Debt_Total_Median_USD'].sum() / 1e9 if len(debtors_scenario) > 0 else 0
                        print(f"     -> {n_creditors} creditors to receive ${total_debt:.1f}B total debt")
                        
                        # Create visualizations for this scenario
                        safe_name = scenario_name.replace('°', '').replace('%', 'pct')
                        
                        # Create creditor/debtor map
                        fig_map = create_creditor_debtor_map(creditor_df, debtors_scenario, scenario_name)
                        fig_map.write_html(os.path.join(OUTPUT_DIR, f'creditor_debtor_map_{safe_name}.html'))
                        fig_map.write_image(os.path.join(OUTPUT_DIR, f'creditor_debtor_map_{safe_name}.png'), scale=2)
                        
                        # Create bar chart for top debtors and creditors
                        fig_bar = create_creditor_debtor_bar_chart(creditor_df, debtors_scenario, scenario_name)
                        fig_bar.write_html(os.path.join(OUTPUT_DIR, f'creditor_debtor_top20_{safe_name}.html'))
                        fig_bar.write_image(os.path.join(OUTPUT_DIR, f'creditor_debtor_top20_{safe_name}.png'), scale=2)
    
    # Save creditor results
    print("\n9. Saving creditor share results...")
    combined_creditors = None
    if all_creditor_results:
        combined_creditors = pd.concat(all_creditor_results, ignore_index=True)
        combined_creditors.to_csv(os.path.join(OUTPUT_DIR, 'creditor_shares_all_scenarios.csv'), index=False)
        
        # Create summary
        creditors_only = combined_creditors[combined_creditors['Is_Creditor']]
        
        print(f"\n   Total records: {len(combined_creditors)}")
        print(f"   Unique countries: {combined_creditors['ISO2'].nunique()}")
        print(f"   Countries with positive remaining budget (creditors): {creditors_only['ISO2'].nunique()}")
        print(f"\n   Creditor summary (all scenarios combined):")
        print(f"     Total remaining budget:  {creditors_only['Remaining_Budget_2050_Mt'].sum()/1e3:.2f} Gt CO2")
        print(f"     Total credit received:   ${creditors_only['Credit_Received_USD'].sum()/1e12:.2f} trillion USD")
    
    print("\n" + "=" * 70)
    print("COMPUTATION COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    return carbon_prices_full_df, combined_compensation if all_compensation_results else None, combined_creditors


if __name__ == '__main__':
    carbon_prices, compensation_data, balance_data = main()
