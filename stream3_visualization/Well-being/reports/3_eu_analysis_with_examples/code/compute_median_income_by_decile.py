"""
compute_median_income_by_decile.py

Standalone script to compute the weighted median equivalized disposable income
per (country, year, decile) from the EU-SILC household data with assigned deciles.

Input:  output/0_raw_data_EUROSTAT/0_EU-SILC/1_income_decile/EU_SILC_household_data_with_decile.csv
Output: output/median_income_by_decile.csv

This is slow (~minutes) on the full 4.9M row dataset, so the result is saved
as a CSV to avoid recomputation.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def weighted_quantile(values, weights, quantiles):
    """Compute weighted quantiles (same logic as 0_raw_indicator_EU-SILC.py)."""
    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return np.full(len(quantiles), np.nan)
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = weights[sorter]
    cumsum_weights = np.cumsum(weights_sorted)
    total_weight = cumsum_weights[-1]
    if total_weight == 0:
        return np.full(len(quantiles), np.nan)
    normalized_weights = cumsum_weights / total_weight
    return np.interp(quantiles, normalized_weights, values_sorted)


def main():
    # Resolve paths relative to this script (reports/3_eu_analysis_with_examples/code/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.abspath(os.path.join(script_dir, '..'))
    well_being_dir = os.path.abspath(os.path.join(report_dir, '..', '..'))

    input_path = os.path.join(
        well_being_dir, 'output', '0_raw_data_EUROSTAT', '0_EU-SILC',
        '1_income_decile', 'EU_SILC_household_data_with_decile.csv'
    )

    output_dir = os.path.join(report_dir, 'outputs', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'median_income_by_decile.csv')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading household income data from:\n  {input_path}")
    df = pd.read_csv(
        input_path,
        usecols=['HB010', 'HB020', 'equi_disp_inc', 'DB090', 'decile'],
        low_memory=False
    )
    print(f"  Loaded {len(df):,} household records")

    # Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=['equi_disp_inc', 'DB090', 'decile'])
    print(f"  Dropped {before - len(df):,} rows with missing data, {len(df):,} remaining")
    df['decile'] = df['decile'].astype(int)

    # Compute weighted median per (country, year, decile)
    print("Computing weighted median income per (country, year, decile)...")
    groups = df.groupby(['HB020', 'HB010', 'decile'])
    results = []

    for (country, year, decile), group in tqdm(groups, desc="Processing groups"):
        median_inc = weighted_quantile(
            group['equi_disp_inc'].to_numpy(),
            group['DB090'].to_numpy(),
            np.array([0.5])
        )[0]
        results.append({
            'country': country,
            'year': int(year),
            'decile': int(decile),
            'median_equi_disp_inc': median_inc
        })

    median_df = pd.DataFrame(results)
    median_df = median_df.sort_values(['country', 'year', 'decile']).reset_index(drop=True)

    print(f"\nComputed {len(median_df)} (country, year, decile) combinations")
    print(f"  Countries: {sorted(median_df['country'].unique())}")
    print(f"  Years: {sorted(median_df['year'].unique())}")

    median_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # ------------------------------------------------------------------
    # Country-level median income (regardless of decile)
    # ------------------------------------------------------------------
    print("\nComputing weighted median income per (country, year)...")
    country_groups = df.groupby(['HB020', 'HB010'])
    country_results = []

    for (country, year), group in tqdm(country_groups, desc="Country-level median"):
        median_inc = weighted_quantile(
            group['equi_disp_inc'].to_numpy(),
            group['DB090'].to_numpy(),
            np.array([0.5])
        )[0]
        country_results.append({
            'country': country,
            'year': int(year),
            'median_equi_disp_inc': median_inc
        })

    country_median_df = pd.DataFrame(country_results)
    country_median_df = country_median_df.sort_values(['country', 'year']).reset_index(drop=True)

    country_output_path = os.path.join(output_dir, 'median_income_by_country.csv')
    country_median_df.to_csv(country_output_path, index=False)
    print(f"Computed {len(country_median_df)} (country, year) combinations")
    print(f"Saved to: {country_output_path}")


if __name__ == "__main__":
    main()
