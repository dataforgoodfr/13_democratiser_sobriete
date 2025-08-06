import pandas as pd
import os

def get_top_20_countries_from_1750():
    """
    This script loads the historical responsibility data (from 1750) and
    identifies the top 20 countries with the earliest overshoot year.
    """
    # Define the path to the historical responsibility file
    output_directory = '/Users/louistronel/Desktop/D4G_WSL/13_democratiser_sobriete-1/stream3_visualization/Budget/Output'
    historical_responsibility_path = os.path.join(output_directory, 'historical_responsibility_1750.csv')

    # Load the historical responsibility data
    try:
        historical_df = pd.read_csv(historical_responsibility_path)
        print("Historical responsibility data (from 1750) loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file {historical_responsibility_path} was not found.")
        print("Please run the 'historical_responsibility.py' script first.")
        return

    # --- Analysis ---
    print("\\n--- Top 20 Countries with Earliest Overshoot Year (from 1750) ---")
    
    # Exclude aggregate rows
    countries_only = historical_df[
        ~historical_df['ISO2'].isin(['WLD', 'EU', 'G20']) & 
        (historical_df['Country'] != 'All')
    ].copy()

    # Convert 'Overshoot_year' to numeric, coercing errors for "Not overshot yet" to NaN
    countries_only['Overshoot_year'] = pd.to_numeric(countries_only['Overshoot_year'], errors='coerce')

    # Drop countries that have not overshot their budget yet
    overshot_countries = countries_only.dropna(subset=['Overshoot_year']).copy()
    overshot_countries['Overshoot_year'] = overshot_countries['Overshoot_year'].astype(int)

    # Sort by the 'Overshoot_year' and get the top 20
    top_20 = overshot_countries.sort_values('Overshoot_year').head(20)

    # Print the result
    print(top_20[['Country', 'Overshoot_year']].to_string(index=False))

if __name__ == "__main__":
    get_top_20_countries_from_1750()
