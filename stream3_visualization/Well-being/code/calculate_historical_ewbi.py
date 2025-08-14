import pandas as pd
import json
import os
import numpy as np

# Get the absolute path to the data directory
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))

def load_ewbi_structure():
    """Load the EWBI structure from JSON"""
    with open(os.path.join(DATA_DIR, '..', 'data', 'ewbi_indicators.json'), 'r') as f:
        return json.load(f)['EWBI']

def calculate_historical_ewbi():
    """Calculate historical EWBI scores for all countries over time"""
    
    print("Loading data...")
    
    # Load the time series data
    time_series_df = pd.read_csv(os.path.join(DATA_DIR, 'master_dataframe_time_series.csv'))
    
    # Load the EWBI structure
    ewbi_structure = load_ewbi_structure()
    
    print(f"Loaded {len(time_series_df)} time series records")
    print(f"Available years: {sorted(time_series_df['year'].unique())}")
    print(f"Available countries: {len(time_series_df['country'].unique())}")
    
    # Create a mapping from primary indicators to EU priorities
    primary_to_eu_priority = {}
    for priority in ewbi_structure:
        for component in priority['components']:
            for indicator in component['indicators']:
                primary_to_eu_priority[indicator['code']] = priority['name']
    
    print(f"Created mapping for {len(primary_to_eu_priority)} primary indicators")
    
    # Calculate historical EWBI scores
    print("Calculating historical EWBI scores...")
    
    # Group by country, year, and decile
    historical_ewbi = []
    
    for country in time_series_df['country'].unique():
        print(f"Processing {country}...")
        
        country_data = time_series_df[time_series_df['country'] == country]
        
        for year in sorted(country_data['year'].unique()):
            year_data = country_data[country_data['year'] == year]
            
            for decile in sorted(year_data['decile'].unique()):
                decile_data = year_data[year_data['decile'] == decile]
                
                # Calculate EU priority scores for this decile and year
                eu_priority_scores = {}
                
                for priority in ewbi_structure:
                    priority_name = priority['name']
                    priority_indicators = []
                    
                    # Get all primary indicators for this priority
                    for component in priority['components']:
                        for indicator in component['indicators']:
                            indicator_code = indicator['code']
                            if indicator_code in decile_data['primary_index'].values:
                                indicator_values = decile_data[decile_data['primary_index'] == indicator_code]['value'].values
                                if len(indicator_values) > 0:
                                    # Take the mean if multiple values exist
                                    priority_indicators.extend(indicator_values)
                    
                    if priority_indicators:
                        # Calculate the score for this priority (geometric mean)
                        eu_priority_scores[priority_name] = np.exp(np.mean(np.log(priority_indicators)))
                
                # Calculate overall EWBI score (geometric mean of EU priorities)
                if eu_priority_scores:
                    ewbi_score = np.exp(np.mean(np.log(list(eu_priority_scores.values()))))
                    
                    historical_ewbi.append({
                        'country': country,
                        'year': year,
                        'decile': decile,
                        'ewbi_score': ewbi_score,
                        **eu_priority_scores
                    })
    
    # Convert to DataFrame
    historical_ewbi_df = pd.DataFrame(historical_ewbi)
    
    print(f"Calculated {len(historical_ewbi_df)} historical EWBI records")
    
    # Save the historical EWBI data
    output_file = os.path.join(DATA_DIR, 'historical_ewbi_scores.csv')
    historical_ewbi_df.to_csv(output_file, index=False)
    print(f"Saved historical EWBI scores to: {output_file}")
    
    # Also create a simplified version with just country-year averages (no deciles)
    country_year_ewbi = historical_ewbi_df.groupby(['country', 'year']).agg({
        'ewbi_score': 'mean'
    }).reset_index()
    
    # Add EU Countries Average
    eu_countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    
    eu_ewbi = historical_ewbi_df[historical_ewbi_df['country'].isin(eu_countries)].groupby('year')['ewbi_score'].mean().reset_index()
    eu_ewbi['country'] = 'EU Countries Average'
    
    # Combine individual countries and EU average
    final_ewbi = pd.concat([country_year_ewbi, eu_ewbi], ignore_index=True)
    
    # Save the simplified version
    simplified_file = os.path.join(DATA_DIR, 'ewbi_evolution_over_time.csv')
    final_ewbi.to_csv(simplified_file, index=False)
    print(f"Saved simplified EWBI evolution to: {simplified_file}")
    
    print("\nSummary:")
    print(f"- Historical EWBI scores: {len(historical_ewbi_df)} records")
    print(f"- Simplified evolution: {len(final_ewbi)} records")
    print(f"- Years covered: {sorted(final_ewbi['year'].unique())}")
    print(f"- Countries: {sorted(final_ewbi['country'].unique())}")
    
    return historical_ewbi_df, final_ewbi

if __name__ == "__main__":
    historical_ewbi, simplified_ewbi = calculate_historical_ewbi() 