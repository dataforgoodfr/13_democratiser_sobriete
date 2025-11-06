"""
6_graphs.py - Generate 2D scatter plots for EWBI analysis

This script creates 2D graphs showing the relationship between interdecile ratio 
(ratio of decile 10 to decile 1) and EWBI scores for:
- Level 1 data: Overall EWBI
- Level 2 data: Each EU priority

The graphs plot all countries and EU-27 with the same colors as used in the dashboard.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np

# Color palette consistent with the dashboard
COUNTRY_COLORS = ['#ffd558', '#fb8072', '#b3de69', '#fdb462', '#bebada', '#8dd3c7', '#ffffb3']
EU_27_COLOR = '#80b1d3'

# ISO-2 to full country names mapping (same as in app.py)
ISO2_TO_FULL_NAME = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CH': 'Switzerland', 'CY': 'Cyprus', 'CZ': 'Czech Republic',
    'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland',
    'FR': 'France', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IS': 'Iceland', 'IT': 'Italy',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta', 'NL': 'Netherlands', 'NO': 'Norway',
    'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia', 'SE': 'Sweden', 'SI': 'Slovenia',
    'SK': 'Slovakia', 'UK': 'United Kingdom'
}

def get_country_color(country, all_countries):
    """Assign consistent colors to countries across all charts (same as app.py)"""
    if country == 'All Countries' or country == 'EU-27':
        return EU_27_COLOR
    
    # Get all individual countries (excluding EU-27)
    individual_countries = [c for c in all_countries if c not in ['All Countries', 'EU-27']]
    individual_countries.sort()  # Sort for consistent ordering
    
    if country in individual_countries:
        index = individual_countries.index(country) % len(COUNTRY_COLORS)
        return COUNTRY_COLORS[index]
    
    return COUNTRY_COLORS[0]  # Default color

def load_data():
    """Load the unified PCA data"""
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'output'))
    
    data_path = os.path.join(data_dir, 'unified_all_levels_1_to_5_pca_weighted.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded data with shape: {df.shape}")
    return df

def calculate_interdecile_ratio(df, level, eu_priority=None, year=None):
    """
    Calculate the interdecile ratio (decile 10 / decile 1) for each country
    
    Parameters:
    - df: DataFrame with the data
    - level: 1 for EWBI, 2 for EU priorities
    - eu_priority: EU priority name (required for level 2)
    - year: Year to use (if None, uses latest available year)
    
    Returns:
    - DataFrame with countries, interdecile ratios, and EWBI/priority scores
    """
    
    # Filter data based on level
    if level == 1:
        # Level 1: EWBI overall
        data = df[
            (df['Level'] == 1) & 
            (df['Country'] != 'All Countries') &  # Exclude EU-27 aggregate for ratio calculation
            (df['Decile'].isin(['1.0', '10.0']))  # Only deciles 1 and 10
        ].copy()
        score_name = 'EWBI Score'
    elif level == 2:
        # Level 2: Specific EU priority
        if eu_priority is None:
            raise ValueError("eu_priority must be specified for level 2")
        data = df[
            (df['Level'] == 2) & 
            (df['EU priority'] == eu_priority) &
            (df['Country'] != 'All Countries') &  # Exclude EU-27 aggregate for ratio calculation
            (df['Decile'].isin(['1.0', '10.0']))  # Only deciles 1 and 10
        ].copy()
        score_name = f'{eu_priority} Score'
    else:
        raise ValueError("Level must be 1 or 2")
    
    if data.empty:
        print(f"No data found for level {level}, EU priority: {eu_priority}")
        return pd.DataFrame()
    
    # Use latest year if not specified
    if year is None:
        year = data['Year'].max()
    
    # Filter by year
    data = data[data['Year'] == year].copy()
    
    if data.empty:
        print(f"No data found for year {year}")
        return pd.DataFrame()
    
    print(f"Calculating interdecile ratios for level {level}, year {year}")
    if level == 2:
        print(f"EU priority: {eu_priority}")
    
    # Calculate interdecile ratios
    ratios = []
    
    for country in data['Country'].unique():
        country_data = data[data['Country'] == country]
        
        decile_1 = country_data[country_data['Decile'] == '1.0']['Value']
        decile_10 = country_data[country_data['Decile'] == '10.0']['Value']
        
        if len(decile_1) > 0 and len(decile_10) > 0:
            decile_1_val = decile_1.iloc[0]
            decile_10_val = decile_10.iloc[0]
            
            # Avoid division by zero
            if decile_1_val > 0:
                ratio = decile_10_val / decile_1_val
                
                # Get the "All" decile value for the overall score
                all_decile_data = df[
                    (df['Level'] == level) & 
                    (df['Country'] == country) &
                    (df['Year'] == year) &
                    (df['Decile'] == 'All')
                ]
                
                if level == 2:
                    all_decile_data = all_decile_data[all_decile_data['EU priority'] == eu_priority]
                
                if not all_decile_data.empty:
                    overall_score = all_decile_data['Value'].iloc[0]
                    
                    ratios.append({
                        'Country': country,
                        'Country_Name': ISO2_TO_FULL_NAME.get(country, country),
                        'Interdecile_Ratio': ratio,
                        score_name: overall_score,
                        'Year': year,
                        'Decile_1': decile_1_val,
                        'Decile_10': decile_10_val
                    })
    
    # Add EU-27 data
    eu_data = df[
        (df['Level'] == level) & 
        (df['Country'] == 'All Countries') &
        (df['Year'] == year) &
        (df['Decile'].isin(['1.0', '10.0', 'All'])) &
        (df['Aggregation'] == 'Population-weighted geometric mean')
    ]
    
    if level == 2:
        eu_data = eu_data[eu_data['EU priority'] == eu_priority]
    
    if not eu_data.empty:
        eu_decile_1 = eu_data[eu_data['Decile'] == '1.0']['Value']
        eu_decile_10 = eu_data[eu_data['Decile'] == '10.0']['Value']
        eu_all = eu_data[eu_data['Decile'] == 'All']['Value']
        
        if len(eu_decile_1) > 0 and len(eu_decile_10) > 0 and len(eu_all) > 0:
            eu_decile_1_val = eu_decile_1.iloc[0]
            eu_decile_10_val = eu_decile_10.iloc[0]
            eu_all_val = eu_all.iloc[0]
            
            if eu_decile_1_val > 0:
                eu_ratio = eu_decile_10_val / eu_decile_1_val
                
                ratios.append({
                    'Country': 'EU-27',
                    'Country_Name': 'EU-27',
                    'Interdecile_Ratio': eu_ratio,
                    score_name: eu_all_val,
                    'Year': year,
                    'Decile_1': eu_decile_1_val,
                    'Decile_10': eu_decile_10_val
                })
    
    result_df = pd.DataFrame(ratios)
    print(f"Calculated {len(result_df)} interdecile ratios")
    
    return result_df

def create_scatter_plot(ratios_df, title, x_label, y_label, output_path):
    """Create a 2D scatter plot"""
    
    if ratios_df.empty:
        print(f"No data to plot for {title}")
        return
    
    # Get all countries for consistent coloring
    all_countries = ratios_df['Country'].unique().tolist()
    
    # Create the scatter plot
    fig = go.Figure()
    
    # Add points for each country
    for _, row in ratios_df.iterrows():
        country = row['Country']
        color = get_country_color(country, all_countries)
        
        # Special marker for EU-27, make all dots twice as large
        marker_symbol = 'diamond' if country == 'EU-27' else 'circle'
        marker_size = 24 if country == 'EU-27' else 16  # Doubled from 12/8 to 24/16
        
        # Use ISO2 code for display (country code directly)
        display_name = country  # This will show "FR", "DE", "EU-27", etc.
        
        fig.add_trace(go.Scatter(
            x=[row['Interdecile_Ratio']],
            y=[row[y_label]],
            mode='markers+text',
            name=row['Country_Name'],  # Keep full name for legend
            text=[display_name],  # Show ISO2 code on the graph
            textposition="top center",  # Position text above the marker to avoid overlap
            textfont=dict(size=10, color='black', family="Arial, sans-serif", weight="bold"),
            marker=dict(
                color=color,
                size=marker_size,
                symbol=marker_symbol,
                line=dict(width=2, color='black') if country == 'EU-27' else dict(width=1, color='white')
            ),
            hovertemplate=f'<b>{row["Country_Name"]}</b><br>' +
                         f'Interdecile Ratio: {row["Interdecile_Ratio"]:.2f}<br>' +
                         f'{y_label}: {row[y_label]:.3f}<br>' +
                         f'Year: {int(row["Year"])}<extra></extra>',
            showlegend=False  # Remove legend as requested
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color="#f4d03f", weight="bold", family="Arial, sans-serif"),
            x=0.5
        ),
        xaxis=dict(
            title=x_label,
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=12, family="Arial, sans-serif"),
            dtick=0.5,  # Grid every 0.5 for interdecile ratio
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            title=y_label,
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=12, family="Arial, sans-serif"),
            dtick=0.1,  # Grid every 0.1 for score
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        showlegend=False,  # Remove legend completely
        margin=dict(t=80, b=60, l=60, r=40)  # Adjust margins since no legend needed
    )
    
    # Save the plot
    fig.write_html(output_path)
    print(f"Saved plot: {output_path}")
    
    # Also save as PNG
    png_path = output_path.replace('.html', '.png')
    fig.write_image(png_path, width=1200, height=800, scale=2)
    print(f"Saved plot: {png_path}")

def main():
    """Main function to generate all graphs"""
    
    print("Starting 6_graphs.py - Generating 2D scatter plots")
    print("=" * 60)
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Get output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(current_dir, '..', 'output', 'Graphs'))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get latest year
    latest_year = df['Year'].max()
    print(f"Using latest year: {int(latest_year)}")
    
    # 1. Level 1 - Overall EWBI
    print("\n1. Generating Level 1 (EWBI Overall) scatter plot...")
    level1_ratios = calculate_interdecile_ratio(df, level=1, year=latest_year)
    
    if not level1_ratios.empty:
        title = f"EWBI vs Interdecile Ratio - All Countries ({int(latest_year)})"
        output_path = os.path.join(output_dir, f"ewbi_vs_interdecile_ratio_{int(latest_year)}.html")
        
        create_scatter_plot(
            level1_ratios,
            title=title,
            x_label="Interdecile Ratio (Decile 10 / Decile 1)",
            y_label="EWBI Score",
            output_path=output_path
        )
    
    # 2. Level 2 - Each EU Priority
    print("\n2. Generating Level 2 (EU Priorities) scatter plots...")
    
    # Get EU priorities
    eu_priorities = df[df['Level'] == 2]['EU priority'].unique()
    eu_priorities = [ep for ep in eu_priorities if pd.notna(ep)]
    
    for i, eu_priority in enumerate(eu_priorities, 1):
        print(f"\n2.{i}. Processing EU Priority: {eu_priority}")
        
        level2_ratios = calculate_interdecile_ratio(df, level=2, eu_priority=eu_priority, year=latest_year)
        
        if not level2_ratios.empty:
            title = f"{eu_priority} vs Interdecile Ratio - All Countries ({int(latest_year)})"
            
            # Create safe filename
            safe_name = eu_priority.replace(' ', '_').replace(',', '').replace('/', '_').lower()
            output_path = os.path.join(output_dir, f"{safe_name}_vs_interdecile_ratio_{int(latest_year)}.html")
            
            create_scatter_plot(
                level2_ratios,
                title=title,
                x_label="Interdecile Ratio (Decile 10 / Decile 1)",
                y_label=f"{eu_priority} Score",
                output_path=output_path
            )
    
    print("\n" + "=" * 60)
    print("✅ All graphs generated successfully!")
    print(f"📁 Output directory: {output_dir}")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        if file.endswith(('.html', '.png')):
            print(f"  • {file}")
    
    print("\n" + "=" * 60)
    print("📊 Summary of Generated Graphs:")
    print("=" * 60)
    print("Level 1 - EWBI Overall:")
    print("  • EWBI vs Interdecile Ratio (all countries + EU-27)")
    print("\nLevel 2 - EU Priorities:")
    for i, eu_priority in enumerate(eu_priorities, 1):
        print(f"  {i}. {eu_priority} vs Interdecile Ratio")
    
    print(f"\n📈 Each graph shows:")
    print("  • X-axis: Interdecile Ratio (Decile 10 / Decile 1)")
    print("  • Y-axis: EWBI Score or EU Priority Score")
    print("  • Points: All individual countries + EU-27 (diamond marker)")
    print("  • Colors: Same as dashboard (consistent across all visualizations)")
    print("  • Year: 2023 (latest available data)")
    
    print(f"\n💡 Interpretation:")
    print("  • Higher interdecile ratio = more inequality between income deciles")
    print("  • Y-axis shows well-being score for that country/priority")
    print("  • EU-27 values use population-weighted geometric means")

if __name__ == "__main__":
    main()