"""
Mobility Analysis for Switzerland
Creates visualizations for public transport share and passenger car age distribution
Based on Eurostat data for transport modal split and vehicle fleet composition
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_public_transport_share():
    """Load and process public transport share data"""
    
    base_path = Path(__file__).parent.parent / "external_data"
    file_path = base_path / "eurostat_share_pt.csv"
    
    if not file_path.exists():
        print(f"⚠ Warning: {file_path} not found. Skipping public transport analysis.")
        return None
    
    print(f"Loading public transport share data: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Filter for Switzerland
        df_swiss = df[df['geo'] == 'Switzerland'].copy()
        
        if df_swiss.empty:
            print("⚠ Warning: No data found for Switzerland in public transport dataset")
            return None
        
        # Convert TIME_PERIOD to numeric year
        df_swiss['year'] = pd.to_numeric(df_swiss['TIME_PERIOD'], errors='coerce').astype('Int64')
        df_swiss['value'] = pd.to_numeric(df_swiss['OBS_VALUE'], errors='coerce')
        
        # Remove rows with missing years or values
        df_swiss = df_swiss.dropna(subset=['year', 'value'])
        
        # Pivot to have vehicle types as columns
        df_pivot = df_swiss.pivot_table(
            index='year', 
            columns='vehicle', 
            values='value', 
            aggfunc='first'
        ).fillna(0)
        
        print(f"✓ Loaded public transport data for Switzerland: {len(df_pivot)} years ({df_pivot.index.min()}-{df_pivot.index.max()})")
        print(f"✓ Vehicle types available: {list(df_pivot.columns)}")
        
        return df_pivot
        
    except Exception as e:
        print(f"✗ Error loading public transport data: {e}")
        return None

def load_car_age_data():
    """Load and process passenger car age data for Switzerland and comparison countries"""
    
    base_path = Path(__file__).parent.parent / "external_data"
    file_path = base_path / "eurostat_car_age.csv"
    
    if not file_path.exists():
        print(f"⚠ Warning: {file_path} not found. Skipping car age analysis.")
        return None, None
    
    print(f"Loading car age data: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Try EU-27 first, then individual countries
        comparison_countries = ['EU27_2020', 'France', 'Germany', 'Austria']
        countries_of_interest = ['Switzerland'] + comparison_countries
        df_filtered = df[df['geo'].isin(countries_of_interest)].copy()
        
        if df_filtered.empty:
            print("⚠ Warning: No data found for Switzerland or comparison countries")
            return None, None
        
        # Convert TIME_PERIOD to numeric year
        df_filtered['year'] = pd.to_numeric(df_filtered['TIME_PERIOD'], errors='coerce').astype('Int64')
        df_filtered['value'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
        
        # Remove rows with missing years or values
        df_filtered = df_filtered.dropna(subset=['year', 'value'])
        
        # Separate Switzerland and comparison data
        df_swiss = df_filtered[df_filtered['geo'] == 'Switzerland']
        
        # Create comparison datasets dictionary
        comparison_data = {}
        for country in comparison_countries:
            country_df = df_filtered[df_filtered['geo'] == country]
            if not country_df.empty:
                country_pivot = country_df.pivot_table(
                    index='year', 
                    columns='age', 
                    values='value', 
                    aggfunc='first'
                ).fillna(0)
                comparison_data[country] = country_pivot
                print(f"✓ Loaded car age data for {country}: {len(country_pivot)} years ({country_pivot.index.min()}-{country_pivot.index.max()})")
        
        # Process Switzerland data
        swiss_pivot = None
        if not df_swiss.empty:
            swiss_pivot = df_swiss.pivot_table(
                index='year', 
                columns='age', 
                values='value', 
                aggfunc='first'
            ).fillna(0)
            print(f"✓ Loaded car age data for Switzerland: {len(swiss_pivot)} years ({swiss_pivot.index.min()}-{swiss_pivot.index.max()})")
        
        if swiss_pivot is not None:
            print(f"✓ Age categories available: {list(swiss_pivot.columns)}")
        
        return swiss_pivot, comparison_data
        
    except Exception as e:
        print(f"✗ Error loading car age data: {e}")
        return None, None

def create_public_transport_area_chart(pt_df, output_dir):
    """Create area chart showing share of trains and buses over time"""
    
    if pt_df is None:
        print("⚠ No public transport data available - skipping visualization")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Map actual column names to colors
    color_mapping = {
        'Trains': '#1f77b4',           # Blue for trains
        'Motor coaches, buses and trolley buses': '#ff7f0e',       # Orange for buses  
        'Trains, motor coaches, buses and trolley buses - sum of available data': '#2ca02c' # Green for total
    }
    
    years = pt_df.index
    
    # Create area chart
    ax.set_title('Switzerland: Share of Public Transport in Inland Passenger Transport', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Find available transport modes (excluding total for stacking)
    individual_modes = []
    total_column = None
    
    for col in pt_df.columns:
        if 'sum of available data' in col:
            total_column = col
        elif col in ['Trains', 'Motor coaches, buses and trolley buses']:
            individual_modes.append(col)
    
    # Create stacked area plot if individual modes are available
    if len(individual_modes) > 0:
        mode_data = pt_df[individual_modes]
        ax.stackplot(years, 
                    *[mode_data[col] for col in individual_modes], 
                    labels=individual_modes,
                    colors=[color_mapping.get(col, '#888888') for col in individual_modes],
                    alpha=0.8)
        
        # Also plot total line if available
        if total_column:
            ax.plot(years, pt_df[total_column], 
                   color=color_mapping.get(total_column, '#2ca02c'), linewidth=3, 
                   label='Total (Trains + Buses)', linestyle='--')
    
    elif total_column:
        # Only total available
        ax.fill_between(years, 0, pt_df[total_column], 
                       color=color_mapping.get(total_column, '#2ca02c'), alpha=0.7,
                       label='Public Transport Total')
    
    else:
        print("⚠ No recognizable transport mode columns found")
        return None
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Share of passenger transport (%)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_ylim(bottom=0)
    
    # Add statistics
    if total_column and total_column in pt_df.columns:
        latest_share = pt_df[total_column].iloc[-1]
        earliest_share = pt_df[total_column].iloc[0]
        change = latest_share - earliest_share
        
        ax.text(0.02, 0.98, 
                f'Latest share: {latest_share:.1f}%\\n'
                f'Change from {years[0]}: {change:+.1f}%',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / "switzerland_public_transport_share.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Public transport share visualization saved: {output_file}")
    
    return fig

def create_car_age_comparison(car_age_df, comparison_data, output_dir):
    """Create double bar plot comparing car age distribution with comparison countries overlay"""
    
    if car_age_df is None:
        print("⚠ No car age data available - skipping visualization")
        return None
    
    # Check if we have both years
    available_years = [int(year) for year in car_age_df.index if pd.notna(year)]
    target_years = [2014, 2024]
    
    # Use available years closest to targets if exact years don't exist
    actual_years = []
    for target_year in target_years:
        if target_year in available_years:
            actual_years.append(target_year)
        else:
            # Find closest year
            closest_year = min(available_years, key=lambda x: abs(int(x) - target_year))
            actual_years.append(closest_year)
            print(f"⚠ Using {closest_year} instead of {target_year}")
    
    if len(set(actual_years)) < 2:
        print("⚠ Need at least 2 different years for comparison")
        return None
    
    # Map actual column names to our codes (ordered by growing age)
    age_mapping = {
        'Less than 2 years': 'Y_LT2',
        'From 2 to 5 years': 'Y2-5', 
        'From 5 to 10 years': 'Y5-10',
        'From 10 to 20 years': 'Y10-20',
        'Over 20 years': 'Y_GT20'
    }
    
    age_labels = {
        'Y_LT2': '< 2 years',
        'Y2-5': '2-5 years', 
        'Y5-10': '5-10 years',
        'Y10-20': '10-20 years',
        'Y_GT20': '> 20 years'
    }
    
    # Get available age categories in correct order
    age_categories_ordered = ['Y_LT2', 'Y2-5', 'Y5-10', 'Y10-20', 'Y_GT20']
    available_columns = []
    
    for age_code in age_categories_ordered:
        # Find the actual column name for this age code
        for col_name, code in age_mapping.items():
            if code == age_code and col_name in car_age_df.columns:
                available_columns.append(col_name)
                break
    
    # Get data for comparison years and calculate percentages
    year1, year2 = actual_years[0], actual_years[1]
    
    # Calculate percentages (relative to Total)
    if 'Total' in car_age_df.columns:
        data_year1 = [(car_age_df.loc[year1, col] / car_age_df.loc[year1, 'Total']) * 100 
                     for col in available_columns]
        data_year2 = [(car_age_df.loc[year2, col] / car_age_df.loc[year2, 'Total']) * 100 
                     for col in available_columns]
        ylabel = 'Share of total passenger cars (%)'
        
        # Calculate comparison countries percentages if available
        comparison_countries_data = {}
        available_comparison_countries = []
        
        if comparison_data:
            # Try EU-27 first
            if 'EU27_2020' in comparison_data and 'Total' in comparison_data['EU27_2020'].columns:
                eu27_df = comparison_data['EU27_2020']
                eu27_years = [int(year) for year in eu27_df.index if pd.notna(year)]
                
                if year1 in eu27_years and year2 in eu27_years:
                    common_columns = [col for col in available_columns if col in eu27_df.columns]
                    if common_columns:
                        comparison_countries_data['EU27_2020'] = {
                            'year1': [(eu27_df.loc[year1, col] / eu27_df.loc[year1, 'Total']) * 100 for col in common_columns],
                            'year2': [(eu27_df.loc[year2, col] / eu27_df.loc[year2, 'Total']) * 100 for col in common_columns],
                            'name': 'EU-27',
                            'marker1': 'o', 'marker2': 's',
                            'color1': 'red', 'color2': 'darkred'
                        }
                        available_comparison_countries.append('EU27_2020')
                        available_columns = common_columns  # Use common columns
                        print(f"✓ EU-27 comparison data available for {year1} and {year2}")
            
            # If no EU-27, try individual countries
            if not available_comparison_countries:
                country_configs = {
                    'France': {'name': 'France', 'marker1': '^', 'marker2': 'v', 'color1': 'blue', 'color2': 'darkblue'},
                    'Germany': {'name': 'Germany', 'marker1': 'D', 'marker2': 'd', 'color1': 'green', 'color2': 'darkgreen'},
                    'Austria': {'name': 'Austria', 'marker1': 'P', 'marker2': 'X', 'color1': 'purple', 'color2': 'indigo'}
                }
                
                for country, config in country_configs.items():
                    if country in comparison_data and 'Total' in comparison_data[country].columns:
                        country_df = comparison_data[country]
                        country_years = [int(year) for year in country_df.index if pd.notna(year)]
                        
                        if year1 in country_years and year2 in country_years:
                            common_columns = [col for col in available_columns if col in country_df.columns]
                            if common_columns:
                                comparison_countries_data[country] = {
                                    'year1': [(country_df.loc[year1, col] / country_df.loc[year1, 'Total']) * 100 for col in common_columns],
                                    'year2': [(country_df.loc[year2, col] / country_df.loc[year2, 'Total']) * 100 for col in common_columns],
                                    **config
                                }
                                if not available_comparison_countries:  # First country sets the common columns
                                    available_columns = common_columns
                                available_comparison_countries.append(country)
                                print(f"✓ {country} comparison data available for {year1} and {year2}")
        
        # Recalculate Switzerland data with final available_columns (common ones)
        data_year1 = [(car_age_df.loc[year1, col] / car_age_df.loc[year1, 'Total']) * 100 
                     for col in available_columns]
        data_year2 = [(car_age_df.loc[year2, col] / car_age_df.loc[year2, 'Total']) * 100 
                     for col in available_columns]
        
        # Create labels using the mapping
        labels = [age_labels[age_mapping[col]] for col in available_columns]
    else:
        print("⚠ Warning: No 'Total' column found - cannot calculate percentages")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, data_year1, width, label=f'{year1}', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, data_year2, width, label=f'{year2}', color='#ff7f0e', alpha=0.8)
    
    # Customize the plot
    # Customize the plot title based on available comparisons
    if 'EU27_2020' in available_comparison_countries:
        title = 'Switzerland vs EU-27: Passenger Cars by Age Distribution Comparison'
    elif available_comparison_countries:
        comparison_names = [comparison_countries_data[c]['name'] for c in available_comparison_countries]
        title = f'Switzerland vs {", ".join(comparison_names)}: Passenger Cars by Age Distribution Comparison'
    else:
        title = 'Switzerland: Passenger Cars by Age Distribution Comparison'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Car Age Category', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add Switzerland bars legend
    bars_legend = ax.legend(handles=[bars1[0], bars2[0]], labels=[f'Switzerland {year1}', f'Switzerland {year2}'], 
                           loc='upper left', fontsize=10)
    ax.add_artist(bars_legend)
    
    # Add comparison countries dots/markers
    if available_comparison_countries:
        comparison_handles = []
        comparison_labels = []
        
        for country in available_comparison_countries:
            config = comparison_countries_data[country]
            
            # Add markers on top of bars with slight offset
            dots1 = ax.scatter(x - 0.1, config['year1'], color=config['color1'], s=80, 
                              marker=config['marker1'], label=f"{config['name']} {year1}", 
                              zorder=6, edgecolors='white', linewidth=1)
            dots2 = ax.scatter(x + 0.1, config['year2'], color=config['color2'], s=80, 
                              marker=config['marker2'], label=f"{config['name']} {year2}", 
                              zorder=6, edgecolors='white', linewidth=1)
            
            comparison_handles.extend([dots1, dots2])
            comparison_labels.extend([f"{config['name']} {year1}", f"{config['name']} {year2}"])
        
        # Add comparison legend
        comparison_legend = ax.legend(handles=comparison_handles, labels=comparison_labels, 
                                     loc='upper right', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add value labels on bars
    def add_value_labels(bars, data):
        for bar, value in zip(bars, data):
            height = bar.get_height()
            ax.annotate(f'{value:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1, data_year1)
    add_value_labels(bars2, data_year2)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / "switzerland_car_age_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Car age comparison visualization saved: {output_file}")
    
    return fig

def export_public_transport_to_excel(pt_df, output_dir):
    """Export public transport share data to Excel format"""
    
    if pt_df is None:
        return None
    
    # Create DataFrame with required structure
    export_data = []
    
    for year in pt_df.index:
        for vehicle_type in pt_df.columns:
            row = {
                'visual_number': np.nan,
                'visual_name': 'Switzerland Public Transport Share',
                'year': year,
                'filter': vehicle_type,
                'decile': np.nan,
                'value': pt_df.loc[year, vehicle_type],
                'unit': '% of passenger.kilometer'
            }
            export_data.append(row)
    
    df = pd.DataFrame(export_data)
    
    # Export to Excel
    excel_file = output_dir / "public_transport_share.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"✓ Public transport share Excel exported to: {excel_file}")
    
    return excel_file

def export_car_age_to_excel(car_age_df, output_dir):
    """Export car age data to Excel format"""
    
    if car_age_df is None:
        return None
    
    # Create DataFrame with required structure
    export_data = []
    
    for year in car_age_df.index:
        for age_category in car_age_df.columns:
            row = {
                'visual_number': np.nan,
                'visual_name': 'Switzerland Car Age Distribution',
                'year': year,
                'filter': age_category,
                'decile': np.nan,
                'value': car_age_df.loc[year, age_category],
                'unit': 'Number of cars'
            }
            export_data.append(row)
    
    df = pd.DataFrame(export_data)
    
    # Export to Excel
    excel_file = output_dir / "car_age_distribution.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"✓ Car age distribution Excel exported to: {excel_file}")
    
    return excel_file

def load_motor_energy_data():
    """Load and process detailed motor energy data for Switzerland"""
    
    base_path = Path(__file__).parent.parent / "external_data"
    file_path = base_path / "eurostat_car_motor.csv"
    
    if not file_path.exists():
        print(f"⚠ Warning: {file_path} not found. Skipping motor energy analysis.")
        return None
    
    print(f"Loading motor energy data: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Filter for Switzerland
        df_swiss = df[df['geo'] == 'Switzerland'].copy()
        
        if df_swiss.empty:
            print("⚠ Warning: No data found for Switzerland in motor energy dataset")
            return None
        
        # Define detailed subcategories for decomposition
        detailed_categories = [
            'Petrol (excluding hybrids) \xa0', 
            'Diesel (excluding hybrids) \xa0',
            'Hybrid diesel-electric',
            'Plug-in hybrid diesel-electric \xa0', 
            'Hybrid electric-petrol',
            'Plug-in hybrid petrol-electric \xa0',
            'Electricity',
            'Bi-fuel', 'Biodiesel', 'Bioethanol', 'Natural gas',
            'Hydrogen and fuel cells\xa0', 'Liquefied petroleum gases (LPG)', 'Other',
            'Total'  # Include total for percentage calculation
        ]
        
        # Filter for these categories
        df_filtered = df_swiss[df_swiss['mot_nrg'].isin(detailed_categories)].copy()
        df_filtered['OBS_VALUE'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
        
        # Create pivot table
        pivot_data = df_filtered.pivot_table(
            index='TIME_PERIOD', 
            columns='mot_nrg', 
            values='OBS_VALUE', 
            aggfunc='first'
        ).fillna(0)
        
        # Create the 6 main categories
        motor_categories = pd.DataFrame(index=pivot_data.index)
        
        motor_categories['Petrol (excl. hybrids)'] = pivot_data.get('Petrol (excluding hybrids) \xa0', 0)
        motor_categories['Diesel (excl. hybrids)'] = pivot_data.get('Diesel (excluding hybrids) \xa0', 0)
        
        # Regular hybrids (non plug-in)
        motor_categories['Hybrids (regular)'] = (
            pivot_data.get('Hybrid diesel-electric', 0) + 
            pivot_data.get('Hybrid electric-petrol', 0)
        )
        
        # Plug-in hybrids
        motor_categories['Plug-in hybrids'] = (
            pivot_data.get('Plug-in hybrid diesel-electric \xa0', 0) + 
            pivot_data.get('Plug-in hybrid petrol-electric \xa0', 0)
        )
        
        motor_categories['Electricity'] = pivot_data.get('Electricity', 0)
        
        # Other alternative energy
        other_alt_categories = ['Bi-fuel', 'Biodiesel', 'Bioethanol', 'Natural gas', 
                               'Hydrogen and fuel cells\xa0', 'Liquefied petroleum gases (LPG)', 'Other']
        motor_categories['Other alternative'] = sum(
            pivot_data.get(cat, 0) for cat in other_alt_categories
        )
        
        # Get total for percentage calculation
        total_fleet = pivot_data.get('Total', motor_categories.sum(axis=1))
        
        # Convert to percentages
        motor_percentages = motor_categories.div(total_fleet, axis=0) * 100
        
        # Sort index to ensure chronological order
        motor_percentages = motor_percentages.sort_index()
        
        print(f"Motor energy data loaded: {len(motor_percentages)} years, {len(motor_percentages.columns)} categories")
        print(f"Years available: {motor_percentages.index.min()} - {motor_percentages.index.max()}")
        
        return motor_percentages
        
    except Exception as e:
        print(f"Error loading motor energy data: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_motor_energy_area_chart(motor_data, output_dir):
    """Create area chart showing evolution of motor energy types over time"""
    
    try:
        # Define colors for each category
        colors = {
            'Petrol (excl. hybrids)': '#FF6B6B',    # Red - traditional
            'Diesel (excl. hybrids)': '#4ECDC4',    # Teal - traditional 
            'Hybrids (regular)': '#45B7D1',         # Light blue - transition
            'Plug-in hybrids': '#96CEB4',           # Light green - transition
            'Electricity': '#FFEAA7',               # Yellow - clean energy
            'Other alternative': '#DDA0DD'          # Plum - other clean
        }
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create stacked area chart
        years = motor_data.index
        bottom = np.zeros(len(years))
        
        # Plot areas from bottom to top
        for category in motor_data.columns:
            values = motor_data[category].values
            ax.fill_between(years, bottom, bottom + values, 
                          color=colors.get(category, '#999999'), 
                          alpha=0.8, label=category)
            bottom += values
        
        # Customize the plot
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage of Total Vehicle Fleet (%)', fontsize=12, fontweight='bold')
        ax.set_title('Switzerland Vehicle Fleet by Motor Energy Type\n(2013-2024)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set y-axis to 0-100%
        ax.set_ylim(0, 100)
        ax.set_xlim(years.min(), years.max())
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend with better positioning
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Add percentage labels on y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        
        # Tight layout to fit legend
        plt.tight_layout()
        
        # Save the plot
        plot_file = output_dir / "switzerland_motor_energy_evolution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()  # Close the figure to free memory
        
        print(f"Motor energy evolution chart saved: {plot_file}")
        return plot_file
        
    except Exception as e:
        print(f"Error creating motor energy area chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_motor_energy_to_excel(motor_data, output_dir):
    """Export motor energy data to be compatible with Dash dashboard"""
    
    try:
        # Prepare standardized format for dashboard
        excel_data = []
        
        for energy_type in motor_data.columns:
            for year in motor_data.index:
                excel_data.append({
                    'visual_number': 3,
                    'visual_name': 'switzerland_motor_energy_evolution',
                    'year': year,
                    'filter': energy_type,
                    'decile': None,
                    'value': motor_data.loc[year, energy_type],
                    'unit': '% of total vehicle fleet'
                })
        
        df_excel = pd.DataFrame(excel_data)
        
        # Export to Excel
        excel_file = output_dir / "switzerland_motor_energy_evolution.xlsx"
        df_excel.to_excel(excel_file, index=False, sheet_name='motor_energy_data')
        
        print(f"Motor energy data exported to: {excel_file}")
        return excel_file
        
    except Exception as e:
        print(f"Error exporting motor energy data to Excel: {e}")
        return None

def load_car_weight_data():
    """Load and process car weight data for Switzerland heatmap"""
    
    base_path = Path(__file__).parent.parent / "external_data"
    file_path = base_path / "eurostat_car_weight.csv"
    
    if not file_path.exists():
        print(f"Warning: {file_path} not found. Skipping car weight analysis.")
        return None
    
    print(f"Loading car weight data: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Filter for Switzerland
        df_swiss = df[df['geo'] == 'Switzerland'].copy()
        
        if df_swiss.empty:
            print("Warning: No data found for Switzerland in car weight dataset")
            return None
        
        # Define the weight categories we want
        weight_categories = [
            'Less than 1 000 kg',
            'From 1 000 to 1 249 kg',
            'From 1 250 to 1 499 kg', 
            '1 500 kg or over',
            'Total'  # Need for percentage calculation
        ]
        
        # Filter for these categories
        df_filtered = df_swiss[df_swiss['weight'].isin(weight_categories)].copy()
        
        # Convert to numeric
        df_filtered['OBS_VALUE'] = pd.to_numeric(df_filtered['OBS_VALUE'], errors='coerce')
        
        # Create pivot table
        pivot_data = df_filtered.pivot_table(
            index='TIME_PERIOD', 
            columns='weight', 
            values='OBS_VALUE', 
            aggfunc='first'
        ).fillna(0)
        
        # Calculate percentages (exclude Total from final data)
        weight_cols = [col for col in weight_categories if col != 'Total']
        
        # Calculate percentages for each weight category
        weight_percentages = pd.DataFrame(index=pivot_data.index)
        
        for weight_cat in weight_cols:
            if weight_cat in pivot_data.columns and 'Total' in pivot_data.columns:
                # Calculate percentages safely, handling division by zero
                percentages = []
                for year in pivot_data.index:
                    total = pivot_data.loc[year, 'Total']
                    value = pivot_data.loc[year, weight_cat]
                    if total > 0 and not pd.isna(total) and not pd.isna(value):
                        percentage = round((value / total * 100), 2)
                        percentages.append(percentage)
                    else:
                        percentages.append(np.nan)  # Use NaN for invalid data
                
                weight_percentages[weight_cat] = percentages
        
        # Remove years with any NaN values (invalid data)
        weight_percentages = weight_percentages.dropna()
        
        if len(weight_percentages) == 0:
            print("Warning: No valid car weight percentage data after cleaning")
            return None
        
        # Sort by year
        weight_percentages = weight_percentages.sort_index()
        
        # Create cleaner labels for the heatmap rows
        rename_dict = {
            'Less than 1 000 kg': 'KG_LT1000\n< 1,000 kg',
            'From 1 000 to 1 249 kg': 'KG1000-1249\n1,000-1,249 kg',
            'From 1 250 to 1 499 kg': 'KG1250-1499\n1,250-1,499 kg',
            '1 500 kg or over': 'KG_GE1500\n>= 1,500 kg'
        }
        
        weight_percentages = weight_percentages.rename(columns=rename_dict)
        
        print(f"Car weight data loaded: {len(weight_percentages)} years, {len(weight_percentages.columns)} weight categories")
        print(f"Years available: {weight_percentages.index.min()} - {weight_percentages.index.max()}")
        
        return weight_percentages
        
    except Exception as e:
        print(f"Error loading car weight data: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_car_weight_heatmap(weight_data, output_dir):
    """Create heatmap showing car weight distribution over time"""
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 6))
        
        # Transpose data so weight categories are rows and years are columns
        heatmap_data = weight_data.T
        
        # Create heatmap with better color scaling
        sns.heatmap(
            heatmap_data,
            annot=True,  # Show values in cells
            fmt='.1f',   # Format to 1 decimal place
            cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (red=high%, blue=low%)
            cbar_kws={'label': 'Percentage of Total Fleet (%)'},
            linewidths=0.5,
            vmin=0,  # Set minimum value for consistent color scaling
            vmax=100,  # Set maximum value for consistent color scaling
            ax=ax
        )
        
        # Customize the plot
        ax.set_title('Switzerland: Passenger Cars by Unloaded Weight Distribution\n(Percentage of Total Fleet by Year)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Weight Category', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels([str(int(year)) for year in heatmap_data.columns], rotation=45)
        
        # Adjust y-axis labels
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        # Add text box with data range info
        years_range = f"{int(heatmap_data.columns.min())}-{int(heatmap_data.columns.max())}"
        ax.text(0.02, 0.98, f'Data: {years_range}\n({len(heatmap_data.columns)} years)', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Tight layout
        plt.tight_layout()
        
        # Save the plot
        plot_file = output_dir / "switzerland_car_weight_heatmap.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Car weight heatmap saved: {plot_file}")
        return plot_file
        
    except Exception as e:
        print(f"Error creating car weight heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_car_weight_line_graph(weight_data, output_dir):
    """Create line graph showing evolution of lighter weight categories over time"""
    
    try:
        # Select only the lighter weight categories (exclude >= 1,500 kg)
        lighter_categories = [
            'KG_LT1000\n< 1,000 kg',
            'KG1000-1249\n1,000-1,249 kg', 
            'KG1250-1499\n1,250-1,499 kg'
        ]
        
        # Filter data for lighter categories only
        light_data = weight_data[[col for col in lighter_categories if col in weight_data.columns]]
        
        if light_data.empty:
            print("Warning: No lighter weight categories found in data")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define colors for each category
        colors = {
            'KG_LT1000\n< 1,000 kg': '#2E86C1',        # Blue
            'KG1000-1249\n1,000-1,249 kg': '#28B463',   # Green 
            'KG1250-1499\n1,250-1,499 kg': '#F39C12'    # Orange
        }
        
        # Plot lines for each category
        for category in light_data.columns:
            ax.plot(light_data.index, light_data[category], 
                   color=colors.get(category, '#333333'),
                   linewidth=3, marker='o', markersize=6,
                   label=category.replace('\n', ' '))
        
        # Customize the plot
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage of Total Vehicle Fleet (%)', fontsize=14, fontweight='bold')
        ax.set_title('Switzerland: Evolution of Lighter Vehicle Weight Categories\n(2013-2024)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set axis limits and ticks
        ax.set_xlim(light_data.index.min() - 0.5, light_data.index.max() + 0.5)
        ax.set_ylim(0, max(light_data.max()) * 1.1)  # Add 10% padding at top
        
        # Format y-axis as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # Improve x-axis year labels
        years = sorted(light_data.index)
        ax.set_xticks(years)
        ax.set_xticklabels([str(int(year)) for year in years], rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the plot
        plot_file = output_dir / "switzerland_car_weight_light_categories.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Car weight line graph saved: {plot_file}")
        return plot_file
        
    except Exception as e:
        print(f"Error creating car weight line graph: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_car_weight_light_to_excel(weight_data, output_dir):
    """Export lighter weight categories data to Excel format"""
    
    try:
        # Select only the lighter weight categories
        lighter_categories = [
            'KG_LT1000\n< 1,000 kg',
            'KG1000-1249\n1,000-1,249 kg', 
            'KG1250-1499\n1,250-1,499 kg'
        ]
        
        light_data = weight_data[[col for col in lighter_categories if col in weight_data.columns]]
        
        # Prepare standardized format for dashboard
        excel_data = []
        
        for weight_category in light_data.columns:
            for year in light_data.index:
                excel_data.append({
                    'visual_number': 5,
                    'visual_name': 'switzerland_car_weight_light_categories',
                    'year': year,
                    'filter': weight_category,
                    'decile': None,
                    'value': light_data.loc[year, weight_category],
                    'unit': '% of total vehicle fleet'
                })
        
        df_excel = pd.DataFrame(excel_data)
        
        # Export to Excel
        excel_file = output_dir / "switzerland_car_weight_light_categories.xlsx"
        df_excel.to_excel(excel_file, index=False, sheet_name='light_weight_data')
        
        print(f"Light weight categories data exported to: {excel_file}")
        return excel_file
        
    except Exception as e:
        print(f"Error exporting light weight data to Excel: {e}")
        return None

def export_car_weight_to_excel(weight_data, output_dir):
    """Export car weight data to Excel format"""
    
    try:
        # Prepare standardized format for dashboard
        excel_data = []
        
        for weight_category in weight_data.columns:
            for year in weight_data.index:
                excel_data.append({
                    'visual_number': 4,
                    'visual_name': 'switzerland_car_weight_heatmap',
                    'year': year,
                    'filter': weight_category,
                    'decile': None,
                    'value': weight_data.loc[year, weight_category],
                    'unit': '% of total vehicle fleet'
                })
        
        df_excel = pd.DataFrame(excel_data)
        
        # Export to Excel
        excel_file = output_dir / "switzerland_car_weight_heatmap.xlsx"
        df_excel.to_excel(excel_file, index=False, sheet_name='car_weight_data')
        
        print(f"Car weight data exported to: {excel_file}")
        return excel_file
        
    except Exception as e:
        print(f"Error exporting car weight data to Excel: {e}")
        return None

def main():
    """Main analysis function"""
    print("=" * 70)
    print("SWITZERLAND MOBILITY ANALYSIS")
    print("Switzerland vs EU-27/Comparison Countries: Public Transport & Car Age")
    print("=" * 70)
    
    try:
        # Setup output directory
        output_dir = Path(__file__).parent.parent / "outputs" / "graphs" / "mobility"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
        
        # Load and process public transport data
        print("\nLoading Switzerland public transport share data...")
        pt_data = load_public_transport_share()
        
        # Load and process car age data
        print("Loading Switzerland and comparison countries car age data...")
        car_age_data, comparison_countries_data = load_car_age_data()
        
        # Load and process motor energy data
        print("Loading Switzerland motor energy data...")
        motor_data = load_motor_energy_data()
        
        # Load and process car weight data
        print("Loading Switzerland car weight data...")
        weight_data = load_car_weight_data()
        
        # Create public transport visualization
        if pt_data is not None:
            print("\nCreating public transport share area chart...")
            pt_plot = create_public_transport_area_chart(pt_data, output_dir)
            
            # Export to Excel
            print("Exporting public transport data to Excel...")
            pt_excel = export_public_transport_to_excel(pt_data, output_dir)
            
            # Save CSV backup
            pt_csv_file = output_dir / "switzerland_public_transport_share.csv"
            pt_data.to_csv(pt_csv_file)
            print(f"✓ Public transport data saved to: {pt_csv_file}")
        
        # Create car age visualization
        if car_age_data is not None:
            print("\nCreating car age comparison chart (Switzerland vs comparison countries)...")
            car_plot = create_car_age_comparison(car_age_data, comparison_countries_data, output_dir)
            
            # Export to Excel
            print("Exporting car age data to Excel...")
            car_excel = export_car_age_to_excel(car_age_data, output_dir)
            
            # Save CSV backup
            car_csv_file = output_dir / "switzerland_car_age_distribution.csv"
            car_age_data.to_csv(car_csv_file)
            print(f"✓ Car age data saved to: {car_csv_file}")
        
        # Create motor energy visualization
        if motor_data is not None:
            print("\nCreating motor energy evolution area chart...")
            motor_plot = create_motor_energy_area_chart(motor_data, output_dir)
            
            # Export to Excel
            print("Exporting motor energy data to Excel...")
            motor_excel = export_motor_energy_to_excel(motor_data, output_dir)

            # Save CSV backup
            motor_csv_file = output_dir / "switzerland_motor_energy_evolution.csv"
            motor_data.to_csv(motor_csv_file)
            print(f"✓ Motor energy data saved to: {motor_csv_file}")
        
        # Create car weight heatmap
        if weight_data is not None:
            print("\nCreating car weight distribution heatmap...")
            weight_plot = create_car_weight_heatmap(weight_data, output_dir)
            
            # Create car weight line graph for lighter categories
            print("Creating car weight line graph for lighter categories...")
            light_plot = create_car_weight_line_graph(weight_data, output_dir)
            
            # Export to Excel
            print("Exporting car weight data to Excel...")
            weight_excel = export_car_weight_to_excel(weight_data, output_dir)
            
            print("Exporting light weight categories data to Excel...")
            light_excel = export_car_weight_light_to_excel(weight_data, output_dir)
            
            # Save CSV backup
            weight_csv_file = output_dir / "switzerland_car_weight_heatmap.csv"
            weight_data.to_csv(weight_csv_file)
            print(f"Car weight data saved to: {weight_csv_file}")
        
        # Summary
        data_loaded = [d for d in [pt_data, car_age_data, motor_data, weight_data] if d is not None]
        if data_loaded:
            print(f"\nSwitzerland mobility analysis complete! Check {output_dir} for all outputs.")
            print(f"   - {len(data_loaded)} datasets processed successfully")
        else:
            print(f"\nNo mobility data found for Switzerland. Please add required CSV files to external_data directory.")
        
        return pt_data, car_age_data, motor_data, weight_data
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
