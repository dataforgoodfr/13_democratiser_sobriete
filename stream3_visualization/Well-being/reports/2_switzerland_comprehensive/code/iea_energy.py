"""
Energy Dependency Analysis for Switzerland
Creates visualization of net energy imports as proportion of total energy supply
Based on IEA data for net energy imports and total energy supply by source
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_energy_import_export_data():
    """Load and process the 4 energy import/export datasets"""
    
    base_path = Path(__file__).parent.parent / "external_data"
    
    # Define the 4 energy import files
    energy_files = {
        'Crude Oil': 'iea_Crude oil imports.csv',
        'Coal': 'iea_Coal imports.csv', 
        'Electricity': 'iea_Electricity imports.csv',
        'Natural Gas': 'iea_Natural gas imports.csv'
    }
    
    net_imports_data = {}
    
    print(f"\nLoading energy import/export data:")
    
    for energy_type, filename in energy_files.items():
        file_path = base_path / filename
        print(f"  - {file_path}")
        
        # Load data (skip first 3 metadata rows)
        df = pd.read_csv(file_path, skiprows=3, index_col=0)
        df.index.name = 'Year'
        
        # Remove Units column if it exists
        if 'Units' in df.columns:
            df = df.drop('Units', axis=1)
        
        # Calculate net imports (imports - exports)
        imports = df['Imports'].fillna(0)  # Handle missing import values
        
        if 'Exports' in df.columns:
            exports = df['Exports'].fillna(0)  # Handle missing export values
            # If exports are stored as negative values, convert them to positive for calculation
            if (exports < 0).any():
                exports = -exports  # Convert negative to positive
            net_imports = imports - exports
        else:
            net_imports = imports  # Only imports, no exports
        
        net_imports_data[energy_type] = net_imports
    
    # Combine all into a single DataFrame
    net_imports_df = pd.DataFrame(net_imports_data)
    
    # Fill any missing values with 0
    net_imports_df = net_imports_df.fillna(0)
    
    return net_imports_df

def create_net_imports_area_chart(net_imports_df, output_dir):
    """Create area chart showing net imports by energy type over time"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    years = net_imports_df.index
    
    # Define colors for each energy type (ensuring consistency)
    colors = {
        'Crude Oil': '#000000',      # Black
        'Coal': '#8B4513',           # Brown  
        'Electricity': '#FFD700',    # Gold/Yellow - always this color
        'Natural Gas': '#FF6347'     # Tomato
    }
    
    # Reorder columns: Electricity as base, Coal on top
    column_order = ['Electricity', 'Natural Gas', 'Crude Oil', 'Coal']
    net_imports_ordered = net_imports_df.reindex(columns=column_order)
    
    # Create area chart that can handle negative values
    # Use fill_between for each energy type individually
    bottom = np.zeros(len(years))
    
    for i, energy_type in enumerate(column_order):
        values = net_imports_ordered[energy_type]
        color = colors[energy_type]
        
        # For stacking: positive values stack upward, negative values stack downward
        if i == 0:  # First layer (Electricity)
            ax.fill_between(years, 0, values, color=color, alpha=0.7, 
                           edgecolor=color, linewidth=0,
                           label=f"{energy_type} (Net Imports)")
            bottom = values.copy()
        else:
            # Stack on top of previous layers
            top = bottom + values
            ax.fill_between(years, bottom, top, color=color, alpha=0.7,
                           edgecolor=color, linewidth=0,
                           label=f"{energy_type} (Net Imports)")
            bottom = top.copy()
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    
    ax.set_title('Switzerland: Net Energy Imports by Type (1990-2024)\n(Imports - Exports)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Net Imports (TJ)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Add annotations for interpretation
    ax.text(0.02, 0.98, 'Positive = Net Imports\nNegative = Net Exports', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_dir / "energy_net_imports_by_type_area.png"
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Net imports area chart saved to: {plot_file}")
    
    plt.close(fig)
    return plot_file

def create_individual_net_imports_charts(net_imports_df, output_dir):
    """Create individual line charts for each energy type's net imports"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#000000', '#8B4513', '#FFD700', '#FF6347']
    
    for i, (energy_type, color) in enumerate(zip(net_imports_df.columns, colors)):
        ax = axes[i]
        years = net_imports_df.index
        values = net_imports_df[energy_type]
        
        # Plot the line
        ax.plot(years, values, color=color, linewidth=2, marker='o', markersize=4)
        
        # Fill area above/below zero
        ax.fill_between(years, values, 0, where=(values >= 0), 
                       color=color, alpha=0.3, interpolate=True, label='Net Imports')
        ax.fill_between(years, values, 0, where=(values < 0),
                       color=color, alpha=0.3, interpolate=True, label='Net Exports')
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_title(f'{energy_type}', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Net Imports (TJ)')
        ax.grid(True, alpha=0.3)
        
        # Add summary stats
        mean_val = values.mean()
        latest_val = values.iloc[-1]
        ax.text(0.02, 0.98, f'Avg: {mean_val:,.0f} TJ\n2024: {latest_val:,.0f} TJ', 
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.suptitle('Switzerland: Net Energy Imports by Type (Individual Charts)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_dir / "energy_net_imports_individual_charts.png"
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Individual net imports charts saved to: {plot_file}")
    
    plt.close(fig)
    return plot_file

def load_and_process_data():
    """Load and process the IEA energy data files"""
    
    # Define file paths
    base_path = Path(__file__).parent.parent / "external_data"
    net_imports_file = base_path / "iea_Net energy imports.csv"
    total_supply_file = base_path / "iea_Total energy supply.csv"
    
    print(f"Loading data from:")
    print(f"  - {net_imports_file}")
    print(f"  - {total_supply_file}")
    
    # Load net energy imports (skip first 3 metadata rows)
    net_imports_df = pd.read_csv(net_imports_file, skiprows=3, index_col=0)
    net_imports_df.index.name = 'Year'
    
    # Convert PJ to TJ for consistency (1 PJ = 1000 TJ)
    net_imports_df['Net imports (TJ)'] = net_imports_df['Net imports'] * 1000
    
    # Load total energy supply (skip first 3 metadata rows)
    total_supply_df = pd.read_csv(total_supply_file, skiprows=3, index_col=0)
    total_supply_df.index.name = 'Year'
    
    # Remove the 'Units' column if it exists
    if 'Units' in total_supply_df.columns:
        total_supply_df = total_supply_df.drop('Units', axis=1)
    
    # Calculate total energy supply by summing all energy sources
    energy_sources = [
        'Coal and coal products', 
        'Natural gas', 
        'Hydropower', 
        'Nuclear',
        'Solar, wind and other renewables', 
        'Biofuels and waste', 
        'Oil and oil products'
    ]
    
    total_supply_df['Total Supply (TJ)'] = total_supply_df[energy_sources].sum(axis=1)
    
    # Merge the datasets
    energy_data = pd.merge(
        net_imports_df[['Net imports (TJ)']], 
        total_supply_df[['Total Supply (TJ)'] + energy_sources], 
        left_index=True, 
        right_index=True, 
        how='inner'
    )
    
    # Calculate net import dependency ratio (as percentage)
    energy_data['Import Dependency (%)'] = (
        energy_data['Net imports (TJ)'] / energy_data['Total Supply (TJ)'] * 100
    )
    
    return energy_data

def create_main_visualization(energy_data, output_dir):
    """Create the main visualization showing net imports / total supply over time"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    years = energy_data.index
    
    # Main plot: Net Import Dependency Ratio over time
    ax.plot(years, energy_data['Import Dependency (%)'], 'b-o', linewidth=3, markersize=6, 
            label='Net Import Dependency')
    
    # Add trend line
    z = np.polyfit(years, energy_data['Import Dependency (%)'], 1)
    p = np.poly1d(z)
    ax.plot(years, p(years), "r--", alpha=0.8, linewidth=2, 
            label=f'Trend: {z[0]:.3f}%/year')
    
    ax.set_title('Switzerland: Net Energy Import Dependency (1990-2024)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Net Imports / Total Supply (%)', fontsize=12)
    ax.set_ylim(0, None)  # Ensure y-axis starts at 0
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add annotations for key insights
    max_year = energy_data['Import Dependency (%)'].idxmax()
    max_val = energy_data['Import Dependency (%)'].max()
    min_year = energy_data['Import Dependency (%)'].idxmin() 
    min_val = energy_data['Import Dependency (%)'].min()
    
    ax.annotate(f'Peak: {max_val:.1f}% ({max_year})', 
                xy=(max_year, max_val), xytext=(10, 10),
                textcoords='offset points', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.annotate(f'Minimum: {min_val:.1f}% ({min_year})', 
                xy=(min_year, min_val), xytext=(10, -20),
                textcoords='offset points', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_dir / "energy_import_dependency_main.png"
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Main visualization saved to: {plot_file}")
    
    plt.close(fig)  # Close figure to free memory
    return plot_file

def create_comprehensive_visualization(energy_data, output_dir):
    """Create comprehensive visualization with multiple panels"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Switzerland Energy Analysis: Comprehensive Overview (1990-2024)', 
                 fontsize=16, fontweight='bold')
    
    years = energy_data.index
    
    # Plot 1: Net Import Dependency Ratio over time
    ax1.plot(years, energy_data['Import Dependency (%)'], 'b-o', linewidth=2, markersize=4)
    ax1.set_title('Net Import Dependency Ratio', fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Net Imports / Total Supply (%)')
    ax1.set_ylim(0, None)  # Ensure y-axis starts at 0
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(years, energy_data['Import Dependency (%)'], 1)
    p = np.poly1d(z)
    ax1.plot(years, p(years), "r--", alpha=0.8, linewidth=1, 
             label=f'Trend: {z[0]:.3f}%/year')
    ax1.legend()
    
    # Plot 2: Net Imports and Total Supply (absolute values)
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(years, energy_data['Net imports (TJ)'] / 1000, 'g-o', 
                     linewidth=2, markersize=3, label='Net Imports (PJ)')
    line2 = ax2_twin.plot(years, energy_data['Total Supply (TJ)'] / 1000, 'r-s', 
                          linewidth=2, markersize=3, label='Total Supply (PJ)')
    
    ax2.set_title('Net Imports vs Total Energy Supply', fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Net Imports (PJ)', color='g')
    ax2_twin.set_ylabel('Total Supply (PJ)', color='r')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    # Plot 3: Energy Supply by Source (Stacked Area)
    energy_sources = [
        'Coal and coal products', 
        'Natural gas', 
        'Hydropower', 
        'Nuclear',
        'Solar, wind and other renewables', 
        'Biofuels and waste', 
        'Oil and oil products'
    ]
    
    # Convert to PJ for better readability
    supply_by_source = energy_data[energy_sources] / 1000
    
    colors = ['#8B4513', '#FF6347', '#4169E1', '#FFD700', '#32CD32', '#DEB887', '#000000']
    ax3.stackplot(years, supply_by_source.T, labels=energy_sources, colors=colors, alpha=0.8)
    ax3.set_title('Energy Supply by Source', fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Energy Supply (PJ)')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Recent trends (last 10 years)
    recent_years = years[-10:]
    recent_data = energy_data.loc[recent_years]
    
    ax4.plot(recent_years, recent_data['Import Dependency (%)'], 'b-o', linewidth=3, markersize=6)
    ax4.set_title('Recent Import Dependency Trend (2015-2024)', fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Import Dependency (%)')
    ax4.set_ylim(0, None)  # Ensure y-axis starts at 0
    ax4.grid(True, alpha=0.3)
    
    # Annotate first and last points
    first_val = recent_data['Import Dependency (%)'].iloc[0]
    last_val = recent_data['Import Dependency (%)'].iloc[-1]
    change = last_val - first_val
    
    ax4.annotate(f'{first_val:.1f}%', 
                xy=(recent_years[0], first_val), xytext=(10, 10),
                textcoords='offset points', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax4.annotate(f'{last_val:.1f}%\n({change:+.1f}% change)', 
                xy=(recent_years[-1], last_val), xytext=(-10, -20),
                textcoords='offset points', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_dir / "energy_import_dependency_comprehensive.png"
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive visualization saved to: {plot_file}")
    
    plt.close(fig)  # Close figure to free memory
    return plot_file

def print_summary_statistics(energy_data):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("SWITZERLAND ENERGY DEPENDENCY ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nData Period: {energy_data.index.min()} - {energy_data.index.max()}")
    print(f"Total Years: {len(energy_data)}")
    
    print(f"\nNET IMPORT DEPENDENCY STATISTICS:")
    print(f"Average Import Dependency: {energy_data['Import Dependency (%)'].mean():.1f}%")
    print(f"Minimum Import Dependency: {energy_data['Import Dependency (%)'].min():.1f}% ({energy_data['Import Dependency (%)'].idxmin()})")
    print(f"Maximum Import Dependency: {energy_data['Import Dependency (%)'].max():.1f}% ({energy_data['Import Dependency (%)'].idxmax()})")
    print(f"Standard Deviation: {energy_data['Import Dependency (%)'].std():.1f}%")
    
    # Recent trends
    recent_5_years = energy_data.tail(5)
    recent_avg = recent_5_years['Import Dependency (%)'].mean()
    overall_avg = energy_data['Import Dependency (%)'].mean()
    print(f"Recent 5-year average (2020-2024): {recent_avg:.1f}%")
    print(f"Difference from overall average: {recent_avg - overall_avg:+.1f}%")
    
    # Calculate trend
    years = energy_data.index.values
    dependency = energy_data['Import Dependency (%)'].values
    trend_coef = np.polyfit(years, dependency, 1)[0]
    if trend_coef > 0:
        trend_desc = "increasing"
    else:
        trend_desc = "decreasing"
    print(f"Long-term trend: {trend_coef:.3f}% per year ({trend_desc})")
    
    print(f"\nENERGY VOLUMES (2024):")
    latest_year = energy_data.index[-1]
    latest_data = energy_data.loc[latest_year]
    print(f"Net Imports: {latest_data['Net imports (TJ)']/1000:.0f} PJ")
    print(f"Total Supply: {latest_data['Total Supply (TJ)']/1000:.0f} PJ")
    print(f"Import Dependency: {latest_data['Import Dependency (%)']:.1f}%")
    
    # Energy mix in latest year
    print(f"\nENERGY MIX BREAKDOWN ({latest_year}):")
    energy_sources = [
        'Coal and coal products', 'Natural gas', 'Hydropower', 'Nuclear',
        'Solar, wind and other renewables', 'Biofuels and waste', 'Oil and oil products'
    ]
    
    total_supply = latest_data['Total Supply (TJ)']
    for source in energy_sources:
        percentage = (latest_data[source] / total_supply) * 100
        print(f"  {source}: {percentage:.1f}% ({latest_data[source]/1000:.0f} PJ)")
    
    # Key insights
    print(f"\nKEY INSIGHTS:")
    max_dep_year = energy_data['Import Dependency (%)'].idxmax()
    min_dep_year = energy_data['Import Dependency (%)'].idxmin()
    print(f"• Switzerland reached peak import dependency of {energy_data['Import Dependency (%)'].max():.1f}% in {max_dep_year}")
    print(f"• Lowest import dependency was {energy_data['Import Dependency (%)'].min():.1f}% in {min_dep_year}")
    
    # Compare recent vs historical
    first_decade = energy_data.head(10)['Import Dependency (%)'].mean()
    last_decade = energy_data.tail(10)['Import Dependency (%)'].mean()
    change = last_decade - first_decade
    print(f"• Import dependency has {'increased' if change > 0 else 'decreased'} by {abs(change):.1f}% from the 1990s to 2015-2024")
    
    print("\n" + "="*70)

def export_energy_dependency_to_excel(energy_data, output_dir):
    """Export energy import dependency data to Excel format"""
    
    # Create DataFrame with required structure
    export_data = []
    
    for year in energy_data.index:
        row = {
            'visual_number': np.nan,
            'visual_name': 'Energy Import Dependency Main',
            'year': year,
            'filter': 'Total Energy',
            'decile': np.nan,
            'value': energy_data.loc[year, 'Import Dependency (%)'],
            'unit': '%'
        }
        export_data.append(row)
    
    df = pd.DataFrame(export_data)
    
    # Export to Excel
    excel_file = output_dir / "energy_import_dependency_main.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"✓ Energy import dependency Excel exported to: {excel_file}")
    
    return excel_file

def export_net_imports_to_excel(net_imports_df, output_dir):
    """Export net imports by type data to Excel format"""
    
    # Create DataFrame with required structure
    export_data = []
    
    for year in net_imports_df.index:
        for energy_type in net_imports_df.columns:
            row = {
                'visual_number': np.nan,
                'visual_name': 'Net Imports by Type Area Chart',
                'year': year,
                'filter': energy_type,
                'decile': np.nan,
                'value': net_imports_df.loc[year, energy_type],
                'unit': 'TJ'
            }
            export_data.append(row)
    
    df = pd.DataFrame(export_data)
    
    # Export to Excel
    excel_file = output_dir / "energy_net_imports_by_type_area.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"✓ Net imports by type Excel exported to: {excel_file}")
    
    return excel_file

def print_net_imports_summary(net_imports_df):
    """Print summary statistics for net imports by energy type"""
    print("\n" + "="*70)
    print("NET IMPORTS BY ENERGY TYPE SUMMARY")
    print("="*70)
    
    print(f"\nData Period: {net_imports_df.index.min()} - {net_imports_df.index.max()}")
    
    for energy_type in net_imports_df.columns:
        values = net_imports_df[energy_type]
        print(f"\n{energy_type.upper()}:")
        print(f"  Average Net Imports: {values.mean():,.0f} TJ")
        print(f"  2024 Net Imports: {values.iloc[-1]:,.0f} TJ")
        print(f"  Range: {values.min():,.0f} to {values.max():,.0f} TJ")
        
        # Check if there are net exports (negative values)
        negative_years = (values < 0).sum()
        if negative_years > 0:
            print(f"  Years with Net Exports: {negative_years} out of {len(values)}")
            min_year = values.idxmin()
            print(f"  Largest Net Export: {values.min():,.0f} TJ in {min_year}")
        else:
            print(f"  Always Net Importer (no exports > imports)")
    
    print("\n" + "="*70)

def load_transport_consumption_data():
    """Load and process transport consumption data by subsector"""
    
    base_path = Path(__file__).parent.parent / "external_data"
    file_path = base_path / "iea_Total consumption Transport.csv"
    
    print(f"Loading transport consumption data: {file_path}")
    
    # Load data (skip first 3 metadata rows)
    df = pd.read_csv(file_path, skiprows=3, index_col=0)
    df.index.name = 'Year'
    
    # Remove Units column if it exists
    if 'Units' in df.columns:
        df = df.drop('Units', axis=1)
    
    # Calculate total transport consumption
    df['Total Transport'] = df.sum(axis=1)
    
    print(f"✓ Loaded transport data for {len(df)} years ({df.index.min()}-{df.index.max()})")
    print(f"✓ Subsectors: {list(df.columns[:-1])}")  # Exclude 'Total Transport'
    
    return df

def create_transport_raw_data_visualization(transport_df, output_dir):
    """Create visualization of raw transport consumption data"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
    
    # Colors for different transport subsectors
    colors = {
        'Cars/Light Trucks': '#2E8B57',        # Sea Green
        'Passenger transport': '#4682B4',       # Steel Blue
        'Trucks': '#FF8C00',                   # Dark Orange
        'Freight transport': '#DC143C',        # Crimson
        'Total Transport': '#000000'           # Black
    }
    
    years = transport_df.index
    
    # Plot 1: Individual subsectors
    ax1.set_title('Switzerland Transport Energy Consumption by Subsector\n(Raw Data)', 
                  fontsize=16, fontweight='bold', pad=20)
    
    subsectors = [col for col in transport_df.columns if col != 'Total Transport']
    
    for subsector in subsectors:
        ax1.plot(years, transport_df[subsector], 
                marker='o', linewidth=2.5, markersize=4,
                color=colors[subsector], label=subsector)
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Energy Consumption (PJ)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Add some statistics text
    max_year = transport_df['Total Transport'].idxmax()
    min_year = transport_df['Total Transport'].idxmin()
    ax1.text(0.02, 0.98, 
             f'Peak consumption: {max_year} ({transport_df.loc[max_year, "Total Transport"]:.1f} PJ)\n'
             f'Minimum consumption: {min_year} ({transport_df.loc[min_year, "Total Transport"]:.1f} PJ)',
             transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Stacked area chart
    ax2.set_title('Switzerland Transport Energy Consumption\n(Stacked Area Chart)', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Create stacked area plot
    subsector_data = transport_df[subsectors]
    ax2.stackplot(years, *[subsector_data[col] for col in subsectors], 
                  labels=subsectors, 
                  colors=[colors[col] for col in subsectors],
                  alpha=0.8)
    
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Energy Consumption (PJ)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / "switzerland_transport_consumption_raw.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Transport raw data visualization saved: {output_file}")
    
    return fig

def create_transport_per_capita_visualization(transport_df, output_dir, population_data=None):
    """Create per capita transport consumption visualization"""
    
    if population_data is None:
        print("⚠ No population data provided - creating placeholder visualization")
        # Use estimate Swiss population (around 8.7-8.8 million inhabitants)
        # This is for demonstration - real population data should be used
        swiss_population_estimate = {
            year: 8.0 + (year - 2000) * 0.04  # Population in millions
            for year in transport_df.index
        }
        population_series = pd.Series(swiss_population_estimate, name='Population (millions)')
        print("⚠ Using estimated population data for demonstration")
    else:
        population_series = population_data
    
    # Calculate per capita consumption (PJ per million inhabitants = GJ per inhabitant)
    per_capita_df = transport_df.copy()
    for col in per_capita_df.columns:
        per_capita_df[col] = per_capita_df[col] / population_series  # PJ / million inhabitants = GJ per inhabitant
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Colors for different transport subsectors
    colors = {
        'Cars/Light Trucks': '#2E8B57',        # Sea Green
        'Passenger transport': '#4682B4',       # Steel Blue  
        'Trucks': '#FF8C00',                   # Dark Orange
        'Freight transport': '#DC143C',        # Crimson
        'Total Transport': '#000000'           # Black
    }
    
    years = per_capita_df.index
    
    # Per capita consumption by subsector
    ax.set_title('Switzerland Transport Energy Consumption Per Capita by Subsector', 
                  fontsize=16, fontweight='bold', pad=20)
    
    subsectors = [col for col in per_capita_df.columns if col != 'Total Transport']
    
    for subsector in subsectors:
        ax.plot(years, per_capita_df[subsector], 
                marker='o', linewidth=2.5, markersize=4,
                color=colors[subsector], label=subsector)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Energy Consumption (GJ per inhabitant)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Ensure y-axis includes 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / "switzerland_transport_consumption_per_capita.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Transport per capita visualization saved: {output_file}")
    
    return fig, per_capita_df

def export_transport_raw_to_excel(transport_df, output_dir):
    """Export transport raw consumption data to Excel format"""
    
    # Create DataFrame with required structure
    export_data = []
    
    for year in transport_df.index:
        for subsector in transport_df.columns:
            row = {
                'visual_number': np.nan,
                'visual_name': 'Transport Consumption Raw Data',
                'year': year,
                'filter': subsector,
                'decile': np.nan,
                'value': transport_df.loc[year, subsector],
                'unit': 'PJ'
            }
            export_data.append(row)
    
    df = pd.DataFrame(export_data)
    
    # Export to Excel
    excel_file = output_dir / "transport_consumption_raw.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"✓ Transport raw consumption Excel exported to: {excel_file}")
    
    return excel_file

def export_transport_per_capita_to_excel(per_capita_df, output_dir):
    """Export transport per capita consumption data to Excel format"""
    
    # Create DataFrame with required structure
    export_data = []
    
    for year in per_capita_df.index:
        for subsector in per_capita_df.columns:
            row = {
                'visual_number': np.nan,
                'visual_name': 'Transport Consumption Per Capita',
                'year': year,
                'filter': subsector,
                'decile': np.nan,
                'value': per_capita_df.loc[year, subsector],
                'unit': 'GJ per inhabitant'
            }
            export_data.append(row)
    
    df = pd.DataFrame(export_data)
    
    # Export to Excel
    excel_file = output_dir / "transport_consumption_per_capita.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"✓ Transport per capita consumption Excel exported to: {excel_file}")
    
    return excel_file

def main():
    """Main analysis function"""
    print("=" * 70)
    print("SWITZERLAND ENERGY DEPENDENCY ANALYSIS")
    print("Net Energy Imports / Total Energy Supply Analysis")
    print("=" * 70)
    
    try:
        # Setup output directory
        output_dir = Path(__file__).parent.parent / "outputs" / "graphs" / "energy"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
        
        # Load and process data
        print("\nLoading and processing IEA energy data for Switzerland...")
        energy_data = load_and_process_data()
        print(f"✓ Successfully loaded data for {len(energy_data)} years ({energy_data.index.min()}-{energy_data.index.max()})")
        
        # Load energy import/export data for detailed analysis
        print("\nLoading energy import/export data by type...")
        net_imports_df = load_energy_import_export_data()
        print(f"✓ Successfully loaded import/export data for {len(net_imports_df.columns)} energy types")
        
        # Create net imports area chart
        print("\nCreating net imports area chart...")
        area_plot = create_net_imports_area_chart(net_imports_df, output_dir)
        
        # Create individual charts  
        print("Creating individual net imports charts...")
        individual_plot = create_individual_net_imports_charts(net_imports_df, output_dir)
        
        # Create main visualization (the key graph requested)
        print("\nCreating main visualization...")
        main_plot = create_main_visualization(energy_data, output_dir)
        
        # Create comprehensive visualization
        print("Creating comprehensive analysis...")
        comp_plot = create_comprehensive_visualization(energy_data, output_dir)
        
        # Load and analyze transport consumption data
        print("\n" + "="*50)
        print("TRANSPORT CONSUMPTION ANALYSIS")
        print("="*50)
        
        # Load transport data
        print("Loading transport consumption data...")
        transport_data = load_transport_consumption_data()
        
        # Create raw data visualization
        print("Creating transport raw data visualization...")
        transport_raw_plot = create_transport_raw_data_visualization(transport_data, output_dir)
        
        # Create per capita visualization
        print("Creating transport per capita visualization...")
        transport_per_capita_plot, per_capita_data = create_transport_per_capita_visualization(
            transport_data, output_dir)
        
        # Export transport data to Excel
        print("Exporting transport data to Excel...")
        transport_raw_excel = export_transport_raw_to_excel(transport_data, output_dir)
        transport_per_capita_excel = export_transport_per_capita_to_excel(per_capita_data, output_dir)
        
        # Save transport data to CSV (optional backup)
        transport_file = output_dir / "switzerland_transport_consumption.csv"
        transport_data.to_csv(transport_file)
        print(f"✓ Transport data saved to: {transport_file}")
        
        per_capita_file = output_dir / "switzerland_transport_consumption_per_capita.csv"
        per_capita_data.to_csv(per_capita_file)
        print(f"✓ Per capita transport data saved to: {per_capita_file}")
        
        # Save net imports data
        net_imports_file = output_dir / "switzerland_net_imports_by_type.csv"
        net_imports_df.to_csv(net_imports_file)
        print(f"✓ Net imports data saved to: {net_imports_file}")
        
        # Save processed data
        data_file = output_dir / "switzerland_energy_dependency_data.csv"
        energy_data.to_csv(data_file)
        print(f"✓ Processed data saved to: {data_file}")
        
        # Export Excel files with specified structure
        print("\nExporting Excel files...")
        export_energy_dependency_to_excel(energy_data, output_dir)
        export_net_imports_to_excel(net_imports_df, output_dir)
        
        # Create a summary CSV with key metrics by year
        summary_data = energy_data[['Net imports (TJ)', 'Total Supply (TJ)', 'Import Dependency (%)']].copy()
        summary_data['Net imports (PJ)'] = summary_data['Net imports (TJ)'] / 1000
        summary_data['Total Supply (PJ)'] = summary_data['Total Supply (TJ)'] / 1000
        
        summary_file = output_dir / "switzerland_energy_summary.csv"
        summary_data.to_csv(summary_file)
        print(f"✓ Summary data saved to: {summary_file}")
        
        # Print summary statistics
        print_summary_statistics(energy_data)
        
        # Print net imports summary
        print_net_imports_summary(net_imports_df)
        
        print(f"\n✓ Analysis complete! Check {output_dir} for all outputs.")
        
        return energy_data, net_imports_df, transport_data, per_capita_data
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()