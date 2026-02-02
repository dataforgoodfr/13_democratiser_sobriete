"""
FSO Household Expense Analysis - Switzerland
==============================================
Creates stacked bar plots of household expenses by income quintiles.
Visualizes spending patterns across income groups for key expense categories.

Data Source: FSO (Swiss Federal Statistical Office)
Year: 2020-2021
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_FILE = Path(__file__).parent.parent / "external_data" / "fso_household_expense.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "graphs" / "FSO"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette (from hbs_test_fr_2020.py)
COMPONENT_COLORS = {
    'Housing and Energy': '#ffd558',       # Yellow
    'Transport': '#fb8072',                # Red/Orange
    'Food': '#b3de69',                     # Green
    'Health': '#fdb462',                   # Orange
    'Education': '#bebada',                # Purple
    'Compulsory transfers, taxes and insurance': '#8dd3c7',     # Teal
    'Remaining': '#cccccc'                 # Grey
}

# Subcategory definitions with hatching patterns
SUBCATEGORIES = {
    'Housing and Energy': {
        '5711: Net rent/mortgage': {'code': '5711', 'hatch': None, 'label': 'Net rent/mortgage'},
        '5712: Maintenance': {'code': '5712', 'hatch': '\\', 'label': 'Maintenance'},
        '5713: Energy': {'code': '5713', 'hatch': '|', 'label': 'Energy'},
        '573: Home repairs': {'code': '573', 'hatch': '-', 'label': 'Home repairs'},
        '4201: Property insurance': {'code': '4201', 'hatch': '+', 'label': 'Property insurance'}
    },
    'Transport': {
        '621: Private vehicles': {'code': '621', 'hatch': None, 'label': 'Private vehicles'},
        '622: Transport services': {'code': '622', 'hatch': '\\', 'label': 'Transport services'},
        '4202: Vehicle insurance': {'code': '4202', 'hatch': 'x', 'label': 'Vehicle insurance'}
    },
    'Health': {
        '61: Health expenditure': {'code': '61', 'hatch': None, 'label': 'Health expenditure'},
        '33: Basic health insurance': {'code': '33', 'hatch': '\\', 'label': 'Basic health insurance'},
        '41: Supplementary insurance': {'code': '41', 'hatch': '|', 'label': 'Supplementary insurance'}
    }
}

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_expense_data():
    """Load FSO household expense CSV file."""
    # Read with semicolon delimiter and clean up
    df = pd.read_csv(DATA_FILE, sep=';', decimal=',', encoding='latin-1')
    
    # Strip whitespace from column names and values
    df.columns = df.columns.str.strip()
    
    # Clean up column names: replace special dashes between numbers with regular hyphen
    import re
    new_columns = []
    for col in df.columns:
        # Replace pattern: number + space + special-dash + space + number with number - number
        col_clean = re.sub(r'(\d+)\s+[–—]\s+(\d+)', r'\1 - \2', col)
        new_columns.append(col_clean)
    df.columns = new_columns
    
    # Get first column name (the type column)
    type_col = df.columns[0]
    df[type_col] = df[type_col].str.strip()
    
    # Replace numeric separators in other columns
    for col in df.columns[1:]:
        if isinstance(df[col].iloc[0], str):
            df[col] = df[col].str.replace(' ', '').str.replace(',', '.').astype(float)
    
    print(f"Loaded columns: {list(df.columns)}")
    
    return df


def extract_category_codes(row_name):
    """Extract numeric code from row name (e.g., '5711' from '5711: Net rent...')"""
    parts = row_name.split(':')
    if len(parts) > 0:
        try:
            code = parts[0].strip()
            if code.isdigit() or ('.' in code and code.replace('.', '').isdigit()):
                return code
        except:
            pass
    return None


def aggregate_expenses(df):
    """Aggregate expense categories into main groups with subcategories."""
    
    # Build code mapping from SUBCATEGORIES
    code_to_subcat = {}
    for main_cat, subcats in SUBCATEGORIES.items():
        for subcat_name, info in subcats.items():
            code_to_subcat[info['code']] = (main_cat, subcat_name)
    
    # Categories that will be adjusted for Compulsory transfers
    compulsory_base_codes = ['30', '35', '40']  # To be added
    compulsory_subtract_codes = ['33', '41', '4201', '4202']  # To be subtracted (double-counted)
    
    # Simple categories (no subcategories)
    simple_categories = {
        '51': 'Food',
        '67': 'Education'
    }
    
    # Get the first column name (type column)
    type_col = df.columns[0]
    
    # Extract codes
    df['Code'] = df[type_col].apply(extract_category_codes)
    
    # Get income columns (all except first column and 'Code')
    income_columns = [col for col in df.columns if col not in [type_col, 'Code']]
    
    aggregated = {}
    for col in income_columns:
        aggregated[col] = {'Gross Income': 0}
        
        # Initialize subcategories
        for main_cat, subcats in SUBCATEGORIES.items():
            for subcat_name in subcats.keys():
                aggregated[col][subcat_name] = 0
        
        # Initialize simple categories
        for cat in simple_categories.values():
            aggregated[col][cat] = 0
        
        aggregated[col]['Compulsory transfers, taxes and insurance'] = 0
        aggregated[col]['Remaining'] = 0
        
        for idx, row in df.iterrows():
            row_type = row[type_col]
            code = row['Code']
            value = row[col]
            
            # Get gross income
            if 'Gross income' in row_type:
                aggregated[col]['Gross Income'] = value
            
            # Categorize with subcategories
            if code in code_to_subcat:
                main_cat, subcat_name = code_to_subcat[code]
                aggregated[col][subcat_name] += value
            
            # Simple categories
            if code in simple_categories:
                cat = simple_categories[code]
                aggregated[col][cat] += value
            
            # Compulsory transfers base (to be added)
            if code in compulsory_base_codes:
                aggregated[col]['Compulsory transfers, taxes and insurance'] += value
            
            # Compulsory transfers subtract (double-counted items that are already in Health/Transport)
            if code in compulsory_subtract_codes:
                aggregated[col]['Compulsory transfers, taxes and insurance'] -= value
    
    # Calculate remaining as Gross Income - sum of all expenses
    for col in aggregated.keys():
        total_expenses = 0
        for key, value in aggregated[col].items():
            if key not in ['Gross Income', 'Remaining']:
                total_expenses += value
        aggregated[col]['Remaining'] = aggregated[col]['Gross Income'] - total_expenses
        
        # Debug output for all income groups
        print(f"\n=== DEBUG: {col} ===")
        print(f"Gross Income: {aggregated[col]['Gross Income']:.2f} CHF")
        print(f"Total Expenses: {total_expenses:.2f} CHF")
        print(f"Remaining: {aggregated[col]['Remaining']:.2f} CHF")
        if aggregated[col]['Gross Income'] > 0:
            print(f"Remaining %: {(aggregated[col]['Remaining'] / aggregated[col]['Gross Income'] * 100):.2f}%")
        print("\nBreakdown:")
        for key, value in sorted(aggregated[col].items()):
            if key not in ['Gross Income', 'Remaining'] and value != 0:
                pct = (value / aggregated[col]['Gross Income'] * 100) if aggregated[col]['Gross Income'] > 0 else 0
                print(f"  {key}: {value:.2f} CHF ({pct:.2f}%)")
    
    return aggregated


def create_stacked_barplot(aggregated_data):
    """Create stacked bar plot of household expenses by income quintile with subcategories."""
    
    # Get actual income columns from the data
    income_levels = list(aggregated_data.keys())
    
    # Build expense order with subcategories (bottom to top)
    expense_order = []
    colors = []
    hatches = []
    
    # Housing and Energy subcategories
    for subcat_name, info in SUBCATEGORIES['Housing and Energy'].items():
        expense_order.append(subcat_name)
        colors.append(COMPONENT_COLORS['Housing and Energy'])
        hatches.append(info['hatch'])
    
    # Transport subcategories
    for subcat_name, info in SUBCATEGORIES['Transport'].items():
        expense_order.append(subcat_name)
        colors.append(COMPONENT_COLORS['Transport'])
        hatches.append(info['hatch'])
    
    # Food (no subcategories)
    expense_order.append('Food')
    colors.append(COMPONENT_COLORS['Food'])
    hatches.append(None)
    
    # Health subcategories
    for subcat_name, info in SUBCATEGORIES['Health'].items():
        expense_order.append(subcat_name)
        colors.append(COMPONENT_COLORS['Health'])
        hatches.append(info['hatch'])
    
    # Education (no subcategories)
    expense_order.append('Education')
    colors.append(COMPONENT_COLORS['Education'])
    hatches.append(None)
    
    # Compulsory transfers, taxes and insurance (no subcategories)
    expense_order.append('Compulsory transfers, taxes and insurance')
    colors.append(COMPONENT_COLORS['Compulsory transfers, taxes and insurance'])
    hatches.append(None)
    
    # Remaining
    expense_order.append('Remaining')
    colors.append(COMPONENT_COLORS['Remaining'])
    hatches.append(None)
    
    # Prepare data for plotting
    plot_data = aggregated_data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Convert to DataFrame for easier plotting
    plot_df = pd.DataFrame(plot_data).T
    plot_df = plot_df[expense_order]
    
    print(f"Plot data shape: {plot_df.shape}")
    print(f"Income levels: {list(plot_df.index)}")
    
    # Create stacked bar plot manually to apply hatching
    x = np.arange(len(income_levels))
    width = 0.6
    
    bottom = np.zeros(len(income_levels))
    bars = []
    labels = []
    
    # First, plot positive expenses
    for i, col in enumerate(expense_order):
        values = plot_df[col].values
        
        # Skip Remaining for now (plot it separately at the end)
        if col == 'Remaining':
            continue
        
        bar = ax.bar(x, values, width, bottom=bottom, 
                    color=colors[i], edgecolor='black', linewidth=0.5,
                    hatch=hatches[i], alpha=0.85, label=col)
        bars.append(bar)
        
        # Get label for legend
        if col in SUBCATEGORIES['Housing and Energy']:
            labels.append(f"H&E: {SUBCATEGORIES['Housing and Energy'][col]['label']}")
        elif col in SUBCATEGORIES['Transport']:
            labels.append(f"Trans: {SUBCATEGORIES['Transport'][col]['label']}")
        elif col in SUBCATEGORIES['Health']:
            labels.append(f"Health: {SUBCATEGORIES['Health'][col]['label']}")
        else:
            labels.append(col)
        
        bottom += values
    
    # Now plot Remaining separately (can be positive or negative)
    remaining_idx = expense_order.index('Remaining')
    remaining_values = plot_df['Remaining'].values
    
    # Split into positive and negative parts
    positive_remaining = np.where(remaining_values > 0, remaining_values, 0)
    negative_remaining = np.where(remaining_values < 0, remaining_values, 0)
    
    # Plot positive Remaining on top of the stack
    if np.any(positive_remaining > 0):
        bar = ax.bar(x, positive_remaining, width, bottom=bottom, 
                    color=colors[remaining_idx], edgecolor='black', linewidth=0.5,
                    hatch=hatches[remaining_idx], alpha=0.85, label='Remaining')
        bars.append(bar)
        labels.append('Remaining')
    
    # Plot negative Remaining below baseline
    if np.any(negative_remaining < 0):
        bar = ax.bar(x, negative_remaining, width, bottom=0, 
                    color=colors[remaining_idx], edgecolor='black', linewidth=0.5,
                    hatch=hatches[remaining_idx], alpha=0.85)
        if np.all(positive_remaining == 0):  # Only add label if not already added
            bars.append(bar)
            labels.append('Remaining')
        
        bottom += values
    
    # Formatting
    ax.set_title('Household Expenses by Income Quintile\nSwitzerland, 2020-2021',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Income Group (Monthly gross income in CHF)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Monthly Expense (CHF)', fontsize=12, fontweight='bold')
    
    # Set x-axis labels correctly
    ax.set_xticks(x)
    ax.set_xticklabels(income_levels, rotation=45, ha='right')
    
    # Add grid for readability
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add dotted line at y=0 (no label - don't show in legend)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Add Gross Income value labels (without line)
    gross_income_values = [aggregated_data[level]['Gross Income'] for level in income_levels]
    y_max = max([plot_df.loc[level].sum() for level in income_levels])  # Max stack height
    for i, (xi, yi) in enumerate(zip(x, gross_income_values)):
        ax.text(xi, yi + y_max * 0.02, f'{yi:.0f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkblue',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkblue', alpha=0.8))
    
    # Legend - use bars and labels explicitly
    ax.legend(
        bars,
        labels,
        title='Expense Category',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=9
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "fso_stacked_expenses_by_income.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Stacked bar plot saved: {output_path}")
    
    plt.close()


def create_percentage_barplot(aggregated_data):
    """Create percentage stacked bar plot (% distribution) by income quintile."""
    
    # Get actual income columns from the data
    income_levels = list(aggregated_data.keys())
    
    # Build expense order with subcategories (bottom to top) - SAME AS ABSOLUTE CHART
    expense_order = []
    colors = []
    hatches = []
    
    # Housing and Energy subcategories
    for subcat_name, info in SUBCATEGORIES['Housing and Energy'].items():
        expense_order.append(subcat_name)
        colors.append(COMPONENT_COLORS['Housing and Energy'])
        hatches.append(info['hatch'])
    
    # Transport subcategories
    for subcat_name, info in SUBCATEGORIES['Transport'].items():
        expense_order.append(subcat_name)
        colors.append(COMPONENT_COLORS['Transport'])
        hatches.append(info['hatch'])
    
    # Food (no subcategories)
    expense_order.append('Food')
    colors.append(COMPONENT_COLORS['Food'])
    hatches.append(None)
    
    # Health subcategories
    for subcat_name, info in SUBCATEGORIES['Health'].items():
        expense_order.append(subcat_name)
        colors.append(COMPONENT_COLORS['Health'])
        hatches.append(info['hatch'])
    
    # Education (no subcategories)
    expense_order.append('Education')
    colors.append(COMPONENT_COLORS['Education'])
    hatches.append(None)
    
    # Compulsory transfers, taxes and insurance (no subcategories)
    expense_order.append('Compulsory transfers, taxes and insurance')
    colors.append(COMPONENT_COLORS['Compulsory transfers, taxes and insurance'])
    hatches.append(None)
    
    # Remaining
    expense_order.append('Remaining')
    colors.append(COMPONENT_COLORS['Remaining'])
    hatches.append(None)
    
    # Prepare data for plotting
    plot_data = aggregated_data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Convert to DataFrame - USE EXACT SAME ORDER
    plot_df = pd.DataFrame(plot_data).T
    plot_df = plot_df[expense_order]
    
    # Convert to percentages: divide each value by GROSS INCOME (not sum of categories)
    # This shows each category as % of income
    plot_df_pct = plot_df.copy()
    for idx in plot_df_pct.index:
        gross_income = aggregated_data[idx]['Gross Income']
        if gross_income > 0:
            plot_df_pct.loc[idx] = (plot_df.loc[idx] / gross_income) * 100
        else:
            plot_df_pct.loc[idx] = 0
    
    # Create stacked bar plot manually to apply hatching
    x = np.arange(len(income_levels))
    width = 0.6
    
    bottom = np.zeros(len(income_levels))
    bars = []
    labels = []
    
    # First, plot positive expenses
    for i, col in enumerate(expense_order):
        values = plot_df_pct[col].values
        
        # Skip Remaining for now (plot it separately at the end)
        if col == 'Remaining':
            continue
        
        bar = ax.bar(x, values, width, bottom=bottom, 
                    color=colors[i], edgecolor='black', linewidth=0.5,
                    hatch=hatches[i], alpha=0.85, label=col)
        bars.append(bar)
        
        # Get label for legend
        if col in SUBCATEGORIES['Housing and Energy']:
            labels.append(f"H&E: {SUBCATEGORIES['Housing and Energy'][col]['label']}")
        elif col in SUBCATEGORIES['Transport']:
            labels.append(f"Trans: {SUBCATEGORIES['Transport'][col]['label']}")
        elif col in SUBCATEGORIES['Health']:
            labels.append(f"Health: {SUBCATEGORIES['Health'][col]['label']}")
        else:
            labels.append(col)
        
        bottom += values
    
    # Now plot Remaining separately (can be positive or negative)
    remaining_idx = expense_order.index('Remaining')
    remaining_values = plot_df_pct['Remaining'].values
    
    # Split into positive and negative parts
    positive_remaining = np.where(remaining_values > 0, remaining_values, 0)
    negative_remaining = np.where(remaining_values < 0, remaining_values, 0)
    
    # Plot positive Remaining on top of the stack
    if np.any(positive_remaining > 0):
        bar = ax.bar(x, positive_remaining, width, bottom=bottom, 
                    color=colors[remaining_idx], edgecolor='black', linewidth=0.5,
                    hatch=hatches[remaining_idx], alpha=0.85, label='Remaining')
        bars.append(bar)
        labels.append('Remaining')
    
    # Plot negative Remaining below baseline
    if np.any(negative_remaining < 0):
        bar = ax.bar(x, negative_remaining, width, bottom=0, 
                    color=colors[remaining_idx], edgecolor='black', linewidth=0.5,
                    hatch=hatches[remaining_idx], alpha=0.85)
        if np.all(positive_remaining == 0):  # Only add label if not already added
            bars.append(bar)
            labels.append('Remaining')
        
        bottom += values
    
    # Formatting
    ax.set_title('Household Expenses by Income Quintile (% of Gross Income)\nSwitzerland, 2020-2021',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Income Group (Monthly gross income in CHF)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Share of Gross Income (%)', fontsize=12, fontweight='bold')
    
    # Set x-axis labels correctly
    ax.set_xticks(x)
    ax.set_xticklabels(income_levels, rotation=45, ha='right')
    
    # Add grid for readability
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set y-axis to 0-100%
    ax.set_ylim(0, 100)
    
    # Add percentage labels to bars (skip if too small)
    cumulative_bottom = np.zeros(len(income_levels))
    for col_idx, col in enumerate(expense_order):
        values = plot_df_pct[col].values
        for i, val in enumerate(values):
            if val >= 3.0:  # Only show if >= 3%
                y_pos = cumulative_bottom[i] + val / 2
                ax.text(x[i], y_pos, f'{val:.1f}%', 
                       ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        cumulative_bottom += values
    
    # Legend
    ax.legend(
        labels,
        title='Expense Category',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=9
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "fso_stacked_expenses_percentage.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Percentage stacked bar plot saved: {output_path}")
    
    plt.close()


def create_detailed_comparison(aggregated_data):
    """Create a more detailed comparison showing absolute and percentage values."""
    
    available_columns = list(aggregated_data.keys())
    
    income_order = [
        'Total',
        'Less than 4,669',
        '4 669 – 7 004',
        '7 005 – 9 733',
        '9 734 – 13 716',
        'From 13,717'
    ]
    
    # Filter to only available columns
    income_levels = [col for col in income_order if col in available_columns]
    
    expense_order = [
        'Housing and Energy',
        'Transport',
        'Food',
        'Health',
        'Education',
        'Compulsory Transfers',
        'Remaining'
    ]
    
    # Prepare data
    plot_data = {}
    for col in income_levels:
        if col in aggregated_data:
            plot_data[col] = aggregated_data[col]
    
    plot_df = pd.DataFrame(plot_data).T
    plot_df = plot_df[expense_order]
    
    # Calculate percentages (as share of gross income)
    pct_df = plot_df.copy()
    for col in pct_df.columns:
        gross_income = aggregated_data[income_levels[0]]['Gross Income'] if income_levels[0] in aggregated_data else aggregated_data[list(aggregated_data.keys())[0]]['Gross Income']
        pct_df[col] = (pct_df[col] / gross_income * 100) if col != 'Remaining' else (pct_df[col] / plot_df.sum(axis=1) * 100)
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Absolute values
    plot_df.plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        color=[COMPONENT_COLORS[cat] for cat in expense_order],
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85,
        legend=False
    )
    ax1.set_title('Household Expenses (CHF)\nSwitzerland, 2020-2021', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Income Group', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Monthly Expense (CHF)', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(len(income_levels)))
    ax1.set_xticklabels(income_levels, rotation=45, ha='right')
    ax1.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Percentage values
    plot_df_pct = plot_df.div(plot_df.sum(axis=1), axis=0) * 100
    plot_df_pct.plot(
        kind='bar',
        stacked=True,
        ax=ax2,
        color=[COMPONENT_COLORS[cat] for cat in expense_order],
        edgecolor='black',
        linewidth=1.0,
        alpha=0.85
    )
    ax2.set_title('Expense Share (% of Total Expense)\nSwitzerland, 2020-2021', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Income Group', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Share of Total Expense (%)', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(len(income_levels)))
    ax2.set_xticklabels(income_levels, rotation=45, ha='right')
    ax2.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Combined legend
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
              ncol=4, frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2.legend_.remove()
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "fso_expenses_comparison_absolute_vs_share.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved: {output_path}")
    
    plt.close()


def main():
    """Main analysis function."""
    print("="*70)
    print("FSO Household Expense Analysis - Switzerland")
    print("="*70)
    print(f"Data file: {DATA_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)
    
    # Load data
    print("\nLoading expense data...")
    df = load_expense_data()
    print(f"Loaded {len(df)} expense categories")
    
    # Aggregate expenses
    print("\nAggregating expenses into main categories...")
    aggregated_data = aggregate_expenses(df)
    print(f"Created {len(aggregated_data)} income group aggregations")
    
    # Create visualizations
    print("\nCreating stacked bar plot...")
    create_stacked_barplot(aggregated_data)
    
    print("\nCreating percentage stacked bar plot...")
    create_percentage_barplot(aggregated_data)
    
    # Save aggregated data to CSV
    print("\nSaving aggregated data to CSV...")
    csv_data = []
    for income_level, data in aggregated_data.items():
        for category, value in data.items():
            csv_data.append({
                'Income_Level': income_level,
                'Category': category,
                'Value': value
            })
    df_csv = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / "fso_expenses_aggregated.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"Aggregated data saved: {csv_path}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
