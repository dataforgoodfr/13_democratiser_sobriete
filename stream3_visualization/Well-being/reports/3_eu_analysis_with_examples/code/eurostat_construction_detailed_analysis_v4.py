"""
EUROSTAT Construction Industry Analysis - Detailed NACE v4
===========================================================
Analyzes detailed NACE hierarchical breakdown with productivity metrics
FIXED: Correct year handling for different indicators
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ============================================================================
# SETUP
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'external_data'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'graphs' / 'EUROSTAT_construction_industry'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EUROSTAT CONSTRUCTION DETAILED NACE ANALYSIS v4")
print("="*80)

# Color schemes - Colors grouped by parent NACE hierarchy (from oecd_analysis.py palette)
DETAILED_NACE_COLORS = {
    # F41: Construction of buildings (green shades)
    'Development of building projects': '#2ca02c',
    'Construction of residential and non-residential buildings': '#2ca02c',
    
    # F42: Civil engineering (red shades)
    'Construction of roads and railways': '#d62728',
    'Construction of utility projects': '#d62728',
    'Construction of other civil engineering projects': '#d62728',
    
    # F43: Specialised construction activities (purple shades)
    'Demolition and site preparation': '#9467bd',
    'Electrical, plumbing and other construction installation activities': '#9467bd',
    'Building completion and finishing': '#9467bd',
    'Other specialised construction activities': '#9467bd',
}

HIERARCHY = {
    'Construction of buildings': {
        'name': 'F41: Construction of Buildings',
        'children': ['Development of building projects',
                    'Construction of residential and non-residential buildings']
    },
    'Civil engineering': {
        'name': 'F42: Civil Engineering',
        'children': ['Construction of roads and railways',
                    'Construction of utility projects',
                    'Construction of other civil engineering projects']
    },
    'Specialised construction activities': {
        'name': 'F43: Specialised Construction',
        'children': ['Demolition and site preparation',
                    'Electrical, plumbing and other construction installation activities',
                    'Building completion and finishing',
                    'Other specialised construction activities']
    }
}

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")
const_df = pd.read_csv(DATA_DIR / 'construction_employment_size_bis.csv')
print(f"Loaded {len(const_df)} records")

# Get detailed NACE only (not parent categories)
detailed_nace = []
for parent_info in HIERARCHY.values():
    detailed_nace.extend(parent_info['children'])

print(f"Detailed NACE sectors: {len(detailed_nace)}")

# ============================================================================
# VIS 7: Value Added Per Employee
# ============================================================================

print("\n--- Visualization 7: Value Added per Employee ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Use 2023 for productivity metrics (latest available)
latest_productivity_year = 2023

for idx, country in enumerate(['European Union - 27 countries (from 2020)', 'France']):
    ax = axes[idx]
    
    data = const_df[
        (const_df['TIME_PERIOD'] == latest_productivity_year) &
        (const_df['indic_sbs'] == 'Value added per employee - thousand euro') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['geo'] == country) &
        (const_df['size_emp'] == 'Total')
    ].dropna(subset=['OBS_VALUE']).copy()
    
    if not data.empty:
        data = data.sort_values('OBS_VALUE', ascending=True)
        colors = [DETAILED_NACE_COLORS.get(row['nace_r2'], '#cccccc') for _, row in data.iterrows()]
        
        bars = ax.barh(range(len(data)), data['OBS_VALUE'], color=colors, edgecolor='black', linewidth=1)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([row['nace_r2'][:35] + '...' if len(row['nace_r2']) > 35 else row['nace_r2'] 
                            for _, row in data.iterrows()], fontsize=9)
        ax.set_xlabel('Value Added per Employee (€ thousand)', fontsize=11, fontweight='bold')
        
        country_label = 'EU27' if 'Union' in country else country
        ax.set_title(f'{country_label} ({latest_productivity_year})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (_, row) in enumerate(data.iterrows()):
            ax.text(row['OBS_VALUE'] + 1, i, f"{row['OBS_VALUE']:.1f}€", 
                   va='center', fontsize=8, fontweight='bold')
        
        # Add parent NACE legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', edgecolor='black', label='F41: Construction of buildings'),
            Patch(facecolor='#d62728', edgecolor='black', label='F42: Civil engineering'),
            Patch(facecolor='#9467bd', edgecolor='black', label='F43: Specialised construction')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('Value Added per Employee by Detailed NACE', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '7_value_added_per_employee_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("  OK Saved: 7_value_added_per_employee_detailed.png")

# ============================================================================
# VIS 7B: Labour Productivity
# ============================================================================

print("\n--- Visualization 7B: Labour Productivity ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for idx, country in enumerate(['European Union - 27 countries (from 2020)', 'France']):
    ax = axes[idx]
    
    data = const_df[
        (const_df['TIME_PERIOD'] == latest_productivity_year) &
        (const_df['indic_sbs'] == 'Apparent labour productivity - thousand euro') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['geo'] == country) &
        (const_df['size_emp'] == 'Total')
    ].dropna(subset=['OBS_VALUE']).copy()
    
    if not data.empty:
        data = data.sort_values('OBS_VALUE', ascending=True)
        colors = [DETAILED_NACE_COLORS.get(row['nace_r2'], '#cccccc') for _, row in data.iterrows()]
        
        bars = ax.barh(range(len(data)), data['OBS_VALUE'], color=colors, edgecolor='black', linewidth=1)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([row['nace_r2'][:35] + '...' if len(row['nace_r2']) > 35 else row['nace_r2'] 
                            for _, row in data.iterrows()], fontsize=9)
        ax.set_xlabel('Labour Productivity (€ thousand)', fontsize=11, fontweight='bold')
        
        country_label = 'EU27' if 'Union' in country else country
        ax.set_title(f'{country_label} ({latest_productivity_year})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (_, row) in enumerate(data.iterrows()):
            ax.text(row['OBS_VALUE'] + 1, i, f"{row['OBS_VALUE']:.1f}€", 
                   va='center', fontsize=8, fontweight='bold')
        
        # Add parent NACE legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', edgecolor='black', label='F41: Construction of buildings'),
            Patch(facecolor='#d62728', edgecolor='black', label='F42: Civil engineering'),
            Patch(facecolor='#9467bd', edgecolor='black', label='F43: Specialised construction')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('Labour Productivity by Detailed NACE', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '7b_labour_productivity_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("  OK Saved: 7b_labour_productivity_detailed.png")

# ============================================================================
# VIS 7C: Productivity Trends (top sectors)
# ============================================================================

print("\n--- Visualization 7C: Productivity Trends ---")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for idx, country in enumerate(['European Union - 27 countries (from 2020)', 'France']):
    ax = axes[idx]
    
    # Get latest data to identify top 5 sectors
    data_latest = const_df[
        (const_df['TIME_PERIOD'] == latest_productivity_year) &
        (const_df['indic_sbs'] == 'Apparent labour productivity - thousand euro') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['geo'] == country) &
        (const_df['size_emp'] == 'Total')
    ].dropna(subset=['OBS_VALUE']).copy()
    
    if not data_latest.empty:
        top_nace = data_latest.nlargest(5, 'OBS_VALUE')['nace_r2'].unique()
        
        for nace_code in top_nace:
            trend_data = const_df[
                (const_df['indic_sbs'] == 'Apparent labour productivity - thousand euro') &
                (const_df['nace_r2'] == nace_code) &
                (const_df['geo'] == country) &
                (const_df['size_emp'] == 'Total')
            ].dropna(subset=['OBS_VALUE']).sort_values('TIME_PERIOD')
            
            if not trend_data.empty:
                color = DETAILED_NACE_COLORS.get(nace_code, '#cccccc')
                label = nace_code[:20] + '...' if len(nace_code) > 20 else nace_code
                ax.plot(trend_data['TIME_PERIOD'].astype(int), trend_data['OBS_VALUE'],
                       marker='o', linewidth=2, label=label, color=color, markersize=5, alpha=0.8)
        
        country_label = 'EU27' if 'Union' in country else country
        ax.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Labour Productivity (€ thousand)', fontsize=11, fontweight='bold')
        ax.set_title(f'{country_label}: Top 5 Sectors', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('Labour Productivity Trends - Top 5 Detailed NACE Sectors', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '7c_productivity_trends_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("  OK Saved: 7c_productivity_trends_detailed.png")

# ============================================================================
# VIS 8B: Employment Distribution
# ============================================================================

print("\n--- Visualization 8B: Employment Distribution ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

latest_employment_year = 2023

for idx, country in enumerate(['European Union - 27 countries (from 2020)', 'France']):
    ax = axes[idx]
    
    data = const_df[
        (const_df['TIME_PERIOD'] == latest_employment_year) &
        (const_df['indic_sbs'] == 'Employees - number') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['geo'] == country) &
        (const_df['size_emp'] == 'Total')
    ].dropna(subset=['OBS_VALUE']).copy()
    
    if not data.empty:
        data = data.sort_values('OBS_VALUE', ascending=True)
        colors = [DETAILED_NACE_COLORS.get(row['nace_r2'], '#cccccc') for _, row in data.iterrows()]
        
        bars = ax.barh(range(len(data)), data['OBS_VALUE'], color=colors, edgecolor='black', linewidth=1)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([row['nace_r2'][:35] + '...' if len(row['nace_r2']) > 35 else row['nace_r2'] 
                            for _, row in data.iterrows()], fontsize=9)
        ax.set_xlabel('Number of Employees', fontsize=11, fontweight='bold')
        
        country_label = 'EU27' if 'Union' in country else country
        ax.set_title(f'{country_label} ({latest_employment_year})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M' if x >= 1e6 else f'{int(x/1000)}K'))
        
        # Add parent NACE legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', edgecolor='black', label='F41: Construction of buildings'),
            Patch(facecolor='#d62728', edgecolor='black', label='F42: Civil engineering'),
            Patch(facecolor='#9467bd', edgecolor='black', label='F43: Specialised construction')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('Employees by Detailed NACE', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '8b_employment_distribution_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("  OK Saved: 8b_employment_distribution_detailed.png")

# ============================================================================
# VIS 8C: Hierarchical Decomposition (FIXED)
# ============================================================================

print("\n--- Visualization 8C: Hierarchical Decomposition ---")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

for country_idx, country in enumerate(['European Union - 27 countries (from 2020)', 'France']):
    # Enterprises
    ax = fig.add_subplot(gs[country_idx, 0])
    
    ent_data = const_df[
        (const_df['geo'] == country) &
        (const_df['TIME_PERIOD'] == 2024) &
        (const_df['indic_sbs'] == 'Enterprises - number') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['size_emp'] == 'Total')
    ].dropna(subset=['OBS_VALUE']).copy()
    
    if not ent_data.empty:
        parent_totals = {}
        parent_colors = {'Construction of buildings': '#2ca02c', 
                        'Civil engineering': '#d62728', 
                        'Specialised construction activities': '#9467bd'}
        
        for parent_code, info in HIERARCHY.items():
            children = info['children']
            parent_data = ent_data[ent_data['nace_r2'].isin(children)]
            
            if not parent_data.empty:
                total = parent_data['OBS_VALUE'].sum()
                parent_totals[parent_code] = total
        
        if parent_totals:
            x_pos = np.arange(len(parent_totals))
            colors_list = [parent_colors.get(p, '#cccccc') for p in parent_totals.keys()]
            
            bars = ax.bar(x_pos, list(parent_totals.values()), color=colors_list, 
                         edgecolor='black', linewidth=1.5, alpha=0.8)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([HIERARCHY[p]['name'].split(':')[0] for p in parent_totals.keys()], 
                               fontsize=10, rotation=15, ha='right')
            ax.set_ylabel('Number of Enterprises', fontsize=11, fontweight='bold')
            
            country_label = 'EU27' if 'Union' in country else country
            ax.set_title(f'{country_label}: Enterprises (2024)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    # Employees
    ax = fig.add_subplot(gs[country_idx, 1])
    
    emp_data = const_df[
        (const_df['geo'] == country) &
        (const_df['TIME_PERIOD'] == 2023) &
        (const_df['indic_sbs'] == 'Employees - number') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['size_emp'] == 'Total')
    ].dropna(subset=['OBS_VALUE']).copy()
    
    if not emp_data.empty:
        parent_totals = {}
        parent_colors = {'Construction of buildings': '#2ca02c', 
                        'Civil engineering': '#d62728', 
                        'Specialised construction activities': '#9467bd'}
        
        for parent_code, info in HIERARCHY.items():
            children = info['children']
            parent_data = emp_data[emp_data['nace_r2'].isin(children)]
            
            if not parent_data.empty:
                total = parent_data['OBS_VALUE'].sum()
                parent_totals[parent_code] = total
        
        if parent_totals:
            x_pos = np.arange(len(parent_totals))
            colors_list = [parent_colors.get(p, '#cccccc') for p in parent_totals.keys()]
            
            bars = ax.bar(x_pos, list(parent_totals.values()), color=colors_list, 
                         edgecolor='black', linewidth=1.5, alpha=0.8)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels([HIERARCHY[p]['name'].split(':')[0] for p in parent_totals.keys()], 
                               fontsize=10, rotation=15, ha='right')
            ax.set_ylabel('Number of Employees', fontsize=11, fontweight='bold')
            
            country_label = 'EU27' if 'Union' in country else country
            ax.set_title(f'{country_label}: Employees (2023)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

# Add parent NACE legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', edgecolor='black', label='F41: Construction of buildings'),
    Patch(facecolor='#d62728', edgecolor='black', label='F42: Civil engineering'),
    Patch(facecolor='#9467bd', edgecolor='black', label='F43: Specialised construction')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, framealpha=0.95, bbox_to_anchor=(0.5, -0.05))

plt.suptitle('Hierarchical NACE Decomposition by Country', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '8c_nace_hierarchy_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()
print("  OK Saved: 8c_nace_hierarchy_decomposition.png")

# ============================================================================
# VIS 9: SCATTER - Value Added vs Productivity
# ============================================================================

print("\n--- Visualization 9: Value Added vs Productivity Scatter ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for idx, country in enumerate(['European Union - 27 countries (from 2020)', 'France']):
    ax = axes[idx]
    
    # Get both metrics
    va_data = const_df[
        (const_df['geo'] == country) &
        (const_df['TIME_PERIOD'] == latest_productivity_year) &
        (const_df['indic_sbs'] == 'Value added per employee - thousand euro') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['size_emp'] == 'Total')
    ][['nace_r2', 'OBS_VALUE']].dropna().copy()
    va_data.columns = ['nace_r2', 'va']
    
    prod_data = const_df[
        (const_df['geo'] == country) &
        (const_df['TIME_PERIOD'] == latest_productivity_year) &
        (const_df['indic_sbs'] == 'Apparent labour productivity - thousand euro') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['size_emp'] == 'Total')
    ][['nace_r2', 'OBS_VALUE']].dropna().copy()
    prod_data.columns = ['nace_r2', 'prod']
    
    scatter_data = va_data.merge(prod_data, on='nace_r2')
    
    # Get employee counts for bubble size
    emp_data = const_df[
        (const_df['geo'] == country) &
        (const_df['TIME_PERIOD'] == latest_employment_year) &
        (const_df['indic_sbs'] == 'Employees - number') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['size_emp'] == 'Total')
    ][['nace_r2', 'OBS_VALUE']].dropna().copy()
    emp_data.columns = ['nace_r2', 'employees']
    
    scatter_data = scatter_data.merge(emp_data, on='nace_r2')
    scatter_data['bubble_size'] = (scatter_data['employees'] / scatter_data['employees'].max()) * 500 + 100
    
    if not scatter_data.empty:
        colors = [DETAILED_NACE_COLORS.get(row['nace_r2'], '#cccccc') for _, row in scatter_data.iterrows()]
        
        scatter = ax.scatter(scatter_data['va'], scatter_data['prod'],
                            s=scatter_data['bubble_size'], c=colors, alpha=0.6,
                            edgecolors='black', linewidth=1.5)
        
        # Add labels
        for _, row in scatter_data.iterrows():
            label = row['nace_r2'][:10] + '...' if len(row['nace_r2']) > 10 else row['nace_r2']
            ax.annotate(label, (row['va'], row['prod']),
                       fontsize=7, fontweight='bold', ha='center', va='center')
        
        ax.set_xlabel('Value Added per Employee (€ thousand)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Labour Productivity (€ thousand)', fontsize=11, fontweight='bold')
        
        country_label = 'EU27' if 'Union' in country else country
        ax.set_title(f'{country_label}: Bubble size = Employees', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant lines
        median_va = scatter_data['va'].median()
        median_prod = scatter_data['prod'].median()
        ax.axvline(median_va, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(median_prod, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

# Add parent NACE legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', edgecolor='black', label='F41: Construction of buildings'),
    Patch(facecolor='#d62728', edgecolor='black', label='F42: Civil engineering'),
    Patch(facecolor='#9467bd', edgecolor='black', label='F43: Specialised construction')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, framealpha=0.95, bbox_to_anchor=(0.5, -0.05))

plt.suptitle('Value Added vs Labour Productivity by Detailed NACE', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '9_value_added_vs_productivity_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("  OK Saved: 9_value_added_vs_productivity_scatter.png")

# ============================================================================
# VIS 10: Firm Size Distribution by Detailed NACE
# ============================================================================

print("\n--- Visualization 10: Firm Size Distribution by Detailed NACE ---")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

size_class_mapping = {
    'From 0 to 9 persons employed': '0-9',
    'From 10 to 19 persons employed': '10-19',
    'From 20 to 49 persons employed': '20-49',
    'From 50 to 249 persons employed': '50-249',
    '250 persons employed or more': '250+',
}

size_order = ['From 0 to 9 persons employed', 'From 10 to 19 persons employed',
              'From 20 to 49 persons employed', 'From 50 to 249 persons employed',
              '250 persons employed or more']
size_labels = ['0-9', '10-19', '20-49', '50-249', '250+']
size_colors = ['#e8f4f8', '#b3dce6', '#7fc9db', '#4db1ce', '#1a88c0']

for idx, country in enumerate(['European Union - 27 countries (from 2020)', 'France']):
    ax = axes[idx]
    
    # Get enterprise distribution by size class for latest year
    size_data = const_df[
        (const_df['geo'] == country) &
        (const_df['TIME_PERIOD'] == 2024) &
        (const_df['indic_sbs'] == 'Enterprises - number') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['size_emp'].isin(size_order))
    ].copy()
    
    if not size_data.empty:
        # Pivot to get size distribution by NACE
        pivot = size_data.pivot_table(
            index='nace_r2',
            columns='size_emp',
            values='OBS_VALUE',
            aggfunc='first'
        )
        
        # Reorder columns
        pivot = pivot[[c for c in size_order if c in pivot.columns]]
        
        # Normalize to 100%
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        
        # Create stacked bar
        pivot_pct.plot(kind='barh', stacked=True, ax=ax, color=size_colors[:len(pivot_pct.columns)],
                      edgecolor='black', linewidth=0.5, width=0.7)
        
        ax.set_xlabel('% of Enterprises', fontsize=11, fontweight='bold')
        ax.set_ylabel('')
        country_label = 'EU27' if 'Union' in country else country
        ax.set_title(f'{country_label} (2024): Enterprise Distribution by Size Class', fontsize=12, fontweight='bold')
        ax.legend(['0-9', '10-19', '20-49', '50-249', '250+'], title='Firm Size', 
                 bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Set y-tick labels
        ax.set_yticklabels([nace[:35] + '...' if len(nace) > 35 else nace for nace in pivot_pct.index], fontsize=9)

plt.suptitle('Firm Size Distribution by Detailed NACE (Normalized %)', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_firm_size_distribution_detailed_nace.png', dpi=300, bbox_inches='tight')
plt.close()
print("  OK Saved: 10_firm_size_distribution_detailed_nace.png")

# ============================================================================
# VIS 11: Average Employees per Firm by Detailed NACE
# ============================================================================

print("\n--- Visualization 11: Average Employees per Firm by Detailed NACE ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for idx, country in enumerate(['European Union - 27 countries (from 2020)', 'France']):
    ax = axes[idx]
    
    # Get total employees and enterprises for each detailed NACE (Total size class)
    emp_data = const_df[
        (const_df['geo'] == country) &
        (const_df['TIME_PERIOD'] == 2023) &
        (const_df['indic_sbs'] == 'Employees - number') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['size_emp'] == 'Total')
    ][['nace_r2', 'OBS_VALUE']].copy()
    emp_data.columns = ['nace_r2', 'employees']
    
    ent_data = const_df[
        (const_df['geo'] == country) &
        (const_df['TIME_PERIOD'] == 2024) &
        (const_df['indic_sbs'] == 'Enterprises - number') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['size_emp'] == 'Total')
    ][['nace_r2', 'OBS_VALUE']].copy()
    ent_data.columns = ['nace_r2', 'enterprises']
    
    # Merge
    merged = emp_data.merge(ent_data, on='nace_r2')
    merged['avg_employees'] = merged['employees'] / merged['enterprises']
    merged = merged.sort_values('avg_employees', ascending=True)
    
    if not merged.empty:
        colors = [DETAILED_NACE_COLORS.get(row['nace_r2'], '#cccccc') for _, row in merged.iterrows()]
        bars = ax.barh(range(len(merged)), merged['avg_employees'], color=colors, edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(merged)))
        ax.set_yticklabels([nace[:35] + '...' if len(nace) > 35 else nace for nace in merged['nace_r2']], fontsize=9)
        ax.set_xlabel('Average Employees per Enterprise', fontsize=11, fontweight='bold')
        
        country_label = 'EU27' if 'Union' in country else country
        ax.set_title(f'{country_label}: Average Firm Size', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (_, row) in enumerate(merged.iterrows()):
            ax.text(row['avg_employees'] + 0.1, i, f"{row['avg_employees']:.1f}", 
                   va='center', fontsize=8, fontweight='bold')
        
        # Add parent NACE legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', edgecolor='black', label='F41: Construction of buildings'),
            Patch(facecolor='#d62728', edgecolor='black', label='F42: Civil engineering'),
            Patch(facecolor='#9467bd', edgecolor='black', label='F43: Specialised construction')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

plt.suptitle('Average Employees per Enterprise by Detailed NACE', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '11_average_employees_per_firm_detailed_nace.png', dpi=300, bbox_inches='tight')
plt.close()
print("  OK Saved: 11_average_employees_per_firm_detailed_nace.png")

# ============================================================================
# VIS 12: Value Added by Firm Size Class (Line graph by NACE)
# ============================================================================

print("\n--- Visualization 12: Value Added per Employee by Firm Size Class ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Parent NACE colors and mapping
parent_nace_colors = {
    'F41': '#2ca02c',
    'F42': '#d62728',
    'F43': '#9467bd',
}

parent_nace_names = {
    'F41': 'F41: Construction of buildings',
    'F42': 'F42: Civil engineering',
    'F43': 'F43: Specialised construction',
}

def get_parent_nace(nace_sector):
    """Determine parent NACE category for a detailed NACE sector."""
    for parent, info in HIERARCHY.items():
        if nace_sector in info['children']:
            if parent == 'Construction of buildings':
                return 'F41'
            elif parent == 'Civil engineering':
                return 'F42'
            elif parent == 'Specialised construction activities':
                return 'F43'
    return None

for idx, country in enumerate(['European Union - 27 countries (from 2020)', 'France']):
    ax = axes[idx]
    
    # Get value added by firm size class for each detailed NACE
    va_data = const_df[
        (const_df['geo'] == country) &
        (const_df['TIME_PERIOD'] == latest_productivity_year) &
        (const_df['indic_sbs'] == 'Value added per employee - thousand euro') &
        (const_df['nace_r2'].isin(detailed_nace)) &
        (const_df['size_emp'].isin(size_order))
    ].copy()
    
    if not va_data.empty:
        # Sort by size class
        va_data['size_order'] = va_data['size_emp'].map({s: i for i, s in enumerate(size_order)})
        va_data = va_data.sort_values('size_order')
        
        # Plot line for each detailed NACE sector
        for nace in detailed_nace:
            nace_data = va_data[va_data['nace_r2'] == nace].copy()
            
            if not nace_data.empty:
                parent_cat = get_parent_nace(nace)
                color = parent_nace_colors.get(parent_cat, '#cccccc')
                
                # Create x positions based on size class order
                x_pos = range(len(nace_data))
                
                line = ax.plot(x_pos, nace_data['OBS_VALUE'], marker='o', linewidth=2,
                              color=color, markersize=6, alpha=0.8, label=nace)
                
                # Add NACE name label at end of line
                last_point = nace_data.iloc[-1]
                ax.text(len(nace_data) - 1, last_point['OBS_VALUE'], 
                       nace[:20] + '...' if len(nace) > 20 else nace,
                       fontsize=8, ha='left', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor='none'))
        
        # Set x-axis labels
        size_labels = [size_class_mapping.get(s, s) for s in size_order]
        ax.set_xticks(range(len(size_order)))
        ax.set_xticklabels(size_labels, fontsize=11, fontweight='bold')
        ax.set_ylabel('Value Added per Employee (€ thousand)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Firm Size Class', fontsize=11, fontweight='bold')
        
        country_label = 'EU27' if 'Union' in country else country
        ax.set_title(f'{country_label} ({latest_productivity_year})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

# Create parent NACE legend at the right
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=parent_nace_colors['F41'], edgecolor='black', label=parent_nace_names['F41']),
    Patch(facecolor=parent_nace_colors['F42'], edgecolor='black', label=parent_nace_names['F42']),
    Patch(facecolor=parent_nace_colors['F43'], edgecolor='black', label=parent_nace_names['F43'])
]
fig.legend(handles=legend_elements, loc='center right', fontsize=10, framealpha=0.95, 
          bbox_to_anchor=(1.12, 0.5))

plt.suptitle('Value Added per Employee by Firm Size Class', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '12_value_added_by_firm_size_class.png', dpi=300, bbox_inches='tight')
plt.close()
print("  OK Saved: 12_value_added_by_firm_size_class.png")

print("\n" + "="*80)
print("DETAILED NACE ANALYSIS COMPLETED SUCCESSFULLY")
print(f"Output directory: {OUTPUT_DIR}")
print("="*80)
