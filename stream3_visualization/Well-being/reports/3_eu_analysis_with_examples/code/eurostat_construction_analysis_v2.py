"""
EUROSTAT Construction Industry Analysis v2
============================================
Analyzes:
1. 4-firm concentration ratio by NACE activity (2023)
2. Construction employment by size class (2005-2020)
   - Includes both enterprises and employees
   - Normalized comparisons
   - Market structure comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
print("EUROSTAT CONSTRUCTION INDUSTRY ANALYSIS v2")
print("="*80)

# Color scheme (from oecd_analysis.py)
COLORS = {
    'EU27_2020': '#80b1d3',  # EU blue
    'EU27': '#80b1d3',       # EU blue
    'FR': '#ffd558',         # yellow
    'concentration': '#d62728',  # red
}

SIZE_COLORS = {
    '[TOTAL]': '#000000',
    '[0-9]': '#e8f4f8',
    '[10-19]': '#b3dce6',
    '[20-49]': '#7fc9db',
    '[50-249]': '#4db1ce',
    '[GE250]': '#1a88c0',
}

NACE_COLORS = {
    'F41': '#2ca02c',     # green (parent category)
    'F42': '#d62728',     # red (parent category)
    'F43': '#9467bd',     # purple (parent category)
}

# Population data for normalization (thousands, 2020 estimates)
POPULATION = {
    'EU27_2020': 447706,
    'FR': 67750,
}

# ============================================================================
# 1. CONCENTRATION RATIO ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("1. CONCENTRATION RATIO ANALYSIS")
print("="*80)

conc_df = pd.read_csv(DATA_DIR / 'concentration_NACE2_2023.csv')
print(f"\nLoaded {len(conc_df)} records from concentration_NACE2_2023.csv")

# Display concentration by NACE
conc_display = conc_df[['nace_r2', 'OBS_VALUE']].copy()
conc_display = conc_display.sort_values('OBS_VALUE', ascending=False)
print("\n4-Firm Concentration Ratio by NACE (2023):")
print(conc_display.to_string(index=False))

# Create concentration visualization
fig, ax = plt.subplots(figsize=(14, 10))
sorted_df = conc_df.sort_values('OBS_VALUE', ascending=True)

bars = ax.barh(range(len(sorted_df)), sorted_df['OBS_VALUE'].values, 
               color='#d62728', alpha=0.8, edgecolor='black', linewidth=1)

# Highlight construction sector
construction_idx = sorted_df[sorted_df['nace_r2'] == 'Construction'].index[0]
construction_pos = list(range(len(sorted_df))).index(construction_idx)
bars[construction_pos].set_color('#2ca02c')
bars[construction_pos].set_linewidth(2)

ax.set_yticks(range(len(sorted_df)))
ax.set_yticklabels(sorted_df['nace_r2'].values, fontsize=10)
ax.set_xlabel('4-Firm Concentration Ratio (% of total employment)', fontsize=12, fontweight='bold')
ax.set_title('Market Concentration by NACE Activity (EU-27, 2023)\nFour Largest Enterprises Share', 
             fontsize=13, fontweight='bold', pad=15)

# Add value labels
for i, (idx, row) in enumerate(sorted_df.iterrows()):
    ax.text(row['OBS_VALUE'] + 0.3, i, f"{row['OBS_VALUE']:.1f}%", 
            va='center', fontweight='bold', fontsize=9)

ax.set_xlim(0, max(sorted_df['OBS_VALUE']) * 1.15)
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(conc_df[conc_df['nace_r2'] == 'Construction']['OBS_VALUE'].values[0], 
           color='#2ca02c', linestyle='--', linewidth=2, alpha=0.7, label='Construction: 2.6%')
ax.legend(fontsize=11, loc='lower right')

plt.tight_layout()
output_file = OUTPUT_DIR / 'concentration_by_nace_2023.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nOK Saved: {output_file}")
plt.close()

# ============================================================================
# 2. CONSTRUCTION EMPLOYMENT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("2. CONSTRUCTION EMPLOYMENT ANALYSIS")
print("="*80)

const_df = pd.read_csv(DATA_DIR / 'construction_employment_size.csv')
print(f"\nLoaded {len(const_df)} records from construction_employment_size.csv")

# First, check what geo values are available
print(f"\nAvailable geo values (sample): {const_df['geo'].unique()[:10]}")

# Check if EU27_2020 exists
has_eu27 = 'EU27_2020' in const_df['geo'].values
print(f"EU27_2020 in data: {has_eu27}")

# EU27 member countries (2020)
eu27_countries = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
                  'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
                  'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
                  'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
                  'Spain', 'Sweden']

# Calculate EU27 aggregate from individual countries if not present
if not has_eu27:
    print("\nCalculating EU27_2020 aggregate from member countries...")
    eu27_data = const_df[const_df['geo'].isin(eu27_countries)].copy()
    eu27_agg = eu27_data.groupby(['TIME_PERIOD', 'nace_r2', 'indic_sb', 'size_emp']).agg({
        'OBS_VALUE': 'sum'
    }).reset_index()
    eu27_agg['geo'] = 'EU27_2020'
    const_df = pd.concat([const_df, eu27_agg], ignore_index=True)
    print(f"Added {len(eu27_agg)} EU27_2020 records")

# Filter for EU27 and FR, indicators of interest
eu27_const = const_df[const_df['geo'].isin(['EU27_2020', 'France', 'FR'])].copy()

# Standardize country names
eu27_const['geo'] = eu27_const['geo'].replace({
    'France': 'FR'
})

# Map to codes
size_mapping = {
    'Total': '[TOTAL]',
    'From 0 to 9 persons employed': '[0-9]',
    'From 10 to 19 persons employed': '[10-19]',
    'From 20 to 49 persons employed': '[20-49]',
    'From 50 to 249 persons employed': '[50-249]',
    '250 persons employed or more': '[GE250]',
}

nace_mapping = {
    'Construction': 'F',
    'Construction of buildings': 'F41',
    'Civil engineering': 'F42',
    'Specialised construction activities': 'F43',
}

eu27_const['size_emp_code'] = eu27_const['size_emp'].map(size_mapping)
eu27_const['nace_code'] = eu27_const['nace_r2'].map(nace_mapping)

eu27_const = eu27_const[
    (eu27_const['size_emp_code'].notna()) & 
    (eu27_const['nace_code'].notna())
].copy()

print(f"Filtered to {len(eu27_const)} records")
print(f"Countries in data: {eu27_const['geo'].unique()}")
print(f"Indicators: {eu27_const['indic_sb'].unique()}")
latest_year = eu27_const['TIME_PERIOD'].max()
print(f"Using latest year: {latest_year}")

# ============================================================================
# VIS 1: Enterprises by NACE over time
# ============================================================================

print("\n--- Visualization 1: Enterprises by NACE (2005-2020) ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, nace_code in enumerate(['F', 'F41', 'F42', 'F43']):
    ax = axes[idx]
    
    ent_data = eu27_const[
        (eu27_const['nace_code'] == nace_code) & 
        (eu27_const['indic_sb'] == 'Enterprises - number')
    ].copy()
    
    for country in ['EU27_2020', 'FR']:
        country_data = ent_data[ent_data['geo'] == country].copy()
        if country_data.empty:
            continue
        
        total_data = country_data[country_data['size_emp_code'] == '[TOTAL]'].sort_values('TIME_PERIOD')
        ax.plot(total_data['TIME_PERIOD'].astype(int), total_data['OBS_VALUE'], 
               marker='o', linewidth=2.5, label=f'{country}', 
               color=COLORS[country], markersize=6)
    
    nace_name = {
        'F': 'All Construction',
        'F41': 'Construction of Buildings',
        'F42': 'Civil Engineering',
        'F43': 'Specialised Construction'
    }[nace_code]
    
    ax.set_title(f'{nace_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Number of Enterprises', fontsize=11)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2004, 2021)

plt.suptitle('Enterprises by NACE Subsector (2005-2020)', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
output_file = OUTPUT_DIR / '1_enterprises_by_nace_over_time.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"OK Saved: {output_file}")
plt.close()

# ============================================================================
# VIS 1B: Employees by NACE (normalized)
# ============================================================================

print("\n--- Visualization 1B: Employees by NACE (Normalized) ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, nace_code in enumerate(['F', 'F41', 'F42', 'F43']):
    ax = axes[idx]
    
    emp_data = eu27_const[
        (eu27_const['nace_code'] == nace_code) & 
        (eu27_const['indic_sb'] == 'Employees - number')
    ].copy()
    
    for country in ['EU27_2020', 'FR']:
        country_data = emp_data[emp_data['geo'] == country].copy()
        if country_data.empty:
            continue
        
        total_data = country_data[country_data['size_emp_code'] == '[TOTAL]'].sort_values('TIME_PERIOD')
        if not total_data.empty:
            pop = POPULATION.get(country, 1)
            normalized = total_data['OBS_VALUE'] / pop * 1000  # per 1000 inhabitants
            ax.plot(total_data['TIME_PERIOD'].astype(int), normalized, 
                   marker='o', linewidth=2.5, label=f'{country}', 
                   color=COLORS[country], markersize=6)
    
    nace_name = {
        'F': 'All Construction',
        'F41': 'Construction of Buildings',
        'F42': 'Civil Engineering',
        'F43': 'Specialised Construction'
    }[nace_code]
    
    ax.set_title(f'{nace_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Employees per 1,000 Inhabitants', fontsize=11)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2004, 2021)

plt.suptitle('Employees by NACE Subsector (2005-2020) - Normalized', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
output_file = OUTPUT_DIR / '1b_employees_by_nace_normalized.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"OK Saved: {output_file}")
plt.close()

# ============================================================================
# VIS 2: Size class distribution (enterprises) 2020
# ============================================================================

print("\n--- Visualization 2: Size Distribution (Enterprises 2020) ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, country in enumerate(['EU27_2020', 'FR']):
    ax = axes[idx]
    
    size_dist = eu27_const[
        (eu27_const['geo'] == country) & 
        (eu27_const['TIME_PERIOD'] == latest_year) &
        (eu27_const['nace_code'] == 'F') &
        (eu27_const['indic_sb'] == 'Enterprises - number') &
        (eu27_const['size_emp_code'] != '[TOTAL]')
    ].copy()
    
    if size_dist.empty:
        continue
    
    size_order = ['[0-9]', '[10-19]', '[20-49]', '[50-249]', '[GE250]']
    size_labels = ['0-9', '10-19', '20-49', '50-249', '250+']
    
    size_dist = size_dist.sort_values('size_emp_code', key=lambda x: x.map({s: i for i, s in enumerate(size_order)}))
    
    colors = [SIZE_COLORS[s] for s in size_dist['size_emp_code'].values]
    bars = ax.bar(range(len(size_dist)), size_dist['OBS_VALUE'].values, 
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_xticks(range(len(size_dist)))
    ax.set_xticklabels(size_labels, fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Enterprises', fontsize=11, fontweight='bold')
    ax.set_title(f'{country}', fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height/1000)}K',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))

plt.suptitle(f'Enterprises by Size Class ({latest_year})', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
output_file = OUTPUT_DIR / '2_size_distribution_enterprises.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"OK Saved: {output_file}")
plt.close()

# ============================================================================
# VIS 2B: Size class distribution (employees) 2020
# ============================================================================

print("\n--- Visualization 2B: Size Distribution (Employees 2020) ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, country in enumerate(['EU27_2020', 'FR']):
    ax = axes[idx]
    
    size_dist = eu27_const[
        (eu27_const['geo'] == country) & 
        (eu27_const['TIME_PERIOD'] == latest_year) &
        (eu27_const['nace_code'] == 'F') &
        (eu27_const['indic_sb'] == 'Employees - number') &
        (eu27_const['size_emp_code'] != '[TOTAL]')
    ].copy()
    
    if size_dist.empty:
        continue
    
    size_order = ['[0-9]', '[10-19]', '[20-49]', '[50-249]', '[GE250]']
    size_labels = ['0-9', '10-19', '20-49', '50-249', '250+']
    
    size_dist = size_dist.sort_values('size_emp_code', key=lambda x: x.map({s: i for i, s in enumerate(size_order)}))
    
    colors = [SIZE_COLORS[s] for s in size_dist['size_emp_code'].values]
    bars = ax.bar(range(len(size_dist)), size_dist['OBS_VALUE'].values, 
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_xticks(range(len(size_dist)))
    ax.set_xticklabels(size_labels, fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Employees', fontsize=11, fontweight='bold')
    ax.set_title(f'{country}', fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height/1e6)}M' if height >= 1e6 else f'{int(height/1000)}K',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M' if x >= 1e6 else f'{int(x/1000)}K'))

plt.suptitle(f'Employees by Size Class ({latest_year})', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
output_file = OUTPUT_DIR / '2b_size_distribution_employees.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"OK Saved: {output_file}")
plt.close()

# ============================================================================
# VIS 3: Enterprises vs Employees trend (normalized)
# ============================================================================

print("\n--- Visualization 3: Enterprises vs Employees (Normalized) ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, country in enumerate(['EU27_2020', 'FR']):
    ax = axes[idx]
    
    total_ent = eu27_const[
        (eu27_const['geo'] == country) & 
        (eu27_const['nace_code'] == 'F') &
        (eu27_const['size_emp_code'] == '[TOTAL]') &
        (eu27_const['indic_sb'] == 'Enterprises - number')
    ].sort_values('TIME_PERIOD')
    
    total_emp = eu27_const[
        (eu27_const['geo'] == country) & 
        (eu27_const['nace_code'] == 'F') &
        (eu27_const['size_emp_code'] == '[TOTAL]') &
        (eu27_const['indic_sb'] == 'Employees - number')
    ].sort_values('TIME_PERIOD')
    
    if total_ent.empty or total_emp.empty:
        continue
    
    ax2 = ax.twinx()
    
    # Normalize enterprises
    pop = POPULATION.get(country, 1)
    ent_norm = total_ent['OBS_VALUE'] / pop * 1000
    emp_norm = total_emp['OBS_VALUE'] / pop * 1000
    
    line1 = ax.plot(total_ent['TIME_PERIOD'].astype(int), ent_norm, 
                   marker='o', linewidth=2.5, label='Enterprises', 
                   color='#1f77b4', markersize=6)
    line2 = ax2.plot(total_emp['TIME_PERIOD'].astype(int), emp_norm,
                    marker='s', linewidth=2.5, label='Employees',
                    color='#ff7f0e', markersize=6)
    
    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Enterprises per 1,000 Inhabitants', fontsize=11, fontweight='bold', color='#1f77b4')
    ax2.set_ylabel('Employees per 1,000 Inhabitants', fontsize=11, fontweight='bold', color='#ff7f0e')
    ax.set_title(f'{country}', fontsize=12, fontweight='bold')
    
    ax.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2004, 2021)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=10, loc='upper left')

plt.suptitle('Construction: Enterprises vs Employees (Normalized 2005-2020)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
output_file = OUTPUT_DIR / '3_enterprises_vs_employees_normalized.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"OK Saved: {output_file}")
plt.close()

# ============================================================================
# VIS 4: Size structure evolution (stacked area)
# ============================================================================

print("\n--- Visualization 4: Size Structure Evolution (Stacked) ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, country in enumerate(['EU27_2020', 'FR']):
    ax = axes[idx]
    
    size_evolution = eu27_const[
        (eu27_const['geo'] == country) & 
        (eu27_const['nace_code'] == 'F') &
        (eu27_const['indic_sb'] == 'Enterprises - number') &
        (eu27_const['size_emp_code'] != '[TOTAL]')
    ].copy()
    
    if size_evolution.empty:
        continue
    
    pivot = size_evolution.pivot_table(
        index='TIME_PERIOD',
        columns='size_emp_code',
        values='OBS_VALUE',
        aggfunc='first'
    )
    
    size_order = ['[0-9]', '[10-19]', '[20-49]', '[50-249]', '[GE250]']
    pivot = pivot[[c for c in size_order if c in pivot.columns]]
    
    colors_list = [SIZE_COLORS[c] for c in pivot.columns]
    ax.stackplot(pivot.index.astype(int), pivot.T.values, 
                labels=['0-9', '10-19', '20-49', '50-249', '250+'],
                colors=colors_list, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Enterprises', fontsize=11, fontweight='bold')
    ax.set_title(f'{country}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, title='Enterprise Size')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2004, 2021)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))

plt.suptitle('Size Structure Evolution (2005-2020)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
output_file = OUTPUT_DIR / '4_size_structure_evolution.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"OK Saved: {output_file}")
plt.close()

# ============================================================================
# VIS 5: Market structure - pie charts (enterprises vs employees)
# ============================================================================

print("\n--- Visualization 5: Market Structure Comparison (2020) ---")

fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Row 1: Enterprises (EU and FR)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# Row 2: Employees (EU and FR)
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

axes_list = [(ax1, 'EU27_2020', 'Enterprises'), (ax2, 'FR', 'Enterprises'),
             (ax3, 'EU27_2020', 'Employees'), (ax4, 'FR', 'Employees')]

for ax, country, metric_type in axes_list:
    if metric_type == 'Enterprises':
        indic = 'Enterprises - number'
    else:
        indic = 'Employees - number'
    
    data_2020 = eu27_const[
        (eu27_const['geo'] == country) &
        (eu27_const['TIME_PERIOD'] == latest_year) &
        (eu27_const['nace_code'].isin(['F41', 'F42', 'F43'])) &
        (eu27_const['indic_sb'] == indic) &
        (eu27_const['size_emp_code'] == '[TOTAL]')
    ].copy()
    
    if not data_2020.empty:
        plot_data = data_2020.groupby('nace_code')['OBS_VALUE'].sum()
        labels = ['Construction of Buildings\n(F41)', 'Civil Engineering\n(F42)', 'Specialised Construction\n(F43)']
        colors = [NACE_COLORS[c] for c in plot_data.index]
        
        wedges, texts, autotexts = ax.pie(
            plot_data.values, 
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            explode=(0.05, 0.05, 0.05),
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title(f'{country}: {metric_type} by NACE Subsector', 
                    fontsize=11, fontweight='bold', pad=15)

plt.suptitle('Construction Market Structure (2020)\nEnterprise Count vs Employee Count', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
output_file = OUTPUT_DIR / '5_market_structure_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"OK Saved: {output_file}")
plt.close()

# ============================================================================
# VIS 6: NACE subsector comparison (normalized percentages)
# ============================================================================

print("\n--- Visualization 6: NACE Subsector Comparison (Normalized %) ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

nace_info = {
    'F': 'All Construction',
    'F41': 'Construction of Buildings',
    'F42': 'Civil Engineering',
    'F43': 'Specialised Construction'
}

for idx, (nace_code, nace_name) in enumerate(nace_info.items()):
    ax = axes[idx]
    
    latest_data = eu27_const[
        (eu27_const['nace_code'] == nace_code) &
        (eu27_const['TIME_PERIOD'] == latest_year) &
        (eu27_const['indic_sb'] == 'Enterprises - number') &
        (eu27_const['size_emp_code'] != '[TOTAL]')
    ].copy()
    
    if latest_data.empty:
        continue
    
    x_pos = np.arange(5)
    width = 0.35
    
    size_order = ['[0-9]', '[10-19]', '[20-49]', '[50-249]', '[GE250]']
    size_labels = ['0-9', '10-19', '20-49', '50-249', '250+']
    
    eu_data_list = []
    fr_data_list = []
    
    for country in ['EU27_2020', 'FR']:
        country_data = latest_data[latest_data['geo'] == country].sort_values(
            'size_emp_code', 
            key=lambda x: x.map({s: i for i, s in enumerate(size_order)})
        )
        
        if country_data.empty:
            continue
        
        values = country_data.set_index('size_emp_code').loc[size_order, 'OBS_VALUE'].values
        total = values.sum()
        percentages = (values / total * 100)
        
        if country == 'EU27_2020':
            eu_data_list = percentages
        else:
            fr_data_list = percentages
    
    if len(eu_data_list) > 0 and len(fr_data_list) > 0:
        ax.bar(x_pos - width/2, eu_data_list, width, 
              label='EU27_2020', color=COLORS['EU27_2020'], alpha=0.8, edgecolor='black', linewidth=1)
        ax.bar(x_pos + width/2, fr_data_list, width, 
              label='FR', color=COLORS['FR'], alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Percentage of Enterprises (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{nace_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(size_labels, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
        
        # Add percentage labels on bars
        for i, (eu_val, fr_val) in enumerate(zip(eu_data_list, fr_data_list)):
            if eu_val > 2:
                ax.text(i - width/2, eu_val + 1, f'{eu_val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
            if fr_val > 2:
                ax.text(i + width/2, fr_val + 1, f'{fr_val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.suptitle(f'Construction Subsectors: Enterprise Distribution by Size ({latest_year})\nNormalized to 100%', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
output_file = OUTPUT_DIR / '6_nace_subsector_comparison_normalized.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"OK Saved: {output_file}")
plt.close()

print("\n" + "="*80)
print(f"ANALYSIS COMPLETED SUCCESSFULLY")
print(f"Output directory: {OUTPUT_DIR}")
print("="*80)
