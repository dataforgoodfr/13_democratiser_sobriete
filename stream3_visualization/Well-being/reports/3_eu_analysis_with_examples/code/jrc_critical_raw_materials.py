"""
JRC Critical Raw Materials — Stacked Bar Charts
Compares baseline supply/demand with projected scenario demand (2030, 2050)
for each material, stacked by transition technology.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

plt.rcParams['font.family'] = 'Arial'

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'external_data',
                         'jrc_critical_raw_material.xlsx')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'graphs', 'JRC_CRM')
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# NAME MAPPING (scenarios → baseline)
# ============================================================================
MATERIAL_TO_BASELINE = {
    'Borates': 'Borate',
    'Graphite': 'Natural graphite',
    'Steel': 'Iron ore',
}

# Short labels for baselines
SOURCE_SHORT = {
    'Average of 2012-2017 global supply x 22% EU share of global GDP in tonnes':
        'Global supply\n× EU GDP share',
    'EU consumption in tonnes\n(less reliable for some materials)':
        'EU\nconsumption',
    'EU domestic production in tonnes':
        'EU domestic\nproduction',
}

# Short labels for transition technologies
TECH_SHORT = {
    'fuel cells for e-mobility': 'FC e-mob',
    'fuel cells for renewables': 'FC renew',
    'batteries for e-mobility': 'Bat e-mob',
    'batteries for renewables': 'Bat renew',
    'Traction Motors': 'Traction',
    'PV': 'PV',
    'Wind Turbines': 'Wind',
}

# Colors for transition technologies
TECH_COLORS = {
    'fuel cells for e-mobility': '#e41a1c',
    'fuel cells for renewables': '#ff7f00',
    'batteries for e-mobility': '#377eb8',
    'batteries for renewables': '#4daf4a',
    'Traction Motors': '#984ea3',
    'PV': '#ffff33',
    'Wind Turbines': '#a65628',
}

BASELINE_COLOR = '#888888'


# ============================================================================
# CHART GROUPS
# ============================================================================
CHART_GROUPS = [
    {
        'materials': ['Germanium', 'Silicon metal', 'Gallium',
                      'Indium'],
        'title': 'PV Materials',
        'filename': 'crm_pv_materials.png',
    },
    {
        'materials': ['Dysprosium', 'Neodymium',
                      'Praseodymium', 'Silicon metal'],
        'title': 'Wind & Traction Motor Materials',
        'filename': 'crm_wind_traction_materials.png',
    },
    {
        'materials': ['Graphite', 'Cobalt', 'Lithium'],
        'title': 'Battery Materials',
        'filename': 'crm_battery_materials.png',
    },
    {
        'materials': ['Cobalt', 'Palladium', 'Platinum', 'Graphite',
                      'Strontium', 'Titanium'],
        'title': 'Fuel Cell Materials',
        'filename': 'crm_fuel_cell_materials.png',
        'n_cols': 3,
    },
]


def load_data():
    """Load both sheets and resolve baselines (Stage 1 if available, else Stage 2)."""
    scenarios = pd.read_excel(DATA_PATH, sheet_name='scenarios')
    baseline_raw = pd.read_excel(DATA_PATH, sheet_name='baseline')

    sources = baseline_raw['source'].unique()

    # Resolve baseline: prefer Stage 1, fallback to Stage 2
    baseline_records = []
    stage_used = {}  # (material, source) → 'Stage 1' or 'Stage 2'
    for source in sources:
        for material in baseline_raw['material'].unique():
            s1 = baseline_raw[
                (baseline_raw['material'] == material) &
                (baseline_raw['source'] == source) &
                (baseline_raw['stage'] == 'Stage 1')
            ]['value']
            s2 = baseline_raw[
                (baseline_raw['material'] == material) &
                (baseline_raw['source'] == source) &
                (baseline_raw['stage'] == 'Stage 2')
            ]['value']

            val_s1 = s1.iloc[0] if len(s1) > 0 else np.nan
            val_s2 = s2.iloc[0] if len(s2) > 0 else np.nan

            if not np.isnan(val_s1):
                baseline_records.append({
                    'material': material, 'source': source,
                    'value': val_s1, 'stage_used': 'Stage 1'})
                stage_used[(material, source)] = 'Stage 1'
            elif not np.isnan(val_s2):
                baseline_records.append({
                    'material': material, 'source': source,
                    'value': val_s2, 'stage_used': 'Stage 2'})
                stage_used[(material, source)] = 'Stage 2'
            else:
                baseline_records.append({
                    'material': material, 'source': source,
                    'value': np.nan, 'stage_used': None})

    baseline = pd.DataFrame(baseline_records)
    return scenarios, baseline, sources, stage_used


def make_chart(scenarios, baseline, sources, stage_used,
               materials, group_title, filename, n_cols=None):
    """Create one subplot per material with independent y-scales."""
    n_mats = len(materials)

    # All transition technologies (ordered)
    all_techs = [
        'fuel cells for e-mobility', 'fuel cells for renewables',
        'batteries for e-mobility', 'batteries for renewables',
        'Traction Motors', 'PV', 'Wind Turbines',
    ]

    # Layout: grid of subplots
    if n_cols is None:
        n_cols = min(n_mats, 4)
    n_rows = int(np.ceil(n_mats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    # x-axis positions: 3 baselines + 3 scenarios@2030 + 3 scenarios@2050 = 9
    x_labels_raw = []
    for src in sources:
        x_labels_raw.append(('baseline', src))
    for year in [2030, 2050]:
        for scen in ['LDS', 'MDS', 'HDS']:
            x_labels_raw.append(('scenario', year, scen))

    n_bars = len(x_labels_raw)
    x = np.arange(n_bars)
    bar_width = 0.7
    stage2_per_material = {}  # material → list of stage2 source labels

    def _fmt_val(v):
        """Format a value for bar label."""
        if v >= 1e6:
            return f'{v / 1e6:.1f}M'
        if v >= 1e3:
            return f'{v / 1e3:.1f}k'
        return f'{v:.0f}'

    for idx, material in enumerate(materials):
        ax = axes_flat[idx]
        bl_name = MATERIAL_TO_BASELINE.get(material, material)

        # --- Baselines (solid gray bars) ---
        stage2_sources = []
        bar_tops = {}  # bar_index → total height
        for bi, source in enumerate(sources):
            row = baseline[
                (baseline['material'] == bl_name) &
                (baseline['source'] == source)
            ]
            val = row['value'].iloc[0] if len(row) > 0 else np.nan
            stage = stage_used.get((bl_name, source))

            if not np.isnan(val):
                ax.bar(bi, val, bar_width, color=BASELINE_COLOR,
                       edgecolor='white', linewidth=0.5)
                bar_tops[bi] = val
                if stage == 'Stage 2':
                    stage2_sources.append(
                        SOURCE_SHORT.get(source, source[:12]).replace('\n', ' '))

        # --- Scenarios (stacked by technology) ---
        for si, (year, scen) in enumerate(
                [(2030, 'LDS'), (2030, 'MDS'), (2030, 'HDS'),
                 (2050, 'LDS'), (2050, 'MDS'), (2050, 'HDS')]):
            bar_idx = len(sources) + si
            bottom = 0
            for tech in all_techs:
                val = scenarios[
                    (scenarios['material'] == material) &
                    (scenarios['year'] == year) &
                    (scenarios['scenario'] == scen) &
                    (scenarios['transition_innovation'] == tech)
                ]['value'].sum()
                if val > 0:
                    ax.bar(bar_idx, val, bar_width, bottom=bottom,
                           color=TECH_COLORS[tech], edgecolor='white',
                           linewidth=0.3)
                    bottom += val
            if bottom > 0:
                bar_tops[bar_idx] = bottom

        # Add total value labels on top of every bar
        for bi, top_val in bar_tops.items():
            ax.text(bi, top_val, _fmt_val(top_val), ha='center',
                    va='bottom', fontsize=5, fontweight='bold',
                    color='#333333')

        # Collect Stage 2 info per material
        stage2_per_material[material] = stage2_sources

        # Formatting
        ax.set_title(material, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        x_tick_labels = []
        for item in x_labels_raw:
            if item[0] == 'baseline':
                x_tick_labels.append(SOURCE_SHORT.get(item[1], item[1][:12]))
            else:
                x_tick_labels.append(f"{item[1]}\n{item[2]}")
        ax.set_xticklabels(x_tick_labels, fontsize=5.5, rotation=0)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f'{v / 1e6:.1f}M' if v >= 1e6 else (
                f'{v / 1e3:.1f}k' if v >= 1e3 else f'{v:.0f}')))
        ax.tick_params(axis='y', labelsize=7)
        ax.set_ylabel('tonnes', fontsize=7, fontstyle='italic')
        ax.grid(axis='y', alpha=0.2)

        # Visual separators between baseline / 2030 / 2050
        for sep_x in [len(sources) - 0.5, len(sources) + 3 - 0.5]:
            ax.axvline(sep_x, color='#aaaaaa', linewidth=0.8,
                       linestyle='--', alpha=0.6)

        # Group labels at bottom
        ax.text((len(sources) - 1) / 2, -0.18, 'Baseline',
                ha='center', va='top', fontsize=6.5, fontweight='bold',
                color='#666', transform=ax.get_xaxis_transform())
        ax.text(len(sources) + 1, -0.18, '2030',
                ha='center', va='top', fontsize=6.5, fontweight='bold',
                color='#666', transform=ax.get_xaxis_transform())
        ax.text(len(sources) + 4, -0.18, '2050',
                ha='center', va='top', fontsize=6.5, fontweight='bold',
                color='#666', transform=ax.get_xaxis_transform())

    # Hide unused axes
    for idx in range(n_mats, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Stage II footnotes: if all materials share same stage2 sources,
    # put a single footnote; otherwise annotate per subplot
    all_stage2_sets = [frozenset(s) for s in stage2_per_material.values()]
    if all_stage2_sets and all(s == all_stage2_sets[0] for s in all_stage2_sets) \
            and all_stage2_sets[0]:
        # All materials have same Stage 2 sources → single figure footnote
        note = 'Note: all baselines use Stage II data (' + \
               ', '.join(sorted(all_stage2_sets[0])) + ')'
        fig.text(0.5, -0.02, note, ha='center', va='top', fontsize=8,
                 color='black', fontstyle='italic')
    else:
        # Per-subplot notes for materials that have Stage 2
        for idx, material in enumerate(materials):
            s2 = stage2_per_material.get(material, [])
            if s2:
                ax = axes_flat[idx]
                note = 'Stage II used for: ' + ', '.join(s2)
                ax.text(0.5, -0.28, note, ha='center', va='top',
                        fontsize=5.5, color='black', fontstyle='italic',
                        transform=ax.transAxes)

    # Build legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=BASELINE_COLOR, label='Baseline')]
    for tech in all_techs:
        handles.append(Patch(facecolor=TECH_COLORS[tech],
                             label=TECH_SHORT[tech]))

    fig.legend(handles=handles, loc='lower center',
               ncol=len(handles), fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f'EU Critical Raw Materials — {group_title}\n'
        'Baseline vs. Projected Demand by Transition Technology '
        '(LDS / MDS / HDS, tonnes)\n'
        'Baseline uses Stage I when available, else Stage II',
        fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200,
                bbox_inches='tight')
    svg_name = filename.replace('.png', '.svg')
    fig.savefig(os.path.join(OUTPUT_DIR, svg_name), format='svg',
                bbox_inches='tight')
    print(f"Saved {filename} + {svg_name}")
    plt.close(fig)


def main():
    print("=" * 50)
    print("JRC Critical Raw Materials Analysis")
    print("=" * 50)
    scenarios, baseline, sources, stage_used = load_data()
    print(f"Scenarios: {len(scenarios)} rows, "
          f"{scenarios['material'].nunique()} materials")
    print(f"Baseline: {len(baseline)} rows (resolved)")

    for group in CHART_GROUPS:
        print(f"\n[{group['title']}] {group['materials']}")
        make_chart(scenarios, baseline, sources, stage_used,
                   group['materials'], group['title'], group['filename'],
                   n_cols=group.get('n_cols'))

    # Excel export
    export_crm_excel(scenarios, baseline, sources, stage_used)

    print("\nDone!")


def export_crm_excel(scenarios, baseline, sources, stage_used):
    """Export one Excel file with one sheet per chart group.
    Each sheet has materials as rows, and columns for baselines +
    scenario totals + per-technology breakdown."""
    all_techs = [
        'fuel cells for e-mobility', 'fuel cells for renewables',
        'batteries for e-mobility', 'batteries for renewables',
        'Traction Motors', 'PV', 'Wind Turbines',
    ]
    scen_combos = [(2030, 'LDS'), (2030, 'MDS'), (2030, 'HDS'),
                   (2050, 'LDS'), (2050, 'MDS'), (2050, 'HDS')]

    source_labels = [SOURCE_SHORT.get(s, s[:20]).replace('\n', ' ')
                     for s in sources]

    out_path = os.path.join(DATA_OUTPUT_DIR,
                            'jrc_critical_raw_materials.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for group in CHART_GROUPS:
            rows = []
            for material in group['materials']:
                bl_name = MATERIAL_TO_BASELINE.get(material, material)
                row = {'Material': material}

                # Baselines
                for si, source in enumerate(sources):
                    lbl = source_labels[si]
                    bl_row = baseline[
                        (baseline['material'] == bl_name) &
                        (baseline['source'] == source)
                    ]
                    val = bl_row['value'].iloc[0] if len(bl_row) > 0 else np.nan
                    stage = stage_used.get((bl_name, source))
                    row[f'Baseline: {lbl}'] = val
                    row[f'Baseline: {lbl} (stage)'] = stage

                # Scenarios: total + per-tech breakdown
                for year, scen in scen_combos:
                    col_prefix = f'{year} {scen}'
                    total = 0
                    for tech in all_techs:
                        val = scenarios[
                            (scenarios['material'] == material) &
                            (scenarios['year'] == year) &
                            (scenarios['scenario'] == scen) &
                            (scenarios['transition_innovation'] == tech)
                        ]['value'].sum()
                        row[f'{col_prefix}: {TECH_SHORT[tech]}'] = val
                        total += val
                    row[f'{col_prefix}: TOTAL'] = total

                rows.append(row)

            df = pd.DataFrame(rows)
            sheet_name = group['title'][:31]  # Excel 31-char limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved Excel: {out_path}")


if __name__ == "__main__":
    main()
