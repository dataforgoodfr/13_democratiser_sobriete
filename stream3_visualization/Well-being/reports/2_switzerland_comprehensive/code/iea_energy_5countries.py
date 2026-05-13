"""
Energy Import Dependency – 5-Country Comparison (CH, FR, IT, DE, AT)
Net energy imports / Total energy supply (%), 1990–2024

Data sources:
  CH:  external_data/iea_Net energy imports.csv, iea_Total energy supply.csv,
       iea_Oil products imports vs. exports - Switzerland.csv (added)
  AT/DE/IT: external_data/iea/Net energy imports - {Country}.csv,
            iea/Total energy supply (TES) by source - {Country}.csv
  FR:  external_data (from report 3)/ individual imports-vs-exports files +
       Domestic energy production by source
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
REPORT2_DATA = Path(__file__).parent.parent / "external_data"        # 2_switzerland_comprehensive
REPORT3_DATA = REPORT2_DATA.parent.parent / "3_eu_analysis_with_examples" / "external_data"
IEA_SUBDIR = REPORT2_DATA / "iea"  # AT/DE/IT files

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "graphs" / "energy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Country display
COUNTRIES = {
    'CH': {'name': 'Switzerland', 'color': '#e41a1c', 'lw': 2.5},
    'FR': {'name': 'France',      'color': '#377eb8', 'lw': 2.0},
    'DE': {'name': 'Germany',     'color': '#4daf4a', 'lw': 2.0},
    'IT': {'name': 'Italy',       'color': '#ff7f00', 'lw': 2.0},
    'AT': {'name': 'Austria',     'color': '#984ea3', 'lw': 2.0},
}

ENERGY_TYPES = ['Coal', 'Crude oil', 'Electricity', 'Natural gas', 'Oil products']


# ============================================================================
# LOADERS
# ============================================================================
def _read_iea(path, skiprows=3):
    """Read an IEA CSV (skip 3 metadata rows), drop Units column."""
    df = pd.read_csv(path, skiprows=skiprows, index_col=0)
    df.index.name = 'Year'
    if 'Units' in df.columns:
        df = df.drop('Units', axis=1)
    return df


def _load_net_imports_from_file(path):
    """Load pre-computed net-imports file (PJ)."""
    df = _read_iea(path)
    return df['Net imports']  # PJ


def _load_net_imports_from_components(file_map):
    """Compute total net imports (TJ) from individual imports-vs-exports files."""
    total = None
    for etype, fpath in file_map.items():
        df = _read_iea(fpath)
        imports = df['Imports'].fillna(0)
        if 'Exports' in df.columns:
            exports = df['Exports'].fillna(0)
            if (exports < 0).any():
                exports = -exports
            net = imports - exports
        else:
            net = imports
        if total is None:
            total = net.copy()
        else:
            total = total.add(net, fill_value=0)
    return total  # TJ


def _load_tes_from_file(path):
    """Load Total Energy Supply from TES-by-source file (TJ)."""
    df = _read_iea(path)
    return df.sum(axis=1)  # TJ


def _load_tes_from_production_and_net_imports(prod_path, net_imports_tj):
    """TES = Domestic production + Net imports (both TJ)."""
    df = _read_iea(prod_path)
    production = df.sum(axis=1)  # TJ
    common = production.index.intersection(net_imports_tj.index)
    return (production.loc[common] + net_imports_tj.loc[common])


def _load_nuclear_production(path):
    """Load 'Nuclear' column from a TES-by-source or production-by-source file (TJ)."""
    df = _read_iea(path)
    if 'Nuclear' in df.columns:
        return df['Nuclear'].fillna(0)
    return pd.Series(dtype=float)


# ============================================================================
# LOAD EACH COUNTRY
# ============================================================================
def load_all_countries():
    results = {}

    # ------------------------------------------------------------------
    # SWITZERLAND  (Net imports in PJ, TES in TJ)
    # ------------------------------------------------------------------
    ch_ni_pj = _load_net_imports_from_file(REPORT2_DATA / 'iea_Net energy imports.csv')
    ch_tes_tj = _load_tes_from_file(REPORT2_DATA / 'iea_Total energy supply.csv')
    ch_ni_tj = ch_ni_pj * 1000  # PJ → TJ
    # Nuclear correction: treat nuclear heat production as imports
    ch_nuclear = _load_nuclear_production(REPORT2_DATA / 'iea_Total energy supply.csv')
    ch_ni_tj = ch_ni_tj.add(ch_nuclear, fill_value=0)
    common = ch_ni_tj.index.intersection(ch_tes_tj.index)
    results['CH'] = pd.DataFrame({
        'Net_imports_TJ': ch_ni_tj.loc[common],
        'TES_TJ': ch_tes_tj.loc[common],
    })
    print(f"CH: {len(common)} years, dependency {(ch_ni_tj.loc[common] / ch_tes_tj.loc[common] * 100).mean():.1f}% avg")

    # ------------------------------------------------------------------
    # AUSTRIA, GERMANY, ITALY  (iea/ subfolder)
    # ------------------------------------------------------------------
    country_names = {'AT': 'Austria', 'DE': 'Germany', 'IT': 'Italy'}
    for iso, cname in country_names.items():
        ni_pj = _load_net_imports_from_file(IEA_SUBDIR / f'Net energy imports - {cname}.csv')
        tes_path = IEA_SUBDIR / f'Total energy supply (TES) by source - {cname}.csv'
        tes_tj = _load_tes_from_file(tes_path)
        ni_tj = ni_pj * 1000
        # Nuclear correction: treat nuclear heat production as imports
        nuclear_tj = _load_nuclear_production(tes_path)
        ni_tj = ni_tj.add(nuclear_tj, fill_value=0)
        common = ni_tj.index.intersection(tes_tj.index)
        results[iso] = pd.DataFrame({
            'Net_imports_TJ': ni_tj.loc[common],
            'TES_TJ': tes_tj.loc[common],
        })
        dep = (ni_tj.loc[common] / tes_tj.loc[common] * 100).mean()
        print(f"{iso}: {len(common)} years, dependency {dep:.1f}% avg")

    # ------------------------------------------------------------------
    # FRANCE  (compute from individual import/export + production files)
    # ------------------------------------------------------------------
    fr_files = {}
    for etype in ENERGY_TYPES:
        fr_files[etype] = REPORT3_DATA / f'iea_{etype} imports vs. exports - France.csv'
    fr_ni_tj = _load_net_imports_from_components(fr_files)

    fr_prod_path = REPORT3_DATA / 'iea_Domestic energy production by source - France.csv'
    fr_tes_tj = _load_tes_from_production_and_net_imports(fr_prod_path, fr_ni_tj)
    # Nuclear correction: treat nuclear heat production as imports
    fr_nuclear = _load_nuclear_production(fr_prod_path)
    fr_ni_tj = fr_ni_tj.add(fr_nuclear, fill_value=0)
    common = fr_ni_tj.index.intersection(fr_tes_tj.index)
    results['FR'] = pd.DataFrame({
        'Net_imports_TJ': fr_ni_tj.loc[common],
        'TES_TJ': fr_tes_tj.loc[common],
    })
    dep = (fr_ni_tj.loc[common] / fr_tes_tj.loc[common] * 100).mean()
    print(f"FR: {len(common)} years, dependency {dep:.1f}% avg")

    return results


# ============================================================================
# VISUALS
# ============================================================================
def create_dependency_comparison(data):
    """Line chart: Import dependency (%) for all 5 countries."""
    fig, ax = plt.subplots(figsize=(14, 8))

    for iso, info in COUNTRIES.items():
        df = data[iso]
        dep = df['Net_imports_TJ'] / df['TES_TJ'] * 100
        ax.plot(dep.index, dep.values, color=info['color'],
                linewidth=info['lw'], label=info['name'],
                marker='o' if iso == 'CH' else None, markersize=3)

    ax.set_title('Net Energy Import Dependency – 5-Country Comparison\n(1990–2024)',
                 fontsize=15, fontweight='bold', color='#2c3e50', pad=15)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Net Imports / Total Energy Supply (%)', fontsize=12)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUTPUT_DIR / f'energy_import_dependency_5countries.{ext}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved energy_import_dependency_5countries.png / .svg")


def create_net_imports_by_type_comparison(data_by_type):
    """2×3 grid: net imports by energy type for all 5 countries (TJ)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes_flat = axes.flatten()

    for i, etype in enumerate(ENERGY_TYPES):
        ax = axes_flat[i]
        for iso, info in COUNTRIES.items():
            if iso in data_by_type and etype in data_by_type[iso]:
                s = data_by_type[iso][etype]
                ax.plot(s.index, s.values / 1000, color=info['color'],
                        linewidth=1.5, label=info['name'])
        ax.set_title(etype, fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Net Imports (PJ)', fontsize=9)
        ax.axhline(0, color='grey', lw=0.5, ls='--')
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 6th panel: total
    ax = axes_flat[5]
    for iso, info in COUNTRIES.items():
        df = data[iso]
        ax.plot(df.index, df['Net_imports_TJ'].values / 1000,
                color=info['color'], linewidth=1.5, label=info['name'])
    ax.set_title('Total Net Imports', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Net Imports (PJ)', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Net Energy Imports by Type – 5-Country Comparison (1990–2024)',
                 fontsize=14, fontweight='bold', color='#2c3e50')
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    for ext in ('png', 'svg'):
        fig.savefig(OUTPUT_DIR / f'energy_net_imports_by_type_5countries.{ext}',
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved energy_net_imports_by_type_5countries.png / .svg")


def load_net_imports_by_type():
    """Load per-type net imports for every country."""
    result = {}

    # CH: original 4 (Crude Oil, Coal, Electricity, Natural Gas) + Oil products
    ch_files = {
        'Crude oil':    REPORT2_DATA / 'iea_Crude oil imports.csv',
        'Coal':         REPORT2_DATA / 'iea_Coal imports.csv',
        'Electricity':  REPORT2_DATA / 'iea_Electricity imports.csv',
        'Natural gas':  REPORT2_DATA / 'iea_Natural gas imports.csv',
        'Oil products': REPORT2_DATA / 'iea_Oil products imports vs. exports - Switzerland.csv',
    }
    ch = {}
    for etype, fpath in ch_files.items():
        df = _read_iea(fpath)
        imports = df['Imports'].fillna(0)
        if 'Exports' in df.columns:
            exports = df['Exports'].fillna(0)
            if (exports < 0).any():
                exports = -exports
            ch[etype] = imports - exports
        else:
            ch[etype] = imports
    result['CH'] = ch

    # AT, DE, IT
    country_names = {'AT': 'Austria', 'DE': 'Germany', 'IT': 'Italy'}
    for iso, cname in country_names.items():
        d = {}
        for etype in ENERGY_TYPES:
            fpath = IEA_SUBDIR / f'{etype} imports vs. exports - {cname}.csv'
            if fpath.exists():
                df = _read_iea(fpath)
                imports = df['Imports'].fillna(0)
                exports = df['Exports'].fillna(0) if 'Exports' in df.columns else 0
                if isinstance(exports, pd.Series) and (exports < 0).any():
                    exports = -exports
                d[etype] = imports - exports
        # Also add Crude oil_NGL if present
        ngl_path = IEA_SUBDIR / f'Crude oil_NGL imports vs. exports - {cname}.csv'
        if ngl_path.exists():
            df = _read_iea(ngl_path)
            ngl_imports = df['Imports'].fillna(0)
            ngl_exports = df['Exports'].fillna(0) if 'Exports' in df.columns else 0
            if isinstance(ngl_exports, pd.Series) and (ngl_exports < 0).any():
                ngl_exports = -ngl_exports
            # Fold NGL into Crude oil
            if 'Crude oil' in d:
                d['Crude oil'] = d['Crude oil'].add(ngl_imports - ngl_exports, fill_value=0)
            else:
                d['Crude oil'] = ngl_imports - ngl_exports
        result[iso] = d

    # FR
    fr = {}
    for etype in ENERGY_TYPES:
        fpath = REPORT3_DATA / f'iea_{etype} imports vs. exports - France.csv'
        df = _read_iea(fpath)
        imports = df['Imports'].fillna(0)
        exports = df['Exports'].fillna(0) if 'Exports' in df.columns else 0
        if isinstance(exports, pd.Series) and (exports < 0).any():
            exports = -exports
        fr[etype] = imports - exports
    result['FR'] = fr

    return result


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('ENERGY IMPORT DEPENDENCY – 5-COUNTRY COMPARISON')
    print('=' * 70)

    data = load_all_countries()
    data_by_type = load_net_imports_by_type()

    print('\nCreating visuals...')
    create_dependency_comparison(data)
    create_net_imports_by_type_comparison(data_by_type)

    # Export summary CSV
    rows = []
    for iso in COUNTRIES:
        df = data[iso]
        dep = df['Net_imports_TJ'] / df['TES_TJ'] * 100
        for year in dep.index:
            rows.append({'Country': iso, 'Year': year,
                         'Net_imports_TJ': df.loc[year, 'Net_imports_TJ'],
                         'TES_TJ': df.loc[year, 'TES_TJ'],
                         'Import_dependency_pct': dep.loc[year]})
    export = pd.DataFrame(rows)
    export.to_csv(OUTPUT_DIR / 'energy_import_dependency_5countries.csv', index=False)
    print(f"Saved energy_import_dependency_5countries.csv")

    print(f'\nDone – outputs in {OUTPUT_DIR}')
