"""
6_health.py — PM2.5 and NO2 exposure maps (YLL per 100k) for 4 reports.

Produces 2 visuals per report (PM2.5 map, NO2 map), each a 1×3 panel
(2005, 2013, 2023) NUTS-3 choropleth with shared legend.

Reports:
  rep_eu   : EU-27 + EFTA view
  rep_ewbi : EU-27 + EFTA view (identical scope)
  rep_fr   : France zoom
  rep_ch   : Switzerland zoom

All outputs: PNG + SVG + Excel.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import mapclassify
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
REPORT_DIR = SCRIPT_DIR.parent
EXTERNAL_DATA_DIR = REPORT_DIR / "external_data"
OUTPUT_BASE = REPORT_DIR / "outputs" / "graphs" / "Health"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Shapefile locations (shared)
REPORT1_DIR = REPORT_DIR.parent / "1_switzerland_vs_eu27_housing_energy"
SHAPEFILE_DIR = REPORT1_DIR / "external_data" / "0_shapefile"
NUTS_SHAPEFILE = EXTERNAL_DATA_DIR / "NUTS_RG_20M_2021_3035.gpkg"
WORLD_SHAPEFILE = (
    SHAPEFILE_DIR / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
)

# Study countries (EU-27 + EFTA + UK)
STUDY_COUNTRIES = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI',
    'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU',
    'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE',
    'UK', 'NO', 'CH', 'IS', 'LI',
]

# NUTS code prefixes (for filtering which regions carry data)
# EEA uses EL for Greece, the shapefile also uses EL
EU27_NUTS_PREFIXES = {
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI',
    'FR', 'DE', 'EL', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU',
    'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE',
}
EFTA_NUTS_PREFIXES = {'CH', 'NO', 'IS', 'LI'}
EWBI_NUTS_PREFIXES = EU27_NUTS_PREFIXES | EFTA_NUTS_PREFIXES

plt.rcParams["font.family"] = "Arial"

# ---------------------------------------------------------------------------
# Report configurations
# ---------------------------------------------------------------------------
REPORT_CONFIGS = {
    'rep_eu': {
        'prefix': 'rep_eu',
        'title_suffix': '(EU-27)',
        'country_filter': None,       # EEA data filter (None = all)
        'data_countries': EU27_NUTS_PREFIXES,  # which NUTS prefixes carry data
        'xlim': (2_200_000, 6_600_000),
        'ylim': (1_200_000, 5_800_000),
    },
    'rep_ewbi': {
        'prefix': 'rep_ewbi',
        'title_suffix': '(EU-27 + EFTA)',
        'country_filter': None,
        'data_countries': EWBI_NUTS_PREFIXES,
        'xlim': (2_200_000, 6_600_000),
        'ylim': (1_200_000, 5_800_000),
    },
    'rep_fr': {
        'prefix': 'rep_fr',
        'title_suffix': '(France)',
        'country_filter': ('FR', 'ES', 'IT', 'CH', 'DE', 'BE'),
        'data_countries': {'FR', 'ES', 'IT', 'CH', 'DE', 'BE'},
        'xlim': (2_400_000, 5_200_000),
        'ylim': (1_300_000, 3_800_000),
    },
    'rep_ch': {
        'prefix': 'rep_ch',
        'title_suffix': '(Switzerland)',
        'country_filter': ('CH', 'FR', 'IT', 'DE', 'AT'),
        'data_countries': {'CH', 'FR', 'IT', 'DE', 'AT'},
        'xlim': (2_800_000, 5_300_000),
        'ylim': (1_500_000, 3_800_000),
    },
}

POLLUTANTS = ["PM2.5", "NO2"]
YEARS = [2005, 2013, 2023]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_fig(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    svg_path = path.with_suffix('.svg') if isinstance(path, Path) else Path(path).with_suffix('.svg')
    fig.savefig(str(svg_path), format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {Path(path).name}")
    print(f"  Saved: {svg_path.name}")


def _save_excel(df, path, sheet_name='Data'):
    if isinstance(path, str):
        path = Path(path)
    if isinstance(df, dict):
        with pd.ExcelWriter(str(path), engine='openpyxl') as writer:
            for name, data in df.items():
                data.to_excel(writer, sheet_name=name[:31], index=False)
    else:
        df.to_excel(str(path), index=False, sheet_name=sheet_name)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_eea_data():
    csv_path = EXTERNAL_DATA_DIR / "eea_dataset.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    return df


def prepare_nuts3_data(df, pollutant, year, country_filter=None):
    mask = (
        (df["Health Indicator"] == "Years of Life Lost (YLL)")
        & (df["Scenario"] == "Baseline from WHO 2021 AQG")
        & (df["Air Pollutant"] == pollutant)
        & (df["Year"] == year)
        & (df["Outcome"] == "All causes")
        & (df["NUTS Code"].str.len() == 5)
    )
    if country_filter:
        if isinstance(country_filter, (list, tuple)):
            mask = mask & df["NUTS Code"].str.startswith(tuple(country_filter))
        else:
            mask = mask & df["NUTS Code"].str.startswith(country_filter)
    filtered = df.loc[
        mask, ["NUTS Code", "NUTS Name", "Year", "Value for 100k Of Affected Population"]
    ].copy()
    filtered.columns = ["geo", "nuts_name", "year", "value"]
    filtered["value"] = pd.to_numeric(filtered["value"], errors="coerce")
    if isinstance(country_filter, (list, tuple)):
        label = f" ({','.join(country_filter)})"
    elif country_filter:
        label = f" ({country_filter})"
    else:
        label = ""
    print(f"    {pollutant} / {year}{label}: {len(filtered)} NUTS-3 regions")
    return filtered


# ---------------------------------------------------------------------------
# Shapefile loading (cached)
# ---------------------------------------------------------------------------
_SHAPEFILE_CACHE = {}

def _load_shapefiles():
    if 'nuts' in _SHAPEFILE_CACHE:
        return _SHAPEFILE_CACHE['nuts'], _SHAPEFILE_CACHE['europe']

    try:
        nuts = gpd.read_file(str(NUTS_SHAPEFILE), layer="NUTS_RG_20M_2021_3035")
    except Exception:
        nuts = gpd.read_file(str(NUTS_SHAPEFILE))

    if "LEVL_CODE" in nuts.columns:
        nuts = nuts[nuts["LEVL_CODE"] == 3].copy()

    nuts_code_col = next(
        (c for c in ["NUTS_ID", "geo", "CNTR_CODE", "id", "CODE"] if c in nuts.columns),
        None,
    )
    if nuts_code_col and nuts_code_col != "geo":
        nuts = nuts.rename(columns={nuts_code_col: "geo"})

    if nuts.crs != "EPSG:3035":
        nuts = nuts.to_crs(epsg=3035)

    world = gpd.read_file(str(WORLD_SHAPEFILE))
    europe_world = world[
        (world["CONTINENT"] == "Europe") | (world["ISO_A2"] == "TR")
    ].copy()
    europe_world = europe_world.to_crs(epsg=3035)

    _SHAPEFILE_CACHE['nuts'] = nuts
    _SHAPEFILE_CACHE['europe'] = europe_world
    return nuts, europe_world


# ---------------------------------------------------------------------------
# Map creation (1×3 panels: 2005, 2013, 2023)
# ---------------------------------------------------------------------------
def create_map(df, pollutant, cfg, out_dir):
    """Create a 1×3 NUTS-3 choropleth (2005 | 2013 | 2023)."""
    country_filter = cfg['country_filter']
    xlim = cfg['xlim']
    ylim = cfg['ylim']

    country_code_map = {"UK": "GB"}
    study_mapped = [country_code_map.get(c, c) for c in STUDY_COUNTRIES]
    # NUTS prefixes that should carry data for this report
    data_prefixes = cfg.get('data_countries', EWBI_NUTS_PREFIXES)

    nuts_base, europe_world = _load_shapefiles()
    non_study_countries = europe_world[~europe_world["ISO_A2"].isin(study_mapped)]

    # Prepare data for all years
    year_datasets = {}
    for yr in YEARS:
        year_datasets[yr] = prepare_nuts3_data(df, pollutant, yr,
                                               country_filter=country_filter)

    if all(year_datasets[yr].empty for yr in YEARS):
        print(f"    No data for {pollutant} – skipping")
        return

    # Shared quantile bins
    all_values = pd.concat([year_datasets[yr]["value"] for yr in YEARS]).dropna()
    k = 6
    classifier = mapclassify.Quantiles(all_values, k=k)
    bins_used = classifier.bins
    colormap = "YlOrRd"
    n_classes = len(bins_used)
    cmap_obj = matplotlib.colormaps.get_cmap(colormap).resampled(n_classes)

    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    plt.subplots_adjust(wspace=0.02)

    for ax, year in zip(axes, YEARS):
        year_data = year_datasets[year]
        nuts = nuts_base.copy()
        nuts = nuts.merge(year_data[["geo", "value"]], on="geo", how="left")
        nuts["nuts_prefix"] = nuts["geo"].astype(str).str[:2]
        nuts["country_code"] = nuts["nuts_prefix"].replace({"EL": "GR"})
        # Only mark regions as study if their NUTS prefix is in this report's scope
        nuts["is_study_region"] = nuts["nuts_prefix"].isin(data_prefixes)

        # Background non-study
        if not non_study_countries.empty:
            non_study_countries.plot(
                ax=ax, color="white", edgecolor="black",
                linewidth=0.3, hatch="///", alpha=0.35,
            )

        # Study regions with data
        study_nuts = nuts[nuts["is_study_region"] & nuts["value"].notna()].copy()
        if not study_nuts.empty:
            study_nuts.plot(
                column="value", cmap=colormap, linewidth=0.1, ax=ax,
                edgecolor="black", legend=False,
                scheme="UserDefined",
                classification_kwds={"bins": list(bins_used)},
                missing_kwds={"color": "lightgrey"},
            )

        # Study regions without data
        study_missing = nuts[nuts["is_study_region"] & nuts["value"].isna()].copy()
        if not study_missing.empty:
            study_missing.plot(
                ax=ax, color="lightgrey", linewidth=0.1, edgecolor="black",
            )

        # Study countries with no NUTS-3 regions (scoped to report)
        nuts_country_codes = nuts["country_code"].unique()
        # Convert NUTS prefixes to ISO-A2 for world borders lookup
        prefix_to_iso = {p: (country_code_map.get(p, p) if p != 'EL' else 'GR')
                         for p in data_prefixes}
        scoped_iso = set(prefix_to_iso.values())
        missing_codes = [c for c in scoped_iso if c not in nuts_country_codes]
        study_no_nuts = europe_world[europe_world["ISO_A2"].isin(missing_codes)]
        if not study_no_nuts.empty:
            study_no_nuts.plot(
                ax=ax, color="lightgrey", edgecolor="black", linewidth=0.3,
            )

        # Country borders
        europe_world.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_axis_off()
        ax.set_title(str(year), fontsize=18, fontweight="bold", pad=12)

    # Shared legend
    vmin = float(all_values.min())
    legend_elements = []
    for i in range(n_classes):
        low = vmin if i == 0 else float(bins_used[i - 1])
        high = float(bins_used[i])
        label = f"{int(round(low)):,} \u2013 {int(round(high)):,}"
        patch = mpatches.Rectangle(
            (0, 0), 1, 1,
            facecolor=cmap_obj(i), edgecolor="black", linewidth=0.3,
        )
        legend_elements.append((patch, label))

    legend_elements.append(
        (mpatches.Rectangle((0, 0), 1, 1, facecolor="lightgrey",
                            edgecolor="black", linewidth=0.3),
         "Missing values")
    )
    legend_elements.append(
        (mpatches.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                            hatch="///", linewidth=0.3, alpha=0.35),
         "Non-study regions")
    )

    value_title = f"YLL per 100k \u2013 {pollutant}"
    axes[2].legend(
        [e[0] for e in legend_elements],
        [e[1] for e in legend_elements],
        loc="upper right",
        title=value_title,
        fontsize=12,
        title_fontsize=13,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
    )

    plt.tight_layout()

    # Save
    safe_pollutant = pollutant.replace('.', '')
    out_png = out_dir / f"{cfg['prefix']}_pm_exposure_YLL_{safe_pollutant}.png"
    _save_fig(fig, out_png)
    plt.close(fig)

    # Excel export
    excel_rows = []
    for yr in YEARS:
        yd = year_datasets[yr]
        if not yd.empty:
            edf = yd.copy()
            edf['pollutant'] = pollutant
            excel_rows.append(edf)
    if excel_rows:
        excel_df = pd.concat(excel_rows, ignore_index=True)
        xlsx_path = out_dir / f"{cfg['prefix']}_pm_exposure_YLL_{safe_pollutant}.xlsx"
        _save_excel(excel_df, xlsx_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_report(report_key, df):
    cfg = REPORT_CONFIGS[report_key]
    print(f"\n{'='*60}")
    print(f"Generating health report: {cfg['prefix']} {cfg['title_suffix']}")
    print(f"{'='*60}")

    out_dir = OUTPUT_BASE / cfg['prefix']
    out_dir.mkdir(parents=True, exist_ok=True)

    for pollutant in POLLUTANTS:
        print(f"\n  [{POLLUTANTS.index(pollutant)+1}] {pollutant} map...")
        create_map(df, pollutant, cfg, out_dir)

    print(f"\n  All outputs saved to: {out_dir}")


def main():
    print("Loading EEA PM/NO2 data...")
    df = load_eea_data()

    for report_key in ['rep_eu', 'rep_ewbi', 'rep_fr', 'rep_ch']:
        generate_report(report_key, df)
    print("\nDone.")


if __name__ == '__main__':
    main()
