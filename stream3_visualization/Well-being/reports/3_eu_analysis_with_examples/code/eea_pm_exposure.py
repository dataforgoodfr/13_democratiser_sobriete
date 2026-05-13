"""
Create NUTS-3 choropleth maps for particulate matter and NO2 exposure
across EU + EFTA countries.

Produces one side-by-side figure per pollutant (PM2.5, NO2) with
2013 on the left and 2023 on the right, sharing a single legend.
Only the "All causes" outcome is used.
Health indicator: Years of Life Lost (YLL) per 100k affected population.
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import geopandas as gpd
import mapclassify
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
REPORT_DIR = SCRIPT_DIR.parent  # 3_eu_analysis_with_examples
EXTERNAL_DATA_DIR = REPORT_DIR / "external_data"
OUTPUT_DIR = REPORT_DIR / "outputs" / "graphs" / "particulate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shapefile locations (shared with report 1)
REPORT1_DIR = REPORT_DIR.parent / "1_switzerland_vs_eu27_housing_energy"
SHAPEFILE_DIR = REPORT1_DIR / "external_data" / "0_shapefile"
NUTS_SHAPEFILE = SHAPEFILE_DIR / "NUTS_RG_10M_2024_3035.gpkg"
WORLD_SHAPEFILE = (
    SHAPEFILE_DIR / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp"
)

# Import study_countries list from report 1
sys.path.insert(0, str(REPORT1_DIR / "code"))
from plot_functions import study_countries  # noqa: E402

plt.rcParams["font.family"] = "Arial"

# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------


def load_eea_pm_data():
    """Load the EEA particulate matter exposure CSV."""
    csv_path = EXTERNAL_DATA_DIR / "eea_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    return df


def prepare_nuts3_data(df, pollutant, year, outcome="All causes",
                       country_filter=None):
    """
    Filter for a single pollutant / year / outcome and return a DataFrame
    with columns ``geo``, ``year``, ``value`` (NUTS-3 level only).

    If *country_filter* is given (e.g. ``"FR"``), only NUTS codes starting
    with that prefix are kept.
    """
    mask = (
        (df["Health Indicator"] == "Years of Life Lost (YLL)")
        & (df["Scenario"] == "Baseline from WHO 2021 AQG")
        & (df["Air Pollutant"] == pollutant)
        & (df["Year"] == year)
        & (df["Outcome"] == outcome)
        & (df["NUTS Code"].str.len() == 5)
    )
    if country_filter:
        mask = mask & df["NUTS Code"].str.startswith(country_filter)
    filtered = df.loc[
        mask, ["NUTS Code", "Year", "Value for 100k Of Affected Population"]
    ].copy()
    filtered.columns = ["geo", "year", "value"]
    filtered["value"] = pd.to_numeric(filtered["value"], errors="coerce")
    label = f" ({country_filter})" if country_filter else ""
    print(f"  {pollutant} / {year}{label}: {len(filtered)} NUTS-3 regions")
    return filtered


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------


def export_data_to_excel(df_all, output_dir):
    """Export processed data to a single Excel file."""
    excel_data = []
    for _, row in df_all.iterrows():
        excel_data.append(
            {
                "visual_number": None,
                "visual_name": (
                    f"YLL due to {row['pollutant']} (All causes) – {row['nuts_name']}"
                ),
                "year": row["year"],
                "decile": None,
                "value": row["value"],
                "unit": "Per 100,000 affected population",
            }
        )
    excel_df = pd.DataFrame(excel_data)
    excel_path = output_dir / "eu_efta_pm_exposure_YLL_data.xlsx"
    excel_df.to_excel(excel_path, index=False, sheet_name="PM_Exposure_Data")
    print(f"Saved Excel data: {excel_path}")


# ---------------------------------------------------------------------------
# Side-by-side map (2013 | 2023) with shared legend
# ---------------------------------------------------------------------------


def _load_shapefiles():
    """Load NUTS-3 and world shapefiles (done once, reused for both panels)."""
    try:
        nuts = gpd.read_file(str(NUTS_SHAPEFILE), layer="NUTS_RG_10M_2024_3035")
    except Exception:
        nuts = gpd.read_file(str(NUTS_SHAPEFILE))

    if "LEVL_CODE" in nuts.columns:
        nuts = nuts[nuts["LEVL_CODE"] == 3].copy()

    nuts_code_col = next(
        (c for c in ["NUTS_ID", "geo", "CNTR_CODE", "id", "CODE"] if c in nuts.columns),
        None,
    )
    if nuts_code_col is None:
        raise ValueError("Could not find a NUTS code column in the shapefile.")
    if nuts_code_col != "geo":
        nuts = nuts.rename(columns={nuts_code_col: "geo"})

    if nuts.crs != "EPSG:3035":
        nuts = nuts.to_crs(epsg=3035)

    world = gpd.read_file(str(WORLD_SHAPEFILE))
    europe_world = world[
        (world["CONTINENT"] == "Europe") | (world["ISO_A2"] == "TR")
    ].copy()
    europe_world = europe_world.to_crs(epsg=3035)

    return nuts, europe_world


def create_side_by_side_map(df, pollutant, outcome="All causes",
                            country_filter=None, xlim=None, ylim=None,
                            filename_prefix="eu_efta"):
    """
    Create a 1×3 figure with 2005 (left), 2013 (centre) and 2023 (right)
    NUTS-3 maps sharing a single legend and using integer labels.

    If *country_filter* is set (e.g. ``"FR"``), only that country's
    NUTS-3 regions carry data; *xlim*/*ylim* let you zoom the map.
    """
    country_code_map = {"UK": "GB"}
    study_mapped = [country_code_map.get(c, c) for c in study_countries]

    nuts_base, europe_world = _load_shapefiles()
    non_study_countries = europe_world[~europe_world["ISO_A2"].isin(study_mapped)]

    # Prepare data for all three years
    years = [2005, 2013, 2023]
    year_datasets = {}
    for yr in years:
        year_datasets[yr] = prepare_nuts3_data(df, pollutant, yr, outcome=outcome,
                                               country_filter=country_filter)

    if all(year_datasets[yr].empty for yr in years):
        print(f"  ⚠ No data for {pollutant} – skipping")
        return

    # Compute shared quantile bins across all years
    all_values = pd.concat([year_datasets[yr]["value"] for yr in years]).dropna()
    k = 6
    classifier = mapclassify.Quantiles(all_values, k=k)
    bins_used = classifier.bins
    colormap = "YlOrRd"
    n_classes = len(bins_used)
    cmap_obj = cm.get_cmap(colormap, n_classes)

    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    plt.subplots_adjust(wspace=0.02)

    for ax, year in zip(axes, years):
        year_data = year_datasets[year]
        nuts = nuts_base.copy()
        nuts = nuts.merge(
            year_data[["geo", "value"]], on="geo", how="left"
        )
        # Map NUTS prefix 'EL' (Greece) back to ISO 'GR' used in study_countries
        nuts["country_code"] = (
            nuts["geo"].astype(str).str[:2].replace({"EL": "GR"})
        )
        nuts["is_study_region"] = nuts["country_code"].isin(study_mapped)

        # Background: non-study countries
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

        # Study countries with no NUTS-3 regions at all (e.g. UK in 2024 shapefile)
        nuts_country_codes = nuts["country_code"].unique()
        missing_country_codes = [
            c for c in study_mapped if c not in nuts_country_codes
        ]
        study_no_nuts = europe_world[
            europe_world["ISO_A2"].isin(missing_country_codes)
        ]
        if not study_no_nuts.empty:
            study_no_nuts.plot(
                ax=ax, color="lightgrey", edgecolor="black", linewidth=0.3,
            )

        # Country borders
        europe_world.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)

        ax.set_xlim(*(xlim or (2_200_000, 6_600_000)))
        ax.set_ylim(*(ylim or (1_200_000, 5_800_000)))
        ax.set_axis_off()
        ax.set_title(str(year), fontsize=18, fontweight="bold", pad=12)

    # ---- Shared legend (placed on the right panel) ----
    vmin = float(all_values.min())
    legend_elements = []
    for i in range(n_classes):
        low = vmin if i == 0 else float(bins_used[i - 1])
        high = float(bins_used[i])
        label = f"{int(round(low)):,} – {int(round(high)):,}"
        patch = mpatches.Rectangle(
            (0, 0), 1, 1,
            facecolor=cmap_obj(i), edgecolor="black", linewidth=0.3,
        )
        legend_elements.append((patch, label))

    legend_elements.append(
        (
            mpatches.Rectangle(
                (0, 0), 1, 1, facecolor="lightgrey",
                edgecolor="black", linewidth=0.3,
            ),
            "Missing values",
        )
    )
    legend_elements.append(
        (
            mpatches.Rectangle(
                (0, 0), 1, 1, facecolor="white", edgecolor="black",
                hatch="///", linewidth=0.3, alpha=0.35,
            ),
            "Non-study regions",
        )
    )

    value_title = f"YLL per 100k – {pollutant}"
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

    filename = f"{filename_prefix}_pm_exposure_YLL_{pollutant}_2005_2013_2023.png"
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")

    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")

    plt.close(fig)
    print(f"Saved: {out_path.name}")
    print(f"Saved: {svg_path.name}")


# ---------------------------------------------------------------------------
# Interactive HTML map (Plotly choropleth, year dropdown)
# ---------------------------------------------------------------------------


def create_html_map(df, pollutant, outcome="All causes",
                    country_filter=None, filename_prefix="eu_efta"):
    """
    Export an interactive Plotly choropleth as a self-contained HTML file.
    A dropdown lets the viewer switch between 2013 and 2023.
    Geometries are simplified for a compact file size.
    """
    country_code_map = {"UK": "GB"}
    study_mapped = [country_code_map.get(c, c) for c in study_countries]

    nuts_base, _ = _load_shapefiles()

    # Reproject to WGS84 and simplify for web rendering
    nuts_wgs = nuts_base[["geo", "geometry"]].to_crs(epsg=4326).copy()
    nuts_wgs["geometry"] = nuts_wgs["geometry"].simplify(tolerance=0.01)

    # Restrict to relevant countries
    if country_filter:
        nuts_wgs = nuts_wgs[nuts_wgs["geo"].str.startswith(country_filter)]
    else:
        nuts_wgs["_cc"] = nuts_wgs["geo"].str[:2].replace({"EL": "GR"})
        nuts_wgs = nuts_wgs[nuts_wgs["_cc"].isin(study_mapped)].drop(columns="_cc")

    geojson = json.loads(nuts_wgs.to_json())

    # Prepare data for all three years
    html_years = [2005, 2013, 2023]
    html_datasets = {}
    for yr in html_years:
        html_datasets[yr] = prepare_nuts3_data(df, pollutant, yr, outcome=outcome,
                                               country_filter=country_filter)

    if all(html_datasets[yr].empty for yr in html_years):
        print(f"  ⚠ No data for {pollutant} HTML map – skipping")
        return

    all_values = pd.concat([html_datasets[yr]["value"] for yr in html_years]).dropna()
    vmin, vmax = float(all_values.min()), float(all_values.max())

    fig = go.Figure()

    for idx, yr in enumerate(html_years):
        year_data = html_datasets[yr]
        merged = nuts_wgs[["geo"]].merge(year_data[["geo", "value"]], on="geo", how="left")
        fig.add_trace(go.Choropleth(
            geojson=geojson,
            locations=merged["geo"],
            z=merged["value"],
            featureidkey="properties.geo",
            colorscale="YlOrRd",
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="YLL per 100k"),
            name=str(yr),
            visible=(idx == 0),
            hovertemplate="<b>%{location}</b><br>YLL per 100k: %{z:,.0f}<extra></extra>",
        ))

    location_suffix = f" – {country_filter}" if country_filter else ""
    fig.update_layout(
        title=dict(
            text=f"YLL due to {pollutant} (All causes){location_suffix}",
            font=dict(size=18),
        ),
        geo=dict(
            resolution=50,
            showcoastlines=True,
            showland=True,
            landcolor="lightgrey",
            showocean=True,
            oceancolor="aliceblue",
            fitbounds="locations",
            visible=True,
        ),
        updatemenus=[dict(
            type="dropdown",
            buttons=[
                dict(
                    label=str(yr),
                    method="update",
                    args=[
                        {"visible": [j == i for j in range(len(html_years))]},
                        {"title.text": f"YLL due to {pollutant} (All causes){location_suffix} – {yr}"},
                    ],
                )
                for i, yr in enumerate(html_years)
            ],
            direction="down",
            x=0.0,
            y=1.12,
            showactive=True,
        )],
        margin=dict(l=0, r=0, t=80, b=0),
        height=700,
    )

    filename = f"{filename_prefix}_pm_exposure_YLL_{pollutant}_interactive.html"
    out_path = OUTPUT_DIR / filename
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved HTML map: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("EEA PM / NO2 Exposure – EU + EFTA NUTS-3 Maps (YLL, All causes)")
    print("=" * 80)

    df = load_eea_pm_data()

    # ---- Excel export ----
    excel_rows = []
    for pollutant in ["PM2.5", "NO2"]:
        for year in [2005, 2013, 2023]:
            mask = (
                (df["Health Indicator"] == "Years of Life Lost (YLL)")
                & (df["Scenario"] == "Baseline from WHO 2021 AQG")
                & (df["Air Pollutant"] == pollutant)
                & (df["Year"] == year)
                & (df["Outcome"] == "All causes")
                & (df["NUTS Code"].str.len() == 5)
            )
            sub = df.loc[
                mask,
                ["NUTS Code", "NUTS Name", "Year",
                 "Value for 100k Of Affected Population"],
            ].copy()
            sub.columns = ["geo", "nuts_name", "year", "value"]
            sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
            sub["pollutant"] = pollutant
            excel_rows.append(sub)

    if excel_rows:
        df_all = pd.concat(excel_rows, ignore_index=True)
        export_data_to_excel(df_all, OUTPUT_DIR)

    # ---- EU-wide maps (one figure per pollutant, 2013 & 2023 side by side) ----
    for pollutant in ["PM2.5", "NO2"]:
        print(f"\n--- {pollutant} ---")
        create_side_by_side_map(df, pollutant, outcome="All causes")
        create_html_map(df, pollutant, outcome="All causes")

    # ---- France-only maps & Excel exports ----
    print("\n" + "=" * 80)
    print("France-only outputs")
    print("=" * 80)

    # France map bounds in EPSG:3035
    fr_xlim = (3_000_000, 4_400_000)
    fr_ylim = (1_900_000, 3_300_000)

    for pollutant in ["PM2.5", "NO2"]:
        print(f"\n--- France – {pollutant} ---")
        create_side_by_side_map(
            df, pollutant, outcome="All causes",
            country_filter="FR", xlim=fr_xlim, ylim=fr_ylim,
            filename_prefix="france",
        )
        create_html_map(
            df, pollutant, outcome="All causes",
            country_filter="FR", filename_prefix="france",
        )

        # Per-pollutant Excel for France
        fr_excel_rows = []
        for year in [2005, 2013, 2023]:
            mask = (
                (df["Health Indicator"] == "Years of Life Lost (YLL)")
                & (df["Scenario"] == "Baseline from WHO 2021 AQG")
                & (df["Air Pollutant"] == pollutant)
                & (df["Year"] == year)
                & (df["Outcome"] == "All causes")
                & (df["NUTS Code"].str.len() == 5)
                & (df["NUTS Code"].str.startswith("FR"))
            )
            sub = df.loc[
                mask,
                ["Country Or Territory", "NUTS Code", "NUTS Name", "Year",
                 "Air Pollutant",
                 "Value for 100k Of Affected Population"],
            ].copy()
            sub.columns = ["country", "nuts3_code", "nuts3_name", "year",
                           "indicator", "value"]
            sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
            fr_excel_rows.append(sub)

        if fr_excel_rows:
            fr_df = pd.concat(fr_excel_rows, ignore_index=True)
            excel_path = OUTPUT_DIR / f"france_pm_exposure_YLL_{pollutant}_data.xlsx"
            fr_df.to_excel(excel_path, index=False, sheet_name="Data")
            print(f"Saved Excel: {excel_path.name}")

    print("\n" + "=" * 80)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
