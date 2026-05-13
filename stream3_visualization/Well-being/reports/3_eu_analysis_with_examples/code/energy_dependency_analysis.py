"""
Energy Dependency Analysis
==========================
Computes and visualises energy import dependency (net imports / gross available energy)
for the EU27 aggregate and selected EU member states.

Dependency definition:
    dependency = (Imports - Exports) / Gross available energy

Energy sources: solid fossil fuels, natural gas, oil & petroleum products.

Output figures:
  1. EU27 line chart  — one line per energy source
  2. Country line charts — one subplot per energy source, one line per country
  3. Heatmaps — one per energy source (countries × years)

Data source:
    ../external_data/eurostat_nrg_bal_s__custom_20746280_linear.csv
    (Eurostat NRG_BAL_S, values in Terajoule)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "..", "external_data",
    "eurostat_nrg_bal_s__custom_20746280_linear.csv",
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "outputs", "energy_dependency",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EU_GEO = "European Union - 27 countries (from 2020)"

COUNTRIES = {
    "FR": "France",
    "ES": "Spain",
    "BE": "Belgium",
    "NL": "Netherlands",
    "LT": "Lithuania",
    "HU": "Hungary",
    "DE": "Germany",
    "PL": "Poland",
}

ENERGY_SOURCES = {
    "Solid fossil fuels": "Solid Fossil Fuels",
    "Natural gas": "Natural Gas",
    "Oil and petroleum products (excluding biofuel portion)": "Oil & Petroleum Products",
    "Total": "Total Energy",
}

# Cluster colours (from ewbi_visuals.py)
#   Cluster 0 – Low performer / Low EWBI:  FR, ES
#   Cluster 1 – Low performer / High EWBI: BE, NL
#   Cluster 2 – High performer / Low EWBI: LT, HU
#   Cluster 3 – High performer / High EWBI: DE, PL
CLUSTER_COLORS = {
    0: "#fb8072",  # salmon
    1: "#fdb462",  # orange
    2: "#8dd3c7",  # teal
    3: "#80b1d3",  # blue
}

COUNTRY_COLORS = {
    "France":      CLUSTER_COLORS[0],
    "Spain":       CLUSTER_COLORS[0],
    "Belgium":     CLUSTER_COLORS[1],
    "Netherlands": CLUSTER_COLORS[1],
    "Lithuania":   CLUSTER_COLORS[2],
    "Hungary":     CLUSTER_COLORS[2],
    "Germany":     CLUSTER_COLORS[3],
    "Poland":      CLUSTER_COLORS[3],
}

# Marker symbols for countries — distinct within each cluster pair
COUNTRY_MARKERS = {
    "France":      "o",
    "Spain":       "s",
    "Belgium":     "o",
    "Netherlands": "s",
    "Lithuania":   "o",
    "Hungary":     "s",
    "Germany":     "o",
    "Poland":      "s",
}

# Colour palette for energy sources (consistent across plots)
SOURCE_COLORS = {
    "Solid fossil fuels": "#5a5a5a",
    "Natural gas":        "#e67e22",
    "Oil and petroleum products (excluding biofuel portion)": "#2980b9",
    "Total":              "#2c3e50",
}

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw Eurostat CSV and return a clean DataFrame."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "value"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"])
    df["year"] = df["year"].astype(int)
    return df


def compute_dependency(df: pd.DataFrame, geo_filter) -> pd.DataFrame:
    """
    Compute energy dependency ratios for the given geo filter.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    geo_filter : str or list of str
        Value(s) to match in the 'geo' column.
        Pass a single string to aggregate across all matching rows (e.g. EU27).
        Pass a list of strings to keep one row per country.

    Returns
    -------
    pd.DataFrame with columns: (geo,) year, siec, imports, exports,
                                gross_available, net_imports, dependency
    """
    single_geo = isinstance(geo_filter, str)
    if single_geo:
        geo_filter = [geo_filter]

    subset = df[df["geo"].isin(geo_filter)].copy()

    # Group-by keys: include geo when multiple countries are requested
    group_keys = ["year", "siec", "nrg_bal"] if single_geo else ["geo", "year", "siec", "nrg_bal"]

    # Pivot on nrg_bal to get imports, exports and gross available side by side
    pivot = (
        subset
        .groupby(group_keys)["value"]
        .sum()
        .unstack("nrg_bal")
        .reset_index()
    )

    # Rename columns to safe Python names
    col_map = {
        "Imports":               "imports",
        "Exports":               "exports",
        "Gross available energy": "gross_available",
    }
    pivot = pivot.rename(columns={k: v for k, v in col_map.items() if k in pivot.columns})

    for col in ("imports", "exports", "gross_available"):
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot["net_imports"] = pivot["imports"].fillna(0) - pivot["exports"].fillna(0)

    # --- Nuclear correction: treat nuclear heat primary production as imports ---
    # Uranium fuel is imported, so domestic nuclear PP overstates energy independence
    merge_keys = ["year"] if single_geo else ["geo", "year"]
    nuclear_pp = (
        subset[(subset["siec"] == "Nuclear heat") &
               (subset["nrg_bal"] == "Primary production")]
        .groupby(merge_keys)["value"].sum()
        .reset_index()
        .rename(columns={"value": "nuclear_pp"})
    )
    if not nuclear_pp.empty:
        pivot = pivot.merge(nuclear_pp, on=merge_keys, how="left")
        pivot["nuclear_pp"] = pivot["nuclear_pp"].fillna(0)
        pivot.loc[pivot["siec"] == "Total", "net_imports"] += (
            pivot.loc[pivot["siec"] == "Total", "nuclear_pp"]
        )
        pivot = pivot.drop(columns=["nuclear_pp"])

    pivot["dependency"] = np.where(
        pivot["gross_available"].notna() & (pivot["gross_available"] != 0),
        pivot["net_imports"] / pivot["gross_available"],
        np.nan,
    )

    # Keep only the energy sources of interest
    pivot = pivot[pivot["siec"].isin(ENERGY_SOURCES.keys())]
    return pivot.sort_values(["siec", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot 1 — EU27 line chart
# ---------------------------------------------------------------------------

def plot_eu_dependency(df: pd.DataFrame) -> None:
    """Line chart of EU27 dependency for each energy source."""
    eu_dep = compute_dependency(df, EU_GEO)

    fig, ax = plt.subplots(figsize=(12, 6))

    for siec, label in ENERGY_SOURCES.items():
        data = eu_dep[eu_dep["siec"] == siec].dropna(subset=["dependency"])
        if data.empty:
            continue
        ax.plot(
            data["year"], data["dependency"] * 100,
            marker="o", linewidth=2.5, markersize=5,
            label=label, color=SOURCE_COLORS[siec],
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Dependency ratio (%)", fontsize=12)
    ax.set_title("EU27 Energy Import Dependency\n(Net Imports / Gross Available Energy)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "1_eu_energy_dependency")
    fig.savefig(out_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    print(f"Saved: {out_path}.png / .svg")

    # Excel export: rows = years, columns = energy sources
    excel_data = (
        eu_dep[["year", "siec", "dependency"]]
        .assign(dependency=lambda d: d["dependency"] * 100)
        .pivot(index="year", columns="siec", values="dependency")
        .rename(columns=ENERGY_SOURCES)
    )
    excel_data.index.name = "Year"
    excel_path = os.path.join(OUTPUT_DIR, "1_eu_energy_dependency.xlsx")
    excel_data.to_excel(excel_path)
    print(f"Saved: {excel_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — Country line charts (one subplot per energy source)
# ---------------------------------------------------------------------------

def plot_country_dependency(df: pd.DataFrame) -> None:
    """
    Four-panel line chart: one subplot per energy source (incl. Total),
    one line per selected country, with distinct colours and markers.
    """
    country_names = list(COUNTRIES.values())
    country_dep = compute_dependency(df, country_names)

    sources = list(ENERGY_SOURCES.keys())
    n_sources = len(sources)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes_flat = axes.flatten()

    for ax, siec in zip(axes_flat, sources):
        source_data = country_dep[country_dep["siec"] == siec]
        for country in country_names:
            cdata = source_data[source_data["geo"] == country].dropna(subset=["dependency"])
            if cdata.empty:
                continue
            ax.plot(
                cdata["year"], cdata["dependency"] * 100,
                marker=COUNTRY_MARKERS.get(country, "o"),
                linewidth=2, markersize=5,
                label=country, color=COUNTRY_COLORS[country],
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(ENERGY_SOURCES[siec], fontsize=12)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Dependency ratio (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Shared legend below the figure
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 8), fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Energy Import Dependency by Country\n(Net Imports / Gross Available Energy)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    out_path = os.path.join(OUTPUT_DIR, "2_country_energy_dependency")
    fig.savefig(out_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    print(f"Saved: {out_path}.png / .svg")

    # Excel export: one sheet per energy source, rows = years, columns = countries
    excel_path = os.path.join(OUTPUT_DIR, "2_country_energy_dependency.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for siec, label in ENERGY_SOURCES.items():
            sheet_data = (
                country_dep[country_dep["siec"] == siec][["geo", "year", "dependency"]]
                .assign(dependency=lambda d: d["dependency"] * 100)
                .pivot(index="year", columns="geo", values="dependency")
                .reindex(columns=[c for c in country_names if c in country_dep["geo"].unique()])
            )
            sheet_data.index.name = "Year"
            sheet_data.to_excel(writer, sheet_name=label[:31])
    print(f"Saved: {excel_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3 — Heatmaps (one per energy source)
# ---------------------------------------------------------------------------

def plot_dependency_heatmaps(df: pd.DataFrame) -> None:
    """
    One heatmap per energy source: countries (rows) × years (columns),
    coloured by dependency ratio.
    """
    country_names = list(COUNTRIES.values())
    country_dep = compute_dependency(df, country_names)

    sources = list(ENERGY_SOURCES.keys())
    n_sources = len(sources)

    fig, axes = plt.subplots(1, n_sources, figsize=(7 * n_sources, 5))

    # Symmetric diverging colormap centred on 0
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    for ax, siec in zip(axes, sources):
        source_data = country_dep[country_dep["siec"] == siec].copy()
        source_data["dependency_pct"] = source_data["dependency"] * 100

        matrix = source_data.pivot(index="geo", columns="year", values="dependency_pct")
        # Keep only countries in our selection, in order
        matrix = matrix.reindex([c for c in country_names if c in matrix.index])

        # Compute symmetric colour scale
        abs_max = matrix.abs().max().max()
        vmin, vmax = -abs_max, abs_max

        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.3,
            linecolor="white",
            annot=False,
            cbar_kws={"label": "Dependency (%)"},
            xticklabels=5,  # show every 5th year to avoid crowding
        )
        ax.set_title(ENERGY_SOURCES[siec], fontsize=12)
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.suptitle(
        "Energy Import Dependency Heatmap\n(Net Imports / Gross Available Energy, %)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "3_energy_dependency_heatmap")
    fig.savefig(out_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    print(f"Saved: {out_path}.png / .svg")

    # Excel export: one sheet per energy source, rows = years, columns = countries
    excel_path = os.path.join(OUTPUT_DIR, "3_energy_dependency_heatmap.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for siec, label in ENERGY_SOURCES.items():
            sheet_data = (
                country_dep[country_dep["siec"] == siec][["geo", "year", "dependency"]]
                .assign(dependency=lambda d: d["dependency"] * 100)
                .pivot(index="year", columns="geo", values="dependency")
                .reindex(columns=[c for c in country_names if c in country_dep["geo"].unique()])
            )
            sheet_data.index.name = "Year"
            sheet_data.to_excel(writer, sheet_name=label[:31])
    print(f"Saved: {excel_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4 — Decomposition of total dependency by source (stacked area)
# ---------------------------------------------------------------------------

# Sources to decompose (order = bottom to top in stack)
DECOMPOSITION_SOURCES = [
    "Oil and petroleum products (excluding biofuel portion)",
    "Natural gas",
    "Solid fossil fuels",
    "Renewables and biofuels",
    "Electricity",
    "Nuclear heat",
]

DECOMPOSITION_LABELS = {
    "Oil and petroleum products (excluding biofuel portion)": "Oil & Petroleum",
    "Natural gas": "Natural Gas",
    "Solid fossil fuels": "Solid Fossil Fuels",
    "Renewables and biofuels": "Renewables & Biofuels",
    "Electricity": "Electricity",
    "Nuclear heat": "Nuclear Heat (PP as import)",
    "Other": "Other",
}

DECOMPOSITION_COLORS = {
    "Oil and petroleum products (excluding biofuel portion)": "#2980b9",
    "Natural gas": "#e67e22",
    "Solid fossil fuels": "#5a5a5a",
    "Renewables and biofuels": "#27ae60",
    "Electricity": "#f1c40f",
    "Nuclear heat": "#9b59b6",
    "Other": "#bdc3c7",
}


def _compute_decomposition(df: pd.DataFrame, geo_filter) -> pd.DataFrame:
    """
    Compute each source's contribution to total dependency.
    contribution_i = net_imports_i / gross_available_total
    """
    single_geo = isinstance(geo_filter, str)
    if single_geo:
        geo_filter = [geo_filter]

    subset = df[df["geo"].isin(geo_filter)].copy()
    group_base = [] if single_geo else ["geo"]

    # Get total gross available per year
    total_ga = (
        subset[subset["siec"] == "Total"]
        .groupby(group_base + ["year", "nrg_bal"])["value"].sum()
        .unstack("nrg_bal")
        .reset_index()
    )
    if "Gross available energy" not in total_ga.columns:
        return pd.DataFrame()
    total_ga = total_ga[group_base + ["year", "Gross available energy"]].rename(
        columns={"Gross available energy": "total_ga"}
    )

    # Get net imports per source per year
    all_sources = DECOMPOSITION_SOURCES
    available_sources = [s for s in all_sources if s in subset["siec"].unique()]
    other_sources = [s for s in subset["siec"].unique()
                     if s not in all_sources and s != "Total"]

    rows = []
    for siec in available_sources + ["_other"]:
        if siec == "_other":
            src_data = subset[subset["siec"].isin(other_sources)]
            label = "Other"
        else:
            src_data = subset[subset["siec"] == siec]
            label = siec

        pivot = (
            src_data
            .groupby(group_base + ["year", "nrg_bal"])["value"].sum()
            .unstack("nrg_bal")
            .reset_index()
        )
        for col in ("Imports", "Exports"):
            if col not in pivot.columns:
                pivot[col] = 0
        pivot["net_imports"] = pivot["Imports"].fillna(0) - pivot["Exports"].fillna(0)
        # For nuclear heat, treat primary production as import-equivalent
        if siec == "Nuclear heat" and "Primary production" in pivot.columns:
            pivot["net_imports"] += pivot["Primary production"].fillna(0)
        pivot["siec"] = label
        rows.append(pivot[group_base + ["year", "siec", "net_imports"]])

    result = pd.concat(rows, ignore_index=True)
    result = result.merge(total_ga, on=group_base + ["year"])
    result["contribution"] = np.where(
        result["total_ga"] != 0,
        result["net_imports"] / result["total_ga"] * 100,
        np.nan,
    )
    return result


def plot_dependency_decomposition(df: pd.DataFrame) -> None:
    """
    Stacked area chart decomposing EU27 total dependency into source contributions.
    Each source's contribution = source net imports / total gross available energy.
    """
    decomp = _compute_decomposition(df, EU_GEO)
    if decomp.empty:
        print("  No data for decomposition – skipping.")
        return

    years = sorted(decomp["year"].unique())
    source_order = DECOMPOSITION_SOURCES + ["Other"]
    available = [s for s in source_order if s in decomp["siec"].unique()]

    stack_data = []
    labels = []
    colors = []
    for siec in available:
        series = decomp[decomp["siec"] == siec].set_index("year")["contribution"].reindex(years).fillna(0)
        stack_data.append(series.values)
        labels.append(DECOMPOSITION_LABELS.get(siec, siec))
        colors.append(DECOMPOSITION_COLORS.get(siec, "#bdc3c7"))

    fig, ax = plt.subplots(figsize=(12, 6))
    years_arr = np.array(years)

    # Separate positive and negative stacking so negatives go below 0
    pos_bottom = np.zeros(len(years))
    neg_bottom = np.zeros(len(years))
    for vals, lbl, clr in zip(stack_data, labels, colors):
        pos_part = np.maximum(vals, 0)
        neg_part = np.minimum(vals, 0)
        has_pos = np.any(pos_part != 0)
        has_neg = np.any(neg_part != 0)
        if has_pos:
            ax.fill_between(years_arr, pos_bottom, pos_bottom + pos_part,
                            color=clr, alpha=0.85, label=lbl)
            pos_bottom = pos_bottom + pos_part
        if has_neg:
            ax.fill_between(years_arr, neg_bottom + neg_part, neg_bottom,
                            color=clr, alpha=0.85, label=lbl if not has_pos else None)
            neg_bottom = neg_bottom + neg_part

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Contribution to total dependency (%)", fontsize=12)
    ax.set_title(
        "EU27 Energy Dependency Decomposition by Source\n"
        "(Source Net Imports / Total Gross Available Energy)",
        fontsize=14,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "4_eu_dependency_decomposition")
    fig.savefig(out_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    print(f"Saved: {out_path}.png / .svg")

    # Excel export: rows = years, columns = sources
    excel_rows = []
    for year in years:
        row = {"Year": year}
        for siec in available:
            val = decomp[(decomp["siec"] == siec) & (decomp["year"] == year)]["contribution"]
            row[DECOMPOSITION_LABELS.get(siec, siec)] = round(val.values[0], 4) if len(val) else 0
        excel_rows.append(row)
    excel_df = pd.DataFrame(excel_rows).set_index("Year")
    excel_path = os.path.join(OUTPUT_DIR, "4_eu_dependency_decomposition.xlsx")
    excel_df.to_excel(excel_path)
    print(f"Saved: {excel_path}")

    plt.close(fig)


def plot_dependency_decomposition_countries(df: pd.DataFrame) -> None:
    """
    3×3 stacked area chart: EU-27 + 8 EWBI cluster countries.
    Each subplot decomposes total dependency into source contributions.
    """
    from matplotlib.patches import Patch

    reporters = [("EU-27", EU_GEO)] + [
        (name, name) for name in COUNTRIES.values()
    ]

    source_order = DECOMPOSITION_SOURCES + ["Other"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharey=True)
    axes_flat = axes.flatten()

    for idx, (label, geo) in enumerate(reporters):
        ax = axes_flat[idx]
        decomp = _compute_decomposition(df, geo)
        if decomp.empty:
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        years = sorted(decomp["year"].unique())
        available = [s for s in source_order if s in decomp["siec"].unique()]

        stack_data = []
        colors = []
        for siec in available:
            series = (decomp[decomp["siec"] == siec]
                      .set_index("year")["contribution"]
                      .reindex(years).fillna(0))
            stack_data.append(series.values)
            colors.append(DECOMPOSITION_COLORS.get(siec, "#bdc3c7"))

        years_arr = np.array(years)
        pos_bottom = np.zeros(len(years))
        neg_bottom = np.zeros(len(years))
        for vals, clr in zip(stack_data, colors):
            pos_part = np.maximum(vals, 0)
            neg_part = np.minimum(vals, 0)
            if np.any(pos_part != 0):
                ax.fill_between(years_arr, pos_bottom, pos_bottom + pos_part,
                                color=clr, alpha=0.85)
                pos_bottom = pos_bottom + pos_part
            if np.any(neg_part != 0):
                ax.fill_between(years_arr, neg_bottom + neg_part, neg_bottom,
                                color=clr, alpha=0.85)
                neg_bottom = neg_bottom + neg_part

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks(years[::4])
        ax.set_xticklabels([str(y) for y in years[::4]], rotation=45, fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        if idx % 3 == 0:
            ax.set_ylabel("Contribution (%)", fontsize=9)

    # Hide unused subplots
    for idx in range(len(reporters), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared legend
    legend_handles = [
        Patch(facecolor=DECOMPOSITION_COLORS.get(s, "#bdc3c7"), alpha=0.85,
              label=DECOMPOSITION_LABELS.get(s, s))
        for s in source_order
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Energy Dependency Decomposition by Source\n"
        "(Source Net Imports / Total Gross Available Energy)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, "5_country_dependency_decomposition")
    fig.savefig(out_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    print(f"Saved: {out_path}.png / .svg")

    # Excel export: one sheet per reporter
    excel_path = os.path.join(OUTPUT_DIR, "5_country_dependency_decomposition.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for label, geo in reporters:
            decomp = _compute_decomposition(df, geo)
            if decomp.empty:
                continue
            years = sorted(decomp["year"].unique())
            available = [s for s in source_order if s in decomp["siec"].unique()]
            rows = []
            for year in years:
                row = {"Year": year}
                for siec in available:
                    val = decomp[(decomp["siec"] == siec) & (decomp["year"] == year)]["contribution"]
                    row[DECOMPOSITION_LABELS.get(siec, siec)] = round(val.values[0], 4) if len(val) else 0
                rows.append(row)
            pd.DataFrame(rows).set_index("Year").to_excel(writer, sheet_name=label[:31])
    print(f"Saved: {excel_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data …")
    df = load_data(DATA_FILE)
    print(f"  {len(df):,} rows loaded.")

    print("\n[1/5] EU27 dependency line chart …")
    plot_eu_dependency(df)

    print("\n[2/5] Country dependency line charts …")
    plot_country_dependency(df)

    print("\n[3/5] Country dependency heatmaps …")
    plot_dependency_heatmaps(df)

    print("\n[4/5] EU27 dependency decomposition …")
    plot_dependency_decomposition(df)

    print("\n[5/5] Country dependency decomposition (3×3) …")
    plot_dependency_decomposition_countries(df)

    print("\nDone. All figures saved to:", OUTPUT_DIR)
