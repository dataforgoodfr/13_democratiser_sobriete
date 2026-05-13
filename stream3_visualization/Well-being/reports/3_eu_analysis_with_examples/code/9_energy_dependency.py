"""
Energy Import Dependency – Report Variants
===========================================
Line charts of energy dependency ratios (net imports / gross available energy)
for selected countries, one subplot per energy source.

Additionally: Excel summary with dependency ratios, total net energy imports
(EUR), and net energy imports as % of GDP.

Data sources:
    ../external_data/eurostat_nrg_bal_s__custom_20746280_linear.csv  (Energy balance, TJ)
    ../external_data/eurostat_imports_ds-059331__custom_20825822_linear.csv  (Imports, EUR)
    ../external_data/eurostat_exports_ds-059331__custom_20825848_linear.csv  (Exports, EUR)
    ../external_data/eurostat_gdp_current_price.csv  (GDP, million EUR current prices)

Outputs: PNG, SVG, XLSX per report variant.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, "external_data")
OUTPUT_BASE = os.path.join(BASE_DIR, "outputs", "graphs", "Energy_Dependency")
os.makedirs(OUTPUT_BASE, exist_ok=True)

plt.rcParams["font.family"] = "Arial"

# ---------------------------------------------------------------------------
# Geo identifiers across datasets
# ---------------------------------------------------------------------------
EU_GEO_BALANCE = "European Union - 27 countries (from 2020)"
EU_GEO_GDP = "European Union - 27 countries (from 2020)"
# Trade data has a different verbose EU27 name — resolved via substring match

# ---------------------------------------------------------------------------
# Country / cluster configuration
# ---------------------------------------------------------------------------
# Base 8 example countries (used for rep_fr, rep_ch)
BASE_COUNTRIES = {
    "FR": "France",
    "ES": "Spain",
    "BE": "Belgium",
    "NL": "Netherlands",
    "LT": "Lithuania",
    "HU": "Hungary",
    "DE": "Germany",
    "PL": "Poland",
}

# France report: neighbouring countries
FR_COUNTRIES = {
    "FR": "France",
    "ES": "Spain",
    "IT": "Italy",
    "DE": "Germany",
    "LU": "Luxembourg",
    "BE": "Belgium",
}

# EU / EWBI variant: NL replaced by DK
EU_COUNTRIES = {
    "FR": "France",
    "ES": "Spain",
    "DK": "Denmark",
    "BE": "Belgium",
    "LT": "Lithuania",
    "HU": "Hungary",
    "DE": "Germany",
    "PL": "Poland",
}

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
    "Denmark":     CLUSTER_COLORS[1],  # replaces NL in cluster 1
    "Lithuania":   CLUSTER_COLORS[2],
    "Hungary":     CLUSTER_COLORS[2],
    "Germany":     CLUSTER_COLORS[3],
    "Poland":      CLUSTER_COLORS[3],
    "Italy":       "#e41a1c",
    "Luxembourg":  "#984ea3",
}

COUNTRY_MARKERS = {
    "France":      "o",
    "Spain":       "s",
    "Belgium":     "o",
    "Netherlands": "s",
    "Denmark":     "s",
    "Lithuania":   "o",
    "Hungary":     "s",
    "Germany":     "o",
    "Poland":      "s",
    "Italy":       "D",
    "Luxembourg":  "^",
}

EU27_COLOR = "#2c3e50"

# All EU-27 + EFTA countries available in energy balance data
ALL_EU_EFTA = {
    "AT": "Austria",   "BE": "Belgium",  "BG": "Bulgaria",
    "HR": "Croatia",   "CY": "Cyprus",   "CZ": "Czechia",
    "DK": "Denmark",   "EE": "Estonia",  "FI": "Finland",
    "FR": "France",    "DE": "Germany",  "GR": "Greece",
    "HU": "Hungary",   "IE": "Ireland",  "IT": "Italy",
    "LV": "Latvia",    "LT": "Lithuania","LU": "Luxembourg",
    "MT": "Malta",     "NL": "Netherlands", "PL": "Poland",
    "PT": "Portugal",  "RO": "Romania",  "SK": "Slovakia",
    "SI": "Slovenia",  "ES": "Spain",    "SE": "Sweden",
    # EFTA (CH and LI not in energy balance data)
    "NO": "Norway",    "IS": "Iceland",
}

# ---------------------------------------------------------------------------
# Energy sources
# ---------------------------------------------------------------------------
ENERGY_SOURCES = {
    "Solid fossil fuels": "Solid Fossil Fuels",
    "Natural gas": "Natural Gas",
    "Oil and petroleum products (excluding biofuel portion)": "Oil & Petroleum Products",
    "Total": "Total Energy",
}

SOURCE_COLORS = {
    "Solid fossil fuels": "#5a5a5a",
    "Natural gas":        "#e67e22",
    "Oil and petroleum products (excluding biofuel portion)": "#2980b9",
    "Total":              "#2c3e50",
}

# ---------------------------------------------------------------------------
# Report configs
# ---------------------------------------------------------------------------
REPORT_CONFIGS = {
    "rep_eu": {
        "prefix": "rep_eu",
        "title_suffix": "(EU-27)",
        "countries": EU_COUNTRIES,
        "show_eu27": True,
    },
    "rep_ewbi": {
        "prefix": "rep_ewbi",
        "title_suffix": "(EU-27 + EFTA)",
        "countries": EU_COUNTRIES,
        "show_eu27": True,
    },
    "rep_fr": {
        "prefix": "rep_fr",
        "title_suffix": "(France)",
        "countries": FR_COUNTRIES,
        "show_eu27": True,
    },
    "rep_ch": {
        "prefix": "rep_ch",
        "title_suffix": "(Switzerland)",
        "countries": BASE_COUNTRIES,
        "show_eu27": True,
    },
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_energy_balance() -> pd.DataFrame:
    """Load Eurostat NRG_BAL_S (values in Terajoule)."""
    path = os.path.join(EXTERNAL_DATA_DIR,
                        "eurostat_nrg_bal_s__custom_20746280_linear.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "value"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "value"])
    df["year"] = df["year"].astype(int)
    return df


def load_trade_data() -> pd.DataFrame:
    """
    Load imports & exports trade data (EUR), aggregate across partners.
    Returns a DataFrame with columns: reporter, year, product, imports_eur,
    exports_eur, net_imports_eur.
    """
    imp_path = os.path.join(
        EXTERNAL_DATA_DIR,
        "eurostat_imports_ds-059331__custom_20825822_linear.csv",
    )
    exp_path = os.path.join(
        EXTERNAL_DATA_DIR,
        "eurostat_exports_ds-059331__custom_20825848_linear.csv",
    )

    imp = pd.read_csv(imp_path)
    exp = pd.read_csv(exp_path)
    imp["OBS_VALUE"] = pd.to_numeric(imp["OBS_VALUE"], errors="coerce")
    exp["OBS_VALUE"] = pd.to_numeric(exp["OBS_VALUE"], errors="coerce")

    # Sum across partners per reporter/year/product
    imp_agg = (
        imp.groupby(["reporter", "TIME_PERIOD", "product"])["OBS_VALUE"]
        .sum().reset_index()
    )
    imp_agg.columns = ["reporter", "year", "product", "imports_eur"]

    exp_agg = (
        exp.groupby(["reporter", "TIME_PERIOD", "product"])["OBS_VALUE"]
        .sum().reset_index()
    )
    exp_agg.columns = ["reporter", "year", "product", "exports_eur"]

    trade = imp_agg.merge(exp_agg, on=["reporter", "year", "product"], how="outer")
    trade["imports_eur"] = trade["imports_eur"].fillna(0)
    trade["exports_eur"] = trade["exports_eur"].fillna(0)
    trade["net_imports_eur"] = trade["imports_eur"] - trade["exports_eur"]
    return trade


def load_gdp_eur() -> pd.DataFrame:
    """Load GDP at current prices in million EUR."""
    path = os.path.join(EXTERNAL_DATA_DIR, "eurostat_gdp_current_price.csv")
    df = pd.read_csv(path)
    df = df[["geo", "TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["geo", "year", "gdp_meur"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["gdp_meur"] = pd.to_numeric(df["gdp_meur"], errors="coerce")
    df = df.dropna(subset=["year", "gdp_meur"])
    df["year"] = df["year"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Dependency computation (from energy balance, TJ)
# ---------------------------------------------------------------------------

def compute_dependency(df: pd.DataFrame, geo_names: list) -> pd.DataFrame:
    """
    Compute energy dependency ratios for given geo names.
    dependency = net_imports / gross_available
    """
    subset = df[df["geo"].isin(geo_names)].copy()
    pivot = (
        subset
        .groupby(["geo", "year", "siec", "nrg_bal"])["value"]
        .sum()
        .unstack("nrg_bal")
        .reset_index()
    )

    col_map = {
        "Imports": "imports",
        "Exports": "exports",
        "Gross available energy": "gross_available",
    }
    pivot = pivot.rename(columns={k: v for k, v in col_map.items()
                                  if k in pivot.columns})

    for col in ("imports", "exports", "gross_available"):
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot["net_imports"] = pivot["imports"].fillna(0) - pivot["exports"].fillna(0)

    # --- Nuclear correction: treat nuclear heat primary production as imports ---
    # Uranium fuel is imported, so domestic nuclear PP overstates energy independence
    nuclear_pp = (
        subset[(subset["siec"] == "Nuclear heat") &
               (subset["nrg_bal"] == "Primary production")]
        .groupby(["geo", "year"])["value"].sum()
        .reset_index()
        .rename(columns={"value": "nuclear_pp"})
    )
    if not nuclear_pp.empty:
        pivot = pivot.merge(nuclear_pp, on=["geo", "year"], how="left")
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

    pivot = pivot[pivot["siec"].isin(ENERGY_SOURCES.keys())]
    return pivot.sort_values(["siec", "year"]).reset_index(drop=True)


def compute_eu27_dependency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EU-27 aggregate dependency."""
    subset = df[df["geo"] == EU_GEO_BALANCE].copy()
    pivot = (
        subset
        .groupby(["year", "siec", "nrg_bal"])["value"]
        .sum()
        .unstack("nrg_bal")
        .reset_index()
    )

    col_map = {
        "Imports": "imports",
        "Exports": "exports",
        "Gross available energy": "gross_available",
    }
    pivot = pivot.rename(columns={k: v for k, v in col_map.items()
                                  if k in pivot.columns})
    for col in ("imports", "exports", "gross_available"):
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot["net_imports"] = pivot["imports"].fillna(0) - pivot["exports"].fillna(0)

    # --- Nuclear correction: treat nuclear heat primary production as imports ---
    nuclear_pp = (
        subset[(subset["siec"] == "Nuclear heat") &
               (subset["nrg_bal"] == "Primary production")]
        .groupby(["year"])["value"].sum()
        .reset_index()
        .rename(columns={"value": "nuclear_pp"})
    )
    if not nuclear_pp.empty:
        pivot = pivot.merge(nuclear_pp, on=["year"], how="left")
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
    pivot["geo"] = "EU-27"
    pivot = pivot[pivot["siec"].isin(ENERGY_SOURCES.keys())]
    return pivot.sort_values(["siec", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trade helpers (EUR)
# ---------------------------------------------------------------------------

def _resolve_trade_reporter(reporters, country_name: str):
    """Find trade-data reporter matching a country name (substring)."""
    matches = [r for r in reporters if country_name.lower() in r.lower()]
    return matches[0] if matches else None


def _total_net_imports_eur(trade_df: pd.DataFrame, reporter: str,
                           year: int) -> float:
    """Sum net imports across all products for a given reporter and year."""
    t = trade_df[(trade_df["reporter"] == reporter) &
                 (trade_df["year"] == year)]
    return t["net_imports_eur"].sum() if not t.empty else np.nan


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_country_dependency(dep_df, eu27_dep, cfg, out_dir):
    """4-panel line chart: one subplot per energy source, one line per country."""
    countries = cfg["countries"]
    prefix = cfg["prefix"]
    title_suffix = cfg["title_suffix"]

    sources = list(ENERGY_SOURCES.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes_flat = axes.flatten()

    for ax, siec in zip(axes_flat, sources):
        source_data = dep_df[dep_df["siec"] == siec]

        for _iso2, country_name in countries.items():
            cdata = source_data[source_data["geo"] == country_name].dropna(
                subset=["dependency"]
            )
            if cdata.empty:
                continue
            ax.plot(
                cdata["year"], cdata["dependency"] * 100,
                marker=COUNTRY_MARKERS.get(country_name, "o"),
                linewidth=2, markersize=5, markevery=3,
                label=country_name, color=COUNTRY_COLORS[country_name],
            )

        # EU-27 reference line
        if cfg["show_eu27"]:
            eu_src = eu27_dep[eu27_dep["siec"] == siec].dropna(
                subset=["dependency"]
            )
            if not eu_src.empty:
                ax.plot(
                    eu_src["year"], eu_src["dependency"] * 100,
                    linewidth=2.5, linestyle="--", color=EU27_COLOR,
                    label="EU-27", zorder=0,
                )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(ENERGY_SOURCES[siec], fontsize=12)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Dependency ratio (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0f}%")
        )

    # Shared legend below
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 9), fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Energy Import Dependency by Country {title_suffix}\n"
        "(Net Imports / Gross Available Energy)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    out_path = os.path.join(out_dir, f"{prefix}_energy_dependency")
    fig.savefig(out_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    print(f"  Saved: {prefix}_energy_dependency.png / .svg")

    # Excel time series: one sheet per energy source
    excel_path = os.path.join(out_dir,
                              f"{prefix}_energy_dependency_timeseries.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for siec, label in ENERGY_SOURCES.items():
            src_data = dep_df[dep_df["siec"] == siec]
            sheet_rows = {}
            for _iso2, cname in countries.items():
                series = (
                    src_data[src_data["geo"] == cname]
                    .set_index("year")["dependency"] * 100
                )
                sheet_rows[cname] = series
            if cfg["show_eu27"]:
                eu_series = (
                    eu27_dep[eu27_dep["siec"] == siec]
                    .set_index("year")["dependency"] * 100
                )
                sheet_rows["EU-27"] = eu_series
            sheet_df = pd.DataFrame(sheet_rows)
            sheet_df.index.name = "Year"
            sheet_df.to_excel(writer, sheet_name=label[:31])
    print(f"  Saved: {prefix}_energy_dependency_timeseries.xlsx")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary Excel
# ---------------------------------------------------------------------------

def export_summary_excel(dep_all, eu27_dep, trade_df, gdp_df, cfg, out_dir):
    """
    Excel summary table for ALL EU-27 + EFTA countries.
    Rows   = countries (all EU+EFTA)
    Columns = dependency ratio per energy source (%), total net energy
              imports (EUR), net energy imports as % of GDP.
    Uses the latest year with energy-balance data.
    """
    prefix = cfg["prefix"]

    latest_year = int(dep_all["year"].max())
    reporters = trade_df["reporter"].unique()

    rows = []
    for _iso2, country_name in ALL_EU_EFTA.items():
        row = {"Country": country_name}

        # Dependency ratios
        for siec, label in ENERGY_SOURCES.items():
            val = dep_all[
                (dep_all["geo"] == country_name)
                & (dep_all["siec"] == siec)
                & (dep_all["year"] == latest_year)
            ]["dependency"]
            row[f"Dependency: {label} (%)"] = (
                round(val.values[0] * 100, 2) if len(val) else np.nan
            )

        # Net energy imports in EUR
        reporter = _resolve_trade_reporter(reporters, country_name)
        net_eur = (
            _total_net_imports_eur(trade_df, reporter, latest_year)
            if reporter else np.nan
        )
        row["Net Energy Imports (EUR)"] = net_eur

        # GDP lookup
        gdp_row = gdp_df[
            (gdp_df["geo"] == country_name) & (gdp_df["year"] == latest_year)
        ]
        gdp_eur = gdp_row["gdp_meur"].values[0] * 1e6 if not gdp_row.empty else np.nan
        row["Net Energy Imports (% GDP)"] = (
            round(net_eur / gdp_eur * 100, 2)
            if not np.isnan(net_eur) and not np.isnan(gdp_eur) and gdp_eur != 0
            else np.nan
        )
        rows.append(row)

    # EU-27 row
    if cfg["show_eu27"]:
        eu_row = {"Country": "EU-27"}
        for siec, label in ENERGY_SOURCES.items():
            val = eu27_dep[
                (eu27_dep["siec"] == siec) & (eu27_dep["year"] == latest_year)
            ]["dependency"]
            eu_row[f"Dependency: {label} (%)"] = (
                round(val.values[0] * 100, 2) if len(val) else np.nan
            )

        eu_reporter = _resolve_trade_reporter(reporters, "European Union")
        eu_net = (
            _total_net_imports_eur(trade_df, eu_reporter, latest_year)
            if eu_reporter else np.nan
        )
        eu_row["Net Energy Imports (EUR)"] = eu_net

        eu_gdp_row = gdp_df[
            (gdp_df["geo"] == EU_GEO_GDP) & (gdp_df["year"] == latest_year)
        ]
        eu_gdp = (
            eu_gdp_row["gdp_meur"].values[0] * 1e6
            if not eu_gdp_row.empty else np.nan
        )
        eu_row["Net Energy Imports (% GDP)"] = (
            round(eu_net / eu_gdp * 100, 2)
            if not np.isnan(eu_net) and not np.isnan(eu_gdp) and eu_gdp != 0
            else np.nan
        )
        rows.append(eu_row)

    summary_df = pd.DataFrame(rows).set_index("Country")
    excel_path = os.path.join(out_dir,
                              f"{prefix}_energy_dependency_summary.xlsx")
    summary_df.to_excel(excel_path)
    print(f"  Saved: {prefix}_energy_dependency_summary.xlsx  (year={latest_year})")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(report_key, df_bal, trade_df, gdp_df):
    cfg = REPORT_CONFIGS[report_key]
    prefix = cfg["prefix"]

    out_dir = os.path.join(OUTPUT_BASE, prefix)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Generating energy dependency report: {prefix} {cfg['title_suffix']}")
    print(f"{'=' * 60}")

    # Country-level dependency (energy balance) — for chart
    country_names = list(cfg["countries"].values())
    dep_df = compute_dependency(df_bal, country_names)

    # All EU+EFTA dependency — for summary Excel
    all_names = list(ALL_EU_EFTA.values())
    dep_all = compute_dependency(df_bal, all_names)

    # EU-27 dependency
    eu27_dep = compute_eu27_dependency(df_bal)

    # Plot
    plot_country_dependency(dep_df, eu27_dep, cfg, out_dir)

    # Summary Excel (all EU+EFTA countries)
    export_summary_excel(dep_all, eu27_dep, trade_df, gdp_df, cfg, out_dir)

    print(f"  All outputs saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading energy balance data...")
    df_bal = load_energy_balance()
    print(f"  {len(df_bal):,} rows")

    print("Loading trade data (imports + exports in EUR)...")
    trade_df = load_trade_data()
    print(f"  {len(trade_df):,} rows")

    print("Loading GDP data (current prices, million EUR)...")
    gdp_df = load_gdp_eur()
    print(f"  {len(gdp_df):,} rows")

    for rk in REPORT_CONFIGS:
        generate_report(rk, df_bal, trade_df, gdp_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
