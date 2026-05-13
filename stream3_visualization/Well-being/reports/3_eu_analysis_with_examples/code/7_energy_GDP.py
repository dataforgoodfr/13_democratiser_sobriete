"""
Energy Intensity of GDP (Energy / GDP) over time
=================================================
Line chart of energy intensity (Final Energy Consumption / GDP PPP)
for the 8 EWBI cluster example countries + EU-27 average.

Data sources:
    ../external_data/eurostat_final_energy.csv   (Final consumption, GWh)
    ../external_data/worldbank_gdp_ppp_$2021.csv (GDP PPP constant 2021 int$, converted to €)

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
OUTPUT_BASE = os.path.join(BASE_DIR, "outputs", "graphs", "Energy_GDP")
os.makedirs(OUTPUT_BASE, exist_ok=True)

plt.rcParams["font.family"] = "Arial"

# Exchange rate: 1 USD = 1.1827 USD/EUR → multiply $ values by 1/1.1827
USD_TO_EUR_2021 = 1.0 / 1.1827

# ---------------------------------------------------------------------------
# Country mappings
# ---------------------------------------------------------------------------
EUROSTAT_TO_CODE = {
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR",
    "Cyprus": "CY", "Czechia": "CZ", "Denmark": "DK", "Estonia": "EE",
    "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "GR",
    "Hungary": "HU", "Ireland": "IE", "Italy": "IT", "Latvia": "LV",
    "Lithuania": "LT", "Luxembourg": "LU", "Malta": "MT", "Netherlands": "NL",
    "Poland": "PL", "Portugal": "PT", "Romania": "RO", "Slovakia": "SK",
    "Slovenia": "SI", "Spain": "ES", "Sweden": "SE", "Norway": "NO",
    "Switzerland": "CH", "Iceland": "IS", "Liechtenstein": "LI",
    "United Kingdom": "UK",
}

STUDY_ISO2_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "HR": "HRV", "CY": "CYP",
    "CZ": "CZE", "DK": "DNK", "EE": "EST", "FI": "FIN", "FR": "FRA",
    "DE": "DEU", "GR": "GRC", "HU": "HUN", "IE": "IRL", "IT": "ITA",
    "LV": "LVA", "LT": "LTU", "LU": "LUX", "MT": "MLT", "NL": "NLD",
    "PL": "POL", "PT": "PRT", "RO": "ROU", "SK": "SVK", "SI": "SVN",
    "ES": "ESP", "SE": "SWE", "UK": "GBR", "NO": "NOR", "CH": "CHE",
    "IS": "ISL", "LI": "LIE",
}
STUDY_ISO3_TO_ISO2 = {v: k for k, v in STUDY_ISO2_TO_ISO3.items()}

EU27_CODES = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI",
    "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU",
    "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
]

# ---------------------------------------------------------------------------
# Example countries & clusters (same as energy_dependency_analysis.py)
# ---------------------------------------------------------------------------
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

EU27_COLOR = "#2c3e50"

# ---------------------------------------------------------------------------
# Report configs
# ---------------------------------------------------------------------------
REPORT_CONFIGS = {
    "rep_eu": {
        "prefix": "rep_eu",
        "title_suffix": "(EU-27)",
        "comparison_countries": list(COUNTRIES.values()),
        "show_eu27": True,
    },
    "rep_ewbi": {
        "prefix": "rep_ewbi",
        "title_suffix": "(EU-27 + EFTA)",
        "comparison_countries": list(COUNTRIES.values()),
        "show_eu27": True,
    },
    "rep_fr": {
        "prefix": "rep_fr",
        "title_suffix": "(France)",
        "comparison_countries": list(COUNTRIES.values()),
        "show_eu27": True,
    },
    "rep_ch": {
        "prefix": "rep_ch",
        "title_suffix": "(Switzerland)",
        "comparison_countries": list(COUNTRIES.values()),
        "show_eu27": True,
    },
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_energy() -> pd.DataFrame:
    """Load Eurostat final energy consumption (GWh)."""
    path = os.path.join(EXTERNAL_DATA_DIR, "eurostat_final_energy.csv")
    df = pd.read_csv(path)
    df = df[["geo", "TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["geo_name", "year", "energy_gwh"]
    df["geo_code"] = df["geo_name"].map(EUROSTAT_TO_CODE)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["energy_gwh"] = pd.to_numeric(df["energy_gwh"], errors="coerce")
    df = df.dropna(subset=["geo_code", "year", "energy_gwh"])
    df["year"] = df["year"].astype(int)
    return df[["geo_code", "year", "energy_gwh"]]


def load_gdp_ppp() -> pd.DataFrame:
    """Load World Bank GDP PPP (constant 2021 international $)."""
    path = os.path.join(EXTERNAL_DATA_DIR, "worldbank_gdp_ppp_$2021.csv")
    df_raw = pd.read_csv(path, skiprows=4)
    df_raw = df_raw[df_raw["Indicator Code"] == "NY.GDP.MKTP.PP.KD"].copy()
    year_cols = [c for c in df_raw.columns if c.isdigit()]
    df = df_raw.melt(
        id_vars=["Country Code"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp_ppp",
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["gdp_ppp"] = pd.to_numeric(df["gdp_ppp"], errors="coerce")
    df["geo_code"] = df["Country Code"].map(STUDY_ISO3_TO_ISO2)
    df = df.dropna(subset=["geo_code", "year", "gdp_ppp"])
    df["year"] = df["year"].astype(int)
    # Convert from constant 2021 int$ to constant 2021 €
    df["gdp_ppp"] = df["gdp_ppp"] * USD_TO_EUR_2021
    return df[["geo_code", "year", "gdp_ppp"]]


def compute_energy_intensity(df_energy, df_gdp, countries_iso2):
    """
    Merge energy and GDP, compute intensity = energy_gwh / gdp_ppp.
    Returns a DataFrame with columns: geo_code, year, energy_gwh, gdp_ppp, intensity.
    """
    subset_e = df_energy[df_energy["geo_code"].isin(countries_iso2)]
    subset_g = df_gdp[df_gdp["geo_code"].isin(countries_iso2)]
    merged = subset_e.merge(subset_g, on=["geo_code", "year"], how="inner")
    merged = merged[(merged["energy_gwh"] > 0) & (merged["gdp_ppp"] > 0)].copy()
    # Intensity in Wh per € PPP (GWh / billion € = Wh / €)
    merged["intensity"] = merged["energy_gwh"] / (merged["gdp_ppp"] / 1e9)
    return merged.sort_values(["geo_code", "year"]).reset_index(drop=True)


def compute_eu27_intensity(df_energy, df_gdp):
    """Compute EU-27 aggregate energy intensity (sum energy / sum GDP)."""
    e27 = df_energy[df_energy["geo_code"].isin(EU27_CODES)].groupby("year")["energy_gwh"].sum()
    g27 = df_gdp[df_gdp["geo_code"].isin(EU27_CODES)].groupby("year")["gdp_ppp"].sum()
    merged = pd.DataFrame({"energy_gwh": e27, "gdp_ppp": g27}).dropna()
    merged = merged[(merged["energy_gwh"] > 0) & (merged["gdp_ppp"] > 0)]
    merged["intensity"] = merged["energy_gwh"] / (merged["gdp_ppp"] / 1e9)
    merged = merged.reset_index()
    merged["geo_code"] = "EU-27"
    return merged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_energy_intensity(df_countries, df_eu27, cfg, out_dir):
    """
    Line chart: energy intensity over time for example countries + EU-27.
    """
    countries = cfg["comparison_countries"]
    prefix = cfg["prefix"]
    title_suffix = cfg["title_suffix"]

    fig, ax = plt.subplots(figsize=(14, 7))

    # --- Country lines ---
    for country_name in countries:
        iso2 = [k for k, v in COUNTRIES.items() if v == country_name][0]
        cdata = df_countries[df_countries["geo_code"] == iso2].sort_values("year")
        if cdata.empty:
            continue
        ax.plot(
            cdata["year"], cdata["intensity"],
            marker=COUNTRY_MARKERS.get(country_name, "o"),
            linewidth=2, markersize=5, markevery=3,
            label=country_name, color=COUNTRY_COLORS[country_name],
        )

    # --- EU-27 line ---
    if cfg["show_eu27"] and not df_eu27.empty:
        ax.plot(
            df_eu27["year"], df_eu27["intensity"],
            linewidth=2.5, linestyle="--", color=EU27_COLOR,
            label="EU-27", zorder=0,
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Energy Intensity (Wh per € PPP 2021)", fontsize=12)
    ax.set_title(
        f"Energy Intensity of GDP {title_suffix}\n"
        "(Final Energy Consumption / GDP PPP, constant 2021 €)",
        fontsize=14,
    )
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save
    out_path = os.path.join(out_dir, f"{prefix}_energy_intensity_gdp")
    fig.savefig(out_path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_path + ".svg", bbox_inches="tight")
    print(f"  Saved: {prefix}_energy_intensity_gdp.png / .svg")

    # Excel
    excel_rows = {}
    for country_name in countries:
        iso2 = [k for k, v in COUNTRIES.items() if v == country_name][0]
        series = (
            df_countries[df_countries["geo_code"] == iso2]
            .set_index("year")["intensity"]
        )
        excel_rows[country_name] = series
    if cfg["show_eu27"] and not df_eu27.empty:
        excel_rows["EU-27"] = df_eu27.set_index("year")["intensity"]
    excel_df = pd.DataFrame(excel_rows)
    excel_df.index.name = "Year"
    excel_path = os.path.join(out_dir, f"{prefix}_energy_intensity_gdp.xlsx")
    excel_df.to_excel(excel_path)
    print(f"  Saved: {prefix}_energy_intensity_gdp.xlsx")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(report_key, df_energy, df_gdp):
    cfg = REPORT_CONFIGS[report_key]
    prefix = cfg["prefix"]
    title_suffix = cfg["title_suffix"]

    out_dir = os.path.join(OUTPUT_BASE, prefix)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Generating energy intensity report: {prefix} {title_suffix}")
    print(f"{'=' * 60}")

    # Country-level data
    country_iso2 = [k for k, v in COUNTRIES.items() if v in cfg["comparison_countries"]]
    df_countries = compute_energy_intensity(df_energy, df_gdp, country_iso2)

    # EU-27 aggregate
    df_eu27 = compute_eu27_intensity(df_energy, df_gdp)

    plot_energy_intensity(df_countries, df_eu27, cfg, out_dir)
    print(f"  All outputs saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading energy data...")
    df_energy = load_energy()
    print(f"  {len(df_energy)} rows")

    print("Loading World Bank GDP PPP data...")
    df_gdp = load_gdp_ppp()
    print(f"  {len(df_gdp)} rows")

    for rk in REPORT_CONFIGS:
        generate_report(rk, df_energy, df_gdp)

    print("\nDone.")


if __name__ == "__main__":
    main()
