"""
Supply Concentration – CR3 & CR5 over time
==========================================
For each energy product (Coal, Petroleum, Gas), compute the share of
extra-EU imports covered by the top 3 and top 5 suppliers over time.

Two graphs per report:
  1. CR3 — concentration ratio top 3 suppliers (one line per country)
  2. CR5 — concentration ratio top 5 suppliers (one line per country)

Data sources:
    EU-27 level:  ../external_data/eurostat_trade_all_year.csv (QUANTITY_KG)
    Per-country:  ../external_data/eurostat_imports_ds-059331__custom_20825822_linear.csv (VALUE_EUR)

Since per-country data only provides VALUE_EUR (no volume), we use EUR
for all reporters to keep the metric consistent.

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
OUTPUT_BASE = os.path.join(BASE_DIR, "outputs", "graphs", "Supply_Concentration")
os.makedirs(OUTPUT_BASE, exist_ok=True)

plt.rcParams["font.family"] = "Arial"

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
# All-year file: EU-27 & France, QUANTITY_KG + VALUE_EUR, 2002-2025
ALL_YEAR_PATH = os.path.join(EXTERNAL_DATA_DIR, "eurostat_trade_all_year.csv")
# Per-country imports: all EU member states, VALUE_EUR only, 2002-2025
PER_COUNTRY_PATH = os.path.join(
    EXTERNAL_DATA_DIR,
    "eurostat_imports_ds-059331__custom_20825822_linear.csv",
)

# ---------------------------------------------------------------------------
# Energy products
# ---------------------------------------------------------------------------
PRODUCTS = [
    "Coal, coke and briquettes",
    "Petroleum, petroleum products and related materials",
    "Gas, natural and manufactured",
]

PRODUCT_SHORT = {
    "Coal, coke and briquettes": "Coal & Coke",
    "Petroleum, petroleum products and related materials": "Petroleum",
    "Gas, natural and manufactured": "Gas",
}

# ---------------------------------------------------------------------------
# Country / cluster configuration
# ---------------------------------------------------------------------------
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
    "Denmark":     CLUSTER_COLORS[1],
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
    "Denmark":     "s",
    "Lithuania":   "o",
    "Hungary":     "s",
    "Germany":     "o",
    "Poland":      "s",
}

EU27_COLOR = "#2c3e50"

# Partners to exclude (aggregates, not individual countries)
AGGREGATE_PARTNERS = {
    "All countries of the world",
    "Extra-EU27 (from 2020)",
    "Extra-EU",
    "Extra-euro area",
    "Extra-euro area - 21 countries (from 2026)",
    "Intra-EU27 (from 2020)",
    "Intra-EU",
    "Intra-euro area",
    "Intra-euro area - 21 countries (from 2026)",
    "Countries and territories not specified",
    "Countries and territories not specified for commercial or military reasons in the framework of extra-Union trade",
    "Countries and territories not specified for commercial or military reasons in the framework of intra-Union trade",
    "Countries and territories not specified within the framework of extra-Union trade",
    "Countries and territories not specified within the framework of intra-Union trade",
    "Stores and provisions",
    "Stores and provisions within the framework of extra-Union trade",
    "Stores and provisions within the framework of intra-Union trade",
    "High seas",
    "United States Minor Outlying Islands",
}

# EU-27 country name fragments (for detecting intra-EU trade)
EU27_PARTNER_FRAGMENTS = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
    "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia",
    "Slovenia", "Spain", "Sweden",
]

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
        "countries": BASE_COUNTRIES,
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
# Helpers
# ---------------------------------------------------------------------------

def _is_eu27_partner(partner_name: str) -> bool:
    for frag in EU27_PARTNER_FRAGMENTS:
        if frag in partner_name:
            return True
    return False


def _is_individual_partner(partner: str) -> bool:
    return partner not in AGGREGATE_PARTNERS and not _is_eu27_partner(partner)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_year() -> pd.DataFrame:
    """Load the all-year trade dataset (EU-27 + France reporters)."""
    df = pd.read_csv(ALL_YEAR_PATH)
    df = df.rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


def load_per_country() -> pd.DataFrame:
    """Load per-country imports dataset (VALUE_EUR only)."""
    df = pd.read_csv(PER_COUNTRY_PATH)
    df = df.rename(columns={"TIME_PERIOD": "year", "OBS_VALUE": "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


# ---------------------------------------------------------------------------
# Concentration ratio computation
# ---------------------------------------------------------------------------

def compute_cr(df_reporter: pd.DataFrame, product: str, n_top: int,
               indicator: str = "VALUE_EUR") -> pd.DataFrame:
    """
    Compute the CR_n concentration ratio over time for a single reporter
    and product.

    CR_n = sum of top-n partners / total (extra-EU individual partners).

    Returns DataFrame with columns: year, cr_pct.
    """
    sub = df_reporter[
        (df_reporter["product"] == product)
        & (df_reporter["indicators"] == indicator)
        & (df_reporter["flow"] == "IMPORT")
        & (df_reporter["partner"].apply(_is_individual_partner))
    ].copy()

    rows = []
    for year in sorted(sub["year"].unique()):
        yr_data = sub[sub["year"] == year]
        total = yr_data["value"].sum()
        if total <= 0:
            continue
        top_n_sum = yr_data.nlargest(n_top, "value")["value"].sum()
        rows.append({"year": year, "cr_pct": top_n_sum / total * 100})
    return pd.DataFrame(rows)


def compute_cr_per_country_simple(df: pd.DataFrame, product: str,
                                  n_top: int) -> pd.DataFrame:
    """
    For per-country imports data (VALUE_EUR only, no 'indicators' column
    to filter — all rows are VALUE_EUR IMPORT).
    """
    sub = df[
        (df["product"] == product)
        & (df["partner"].apply(_is_individual_partner))
    ].copy()

    rows = []
    for year in sorted(sub["year"].unique()):
        yr_data = sub[sub["year"] == year]
        total = yr_data["value"].sum()
        if total <= 0:
            continue
        top_n_sum = yr_data.nlargest(n_top, "value")["value"].sum()
        rows.append({"year": year, "cr_pct": top_n_sum / total * 100})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cr(cr_data: dict, eu27_cr: pd.DataFrame, n_top: int,
            product: str, cfg: dict, out_dir: str):
    """
    Line chart of CR_n over time for example countries + EU-27.
    cr_data: {country_name: DataFrame(year, cr_pct)}
    """
    countries = cfg["countries"]
    prefix = cfg["prefix"]
    title_suffix = cfg["title_suffix"]
    short_prod = PRODUCT_SHORT.get(product, product)

    fig, ax = plt.subplots(figsize=(14, 7))

    for _iso2, country_name in countries.items():
        if country_name not in cr_data or cr_data[country_name].empty:
            continue
        cdata = cr_data[country_name].sort_values("year")
        ax.plot(
            cdata["year"], cdata["cr_pct"],
            marker=COUNTRY_MARKERS.get(country_name, "o"),
            linewidth=2, markersize=5, markevery=2,
            label=country_name, color=COUNTRY_COLORS[country_name],
        )

    if cfg["show_eu27"] and not eu27_cr.empty:
        ax.plot(
            eu27_cr["year"], eu27_cr["cr_pct"],
            linewidth=2.5, linestyle="--", color=EU27_COLOR,
            label="EU-27", zorder=0,
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(f"CR{n_top} — Share of imports (%)", fontsize=12)
    ax.set_title(
        f"{short_prod}: Top-{n_top} Supplier Concentration {title_suffix}\n"
        f"(Share of extra-EU imports covered by {n_top} largest suppliers, EUR)",
        fontsize=14,
    )
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%")
    )
    fig.tight_layout()

    safe_prod = short_prod.replace(" ", "_").replace("&", "and")
    fname = f"{prefix}_CR{n_top}_{safe_prod}"
    fig.savefig(os.path.join(out_dir, fname + ".png"), dpi=150,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, fname + ".svg"), bbox_inches="tight")
    print(f"  Saved: {fname}.png / .svg")

    plt.close(fig)
    return fname


def plot_cr_combined(cr_data_all: dict, eu_cr_all: dict,
                     n_top: int, cfg: dict, out_dir: str):
    """
    Combined figure for a single CR level: 3 products × 1 column = 3 subplots.
    """
    countries = cfg["countries"]
    prefix = cfg["prefix"]
    title_suffix = cfg["title_suffix"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharey=True)

    for row_idx, product in enumerate(PRODUCTS):
        short = PRODUCT_SHORT.get(product, product)
        ax = axes[row_idx]
        cr_data = cr_data_all[product]
        eu_cr = eu_cr_all[product]

        for _iso2, country_name in countries.items():
            if country_name not in cr_data or cr_data[country_name].empty:
                continue
            cdata = cr_data[country_name].sort_values("year")
            ax.plot(
                cdata["year"], cdata["cr_pct"],
                marker=COUNTRY_MARKERS.get(country_name, "o"),
                linewidth=2, markersize=4, markevery=3,
                label=country_name, color=COUNTRY_COLORS[country_name],
            )

        if cfg["show_eu27"] and not eu_cr.empty:
            ax.plot(
                eu_cr["year"], eu_cr["cr_pct"],
                linewidth=2.5, linestyle="--", color=EU27_COLOR,
                label="EU-27", zorder=0,
            )

        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0f}%")
        )
        if row_idx == 2:
            ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel(f"{short}\nShare (%)", fontsize=11)
        ax.set_title(f"CR{n_top} — {short}", fontsize=11)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 9), fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Top-{n_top} Supply Concentration {title_suffix}\n"
        f"(Share of extra-EU imports covered by {n_top} largest suppliers, EUR)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    fname = f"{prefix}_supply_concentration_CR{n_top}_combined"
    fig.savefig(os.path.join(out_dir, fname + ".png"), dpi=150,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, fname + ".svg"), bbox_inches="tight")
    print(f"  Saved: {fname}.png / .svg")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def export_excel(cr3_data: dict, cr5_data: dict,
                 eu27_cr3: dict, eu27_cr5: dict,
                 cfg: dict, out_dir: str):
    """
    One Excel file per report.
    Sheets: one per product, rows = years, columns = CR3/CR5 per country.
    """
    countries = cfg["countries"]
    prefix = cfg["prefix"]

    excel_path = os.path.join(out_dir,
                              f"{prefix}_supply_concentration.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for product in PRODUCTS:
            short = PRODUCT_SHORT.get(product, product)
            rows = {}

            # Collect all years
            all_years = set()
            for d in cr3_data[product].values():
                if not d.empty:
                    all_years.update(d["year"].tolist())
            if not eu27_cr3[product].empty:
                all_years.update(eu27_cr3[product]["year"].tolist())
            all_years = sorted(all_years)

            for _iso2, cname in countries.items():
                for n_top, cr_dict in [(3, cr3_data), (5, cr5_data)]:
                    col = f"{cname} CR{n_top}"
                    if cname in cr_dict[product] and not cr_dict[product][cname].empty:
                        series = cr_dict[product][cname].set_index("year")["cr_pct"]
                        rows[col] = series
                    else:
                        rows[col] = pd.Series(dtype=float)

            if cfg["show_eu27"]:
                for n_top, eu_dict in [(3, eu27_cr3), (5, eu27_cr5)]:
                    col = f"EU-27 CR{n_top}"
                    if not eu_dict[product].empty:
                        rows[col] = eu_dict[product].set_index("year")["cr_pct"]

            sheet_df = pd.DataFrame(rows)
            sheet_df.index.name = "Year"
            sheet_df = sheet_df.reindex(all_years)
            sheet_df.to_excel(writer, sheet_name=short[:31])

    print(f"  Saved: {prefix}_supply_concentration.xlsx")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _resolve_reporter(reporters, country_name: str):
    """Find reporter string matching a country name (substring)."""
    matches = [r for r in reporters if country_name.lower() in r.lower()]
    return matches[0] if matches else None


def generate_report(report_key, df_all_year, df_per_country):
    cfg = REPORT_CONFIGS[report_key]
    prefix = cfg["prefix"]

    out_dir = os.path.join(OUTPUT_BASE, prefix)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Generating supply concentration report: {prefix} {cfg['title_suffix']}")
    print(f"{'=' * 60}")

    # -- EU-27 CR from all_year data (VALUE_EUR for consistency) --
    eu27_reporter = None
    for r in df_all_year["reporter"].unique():
        if "European Union" in r:
            eu27_reporter = r
            break
    df_eu = df_all_year[df_all_year["reporter"] == eu27_reporter]

    eu27_cr3 = {}
    eu27_cr5 = {}
    for product in PRODUCTS:
        eu27_cr3[product] = compute_cr(df_eu, product, 3, "VALUE_EUR")
        eu27_cr5[product] = compute_cr(df_eu, product, 5, "VALUE_EUR")

    # -- Per-country CR from per-country imports data --
    reporters = df_per_country["reporter"].unique()
    countries = cfg["countries"]

    cr3_data = {p: {} for p in PRODUCTS}
    cr5_data = {p: {} for p in PRODUCTS}

    for _iso2, country_name in countries.items():
        reporter = _resolve_reporter(reporters, country_name)
        if reporter is None:
            for product in PRODUCTS:
                cr3_data[product][country_name] = pd.DataFrame()
                cr5_data[product][country_name] = pd.DataFrame()
            continue
        df_c = df_per_country[df_per_country["reporter"] == reporter]
        for product in PRODUCTS:
            cr3_data[product][country_name] = compute_cr_per_country_simple(
                df_c, product, 3)
            cr5_data[product][country_name] = compute_cr_per_country_simple(
                df_c, product, 5)

    # -- Individual plots per product --
    for product in PRODUCTS:
        plot_cr(cr3_data[product], eu27_cr3[product], 3, product, cfg, out_dir)
        plot_cr(cr5_data[product], eu27_cr5[product], 5, product, cfg, out_dir)

    # -- Combined figures (one for CR3, one for CR5) --
    plot_cr_combined(cr3_data, eu27_cr3, 3, cfg, out_dir)
    plot_cr_combined(cr5_data, eu27_cr5, 5, cfg, out_dir)

    # -- Excel --
    export_excel(cr3_data, cr5_data, eu27_cr3, eu27_cr5, cfg, out_dir)

    print(f"  All outputs saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading all-year trade data (EU-27 + France)...")
    df_all_year = load_all_year()
    print(f"  {len(df_all_year):,} rows")

    print("Loading per-country imports data...")
    df_per_country = load_per_country()
    print(f"  {len(df_per_country):,} rows")

    for rk in REPORT_CONFIGS:
        generate_report(rk, df_all_year, df_per_country)

    print("\nDone.")


if __name__ == "__main__":
    main()
