"""
1_expense.py — Consumption Breakdown Comparison
================================================
Switzerland vs. EU neighbours (DE, FR, AT, IT).

EU countries (DE, FR, AT, IT):
    HBS 2020, income quintiles (Q1-Q5), annual PPS per adult equivalent.
    Uses hbs_cluster_comparison / hbs_data_loader PPS pipeline.

Switzerland:
    FSO 2020-2021, income quintiles (Q1-Q5), monthly CHF per household
    annualised (×12) and converted to PPP using Eurostat PRC_PPP_IND_1
    (average of 2020 and 2021 values).

Components: Housing · Transport · Food & Beverage · Health · Education · Other.

Note: CH values are per household; EU values are per adult equivalent.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from concurrent.futures import ThreadPoolExecutor, as_completed

from hbs_data_loader import setup_directories, load_pps_data
from hbs_cluster_comparison import build_country_components, assign_simple_deciles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, "external_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "graphs", "HBS_expense")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FSO raw data (from the Switzerland report)
FSO_DATA_PATH = os.path.abspath(os.path.join(
    CURRENT_DIR, "..", "..",
    "1_switzerland_vs_eu27_housing_energy", "external_data",
    "fso_household_expense.csv",
))

# Eurostat PPP factor for Switzerland
CH_PPP_PATH = os.path.join(
    EXTERNAL_DATA_DIR,
    "eurostat_ch_prc_ppp_ind_1__custom_20868951_linear.csv",
)

# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------
COUNTRIES = {
    "CH": "Switzerland",
    "DE": "Germany",
    "FR": "France",
    "AT": "Austria",
    "IT": "Italy",
}

# ---------------------------------------------------------------------------
# Display constants (same palette as hbs_cluster_comparison)
# ---------------------------------------------------------------------------
DISPLAY_COMPONENTS = ["Housing", "Transport", "Food & Beverage", "Health", "Education"]
DISPLAY_COLORS = ["#fc8d62", "#b3de69", "#8dd3c7", "#ffffb3", "#bebada"]

plt.rcParams["font.family"] = "Arial"


# ═══════════════════════════════════════════════════════════════════════════
# Switzerland (FSO)
# ═══════════════════════════════════════════════════════════════════════════

def load_ch_ppp_factor() -> float:
    """Return the average 2020-2021 PPP factor for Switzerland."""
    df = pd.read_csv(CH_PPP_PATH)
    df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce")
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    vals = df[df["TIME_PERIOD"].isin([2020, 2021])]["OBS_VALUE"]
    avg = vals.mean()
    print(f"  CH PPP factor (avg 2020–2021): {avg:.5f}")
    return avg


def load_fso_data() -> dict:
    """
    Load FSO household expense CSV and extract absolute monthly CHF per
    quintile.

    Returns
    -------
    dict  {quintile_label: {component: monthly_chf_value, ...}}
    """
    df = pd.read_csv(FSO_DATA_PATH, sep=";", decimal=",", encoding="latin-1")
    df.columns = df.columns.str.strip()

    # Normalise special dashes
    new_cols = []
    for col in df.columns:
        new_cols.append(re.sub(r"(\d+)\s+[–—]\s+(\d+)", r"\1 - \2", col))
    df.columns = new_cols

    type_col = df.columns[0]
    df[type_col] = df[type_col].str.strip()

    # Parse numeric cells (thousands separator = space, decimal = comma)
    for col in df.columns[1:]:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

    # Quintile columns: all data columns except "Total"
    data_cols = [c for c in df.columns if c != type_col]
    quintile_cols = [c for c in data_cols if c.strip().lower() != "total"]

    # Build code → {quintile_col: value}  and gross income
    code_lookup: dict[str, dict[str, float]] = {}
    gross_income: dict[str, float] = {}

    for _, row in df.iterrows():
        text = row[type_col]
        code = None
        if ":" in text:
            candidate = text.split(":")[0].strip()
            if candidate.replace(".", "").isdigit():
                code = candidate

        for col in quintile_cols:
            if code is not None:
                code_lookup.setdefault(code, {})[col] = row[col]
            if "gross income" in text.lower():
                gross_income[col] = row[col]

    # Map to our component names
    result = {}
    for i, col in enumerate(quintile_cols):
        cl = code_lookup
        total_cons = cl.get("50", {}).get(col, 0)   # Consumer expenditure
        housing    = cl.get("57", {}).get(col, 0)   # Housing and energy
        transport  = cl.get("62", {}).get(col, 0)   # Transport
        food       = cl.get("51", {}).get(col, 0)   # Food & non-alc. beverages
        health     = cl.get("61", {}).get(col, 0)   # Health expenditure
        education  = cl.get("67", {}).get(col, 0)   # School & training fees
        comp_sum = housing + transport + food + health + education
        other = max(total_cons - comp_sum, 0)

        result[f"Q{i + 1}"] = {
            "total_consumption": total_cons,
            "gross_income": gross_income.get(col, 0),
            "Housing": housing,
            "Transport": transport,
            "Food & Beverage": food,
            "Health": health,
            "Education": education,
            "Other (Residual)": other,
        }

    return result


def build_ch_components(ppp_factor: float) -> pd.DataFrame:
    """
    Build a DataFrame for Switzerland comparable to the HBS component tables.
    Monthly CHF values are annualised (×12) and converted to PPP.
    """
    fso = load_fso_data()

    value_keys = [
        "total_consumption",
        "Housing",
        "Transport",
        "Food & Beverage",
        "Health",
        "Education",
        "Other (Residual)",
    ]

    rows = []
    for quintile, vals in fso.items():
        row = {"decile": quintile}
        for key in value_keys:
            row[key] = vals[key] * 12 / ppp_factor   # monthly CHF → annual PPP
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  CH: {len(df)} quintiles built (annual PPP per household)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_expense_comparison(all_components: dict) -> None:
    """
    3 columns × 2 rows grid of stacked bar charts.
    Row 1: CH | DE | FR
    Row 2: AT | IT | (empty)
    """
    bar_components = DISPLAY_COMPONENTS + ["Other (Residual)"]
    bar_colors = DISPLAY_COLORS + ["#d3d3d3"]

    country_order = ["CH", "DE", "FR", "AT", "IT"]
    n_cols, n_rows = 3, 2

    # ── global y-max ──────────────────────────────────────────────────
    global_ymax = 0
    for cc in country_order:
        cdf = all_components.get(cc)
        if cdf is None or cdf.empty:
            continue
        row_totals = sum(
            cdf[comp].fillna(0).values
            for comp in bar_components
            if comp in cdf.columns
        )
        global_ymax = max(global_ymax, row_totals.max())
    global_ymax *= 1.08

    fig = plt.figure(figsize=(7 * n_cols, 6.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.40, wspace=0.28)

    fig.suptitle(
        "Consumption Breakdown — Switzerland vs. EU Neighbours\n"
        "Housing · Transport · Food · Health · Education · Residual  (annual PPP)",
        fontsize=15,
        fontweight="bold",
        y=1.005,
    )

    first_ax = None
    for idx, cc in enumerate(country_order):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = fig.add_subplot(gs[row_idx, col_idx])
        if first_ax is None:
            first_ax = ax

        cname = COUNTRIES[cc]
        cdf = all_components.get(cc)

        if cdf is None or cdf.empty:
            ax.text(
                0.5, 0.5, f"{cname}\n(no data)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12,
            )
            ax.set_title(cname, fontsize=12, fontweight="bold")
            ax.set_ylim(0, global_ymax)
            continue

        x = np.arange(len(cdf))
        bottom = np.zeros(len(cdf))

        for comp, color in zip(bar_components, bar_colors):
            if comp not in cdf.columns:
                continue
            vals = cdf[comp].fillna(0).values
            ax.bar(
                x, vals, bottom=bottom, label=comp,
                color=color, edgecolor="white", linewidth=0.8, alpha=0.85,
            )
            for j, v in enumerate(vals):
                if v > global_ymax * 0.04:
                    ax.text(
                        j, bottom[j] + v / 2, f"{int(v)}",
                        ha="center", va="center", fontsize=5.5,
                    )
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(
            cdf["decile"].values,
            fontsize=9,
            rotation=45 if len(cdf) > 5 else 0,
        )

        if cc == "CH":
            xlabel = "Income Quintile"
            subtitle = "(FSO 2020-21, per household)"
        else:
            xlabel = "Income Quintile"
            subtitle = "(HBS 2020, per adult eq.)"

        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
        ax.set_ylabel("Annual PPP", fontsize=10, fontweight="bold")
        ax.set_title(f"{cname}\n{subtitle}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, global_ymax)
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused cells
    for idx in range(len(country_order), n_rows * n_cols):
        ax_empty = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax_empty.set_visible(False)

    # Shared legend
    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="center left", fontsize=10,
            framealpha=0.95, bbox_to_anchor=(1.0, 0.5),
            ncol=1, title="Component", title_fontsize=11,
        )

    out_png = os.path.join(OUTPUT_DIR, "expense_comparison_ch_vs_eu.png")
    out_svg = os.path.join(OUTPUT_DIR, "expense_comparison_ch_vs_eu.svg")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_svg}")


# ═══════════════════════════════════════════════════════════════════════════
# Excel export
# ═══════════════════════════════════════════════════════════════════════════

def export_excel(all_components: dict) -> None:
    """One sheet per country.  Rows = income groups, columns = components."""
    cols_to_export = [
        "decile", "Housing", "Transport", "Food & Beverage",
        "Health", "Education", "Other (Residual)", "total_consumption",
    ]

    xlsx_path = os.path.join(OUTPUT_DIR, "expense_comparison_ch_vs_eu.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for cc, cname in COUNTRIES.items():
            cdf = all_components.get(cc)
            if cdf is None or cdf.empty:
                continue
            sheet = cdf[[c for c in cols_to_export if c in cdf.columns]].copy()
            if "decile" in sheet.columns:
                sheet = sheet.set_index("decile")
            sheet = sheet.round(1)
            sheet.to_excel(writer, sheet_name=f"{cname} ({cc})")

    print(f"  Saved: {xlsx_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Per-country visuals  (all HBS countries, parallel loading)
# ═══════════════════════════════════════════════════════════════════════════

PER_COUNTRY_DIR = os.path.join(OUTPUT_DIR, "per_country")
os.makedirs(PER_COUNTRY_DIR, exist_ok=True)

# Local cache for processed component tables (avoids re-reading 100 MB xlsx files)
_CACHE_DIR = os.path.join(BASE_DIR, "outputs", "data", "hbs_components_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

# Raw household-level parquet cache (written by load_country_2020 / 0_preprocess_hbs_cache.py)
_RAW_CACHE_DIR = os.path.join(BASE_DIR, "outputs", "data", "hbs_raw_cache")

# All countries with HBS 2020 files on disk
HBS_COUNTRIES = {
    "AT": "Austria",     "BE": "Belgium",     "BG": "Bulgaria",
    "CY": "Cyprus",      "DE": "Germany",
    "DK": "Denmark",     "EE": "Estonia",     "EL": "Greece",
    "ES": "Spain",       "FI": "Finland",     "FR": "France",
    "HR": "Croatia",                           "IE": "Ireland",
    "LT": "Lithuania",   "LU": "Luxembourg",
    "LV": "Latvia",      "MT": "Malta",       "NL": "Netherlands",
    "PL": "Poland",
    "SI": "Slovenia",    "SK": "Slovakia",
}  # EU-27 only — Norway excluded (non-EU)

_PC_COMPONENTS = ["Housing", "Transport", "Food & Beverage",
                   "Health", "Education", "Other (Residual)"]
_PC_COLORS     = ["#fc8d62", "#b3de69", "#8dd3c7",
                   "#ffffb3", "#bebada", "#d3d3d3"]
_PC_WORKERS    = 8   # parallel Excel reads — tune to your OneDrive bandwidth


def _load_one_country(cc: str, cname: str, pps_df,
                      force_rebuild: bool = False) -> tuple:
    """
    Worker: return (cc, component_DataFrame).
    Strategy:
      1. If a local parquet cache exists and force_rebuild is False → load it
         instantly (no OneDrive access needed after the first run).
      2. Otherwise read the full xlsx, compute components, save cache.
    This turns a 5-minute per-country xlsx read into a <1 second cache hit.
    """
    cache_path = os.path.join(_CACHE_DIR, f"{cc}_components.parquet")
    if not force_rebuild and os.path.exists(cache_path):
        try:
            cdf = pd.read_parquet(cache_path)
            print(f"  [{cc}]  cache hit")
            return cc, cdf
        except Exception:
            pass  # corrupt cache → rebuild below

    print(f"  [{cc}]  building from xlsx (first run — will be cached)…")
    try:
        cdf = build_country_components(cc, cname, pps_df, n_groups=10)
        if not cdf.empty:
            cdf.to_parquet(cache_path, index=False)
        return cc, cdf
    except Exception as exc:
        print(f"  [{cc}] ERROR: {exc}")
        return cc, pd.DataFrame()


def _plot_one_country(cc: str, cname: str, cdf: pd.DataFrame) -> None:
    """Stacked bar chart for one country — PNG + SVG to PER_COUNTRY_DIR."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x      = np.arange(len(cdf))
    bottom = np.zeros(len(cdf))

    ymax_total = sum(
        cdf[c].fillna(0).values for c in _PC_COMPONENTS if c in cdf.columns
    ).max()
    label_threshold = ymax_total * 0.04

    for comp, color in zip(_PC_COMPONENTS, _PC_COLORS):
        if comp not in cdf.columns:
            continue
        vals = cdf[comp].fillna(0).values
        ax.bar(x, vals, bottom=bottom, label=comp,
               color=color, edgecolor="white", linewidth=0.8, alpha=0.88)
        for j, v in enumerate(vals):
            if v > label_threshold:
                ax.text(j, bottom[j] + v / 2, f"{int(v):,}",
                        ha="center", va="center", fontsize=6.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(cdf["decile"].values, fontsize=9, rotation=30, ha="right")
    ax.set_xlabel("Income group", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annual PPP", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Consumption Breakdown — {cname} ({cc})\nHBS 2020, per adult equivalent",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    plt.tight_layout()
    base = os.path.join(PER_COUNTRY_DIR, f"{cc}_expense")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [{cc}]  → {cc}_expense.png / .svg")


def _export_per_country_excel(all_comps: dict) -> None:
    """One workbook, one sheet per country."""
    cols = ["decile", "Housing", "Transport", "Food & Beverage",
            "Health", "Education", "Other (Residual)", "total_consumption"]
    xlsx_path = os.path.join(PER_COUNTRY_DIR, "expense_per_country.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for cc in sorted(all_comps):
            cdf = all_comps[cc]
            if cdf is None or cdf.empty:
                continue
            sheet = cdf[[c for c in cols if c in cdf.columns]].copy()
            if "decile" in sheet.columns:
                sheet = sheet.set_index("decile")
            sheet_name = f"{HBS_COUNTRIES.get(cc, cc)} ({cc})"[:31]
            sheet.round(1).to_excel(writer, sheet_name=sheet_name)
    print(f"  Saved: {xlsx_path}")


# ═══════════════════════════════════════════════════════════════════════════
# EU snapshot heatmaps — % share & absolute PPS, D1 vs D10
# ═══════════════════════════════════════════════════════════════════════════

_SNAP_COMPS      = ["Housing", "Transport", "Food & Beverage",
                    "Health", "Education", "Other (Residual)"]
_SNAP_COMP_SHORT = ["Housing", "Transport", "Food", "Health", "Educ.", "Other"]
_SNAP_DECILES    = ["D1", "D10"]

# Housing, Energy & Transport all-decile heatmap
_HT_DECILES = [f"D{i}" for i in range(1, 11)]  # D1 … D10
_HT_COMPS   = ["Housing", "Transport"]

_HBS_PERF_CUT  = 0.006972
_HBS_EWBI_CUT  = 0.7
_HBS_M5_STEP   = 5000
_HBS_CL_COLORS = ["#fb8072", "#fdb462", "#8dd3c7", "#80b1d3"]
_HBS_CL_NAMES  = {
    0: "Low perf / Low EWBI",  1: "Low perf / High EWBI",
    2: "High perf / Low EWBI", 3: "High perf / High EWBI",
}


def _hbs_cluster_map() -> dict:
    """Return {iso: cluster_id 0-3} by replicating 0_clustering.py logic."""
    try:
        well_being_dir = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
        data_csv   = os.path.join(well_being_dir, "output", "ewbi_master_aggregated.csv")
        income_csv = os.path.join(BASE_DIR, "outputs", "data", "median_income_by_decile.csv")
        if not (os.path.exists(data_csv) and os.path.exists(income_csv)):
            return {}

        df = pd.read_csv(data_csv, low_memory=False)

        ewbi_d = df[
            (df["Level"] == 1) & (df["Decile"] != "All Deciles") &
            (~df["Country"].isin(["EU-27", "All Countries"]))
        ].copy()
        for c in ("Decile", "Year", "Value"):
            ewbi_d[c] = pd.to_numeric(ewbi_d[c], errors="coerce")
        ewbi_d = ewbi_d.dropna(subset=["Country", "Year", "Decile", "Value"])

        ewbi_all = df[
            (df["Level"] == 1) & (df["Decile"] == "All Deciles") &
            (~df["Country"].isin(["EU-27", "All Countries"]))
        ].copy()
        for c in ("Year", "Value"):
            ewbi_all[c] = pd.to_numeric(ewbi_all[c], errors="coerce")
        ewbi_all = ewbi_all.dropna(subset=["Country", "Year", "Value"])
        ewbi_last = (ewbi_all.sort_values("Year")
                             .groupby("Country").tail(1)[["Country", "Value"]]
                             .rename(columns={"Value": "EWBI_Last"}))

        inc_df = pd.read_csv(income_csv)
        for c in ("year", "decile", "median_equi_disp_inc"):
            inc_df[c] = pd.to_numeric(inc_df[c], errors="coerce")
        inc_df = inc_df.dropna(
            subset=["country", "year", "decile", "median_equi_disp_inc"])

        merged = ewbi_d.merge(
            inc_df, left_on=["Country", "Year", "Decile"],
            right_on=["country", "year", "decile"], how="inner",
        ).dropna(subset=["median_equi_disp_inc", "Value"])

        last_yr = merged.groupby("Country")["Year"].max().reset_index()
        last_yr.columns = ["Country", "Last_Year"]
        pts = merged.merge(last_yr, on="Country")
        pts = pts[pts["Year"] == pts["Last_Year"]].copy()

        pts["bin"] = (np.round(pts["median_equi_disp_inc"] / _HBS_M5_STEP)
                      * _HBS_M5_STEP)
        bench = (pts.groupby("bin", as_index=False)["Value"].mean()
                    .sort_values("bin").rename(columns={"Value": "bench"}))
        pts["ewbi_exp"] = np.interp(
            pts["median_equi_disp_inc"].values.astype(float),
            bench["bin"].values.astype(float),
            bench["bench"].values.astype(float),
        )
        pts["residual"] = pts["Value"] - pts["ewbi_exp"]

        perf = pts.groupby("Country", as_index=False).agg(
            Performance_Score=("residual", "mean"))
        feats = perf.merge(ewbi_last, on="Country").dropna(
            subset=["Performance_Score", "EWBI_Last"])

        cv = {("Low performer",  "Low EWBI"):  0,
              ("Low performer",  "High EWBI"): 1,
              ("High performer", "Low EWBI"):  2,
              ("High performer", "High EWBI"): 3}
        feats["pg"] = np.where(feats["Performance_Score"] >= _HBS_PERF_CUT,
                               "High performer", "Low performer")
        feats["eg"] = np.where(feats["EWBI_Last"] >= _HBS_EWBI_CUT,
                               "High EWBI", "Low EWBI")
        feats["Cluster"] = feats.apply(lambda r: cv[(r["pg"], r["eg"])], axis=1)
        return dict(zip(feats["Country"], feats["Cluster"].astype(int)))

    except Exception as exc:
        print(f"  [hbs_cluster] Failed: {exc}")
        return {}


def _build_expense_snapshot_df(all_comps: dict) -> pd.DataFrame:
    """One row per country: D1 and D10 values for both PPS (abs) and % of net income."""
    records = []
    for cc, cdf in all_comps.items():
        if cdf is None or cdf.empty:
            continue
        cdf = cdf.copy()
        cdf["decile"] = cdf["decile"].astype(str).str.strip()
        row = {"Country": cc, "Country_Name": HBS_COUNTRIES.get(cc, cc)}
        for dlabel in _SNAP_DECILES:
            sub = cdf[cdf["decile"] == dlabel]
            if sub.empty:
                row[f"{dlabel}_n"] = np.nan
                for comp in _SNAP_COMPS:
                    row[f"{dlabel}_pps_{comp}"] = np.nan
                    row[f"{dlabel}_pct_{comp}"] = np.nan
                continue
            r      = sub.iloc[0]
            income = r.get("equivalized_income_pps", np.nan)
            row[f"{dlabel}_n"] = r.get("n_households", np.nan)
            for comp in _SNAP_COMPS:
                pps_val = r.get(comp, np.nan)
                row[f"{dlabel}_pps_{comp}"] = pps_val
                row[f"{dlabel}_pct_{comp}"] = (
                    pps_val / income * 100
                    if not (pd.isna(pps_val) or pd.isna(income) or income == 0)
                    else np.nan
                )
        records.append(row)
    return pd.DataFrame(records)


def _plot_expense_heatmap(snap_df: pd.DataFrame, kind: str,
                          cluster_map: dict, out_dir: str) -> None:
    """
    kind='pct' → % of net income for D1 & D10 (12 columns)
    kind='pps' → absolute PPS per adult equivalent for D1 & D10

    Rows = countries sorted by cluster (0→3) then name.
    Each column has independent min-to-max colour normalisation.
    """
    import matplotlib.colors as mcolors

    # ── Sort rows ────────────────────────────────────────────────────────
    df = snap_df.copy()
    df["cl"]   = df["Country"].map(lambda cc: cluster_map.get(cc, -1))
    has_cl = df[df["cl"] >= 0].sort_values(["cl", "Country_Name"])
    no_cl  = df[df["cl"] <  0].sort_values("Country_Name")
    df = pd.concat([has_cl, no_cl], ignore_index=True)
    n_rows = len(df)

    # ── Column spec ───────────────────────────────────────────────────────
    col_meta     = [(d, c, f"{d}_{kind}_{c}")
                    for d in _SNAP_DECILES for c in _SNAP_COMPS]
    n_cols       = len(col_meta)    # 12
    n_per_decile = len(_SNAP_COMPS) # 6

    col_min = {}; col_max = {}
    for _, _, k in col_meta:
        vals = df[k].dropna()
        col_min[k] = float(vals.min()) if len(vals) else 0.0
        col_max[k] = float(vals.max()) if len(vals) else 1.0
        if col_max[k] == col_min[k]:
            col_max[k] = col_min[k] + 1.0

    cmap = plt.cm.RdYlGn_r if kind == "pct" else plt.cm.YlGnBu

    # ── Geometry ─────────────────────────────────────────────────────────
    cell_w  = 1.55
    cell_h  = 0.40
    lmargin = 3.0
    rmargin = 1.8

    fig, ax = plt.subplots(figsize=(lmargin + n_cols * cell_w + rmargin,
                                    n_rows * cell_h + 2.5))
    # set_ylim(large, small) → y increases downward (row 0 visually at top)
    ax.set_xlim(-lmargin, n_cols * cell_w + rmargin)
    ax.set_ylim(n_rows * cell_h + 0.1, -2.5 * cell_h)
    ax.set_axis_off()

    # ── Column headers (y < 0, appear above data) ─────────────────────────
    for ci, (_, c, _) in enumerate(col_meta):
        short = _SNAP_COMP_SHORT[_SNAP_COMPS.index(c)]
        bg    = "#e0e0e0" if ci < n_per_decile else "#c8c8c8"
        ax.add_patch(plt.Rectangle(
            (ci * cell_w, -cell_h), cell_w, cell_h,
            facecolor=bg, edgecolor="white", linewidth=0.5, clip_on=False,
        ))
        ax.text(ci * cell_w + cell_w / 2, -cell_h / 2, short,
                ha="center", va="center", fontsize=7.5, fontweight="bold")

    for bi, dlabel in enumerate(_SNAP_DECILES):
        x0 = bi * n_per_decile * cell_w
        bg = "#3a3a3a" if bi == 0 else "#1a2a4a"
        ax.add_patch(plt.Rectangle(
            (x0, -2 * cell_h), n_per_decile * cell_w, cell_h,
            facecolor=bg, edgecolor="white", linewidth=0.8, clip_on=False,
        ))
        ax.text(x0 + n_per_decile * cell_w / 2, -1.5 * cell_h, dlabel,
                ha="center", va="center", fontsize=12,
                fontweight="bold", color="white")

    # Vertical separator D1 | D10
    ax.plot([n_per_decile * cell_w, n_per_decile * cell_w],
            [-2 * cell_h, n_rows * cell_h],
            color="#333333", linewidth=2.0, clip_on=False)

    # ── Data cells ────────────────────────────────────────────────────────
    prev_cl      = None
    cl_first_row: dict = {}
    cl_last_row:  dict = {}

    for ri in range(n_rows):
        row = df.iloc[ri]
        cl  = int(row["cl"])
        y   = ri * cell_h

        if cl != prev_cl and prev_cl is not None:
            ax.axhline(y, color="#555555", linewidth=1.0, linestyle="--")
        if cl not in cl_first_row:
            cl_first_row[cl] = ri
        cl_last_row[cl] = ri
        prev_cl = cl

        name_col = _HBS_CL_COLORS[cl] if 0 <= cl <= 3 else "#888888"
        ax.text(-0.12, y + cell_h / 2, row["Country_Name"],
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=name_col)

        for ci, (dlabel_ci, _, ckey) in enumerate(col_meta):
            val  = row[ckey]
            n_hh = row.get(f"{dlabel_ci}_n", _MIN_CELL_N)
            x    = ci * cell_w

            if not pd.isna(n_hh) and 0 < n_hh < _MIN_CELL_N:
                face, txt, txt_col = "#cccccc", "n<5", "#666666"
            elif pd.isna(val):
                face, txt, txt_col = "#e0e0e0", "—", "#999999"
            else:
                vmin, vmax = col_min[ckey], col_max[ckey]
                nv   = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
                face = mcolors.to_hex(cmap(nv))
                txt  = f"{val:.1f}%" if kind == "pct" else f"{val:,.0f}"
                r_, g_, b_, _ = mcolors.to_rgba(face)
                lum  = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                txt_col = "white" if lum < 0.45 else "black"

            ax.add_patch(plt.Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=face, edgecolor="white", linewidth=0.5,
            ))
            ax.text(x + cell_w / 2, y + cell_h / 2, txt,
                    ha="center", va="center", fontsize=6.5, color=txt_col)

    # ── Cluster labels in right margin ────────────────────────────────────
    for cl, first in sorted(cl_first_row.items()):
        if cl < 0:
            continue
        mid_y = (first + cl_last_row[cl] + 1) / 2 * cell_h
        ax.text(n_cols * cell_w + 0.15, mid_y,
                _HBS_CL_NAMES[cl], ha="left", va="center",
                fontsize=7, color=_HBS_CL_COLORS[cl],
                fontweight="bold", rotation=90)

    # ── Title & save ──────────────────────────────────────────────────────
    unit = ("% of net income"
            if kind == "pct" else "Annual PPS, per adult equivalent")
    ax.set_title(
        f"Household Expenditure by Component — D1 & D10\n"
        f"({unit})  |  HBS 2020",
        fontsize=11, fontweight="bold", pad=6,
    )
    fname = f"expense_snapshot_{kind}"
    fig.savefig(os.path.join(out_dir, fname + ".png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, fname + ".svg"),
                format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [{kind.upper()}] Saved: {fname}.png / .svg")


def _build_ht_decile_df(all_comps: dict) -> pd.DataFrame:
    """
    One row per country: Housing + Energy + Transport combined, D1–D10.
    Columns per decile:
      HT_{D}_pct  → (Housing + Energy + Transport) / net_income (EUR_HH099) × 100
      HT_{D}_pps  → Housing + Energy + Transport  (annual PPS per adult equivalent)
    """
    records = []
    for cc, cdf in all_comps.items():
        if cdf is None or cdf.empty:
            continue
        cdf = cdf.copy()
        cdf["decile"] = cdf["decile"].astype(str).str.strip()
        row = {"Country": cc, "Country_Name": HBS_COUNTRIES.get(cc, cc)}
        for dlabel in _HT_DECILES:
            sub = cdf[cdf["decile"] == dlabel]
            pct_key = f"HT_{dlabel}_pct"
            pps_key = f"HT_{dlabel}_pps"
            n_key   = f"HT_{dlabel}_n"
            if sub.empty:
                row[pct_key] = np.nan
                row[pps_key] = np.nan
                row[n_key]   = np.nan
            else:
                r      = sub.iloc[0]
                income = r.get("equivalized_income_pps", np.nan)
                h      = r.get("Housing",   np.nan)
                t      = r.get("Transport",  np.nan)
                h_val  = 0.0 if pd.isna(h) else float(h)
                t_val  = 0.0 if pd.isna(t) else float(t)
                ht     = h_val + t_val
                row[pps_key] = ht
                row[pct_key] = (
                    ht / float(income) * 100
                    if not (pd.isna(income) or income == 0)
                    else np.nan
                )
                row[n_key] = r.get("n_households", np.nan)
        records.append(row)
    return pd.DataFrame(records)


def _plot_ht_decile_heatmap(ht_df: pd.DataFrame, cluster_map: dict,
                             out_dir: str, kind: str = "pct") -> None:
    """
    kind='pct': Housing+Energy+Transport as % of net income (EUR_HH099), D1–D10.
    kind='pps': Housing+Energy+Transport in absolute PPS per adult equivalent, D1–D10.
    Rows = countries sorted by cluster then name.
    10 columns, one per decile.  Shared colour scale across all columns.
    """
    import matplotlib.colors as mcolors

    df = ht_df.copy()
    df["cl"] = df["Country"].map(lambda cc: cluster_map.get(cc, -1))
    has_cl = df[df["cl"] >= 0].sort_values(["cl", "Country_Name"])
    no_cl  = df[df["cl"] <  0].sort_values("Country_Name")
    df = pd.concat([has_cl, no_cl], ignore_index=True)
    n_rows = len(df)

    # 10 columns, one per decile
    col_meta = [(d, f"HT_{d}_{kind}") for d in _HT_DECILES]
    n_cols   = len(col_meta)  # 10

    # Shared colour scale across all 10 decile columns
    all_vals = pd.concat([df[k] for _, k in col_meta]).dropna()
    g_min = float(all_vals.min()) if len(all_vals) else 0.0
    g_max = float(all_vals.max()) if len(all_vals) else 1.0
    if g_max == g_min:
        g_max = g_min + 1.0

    cmap    = plt.cm.RdYlGn_r
    hdr_bg  = "#c8d8e8"   # light blue-grey for decile headers
    hdr_col = "#1a3a5a"   # dark navy for block header

    cell_w  = 1.4
    cell_h  = 0.40
    lmargin = 3.0
    rmargin = 1.8

    fig, ax = plt.subplots(figsize=(lmargin + n_cols * cell_w + rmargin,
                                    n_rows * cell_h + 2.5))
    ax.set_xlim(-lmargin, n_cols * cell_w + rmargin)
    ax.set_ylim(n_rows * cell_h + 0.1, -2.5 * cell_h)
    ax.set_axis_off()

    # Decile sub-headers (row -1)
    for ci, (d, _) in enumerate(col_meta):
        ax.add_patch(plt.Rectangle(
            (ci * cell_w, -cell_h), cell_w, cell_h,
            facecolor=hdr_bg, edgecolor="white",
            linewidth=0.5, clip_on=False,
        ))
        ax.text(ci * cell_w + cell_w / 2, -cell_h / 2, d,
                ha="center", va="center", fontsize=7, fontweight="bold")

    # Single block header spanning all columns (row -2)
    ax.add_patch(plt.Rectangle(
        (0, -2 * cell_h), n_cols * cell_w, cell_h,
        facecolor=hdr_col, edgecolor="white", linewidth=0.8, clip_on=False,
    ))
    ax.text(n_cols * cell_w / 2, -1.5 * cell_h,
            "Housing + Energy + Transport",
            ha="center", va="center", fontsize=12,
            fontweight="bold", color="white")

    # Data cells
    prev_cl = None
    cl_first_row: dict = {}
    cl_last_row:  dict = {}

    for ri in range(n_rows):
        row = df.iloc[ri]
        cl  = int(row["cl"])
        y   = ri * cell_h

        if cl != prev_cl and prev_cl is not None:
            ax.axhline(y, color="#555555", linewidth=1.0, linestyle="--")
        if cl not in cl_first_row:
            cl_first_row[cl] = ri
        cl_last_row[cl] = ri
        prev_cl = cl

        name_col = _HBS_CL_COLORS[cl] if 0 <= cl <= 3 else "#888888"
        ax.text(-0.12, y + cell_h / 2, row["Country_Name"],
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=name_col)

        for ci, (d_ci, ckey) in enumerate(col_meta):
            val  = row[ckey]
            n_hh = row.get(f"HT_{d_ci}_n", _MIN_CELL_N)
            x    = ci * cell_w

            if not pd.isna(n_hh) and 0 < n_hh < _MIN_CELL_N:
                face, txt, txt_col = "#cccccc", "n<5", "#666666"
            elif pd.isna(val):
                face, txt, txt_col = "#e0e0e0", "—", "#999999"
            else:
                nv   = max(0.0, min(1.0, (val - g_min) / (g_max - g_min)))
                face = mcolors.to_hex(cmap(nv))
                txt  = f"{val:.1f}%" if kind == "pct" else f"{val:,.0f}"
                r_, g_, b_, _ = mcolors.to_rgba(face)
                lum  = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                txt_col = "white" if lum < 0.45 else "black"

            ax.add_patch(plt.Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=face, edgecolor="white", linewidth=0.5,
            ))
            ax.text(x + cell_w / 2, y + cell_h / 2, txt,
                    ha="center", va="center", fontsize=6.5, color=txt_col)

    # Cluster labels in right margin
    for cl, first in sorted(cl_first_row.items()):
        if cl < 0:
            continue
        mid_y = (first + cl_last_row[cl] + 1) / 2 * cell_h
        ax.text(n_cols * cell_w + 0.15, mid_y,
                _HBS_CL_NAMES[cl], ha="left", va="center",
                fontsize=7, color=_HBS_CL_COLORS[cl],
                fontweight="bold", rotation=90)

    unit = "% of net income" if kind == "pct" else "Annual PPS per adult equivalent"
    ax.set_title(
        f"Housing + Energy + Transport (sum) — {unit}, by Decile D1–D10\n"
        "HBS 2020  |  EU-27 countries  |  shared colour scale",
        fontsize=11, fontweight="bold", pad=6,
    )
    fname = f"expense_housing_energy_transport_deciles_{kind}"
    fig.savefig(os.path.join(out_dir, fname + ".png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, fname + ".svg"),
                format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [HT] Saved: {fname}.png / .svg")


# Raw housing + transport column names in the HBS household file
_HBS_HOUSING_COLS   = ["EUR_HE041", "EUR_HE042", "EUR_HE043", "EUR_HE044", "EUR_HE045"]
_HBS_TRANSPORT_COLS = ["EUR_HE071", "EUR_HE072", "EUR_HE073"]
# Housing + Energy overburden: shelter (04.1+04.2) + energy (04.5)
_HBS_SHELTER_COLS   = ["EUR_HE041", "EUR_HE042"]   # actual + imputed rent
_HBS_ENERGY_COL     = "EUR_HE045"                   # electricity, gas & fuels
_HBS_TOTAL_COL      = "EUR_HE00"
_HBS_WEIGHT_COL     = "HA10"
_OVERBURDEN_THRESHOLD    = 0.50   # 50 % — H+E+T overburden
_OVERBURDEN_THRESHOLD_HE = 0.40   # 40 % — Housing+Energy overburden
_MIN_CELL_N              = 5      # grey out cells with fewer than this many households


def _build_ht_overburden_df() -> pd.DataFrame:
    """
    For each country × decile: share of households (weighted) whose
    Housing + Energy + Transport expenditure exceeds _OVERBURDEN_THRESHOLD of
    their net income (EUR_HH099).

    Reads from the raw household-level parquet cache written by
    load_country_2020() / 0_preprocess_hbs_cache.py.
    Countries without a raw parquet file are skipped.
    """
    records = []
    for cc, cname in HBS_COUNTRIES.items():
        raw_path = os.path.join(_RAW_CACHE_DIR, f"{cc}_hbs2020_raw.parquet")
        if not os.path.exists(raw_path):
            print(f"  [{cc}]  no raw parquet — skipped for overburden")
            continue
        try:
            df = pd.read_parquet(raw_path)
        except Exception as exc:
            print(f"  [{cc}]  parquet read error: {exc}")
            continue

        # Coerce numerics
        for col in _HBS_HOUSING_COLS + _HBS_TRANSPORT_COLS + [
                _HBS_TOTAL_COL, _HBS_WEIGHT_COL, "EUR_HH099", "HB061"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where net income or weight is missing / zero
        df = df[
            df["EUR_HH099"].notna() & (df["EUR_HH099"] > 0) &
            df[_HBS_WEIGHT_COL].notna() & (df[_HBS_WEIGHT_COL] > 0)
        ].copy()
        if df.empty:
            continue

        # H+E+T sum (raw EUR — ratio against income)
        ht = pd.Series(0.0, index=df.index)
        for col in _HBS_HOUSING_COLS + _HBS_TRANSPORT_COLS:
            if col in df.columns:
                ht += df[col].fillna(0.0)
        df["_ht_share"] = ht / df["EUR_HH099"]
        df["_overburdened"] = (df["_ht_share"] > _OVERBURDEN_THRESHOLD).astype(float)

        # Assign income deciles
        df = assign_simple_deciles(df, n_groups=10)
        if "income_decile" not in df.columns or df["income_decile"].notna().sum() == 0:
            continue

        row = {"Country": cc, "Country_Name": cname}
        for dlabel in _HT_DECILES:
            grp = df[df["income_decile"] == dlabel]
            key   = f"OB_{dlabel}_pct"
            n_key = f"OB_{dlabel}_n"
            if grp.empty or grp[_HBS_WEIGHT_COL].sum() == 0:
                row[key]   = np.nan
                row[n_key] = 0
            else:
                w_total = grp[_HBS_WEIGHT_COL].sum()
                w_ob    = (grp[_HBS_WEIGHT_COL] * grp["_overburdened"]).sum()
                row[key]   = w_ob / w_total * 100.0
                row[n_key] = len(grp)
        records.append(row)
        print(f"  [{cc}]  overburden computed")

    return pd.DataFrame(records)


def _plot_ht_overburden_heatmap(ob_df: pd.DataFrame, cluster_map: dict,
                                 out_dir: str) -> None:
    """
    Heatmap: rows = countries, columns = D1–D10.
    Cell value = % of households (weighted) with H+E+T > 50 % of net income.
    Shared colour scale across all columns.  Diverging around 50 %.
    """
    import matplotlib.colors as mcolors

    df = ob_df.copy()
    df["cl"] = df["Country"].map(lambda cc: cluster_map.get(cc, -1))
    has_cl = df[df["cl"] >= 0].sort_values(["cl", "Country_Name"])
    no_cl  = df[df["cl"] <  0].sort_values("Country_Name")
    df = pd.concat([has_cl, no_cl], ignore_index=True)
    n_rows = len(df)

    col_meta = [(d, f"OB_{d}_pct") for d in _HT_DECILES]
    n_cols   = len(col_meta)  # 10

    all_vals = pd.concat([df[k] for _, k in col_meta]).dropna()
    g_min = float(all_vals.min()) if len(all_vals) else 0.0
    g_max = float(all_vals.max()) if len(all_vals) else 100.0
    if g_max == g_min:
        g_max = g_min + 1.0

    cmap    = plt.cm.RdYlGn_r
    hdr_bg  = "#f0e0d0"
    hdr_col = "#8B2500"   # dark red for header

    cell_w  = 1.4
    cell_h  = 0.40
    lmargin = 3.0
    rmargin = 1.8

    fig, ax = plt.subplots(figsize=(lmargin + n_cols * cell_w + rmargin,
                                    n_rows * cell_h + 2.5))
    ax.set_xlim(-lmargin, n_cols * cell_w + rmargin)
    ax.set_ylim(n_rows * cell_h + 0.1, -2.5 * cell_h)
    ax.set_axis_off()

    # Decile sub-headers (row −1)
    for ci, (d, _) in enumerate(col_meta):
        ax.add_patch(plt.Rectangle(
            (ci * cell_w, -cell_h), cell_w, cell_h,
            facecolor=hdr_bg, edgecolor="white", linewidth=0.5, clip_on=False,
        ))
        ax.text(ci * cell_w + cell_w / 2, -cell_h / 2, d,
                ha="center", va="center", fontsize=7, fontweight="bold")

    # Single block header (row −2)
    ax.add_patch(plt.Rectangle(
        (0, -2 * cell_h), n_cols * cell_w, cell_h,
        facecolor=hdr_col, edgecolor="white", linewidth=0.8, clip_on=False,
    ))
    ax.text(n_cols * cell_w / 2, -1.5 * cell_h,
            "H+E+T overburden  (H+E+T > 50 % of net income)",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="white")

    # Data cells
    prev_cl = None
    cl_first_row: dict = {}
    cl_last_row:  dict = {}

    for ri in range(n_rows):
        row = df.iloc[ri]
        cl  = int(row["cl"])
        y   = ri * cell_h

        if cl != prev_cl and prev_cl is not None:
            ax.axhline(y, color="#555555", linewidth=1.0, linestyle="--")
        if cl not in cl_first_row:
            cl_first_row[cl] = ri
        cl_last_row[cl] = ri
        prev_cl = cl

        name_col = _HBS_CL_COLORS[cl] if 0 <= cl <= 3 else "#888888"
        ax.text(-0.12, y + cell_h / 2, row["Country_Name"],
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=name_col)

        for ci, (_, ckey) in enumerate(col_meta):
            val  = row[ckey]
            n_hh = row.get(ckey.replace("_pct", "_n"), _MIN_CELL_N)
            x    = ci * cell_w

            if not pd.isna(n_hh) and 0 < n_hh < _MIN_CELL_N:
                face, txt, txt_col = "#cccccc", "n<5", "#666666"
            elif pd.isna(val):
                face, txt, txt_col = "#e0e0e0", "—", "#999999"
            else:
                nv   = max(0.0, min(1.0, (val - g_min) / (g_max - g_min)))
                face = mcolors.to_hex(cmap(nv))
                txt  = f"{val:.1f}%"
                r_, g_, b_, _ = mcolors.to_rgba(face)
                lum  = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                txt_col = "white" if lum < 0.45 else "black"

            ax.add_patch(plt.Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=face, edgecolor="white", linewidth=0.5,
            ))
            ax.text(x + cell_w / 2, y + cell_h / 2, txt,
                    ha="center", va="center", fontsize=6.5, color=txt_col)

    # Cluster labels in right margin
    for cl, first in sorted(cl_first_row.items()):
        if cl < 0:
            continue
        mid_y = (first + cl_last_row[cl] + 1) / 2 * cell_h
        ax.text(n_cols * cell_w + 0.15, mid_y,
                _HBS_CL_NAMES[cl], ha="left", va="center",
                fontsize=7, color=_HBS_CL_COLORS[cl],
                fontweight="bold", rotation=90)

    ax.set_title(
        "Housing + Energy + Transport Overburden Rate by Decile\n"
        "Share of households (%) with H+E+T > 50 % of net income  —  HBS 2020",
        fontsize=11, fontweight="bold", pad=6,
    )
    fname = "expense_het_overburden_deciles"
    fig.savefig(os.path.join(out_dir, fname + ".png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, fname + ".svg"),
                format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OB] Saved: {fname}.png / .svg")


def _build_he_overburden_df() -> pd.DataFrame:
    """
    For each country × decile: share of households (weighted) whose
    Housing (shelter: rent + imputed rent) + Energy expenditure exceeds
    _OVERBURDEN_THRESHOLD_HE (40 %) of net income (EUR_HH099).
    Shelter = EUR_HE041 + EUR_HE042 (COICOP 04.1 + 04.2)
    Energy  = EUR_HE045            (COICOP 04.5)
    """
    he_cols = _HBS_SHELTER_COLS + [_HBS_ENERGY_COL]
    records = []
    for cc, cname in HBS_COUNTRIES.items():
        raw_path = os.path.join(_RAW_CACHE_DIR, f"{cc}_hbs2020_raw.parquet")
        if not os.path.exists(raw_path):
            print(f"  [{cc}]  no raw parquet — skipped for H+E overburden")
            continue
        try:
            df = pd.read_parquet(raw_path)
        except Exception as exc:
            print(f"  [{cc}]  parquet read error: {exc}")
            continue

        for col in he_cols + [_HBS_TOTAL_COL, _HBS_WEIGHT_COL, "EUR_HH099", "HB061"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[
            df["EUR_HH099"].notna() & (df["EUR_HH099"] > 0) &
            df[_HBS_WEIGHT_COL].notna() & (df[_HBS_WEIGHT_COL] > 0)
        ].copy()
        if df.empty:
            continue

        he = pd.Series(0.0, index=df.index)
        for col in he_cols:
            if col in df.columns:
                he += df[col].fillna(0.0)
        df["_he_share"] = he / df["EUR_HH099"]
        df["_overburdened"] = (df["_he_share"] > _OVERBURDEN_THRESHOLD_HE).astype(float)

        df = assign_simple_deciles(df, n_groups=10)
        if "income_decile" not in df.columns or df["income_decile"].notna().sum() == 0:
            continue

        row = {"Country": cc, "Country_Name": cname}
        for dlabel in _HT_DECILES:
            grp = df[df["income_decile"] == dlabel]
            key   = f"HE_{dlabel}_pct"
            n_key = f"HE_{dlabel}_n"
            if grp.empty or grp[_HBS_WEIGHT_COL].sum() == 0:
                row[key]   = np.nan
                row[n_key] = 0
            else:
                w_total = grp[_HBS_WEIGHT_COL].sum()
                w_ob    = (grp[_HBS_WEIGHT_COL] * grp["_overburdened"]).sum()
                row[key]   = w_ob / w_total * 100.0
                row[n_key] = len(grp)
        records.append(row)
        print(f"  [{cc}]  H+E overburden computed")

    return pd.DataFrame(records)


def _plot_he_overburden_heatmap(ob_df: pd.DataFrame, cluster_map: dict,
                                 out_dir: str) -> None:
    """
    Heatmap: rows = countries, columns = D1–D10.
    Cell value = % of households with Housing+Energy > 40 % of net income.
    """
    import matplotlib.colors as mcolors

    df = ob_df.copy()
    df["cl"] = df["Country"].map(lambda cc: cluster_map.get(cc, -1))
    has_cl = df[df["cl"] >= 0].sort_values(["cl", "Country_Name"])
    no_cl  = df[df["cl"] <  0].sort_values("Country_Name")
    df = pd.concat([has_cl, no_cl], ignore_index=True)
    n_rows = len(df)

    col_meta = [(d, f"HE_{d}_pct") for d in _HT_DECILES]
    n_cols   = len(col_meta)

    all_vals = pd.concat([df[k] for _, k in col_meta]).dropna()
    g_min = float(all_vals.min()) if len(all_vals) else 0.0
    g_max = float(all_vals.max()) if len(all_vals) else 100.0
    if g_max == g_min:
        g_max = g_min + 1.0

    cmap    = plt.cm.RdYlGn_r
    hdr_bg  = "#dce8f0"
    hdr_col = "#1a4a6a"   # dark blue for header

    cell_w  = 1.4
    cell_h  = 0.40
    lmargin = 3.0
    rmargin = 1.8

    fig, ax = plt.subplots(figsize=(lmargin + n_cols * cell_w + rmargin,
                                    n_rows * cell_h + 2.5))
    ax.set_xlim(-lmargin, n_cols * cell_w + rmargin)
    ax.set_ylim(n_rows * cell_h + 0.1, -2.5 * cell_h)
    ax.set_axis_off()

    for ci, (d, _) in enumerate(col_meta):
        ax.add_patch(plt.Rectangle(
            (ci * cell_w, -cell_h), cell_w, cell_h,
            facecolor=hdr_bg, edgecolor="white", linewidth=0.5, clip_on=False,
        ))
        ax.text(ci * cell_w + cell_w / 2, -cell_h / 2, d,
                ha="center", va="center", fontsize=7, fontweight="bold")

    ax.add_patch(plt.Rectangle(
        (0, -2 * cell_h), n_cols * cell_w, cell_h,
        facecolor=hdr_col, edgecolor="white", linewidth=0.8, clip_on=False,
    ))
    ax.text(n_cols * cell_w / 2, -1.5 * cell_h,
            "Housing + Energy overburden  (shelter + energy > 40 % of net income)",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="white")

    prev_cl = None
    cl_first_row: dict = {}
    cl_last_row:  dict = {}

    for ri in range(n_rows):
        row = df.iloc[ri]
        cl  = int(row["cl"])
        y   = ri * cell_h

        if cl != prev_cl and prev_cl is not None:
            ax.axhline(y, color="#555555", linewidth=1.0, linestyle="--")
        if cl not in cl_first_row:
            cl_first_row[cl] = ri
        cl_last_row[cl] = ri
        prev_cl = cl

        name_col = _HBS_CL_COLORS[cl] if 0 <= cl <= 3 else "#888888"
        ax.text(-0.12, y + cell_h / 2, row["Country_Name"],
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=name_col)

        for ci, (_, ckey) in enumerate(col_meta):
            val  = row[ckey]
            n_hh = row.get(ckey.replace("_pct", "_n"), _MIN_CELL_N)
            x    = ci * cell_w

            if not pd.isna(n_hh) and 0 < n_hh < _MIN_CELL_N:
                face, txt, txt_col = "#cccccc", "n<5", "#666666"
            elif pd.isna(val):
                face, txt, txt_col = "#e0e0e0", "—", "#999999"
            else:
                nv   = max(0.0, min(1.0, (val - g_min) / (g_max - g_min)))
                face = mcolors.to_hex(cmap(nv))
                txt  = f"{val:.1f}%"
                r_, g_, b_, _ = mcolors.to_rgba(face)
                lum  = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                txt_col = "white" if lum < 0.45 else "black"

            ax.add_patch(plt.Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=face, edgecolor="white", linewidth=0.5,
            ))
            ax.text(x + cell_w / 2, y + cell_h / 2, txt,
                    ha="center", va="center", fontsize=6.5, color=txt_col)

    for cl, first in sorted(cl_first_row.items()):
        if cl < 0:
            continue
        mid_y = (first + cl_last_row[cl] + 1) / 2 * cell_h
        ax.text(n_cols * cell_w + 0.15, mid_y,
                _HBS_CL_NAMES[cl], ha="left", va="center",
                fontsize=7, color=_HBS_CL_COLORS[cl],
                fontweight="bold", rotation=90)

    ax.set_title(
        "Housing + Energy Overburden Rate by Decile\n"
        "Share of households (%) with shelter + energy > 40 % of net income  —  HBS 2020",
        fontsize=11, fontweight="bold", pad=6,
    )
    fname = "expense_he_overburden_deciles"
    fig.savefig(os.path.join(out_dir, fname + ".png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, fname + ".svg"),
                format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [HE] Saved: {fname}.png / .svg")


def _build_overburden_by_tenure_df(
    burden_cols: list, threshold: float, col_prefix: str
) -> pd.DataFrame:
    """
    For each country × decile: overburden rate split by tenure.

    Tenure classification (strict renter = only actual rent reported):
        renter : EUR_HE041 > 0  AND  EUR_HE042 == 0
        other  : everyone else  (owner-occupiers, dual-reporters, zero-rent HHs)

    Parameters
    ----------
    burden_cols : list of HBS column names to sum for the burden numerator
    threshold   : overburden threshold (fraction of net income)
    col_prefix  : short tag for output columns, e.g. "HE" or "HT"
    """
    records = []
    for cc, cname in HBS_COUNTRIES.items():
        raw_path = os.path.join(_RAW_CACHE_DIR, f"{cc}_hbs2020_raw.parquet")
        if not os.path.exists(raw_path):
            continue
        try:
            df = pd.read_parquet(raw_path)
        except Exception as exc:
            print(f"  [{cc}]  parquet read error: {exc}")
            continue

        need_cols = burden_cols + [
            "EUR_HE041", "EUR_HE042", _HBS_WEIGHT_COL, "EUR_HH099", "HB061"
        ]
        for col in need_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[
            df["EUR_HH099"].notna() & (df["EUR_HH099"] > 0) &
            df[_HBS_WEIGHT_COL].notna() & (df[_HBS_WEIGHT_COL] > 0)
        ].copy()
        if df.empty:
            continue

        # Tenure flag
        h41 = df["EUR_HE041"].fillna(0)
        h42 = df["EUR_HE042"].fillna(0)
        df["_is_renter"] = ((h41 > 0) & (h42 == 0))

        # Burden share
        # Note on units: both expenses (EUR_HExx) and income (EUR_HH099) are in
        # raw national EUR for the same household.  The PPP/PPS factor would be
        # identical for numerator and denominator and cancels in the ratio, so
        # the overburden share is currency-invariant.
        burden = pd.Series(0.0, index=df.index)
        for col in burden_cols:
            if col in df.columns:
                burden += df[col].fillna(0.0)
        df["_overburdened"] = (burden / df["EUR_HH099"] > threshold).astype(float)

        df = assign_simple_deciles(df, n_groups=10)
        if "income_decile" not in df.columns or df["income_decile"].notna().sum() == 0:
            continue

        row = {"Country": cc, "Country_Name": cname}
        for dlabel in _HT_DECILES:
            grp = df[df["income_decile"] == dlabel]
            for tenure_key, mask_col, val in [
                ("renter", True, None), ("other", False, None)
            ]:
                sub = grp[grp["_is_renter"] == (tenure_key == "renter")]
                key   = f"{col_prefix}_{dlabel}_{tenure_key}_pct"
                n_key = f"{col_prefix}_{dlabel}_{tenure_key}_n"
                if sub.empty or sub[_HBS_WEIGHT_COL].sum() == 0:
                    row[key]   = np.nan
                    row[n_key] = 0
                else:
                    wt = sub[_HBS_WEIGHT_COL].sum()
                    wo = (sub[_HBS_WEIGHT_COL] * sub["_overburdened"]).sum()
                    row[key]   = wo / wt * 100.0
                    row[n_key] = len(sub)
        records.append(row)
        print(f"  [{cc}]  tenure overburden ({col_prefix}) computed")

    return pd.DataFrame(records)


def _plot_overburden_by_tenure_heatmap(
    ob_df: pd.DataFrame,
    col_prefix: str,
    threshold_pct: int,
    burden_label: str,
    hdr_col: str,
    out_fname: str,
    cluster_map: dict,
    out_dir: str,
) -> None:
    """
    Side-by-side heatmap: left block = Renters, right block = Other (owners/mixed).
    Each block has 10 columns (D1–D10).  Shared colour scale across both blocks.

    Tenure definitions shown in subtitle:
        Renters : EUR_HE041 > 0 and EUR_HE042 == 0
        Other   : all remaining households
    """
    import matplotlib.colors as mcolors

    df = ob_df.copy()
    df["cl"] = df["Country"].map(lambda cc: cluster_map.get(cc, -1))
    has_cl = df[df["cl"] >= 0].sort_values(["cl", "Country_Name"])
    no_cl  = df[df["cl"] <  0].sort_values("Country_Name")
    df = pd.concat([has_cl, no_cl], ignore_index=True)
    n_rows = len(df)

    # Build column metadata for both tenure blocks
    renter_cols = [(d, f"{col_prefix}_{d}_renter_pct") for d in _HT_DECILES]
    other_cols  = [(d, f"{col_prefix}_{d}_other_pct")  for d in _HT_DECILES]
    blocks = [("Renters\n(actual rent only)", renter_cols, "#d6eaf8"),
              ("Other\n(owners / mixed)", other_cols, "#fde8d8")]

    # Shared colour scale
    all_vals = pd.concat(
        [df[k] for _, cols in [(None, renter_cols), (None, other_cols)]
         for _, k in cols]
    ).dropna()
    g_min = float(all_vals.min()) if len(all_vals) else 0.0
    g_max = float(all_vals.max()) if len(all_vals) else 100.0
    if g_max == g_min:
        g_max = g_min + 1.0

    cmap = plt.cm.RdYlGn_r

    cell_w   = 1.4
    cell_h   = 0.40
    gap_w    = 0.6    # gap between the two tenure blocks
    lmargin  = 3.0
    rmargin  = 1.8
    n_cols_each = len(_HT_DECILES)
    total_w  = n_cols_each * cell_w * 2 + gap_w

    fig, ax = plt.subplots(figsize=(lmargin + total_w + rmargin,
                                    n_rows * cell_h + 3.0))
    ax.set_xlim(-lmargin, total_w + rmargin)
    ax.set_ylim(n_rows * cell_h + 0.1, -3.0 * cell_h)
    ax.set_axis_off()

    x_offsets = [0, n_cols_each * cell_w + gap_w]

    # Draw headers for each block
    for bi, (blabel, cols, blk_bg) in enumerate(blocks):
        x0 = x_offsets[bi]
        block_w = n_cols_each * cell_w

        # Sub-headers row −1: D1…D10
        for ci, (d, _) in enumerate(cols):
            ax.add_patch(plt.Rectangle(
                (x0 + ci * cell_w, -cell_h), cell_w, cell_h,
                facecolor=blk_bg, edgecolor="white", linewidth=0.5, clip_on=False,
            ))
            ax.text(x0 + ci * cell_w + cell_w / 2, -cell_h / 2, d,
                    ha="center", va="center", fontsize=7, fontweight="bold")

        # Block label row −2
        ax.add_patch(plt.Rectangle(
            (x0, -2 * cell_h), block_w, cell_h,
            facecolor=hdr_col, edgecolor="white", linewidth=0.8, clip_on=False,
        ))
        # Single-line label stripped of newline for compact display
        ax.text(x0 + block_w / 2, -1.5 * cell_h,
                blabel.replace("\n", "  "),
                ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")

        # Row −3: overarching burden label (only first block gets the left part)
        ax.add_patch(plt.Rectangle(
            (x0, -3 * cell_h), block_w, cell_h,
            facecolor="#444444", edgecolor="white", linewidth=0.8, clip_on=False,
        ))
        if bi == 0:
            ax.text(x0 + block_w / 2, -2.5 * cell_h,
                    f"{burden_label} overburden  (> {threshold_pct} % of net income)",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white")
        # (right block row −3 filled but empty text — continuity)

    # Data cells
    prev_cl = None
    cl_first_row: dict = {}
    cl_last_row:  dict = {}

    for ri in range(n_rows):
        row = df.iloc[ri]
        cl  = int(row["cl"])
        y   = ri * cell_h

        if cl != prev_cl and prev_cl is not None:
            ax.axhline(y, color="#555555", linewidth=1.0, linestyle="--",
                       xmin=0, xmax=1)
        if cl not in cl_first_row:
            cl_first_row[cl] = ri
        cl_last_row[cl] = ri
        prev_cl = cl

        name_col = _HBS_CL_COLORS[cl] if 0 <= cl <= 3 else "#888888"
        ax.text(-0.12, y + cell_h / 2, row["Country_Name"],
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=name_col)

        for bi, (_, cols, _blk_bg) in enumerate(blocks):
            x0 = x_offsets[bi]
            for ci, (_, ckey) in enumerate(cols):
                val  = row[ckey]
                n_hh = row.get(ckey.replace("_pct", "_n"), _MIN_CELL_N)
                x    = x0 + ci * cell_w

                if not pd.isna(n_hh) and 0 < n_hh < _MIN_CELL_N:
                    face, txt, txt_col = "#cccccc", "n<5", "#666666"
                elif pd.isna(val):
                    face, txt, txt_col = "#e0e0e0", "—", "#999999"
                else:
                    nv   = max(0.0, min(1.0, (val - g_min) / (g_max - g_min)))
                    face = mcolors.to_hex(cmap(nv))
                    txt  = f"{val:.1f}%"
                    r_, g_, b_, _ = mcolors.to_rgba(face)
                    lum  = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                    txt_col = "white" if lum < 0.45 else "black"

                ax.add_patch(plt.Rectangle(
                    (x, y), cell_w, cell_h,
                    facecolor=face, edgecolor="white", linewidth=0.5,
                ))
                ax.text(x + cell_w / 2, y + cell_h / 2, txt,
                        ha="center", va="center", fontsize=6.5, color=txt_col)

    # Cluster labels in right margin (after second block)
    x_right = x_offsets[1] + n_cols_each * cell_w + 0.15
    for cl, first in sorted(cl_first_row.items()):
        if cl < 0:
            continue
        mid_y = (first + cl_last_row[cl] + 1) / 2 * cell_h
        ax.text(x_right, mid_y,
                _HBS_CL_NAMES[cl], ha="left", va="center",
                fontsize=7, color=_HBS_CL_COLORS[cl],
                fontweight="bold", rotation=90)

    ax.set_title(
        f"{burden_label} Overburden by Tenure and Income Decile  —  HBS 2020\n"
        "Renters: EUR_HE041 > 0 & EUR_HE042 = 0  |  Other: all remaining households",
        fontsize=10, fontweight="bold", pad=6,
    )
    fig.savefig(os.path.join(out_dir, out_fname + ".png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, out_fname + ".svg"),
                format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [tenure] Saved: {out_fname}.png / .svg")


def _plot_overburden_tenure_d1_heatmap(
    he_tenure_df: pd.DataFrame,
    cluster_map: dict,
    out_dir: str,
) -> None:
    """
    Combined D1 / D10 overburden heatmap for Housing + Energy.
    Rows = countries.  4 columns:
      [H+E Renters D1 | H+E Other D1 | H+E Renters D10 | H+E Other D10]
    Shared colour scale across all 4 columns.
    """
    import matplotlib.colors as mcolors

    # Pull D1 and D10 columns (pct + n) from he_tenure_df
    cols_needed = [
        "Country", "Country_Name",
        "HE_D1_renter_pct",  "HE_D1_renter_n",
        "HE_D1_other_pct",   "HE_D1_other_n",
        "HE_D10_renter_pct", "HE_D10_renter_n",
        "HE_D10_other_pct",  "HE_D10_other_n",
    ]
    avail = [c for c in cols_needed if c in he_tenure_df.columns]
    df = he_tenure_df[avail].copy()

    df["cl"] = df["Country"].map(lambda cc: cluster_map.get(cc, -1))
    has_cl = df[df["cl"] >= 0].sort_values(["cl", "Country_Name"])
    no_cl  = df[df["cl"] <  0].sort_values("Country_Name")
    df = pd.concat([has_cl, no_cl], ignore_index=True)
    n_rows = len(df)

    col_meta = [
        ("Renters",  "HE_D1_renter_pct",  "#1a4a6a",  "#d6eaf8"),
        ("Other",    "HE_D1_other_pct",   "#1a4a6a",  "#d6eaf8"),
        ("Renters",  "HE_D10_renter_pct", "#1a5a3a",  "#d5f0e4"),
        ("Other",    "HE_D10_other_pct",  "#1a5a3a",  "#d5f0e4"),
    ]
    n_cols = len(col_meta)  # 4

    # Shared colour scale
    all_vals = pd.concat([df[k] for _, k, _, _ in col_meta if k in df.columns]).dropna()
    g_min = float(all_vals.min()) if len(all_vals) else 0.0
    g_max = float(all_vals.max()) if len(all_vals) else 100.0
    if g_max == g_min:
        g_max = g_min + 1.0
    cmap = plt.cm.RdYlGn_r

    cell_w  = 2.0
    cell_h  = 0.40
    gap_w   = 0.5   # gap between D1 and D10 blocks
    lmargin = 3.0
    rmargin = 1.8
    x_off = [0, cell_w, 2 * cell_w + gap_w, 3 * cell_w + gap_w]
    total_w = 4 * cell_w + gap_w

    fig, ax = plt.subplots(figsize=(lmargin + total_w + rmargin,
                                    n_rows * cell_h + 3.0))
    ax.set_xlim(-lmargin, total_w + rmargin)
    ax.set_ylim(n_rows * cell_h + 0.1, -3.0 * cell_h)
    ax.set_axis_off()

    # Row −1: column sub-headers
    for ci, (lbl, _, hdr_c, bg_c) in enumerate(col_meta):
        ax.add_patch(plt.Rectangle(
            (x_off[ci], -cell_h), cell_w, cell_h,
            facecolor=bg_c, edgecolor="white", linewidth=0.5, clip_on=False,
        ))
        ax.text(x_off[ci] + cell_w / 2, -cell_h / 2, lbl,
                ha="center", va="center", fontsize=7, fontweight="bold")

    # Row −2: block labels (D1 and D10)
    for x0, label, hdr_c, bw in [
        (x_off[0], f"D1  —  H+E > {int(_OVERBURDEN_THRESHOLD_HE*100)} %", "#1a4a6a", 2 * cell_w),
        (x_off[2], f"D10  —  H+E > {int(_OVERBURDEN_THRESHOLD_HE*100)} %", "#1a5a3a", 2 * cell_w),
    ]:
        ax.add_patch(plt.Rectangle(
            (x0, -2 * cell_h), bw, cell_h,
            facecolor=hdr_c, edgecolor="white", linewidth=0.8, clip_on=False,
        ))
        ax.text(x0 + bw / 2, -1.5 * cell_h, label,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")

    # Row −3: overall title bar
    ax.add_patch(plt.Rectangle(
        (x_off[0], -3 * cell_h), total_w, cell_h,
        facecolor="#333333", edgecolor="white", linewidth=0.8, clip_on=False,
    ))
    ax.text(total_w / 2, -2.5 * cell_h,
            "Housing + Energy Overburden  |  D1 vs D10  |  Renters vs Other  —  HBS 2020",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color="white")

    # Data cells
    prev_cl = None
    cl_first_row: dict = {}
    cl_last_row:  dict = {}

    for ri in range(n_rows):
        row = df.iloc[ri]
        cl  = int(row["cl"])
        y   = ri * cell_h

        if cl != prev_cl and prev_cl is not None:
            ax.axhline(y, color="#555555", linewidth=1.0, linestyle="--")
        if cl not in cl_first_row:
            cl_first_row[cl] = ri
        cl_last_row[cl] = ri
        prev_cl = cl

        name_col = _HBS_CL_COLORS[cl] if 0 <= cl <= 3 else "#888888"
        ax.text(-0.12, y + cell_h / 2, row["Country_Name"],
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=name_col)

        for ci, (_, ckey, _, _) in enumerate(col_meta):
            val  = row.get(ckey, np.nan)
            n_hh = row.get(ckey.replace("_pct", "_n"), _MIN_CELL_N)
            x    = x_off[ci]

            if not pd.isna(n_hh) and 0 < n_hh < _MIN_CELL_N:
                face, txt, txt_col = "#cccccc", "n<5", "#666666"
            elif pd.isna(val):
                face, txt, txt_col = "#e0e0e0", "—", "#999999"
            else:
                nv   = max(0.0, min(1.0, (val - g_min) / (g_max - g_min)))
                face = mcolors.to_hex(cmap(nv))
                txt  = f"{val:.1f}%"
                r_, g_, b_, _ = mcolors.to_rgba(face)
                lum  = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                txt_col = "white" if lum < 0.45 else "black"

            ax.add_patch(plt.Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=face, edgecolor="white", linewidth=0.5,
            ))
            ax.text(x + cell_w / 2, y + cell_h / 2, txt,
                    ha="center", va="center", fontsize=7, color=txt_col)

    # Cluster labels in right margin
    x_right = total_w + 0.15
    for cl, first in sorted(cl_first_row.items()):
        if cl < 0:
            continue
        mid_y = (first + cl_last_row[cl] + 1) / 2 * cell_h
        ax.text(x_right, mid_y,
                _HBS_CL_NAMES[cl], ha="left", va="center",
                fontsize=7, color=_HBS_CL_COLORS[cl],
                fontweight="bold", rotation=90)

    ax.set_title(
        "Housing + Energy Overburden — D1 vs D10 by Tenure  —  HBS 2020\n"
        "Renters: EUR_HE041 > 0 & EUR_HE042 = 0  |  Other: all remaining households",
        fontsize=10, fontweight="bold", pad=6,
    )
    fname = "expense_overburden_d1_d10_by_tenure"
    fig.savefig(os.path.join(out_dir, fname + ".png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, fname + ".svg"),
                format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [D1D10] Saved: {fname}.png / .svg")


# Age-group labels and their classification logic (based on HBS household cols)
# HB055 = persons aged 16-24 who are students
# HB056 = persons aged 25-64
# HB057 = persons aged 65+
# Household age flag = "maximum" age tier present
_AGE_GROUPS = [
    ("65+",   "65p",  "#6a3d9a"),   # label, col_key_tag, header colour
    ("25–64", "2564", "#1f78b4"),
    ("16–24\n(student)", "1624", "#33a02c"),
]


def _assign_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a '_age_group' column based on the maximum age tier present
    in the household:
        65+        if HB057 > 0
        25-64      elif HB056 > 0
        16-24      elif HB055 > 0   (students)
        unclassed  otherwise        (dropped downstream)
    """
    df = df.copy()
    for col in ["HB055", "HB056", "HB057"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    conditions = [
        df["HB057"] > 0,
        (df["HB057"] == 0) & (df["HB056"] > 0),
        (df["HB057"] == 0) & (df["HB056"] == 0) & (df["HB055"] > 0),
    ]
    choices = ["65p", "2564", "1624"]
    df["_age_group"] = np.select(conditions, choices, default="unclassed")
    return df


def _build_he_overburden_tenure_age_df() -> pd.DataFrame:
    """
    For each country: H+E overburden rate (> 40 % of net income) broken down
    by the 2 × 3 = 6 cells of tenure × age-group.

    Columns returned:
        HE_{age_tag}_{tenure}_pct   e.g.  HE_65p_renter_pct, HE_2564_other_pct …

    Tenure:
        renter  → EUR_HE041 > 0  AND  EUR_HE042 == 0
        other   → all remaining households

    Age group (maximum age tier present):
        65p   → HB057 > 0
        2564  → HB057 == 0 AND HB056 > 0
        1624  → HB057 == 0 AND HB056 == 0 AND HB055 > 0
    """
    he_cols = _HBS_SHELTER_COLS + [_HBS_ENERGY_COL]
    records = []
    for cc, cname in HBS_COUNTRIES.items():
        raw_path = os.path.join(_RAW_CACHE_DIR, f"{cc}_hbs2020_raw.parquet")
        if not os.path.exists(raw_path):
            continue
        try:
            df = pd.read_parquet(raw_path)
        except Exception as exc:
            print(f"  [{cc}]  parquet read error: {exc}")
            continue

        need = he_cols + ["EUR_HE041", "EUR_HE042",
                          _HBS_WEIGHT_COL, "EUR_HH099", "HB061",
                          "HB055", "HB056", "HB057"]
        for col in need:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[
            df["EUR_HH099"].notna() & (df["EUR_HH099"] > 0) &
            df[_HBS_WEIGHT_COL].notna() & (df[_HBS_WEIGHT_COL] > 0)
        ].copy()
        if df.empty:
            continue

        # Tenure
        h41 = df["EUR_HE041"].fillna(0)
        h42 = df["EUR_HE042"].fillna(0)
        df["_is_renter"] = (h41 > 0) & (h42 == 0)

        # Burden
        burden = pd.Series(0.0, index=df.index)
        for col in he_cols:
            if col in df.columns:
                burden += df[col].fillna(0.0)
        df["_overburdened"] = (burden / df["EUR_HH099"] > _OVERBURDEN_THRESHOLD_HE).astype(float)

        # Age group
        df = _assign_age_group(df)
        df = df[df["_age_group"] != "unclassed"].copy()
        if df.empty:
            continue

        row = {"Country": cc, "Country_Name": cname}
        for _, age_tag, _ in _AGE_GROUPS:
            age_sub = df[df["_age_group"] == age_tag]
            for tenure_tag, is_renter in [("renter", True), ("other", False)]:
                key   = f"HE_{age_tag}_{tenure_tag}_pct"
                n_key = f"HE_{age_tag}_{tenure_tag}_n"
                sub = age_sub[age_sub["_is_renter"] == is_renter]
                if sub.empty or sub[_HBS_WEIGHT_COL].sum() == 0:
                    row[key]   = np.nan
                    row[n_key] = 0
                else:
                    wt = sub[_HBS_WEIGHT_COL].sum()
                    wo = (sub[_HBS_WEIGHT_COL] * sub["_overburdened"]).sum()
                    row[key]   = wo / wt * 100.0
                    row[n_key] = len(sub)
        records.append(row)
        print(f"  [{cc}]  H+E tenure×age overburden computed")

    return pd.DataFrame(records)


def _plot_he_overburden_tenure_age_heatmap(
    df_in: pd.DataFrame, cluster_map: dict, out_dir: str
) -> None:
    """
    Heatmap — rows = countries, columns = 6 cells:
        [Renters 65+ | Other 65+]  [Renters 25-64 | Other 25-64]  [Renters 16-24 | Other 16-24]
    Three age-group blocks separated by a small gap.
    Shared colour scale across all 6 columns.
    """
    import matplotlib.colors as mcolors

    df = df_in.copy()
    df["cl"] = df["Country"].map(lambda cc: cluster_map.get(cc, -1))
    has_cl = df[df["cl"] >= 0].sort_values(["cl", "Country_Name"])
    no_cl  = df[df["cl"] <  0].sort_values("Country_Name")
    df = pd.concat([has_cl, no_cl], ignore_index=True)
    n_rows = len(df)

    # Build column metadata: list of (sub-label, col_key, hdr_color, bg_color)
    col_meta = []
    for age_lbl, age_tag, age_col in _AGE_GROUPS:
        for tenure_lbl, tenure_tag in [("Renters", "renter"), ("Other", "other")]:
            col_meta.append((
                f"{tenure_lbl}",
                f"HE_{age_tag}_{tenure_tag}_pct",
                age_col,
                mcolors.to_hex(mcolors.to_rgba(age_col, alpha=0.18)[:3]),  # light tint
            ))

    # Shared colour scale
    all_vals = pd.concat([df[k] for _, k, _, _ in col_meta]).dropna()
    g_min = float(all_vals.min()) if len(all_vals) else 0.0
    g_max = float(all_vals.max()) if len(all_vals) else 100.0
    if g_max == g_min:
        g_max = g_min + 1.0
    cmap = plt.cm.RdYlGn_r

    cell_w   = 1.6
    cell_h   = 0.40
    gap_w    = 0.5   # gap between age-group blocks
    lmargin  = 3.0
    rmargin  = 1.8
    n_age    = len(_AGE_GROUPS)     # 3
    n_tenure = 2
    # x offset for each column
    x_off = []
    for ai in range(n_age):
        for ti in range(n_tenure):
            x_off.append(ai * (n_tenure * cell_w + gap_w) + ti * cell_w)
    total_w = n_age * n_tenure * cell_w + (n_age - 1) * gap_w

    fig, ax = plt.subplots(figsize=(lmargin + total_w + rmargin,
                                    n_rows * cell_h + 3.5))
    ax.set_xlim(-lmargin, total_w + rmargin)
    ax.set_ylim(n_rows * cell_h + 0.1, -3.5 * cell_h)
    ax.set_axis_off()

    # Row −1: tenure sub-headers per column
    for ci, (lbl, _, hdr_c, bg_c) in enumerate(col_meta):
        # light background using alpha blend with white
        r_, g_, b_ = mcolors.to_rgb(hdr_c)
        light = tuple(0.18 * v + 0.82 for v in (r_, g_, b_))
        ax.add_patch(plt.Rectangle(
            (x_off[ci], -cell_h), cell_w, cell_h,
            facecolor=light, edgecolor="white", linewidth=0.5, clip_on=False,
        ))
        ax.text(x_off[ci] + cell_w / 2, -cell_h / 2, lbl,
                ha="center", va="center", fontsize=7, fontweight="bold")

    # Row −2: age-group block headers
    for ai, (age_lbl, age_tag, age_col) in enumerate(_AGE_GROUPS):
        bx = ai * (n_tenure * cell_w + gap_w)
        bw = n_tenure * cell_w
        ax.add_patch(plt.Rectangle(
            (bx, -2 * cell_h), bw, cell_h,
            facecolor=age_col, edgecolor="white", linewidth=0.8, clip_on=False,
        ))
        ax.text(bx + bw / 2, -1.5 * cell_h,
                age_lbl.replace("\n", " "),
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")

    # Row −3: overall title bar
    ax.add_patch(plt.Rectangle(
        (0, -3 * cell_h), total_w, cell_h,
        facecolor="#1a4a6a", edgecolor="white", linewidth=0.8, clip_on=False,
    ))
    ax.text(total_w / 2, -2.5 * cell_h,
            f"Housing + Energy overburden (> {int(_OVERBURDEN_THRESHOLD_HE*100)} % of net income)"
            "  |  Tenure × Age group  —  HBS 2020",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color="white")

    # Data cells
    prev_cl = None
    cl_first_row: dict = {}
    cl_last_row:  dict = {}

    for ri in range(n_rows):
        row = df.iloc[ri]
        cl  = int(row["cl"])
        y   = ri * cell_h

        if cl != prev_cl and prev_cl is not None:
            ax.axhline(y, color="#555555", linewidth=1.0, linestyle="--")
        if cl not in cl_first_row:
            cl_first_row[cl] = ri
        cl_last_row[cl] = ri
        prev_cl = cl

        name_col = _HBS_CL_COLORS[cl] if 0 <= cl <= 3 else "#888888"
        ax.text(-0.12, y + cell_h / 2, row["Country_Name"],
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=name_col)

        for ci, (_, ckey, hdr_c, _bg) in enumerate(col_meta):
            val  = row[ckey]
            n_hh = row.get(ckey.replace("_pct", "_n"), _MIN_CELL_N)
            x    = x_off[ci]

            if not pd.isna(n_hh) and 0 < n_hh < _MIN_CELL_N:
                face, txt, txt_col = "#cccccc", "n<5", "#666666"
            elif pd.isna(val):
                face, txt, txt_col = "#e0e0e0", "—", "#999999"
            else:
                nv   = max(0.0, min(1.0, (val - g_min) / (g_max - g_min)))
                face = mcolors.to_hex(cmap(nv))
                txt  = f"{val:.1f}%"
                r_, g_, b_, _ = mcolors.to_rgba(face)
                lum  = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                txt_col = "white" if lum < 0.45 else "black"

            ax.add_patch(plt.Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=face, edgecolor="white", linewidth=0.5,
            ))
            ax.text(x + cell_w / 2, y + cell_h / 2, txt,
                    ha="center", va="center", fontsize=6.5, color=txt_col)

    # Cluster labels in right margin
    x_right = total_w + 0.15
    for cl, first in sorted(cl_first_row.items()):
        if cl < 0:
            continue
        mid_y = (first + cl_last_row[cl] + 1) / 2 * cell_h
        ax.text(x_right, mid_y,
                _HBS_CL_NAMES[cl], ha="left", va="center",
                fontsize=7, color=_HBS_CL_COLORS[cl],
                fontweight="bold", rotation=90)

    ax.set_title(
        "Housing + Energy Overburden by Tenure and Household Age Group  —  HBS 2020\n"
        "Age = maximum tier present in HH  |  Renters: EUR_HE041 > 0 & EUR_HE042 = 0",
        fontsize=10, fontweight="bold", pad=6,
    )
    fname = "expense_he_overburden_tenure_age"
    fig.savefig(os.path.join(out_dir, fname + ".png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, fname + ".svg"),
                format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [TA] Saved: {fname}.png / .svg")


# ─────────────────────────────────────────────────────────────────────────────
# Expense share of net income — heatmap + EWBI scatter (generic)
# ─────────────────────────────────────────────────────────────────────────────

def _build_component_share_df(
    expense_cols: list, col_prefix: str, max_share: float = 200.0
) -> pd.DataFrame:
    """
    For each country × decile: weighted mean of (sum(expense_cols) / net income) × 100.
    Also computes the country-level overall mean ("All").
    Households with share > max_share are dropped (near-zero income artefacts).

    Returns columns: Country, Country_Name,
        {col_prefix}_D1_pct … {col_prefix}_D10_pct,
        {col_prefix}_D1_n  … {col_prefix}_D10_n,
        {col_prefix}_All_pct, {col_prefix}_All_n
    """
    records = []
    for cc, cname in HBS_COUNTRIES.items():
        raw_path = os.path.join(_RAW_CACHE_DIR, f"{cc}_hbs2020_raw.parquet")
        if not os.path.exists(raw_path):
            continue
        try:
            df = pd.read_parquet(raw_path)
        except Exception as exc:
            print(f"  [{cc}]  parquet read error: {exc}")
            continue

        for col in expense_cols + [_HBS_WEIGHT_COL, "EUR_HH099"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[
            df["EUR_HH099"].notna() & (df["EUR_HH099"] > 0) &
            df[_HBS_WEIGHT_COL].notna() & (df[_HBS_WEIGHT_COL] > 0)
        ].copy()
        if df.empty:
            continue

        expense = pd.Series(0.0, index=df.index)
        for col in expense_cols:
            if col in df.columns:
                expense += df[col].fillna(0.0)
        df["_share"] = expense / df["EUR_HH099"] * 100.0  # %

        df = df[df["_share"] <= max_share].copy()
        if df.empty:
            continue

        df = assign_simple_deciles(df, n_groups=10)
        if "income_decile" not in df.columns or df["income_decile"].notna().sum() == 0:
            continue

        row = {"Country": cc, "Country_Name": cname}
        for dlabel in _HT_DECILES:
            grp = df[df["income_decile"] == dlabel]
            key   = f"{col_prefix}_{dlabel}_pct"
            n_key = f"{col_prefix}_{dlabel}_n"
            w = grp[_HBS_WEIGHT_COL].sum() if not grp.empty else 0
            if w == 0:
                row[key]   = np.nan
                row[n_key] = 0
            else:
                row[key]   = (grp[_HBS_WEIGHT_COL] * grp["_share"]).sum() / w
                row[n_key] = len(grp)
        w_all = df[_HBS_WEIGHT_COL].sum()
        row[f"{col_prefix}_All_pct"] = (
            (df[_HBS_WEIGHT_COL] * df["_share"]).sum() / w_all
            if w_all > 0 else np.nan
        )
        row[f"{col_prefix}_All_n"] = len(df)
        records.append(row)
        print(f"  [{cc}]  {col_prefix} share computed")

    return pd.DataFrame(records)


def _plot_component_share_heatmap(
    share_df: pd.DataFrame,
    col_prefix: str,
    title_label: str,
    cluster_map: dict,
    out_dir: str,
    fname: str,
) -> None:
    """
    Heatmap rows = countries, columns = D1–D10.
    Cell = weighted mean share of net income (%) for *title_label* expenditure.
    Shared colour scale (RdYlGn_r: high share = red).
    """
    import matplotlib.colors as mcolors

    df = share_df.copy()
    df["cl"] = df["Country"].map(lambda cc: cluster_map.get(cc, -1))
    has_cl = df[df["cl"] >= 0].sort_values(["cl", "Country_Name"])
    no_cl  = df[df["cl"] <  0].sort_values("Country_Name")
    df = pd.concat([has_cl, no_cl], ignore_index=True)
    n_rows = len(df)

    col_meta = [(d, f"{col_prefix}_{d}_pct") for d in _HT_DECILES]
    n_cols   = len(col_meta)  # 10

    all_vals = pd.concat([df[k] for _, k in col_meta]).dropna()
    g_min = float(all_vals.min()) if len(all_vals) else 0.0
    g_max = float(all_vals.max()) if len(all_vals) else 1.0
    if g_max == g_min:
        g_max = g_min + 1.0

    cmap    = plt.cm.RdYlGn_r
    hdr_bg  = "#c8d8e8"
    hdr_col = "#1a3a5a"
    cell_w  = 1.4
    cell_h  = 0.40
    lmargin = 3.0
    rmargin = 1.8

    fig, ax = plt.subplots(figsize=(lmargin + n_cols * cell_w + rmargin,
                                    n_rows * cell_h + 2.5))
    ax.set_xlim(-lmargin, n_cols * cell_w + rmargin)
    ax.set_ylim(n_rows * cell_h + 0.1, -2.5 * cell_h)
    ax.set_axis_off()

    # Decile sub-headers
    for ci, (d, _) in enumerate(col_meta):
        ax.add_patch(plt.Rectangle(
            (ci * cell_w, -cell_h), cell_w, cell_h,
            facecolor=hdr_bg, edgecolor="white", linewidth=0.5, clip_on=False,
        ))
        ax.text(ci * cell_w + cell_w / 2, -cell_h / 2, d,
                ha="center", va="center", fontsize=7, fontweight="bold")

    # Block header
    ax.add_patch(plt.Rectangle(
        (0, -2 * cell_h), n_cols * cell_w, cell_h,
        facecolor=hdr_col, edgecolor="white", linewidth=0.8, clip_on=False,
    ))
    ax.text(n_cols * cell_w / 2, -1.5 * cell_h, title_label,
            ha="center", va="center", fontsize=12,
            fontweight="bold", color="white")

    prev_cl = None
    cl_first_row: dict = {}
    cl_last_row:  dict = {}

    for ri in range(n_rows):
        row = df.iloc[ri]
        cl  = int(row["cl"])
        y   = ri * cell_h

        if cl != prev_cl and prev_cl is not None:
            ax.axhline(y, color="#555555", linewidth=1.0, linestyle="--")
        if cl not in cl_first_row:
            cl_first_row[cl] = ri
        cl_last_row[cl] = ri
        prev_cl = cl

        name_col = _HBS_CL_COLORS[cl] if 0 <= cl <= 3 else "#888888"
        ax.text(-0.12, y + cell_h / 2, row["Country_Name"],
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=name_col)

        for ci, (d_ci, ckey) in enumerate(col_meta):
            val  = row[ckey]
            n_hh = row.get(f"{col_prefix}_{d_ci}_n", _MIN_CELL_N)
            x    = ci * cell_w

            if not pd.isna(n_hh) and 0 < n_hh < _MIN_CELL_N:
                face, txt, txt_col = "#cccccc", "n<5", "#666666"
            elif pd.isna(val):
                face, txt, txt_col = "#e0e0e0", "—", "#999999"
            else:
                nv   = max(0.0, min(1.0, (val - g_min) / (g_max - g_min)))
                face = mcolors.to_hex(cmap(nv))
                txt  = f"{val:.1f}%"
                r_, g_, b_, _ = mcolors.to_rgba(face)
                lum  = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                txt_col = "white" if lum < 0.45 else "black"

            ax.add_patch(plt.Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=face, edgecolor="white", linewidth=0.5,
            ))
            ax.text(x + cell_w / 2, y + cell_h / 2, txt,
                    ha="center", va="center", fontsize=6.5, color=txt_col)

    for cl, first in sorted(cl_first_row.items()):
        if cl < 0:
            continue
        mid_y = (first + cl_last_row[cl] + 1) / 2 * cell_h
        ax.text(n_cols * cell_w + 0.15, mid_y,
                _HBS_CL_NAMES[cl], ha="left", va="center",
                fontsize=7, color=_HBS_CL_COLORS[cl],
                fontweight="bold", rotation=90)

    ax.set_title(
        f"{title_label} — % of net income, by Decile D1–D10\n"
        "HBS 2020  |  EU-27 countries  |  shared colour scale",
        fontsize=11, fontweight="bold", pad=6,
    )
    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"{fname}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white",
                    format=ext if ext == "svg" else None)
        print(f"  [{col_prefix}] Saved: {os.path.basename(path)}")
    plt.close(fig)


def _plot_share_vs_ewbi_scatter(
    share_df: pd.DataFrame,
    col_prefix: str,
    ewbi_priority_name: str,
    burden_label: str,
    cluster_map: dict,
    out_dir: str,
    fname: str,
) -> None:
    """
    Three-panel scatter (D1 | D10 | All Deciles).
    X = weighted mean share of net income (%) for *burden_label* expenditure.
    Y = EWBI *ewbi_priority_name* priority score (0–1, per decile / all deciles).
    Points coloured by cluster; country codes annotated; regression line + R².
    """
    ewbi_path = os.path.abspath(
        os.path.join(CURRENT_DIR, "..", "..", "..", "output", "ewbi_master_aggregated.csv")
    )
    if not os.path.exists(ewbi_path):
        print(f"  [{col_prefix} scatter] EWBI file not found: {ewbi_path}")
        return

    ewbi = pd.read_csv(ewbi_path, low_memory=False)
    ewbi = ewbi[
        (ewbi["EU priority"] == ewbi_priority_name) &
        (ewbi["Level"] == 2) &
        (ewbi["Year"] == 2020)
    ].copy()
    ewbi["_dec_num"] = pd.to_numeric(ewbi["Decile"], errors="coerce")

    def _get_ewbi(dec):
        if dec == "All":
            sub = ewbi[ewbi["Decile"] == "All Deciles"]
        else:
            sub = ewbi[ewbi["_dec_num"] == float(dec)]
        return dict(zip(sub["Country"], sub["Value"]))

    df = share_df.copy()
    df["cl"] = df["Country"].map(lambda cc: cluster_map.get(cc, -1))

    panels = [
        ("Bottom decile (D1)", f"{col_prefix}_D1_pct",  _get_ewbi(1)),
        ("Top decile (D10)",   f"{col_prefix}_D10_pct", _get_ewbi(10)),
        ("All deciles",        f"{col_prefix}_All_pct", _get_ewbi("All")),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"{burden_label} Share of Net Income vs. EWBI {ewbi_priority_name} Priority"
        "  (HBS 2020 / EU-SILC 2020)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for ax, (title, x_col, ewbi_map) in zip(axes, panels):
        rows = []
        for _, r in df.iterrows():
            score = ewbi_map.get(r["Country"])
            if score is None or pd.isna(r[x_col]):
                continue
            rows.append({"Country": r["Country"], "cl": r["cl"],
                         "x": r[x_col], "y": score})
        sub = pd.DataFrame(rows)

        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue

        for _, row in sub.iterrows():
            cl    = int(row["cl"])
            color = _HBS_CL_COLORS[cl] if 0 <= cl <= 3 else "#888888"
            ax.scatter(row["x"], row["y"], color=color,
                       s=80, edgecolors="white", linewidths=0.6, zorder=3)
            ax.annotate(row["Country"],
                        xy=(row["x"], row["y"]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=7.5, color="#333333", zorder=4)

        x_vals = sub["x"].values.astype(float)
        y_vals = sub["y"].values.astype(float)
        if len(x_vals) >= 3:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            y_pred = np.polyval(coeffs, x_vals)
            ss_res = np.sum((y_vals - y_pred) ** 2)
            ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            x_fit  = np.linspace(x_vals.min(), x_vals.max(), 200)
            ax.plot(x_fit, np.polyval(coeffs, x_fit),
                    color="#555555", linewidth=1.3, linestyle="--", zorder=2)
            ax.text(0.97, 0.05, f"R² = {r2:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color="#555555")

        ax.set_xlabel(f"{burden_label} share of net income  (%)", fontsize=10)
        ax.set_ylabel(f"EWBI {ewbi_priority_name} score\n(0–1, higher = better well-being)",
                      fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_HBS_CL_COLORS[i], label=_HBS_CL_NAMES[i])
        for i in range(4)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"{fname}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  [{col_prefix} scatter] saved {os.path.basename(path)}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# EWBI Energy & Housing vs overburden — scatter plot (H+E and H+E+T)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ewbi_vs_overburden_df(
    ob_df: pd.DataFrame, col_prefix: str
) -> pd.DataFrame:
    """
    Merge overburden rates (D1 & D10, column prefix *col_prefix*) with the
    EWBI Energy and Housing priority score for year 2020, matching deciles.

    *col_prefix* is "HE" for Housing+Energy or "OB" for Housing+Energy+Transport.

    Returns one row per country with columns:
        Country, Country_Name, ob_D1, ob_D10, ewbi_D1, ewbi_D10
    """
    ewbi_path = os.path.abspath(
        os.path.join(CURRENT_DIR, "..", "..", "..", "output", "ewbi_master_aggregated.csv")
    )
    if not os.path.exists(ewbi_path):
        print(f"  [EWBI scatter] file not found: {ewbi_path}")
        return pd.DataFrame()

    ewbi = pd.read_csv(ewbi_path, low_memory=False)
    ewbi = ewbi[
        (ewbi["EU priority"] == "Energy and Housing") &
        (ewbi["Level"] == 2) &
        (ewbi["Year"] == 2020)
    ].copy()
    ewbi["Decile"] = pd.to_numeric(ewbi["Decile"], errors="coerce")

    d1_ewbi  = (ewbi[ewbi["Decile"] == 1.0][["Country", "Value"]]
                .rename(columns={"Value": "ewbi_D1"}))
    d10_ewbi = (ewbi[ewbi["Decile"] == 10.0][["Country", "Value"]]
                .rename(columns={"Value": "ewbi_D10"}))

    d1_col  = f"{col_prefix}_D1_pct"
    d10_col = f"{col_prefix}_D10_pct"
    ob = ob_df[["Country", "Country_Name", d1_col, d10_col]].copy()
    ob = ob.rename(columns={d1_col: "ob_D1", d10_col: "ob_D10"})

    merged = ob.merge(d1_ewbi, on="Country", how="inner")
    merged = merged.merge(d10_ewbi, on="Country", how="inner")
    return merged


def _plot_ewbi_vs_overburden_scatter(
    scatter_df: pd.DataFrame,
    burden_label: str,
    fname: str,
    cluster_map: dict,
    out_dir: str,
) -> None:
    """
    Two-panel scatter: left = bottom decile (D1), right = top decile (D10).
    X = overburden rate (%), Y = EWBI Energy & Housing priority score (0–1).
    Points coloured by cluster; country codes annotated; regression line + R².

    *burden_label* appears in the title and x-axis label (e.g. "H+E" or "H+E+T").
    """
    df = scatter_df.copy()
    df["cl"] = df["Country"].map(lambda cc: cluster_map.get(cc, -1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{burden_label} Overburden vs. EWBI Energy & Housing Priority"
        "  (HBS 2020 / EU-SILC 2020)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    panels = [
        ("Bottom income decile (D1)", "ob_D1",  "ewbi_D1"),
        ("Top income decile (D10)",   "ob_D10", "ewbi_D10"),
    ]

    for ax, (title, x_col, y_col) in zip(axes, panels):
        sub = df[[x_col, y_col, "Country", "cl"]].dropna()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue

        # ── scatter points ────────────────────────────────────────────────
        for _, row in sub.iterrows():
            cl    = int(row["cl"])
            color = _HBS_CL_COLORS[cl] if 0 <= cl <= 3 else "#888888"
            ax.scatter(row[x_col], row[y_col], color=color,
                       s=80, edgecolors="white", linewidths=0.6, zorder=3)
            ax.annotate(
                row["Country"],
                xy=(row[x_col], row[y_col]),
                xytext=(4, 3), textcoords="offset points",
                fontsize=7.5, color="#333333", zorder=4,
            )

        # ── OLS regression line ───────────────────────────────────────────
        x_vals = sub[x_col].values.astype(float)
        y_vals = sub[y_col].values.astype(float)
        if len(x_vals) >= 3:
            coeffs           = np.polyfit(x_vals, y_vals, 1)
            y_pred           = np.polyval(coeffs, x_vals)
            ss_res           = np.sum((y_vals - y_pred) ** 2)
            ss_tot           = np.sum((y_vals - y_vals.mean()) ** 2)
            r2               = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            x_fit            = np.linspace(x_vals.min(), x_vals.max(), 200)
            ax.plot(x_fit, np.polyval(coeffs, x_fit),
                    color="#555555", linewidth=1.3, linestyle="--", zorder=2)
            ax.text(0.97, 0.05, f"R² = {r2:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color="#555555")

        # ── axes formatting ───────────────────────────────────────────────
        ax.set_xlabel(f"{burden_label} overburden rate  (%)", fontsize=10)
        ax.set_ylabel("EWBI Energy & Housing score\n(0–1, higher = better well-being)",
                      fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    # ── cluster legend ────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_HBS_CL_COLORS[i], label=_HBS_CL_NAMES[i])
        for i in range(4)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"{fname}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  [scatter] saved {os.path.basename(path)}")
    plt.close(fig)


def generate_snapshot_heatmaps(all_comps: dict) -> None:
    """Build the EU expenditure snapshot heatmaps from loaded component data."""
    print("\n--- Building expenditure snapshot heatmaps ---")
    cluster_map = _hbs_cluster_map()
    snap_df     = _build_expense_snapshot_df(all_comps)
    if snap_df.empty:
        print("  No data — skipped")
        return
    _plot_expense_heatmap(snap_df, "pct", cluster_map, PER_COUNTRY_DIR)
    _plot_expense_heatmap(snap_df, "pps", cluster_map, PER_COUNTRY_DIR)

    # Housing, Energy & Transport across all 10 deciles — % and absolute PPS
    ht_df = _build_ht_decile_df(all_comps)
    if not ht_df.empty:
        _plot_ht_decile_heatmap(ht_df, cluster_map, PER_COUNTRY_DIR, kind="pct")
        _plot_ht_decile_heatmap(ht_df, cluster_map, PER_COUNTRY_DIR, kind="pps")

    # H+E+T overburden rate (requires raw household-level parquet cache)
    print("\n--- Building H+E+T overburden heatmap (from raw parquet) ---")
    ob_df = _build_ht_overburden_df()
    if not ob_df.empty:
        _plot_ht_overburden_heatmap(ob_df, cluster_map, PER_COUNTRY_DIR)
        # EWBI Energy & Housing vs H+E+T overburden scatter
        print("\n--- Building EWBI vs H+E+T overburden scatter plot ---")
        het_scatter_df = _build_ewbi_vs_overburden_df(ob_df, col_prefix="OB")
        if not het_scatter_df.empty:
            _plot_ewbi_vs_overburden_scatter(
                het_scatter_df, "H+E+T",
                "ewbi_vs_het_overburden_scatter",
                cluster_map, PER_COUNTRY_DIR)
        else:
            print("  No matching EWBI data — H+E+T scatter skipped")
    else:
        print("  No raw parquet files found — run 0_preprocess_hbs_cache.py first")

    # Housing+Energy overburden rate at 40 %
    print("\n--- Building Housing+Energy overburden heatmap (from raw parquet) ---")
    he_df = _build_he_overburden_df()
    if not he_df.empty:
        _plot_he_overburden_heatmap(he_df, cluster_map, PER_COUNTRY_DIR)
        # EWBI Energy & Housing vs H+E overburden scatter
        print("\n--- Building EWBI vs H+E overburden scatter plot ---")
        he_scatter_df = _build_ewbi_vs_overburden_df(he_df, col_prefix="HE")
        if not he_scatter_df.empty:
            _plot_ewbi_vs_overburden_scatter(
                he_scatter_df, "H+E",
                "ewbi_vs_he_overburden_scatter",
                cluster_map, PER_COUNTRY_DIR)
        else:
            print("  No matching EWBI data — H+E scatter skipped")
    else:
        print("  No raw parquet files found — run 0_preprocess_hbs_cache.py first")

    # Overburden by tenure (renters vs other) — H+E and H+E+T
    print("\n--- Building overburden-by-tenure heatmaps (from raw parquet) ---")
    he_tenure_df = _build_overburden_by_tenure_df(
        _HBS_SHELTER_COLS + [_HBS_ENERGY_COL], _OVERBURDEN_THRESHOLD_HE, "HE")
    if not he_tenure_df.empty:
        _plot_overburden_by_tenure_heatmap(
            he_tenure_df, "HE", int(_OVERBURDEN_THRESHOLD_HE * 100),
            "Housing + Energy", "#1a4a6a",
            "expense_he_overburden_by_tenure", cluster_map, PER_COUNTRY_DIR)
    ht_tenure_df = _build_overburden_by_tenure_df(
        _HBS_HOUSING_COLS + _HBS_TRANSPORT_COLS, _OVERBURDEN_THRESHOLD, "HT")
    if not ht_tenure_df.empty:
        _plot_overburden_by_tenure_heatmap(
            ht_tenure_df, "HT", int(_OVERBURDEN_THRESHOLD * 100),
            "Housing + Energy + Transport", "#8B2500",
            "expense_het_overburden_by_tenure", cluster_map, PER_COUNTRY_DIR)

    # D1 vs D10 snapshot: H+E by tenure
    if not he_tenure_df.empty:
        _plot_overburden_tenure_d1_heatmap(
            he_tenure_df, cluster_map, PER_COUNTRY_DIR)

    # H+E overburden by tenure × age group
    print("\n--- Building H+E overburden by tenure × age-group heatmap ---")
    he_tenure_age_df = _build_he_overburden_tenure_age_df()
    if not he_tenure_age_df.empty:
        _plot_he_overburden_tenure_age_heatmap(
            he_tenure_age_df, cluster_map, PER_COUNTRY_DIR)
    else:
        print("  No data — skipped")

    # Expense share of net income: heatmap + EWBI scatter — H+E, Health, Education
    _SHARE_SPECS = [
        (
            _HBS_SHELTER_COLS + [_HBS_ENERGY_COL],
            "HES", "Housing + Energy", "Energy and Housing",
            "expense_he_share_deciles", "ewbi_vs_he_share_scatter",
        ),
        (
            ["EUR_HE06"],
            "HLT", "Health", "Health",
            "expense_health_share_deciles", "ewbi_vs_health_share_scatter",
        ),
        (
            ["EUR_HE10"],
            "EDU", "Education", "Education",
            "expense_education_share_deciles", "ewbi_vs_education_share_scatter",
        ),
    ]
    for exp_cols, prefix, label, ewbi_prio, hmap_fname, scat_fname in _SHARE_SPECS:
        print(f"\n--- Building {label} share heatmap + EWBI scatter ---")
        sdf = _build_component_share_df(exp_cols, prefix)
        if sdf.empty:
            print(f"  No raw parquet files found — {label} skipped")
            continue
        _plot_component_share_heatmap(
            sdf, prefix, label, cluster_map, PER_COUNTRY_DIR, hmap_fname)
        _plot_share_vs_ewbi_scatter(
            sdf, prefix, ewbi_prio, label, cluster_map, PER_COUNTRY_DIR, scat_fname)


def generate_per_country_charts(pps_df) -> None:
    """
    Load all HBS countries in parallel (ThreadPoolExecutor), then render
    one stacked-bar PNG + SVG per country plus a combined Excel workbook.
    Mirrors the speed design of 2_ownership_heatmap_eu-silc.py.
    """
    print("\n" + "=" * 80)
    print("PER-COUNTRY EXPENDITURE CHARTS — all HBS countries")
    print(f"  Cache dir: {_CACHE_DIR}")
    print("  (Pass force_rebuild=True to re-read xlsx files from OneDrive)")
    print("=" * 80)

    # ── parallel loading — cache hits run concurrently; xlsx builds may be serial
    # on slow OneDrive connections but each result is cached for future runs.
    all_comps: dict = {}
    n_cached   = sum(1 for cc in HBS_COUNTRIES
                     if os.path.exists(os.path.join(_CACHE_DIR, f"{cc}_components.parquet")))
    n_to_build = len(HBS_COUNTRIES) - n_cached
    print(f"\n  {n_cached} countries cached, {n_to_build} need xlsx build")
    workers = _PC_WORKERS if n_cached == len(HBS_COUNTRIES) else min(_PC_WORKERS, 4)
    print(f"  Using {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_load_one_country, cc, cname, pps_df): cc
            for cc, cname in HBS_COUNTRIES.items()
        }
        for fut in as_completed(futures):
            cc, cdf = fut.result()
            all_comps[cc] = cdf

    # ── render charts ──────────────────────────────────────────────────────
    print("\nRendering charts...")
    for cc in sorted(all_comps):
        cdf = all_comps[cc]
        if cdf is None or cdf.empty:
            print(f"  [{cc}]  no data — skipped")
            continue
        _plot_one_country(cc, HBS_COUNTRIES.get(cc, cc), cdf)

    # ── Excel ──────────────────────────────────────────────────────────────
    print("\nExporting Excel...")
    _export_per_country_excel(all_comps)

    # ── Snapshot heatmaps (% share & PPS) ─────────────────────────────────
    generate_snapshot_heatmaps(all_comps)
    print("\nDONE — per-country charts.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("1_EXPENSE: Consumption Breakdown — CH vs EU Neighbours")
    print("=" * 80)

    dirs = setup_directories()
    pps_df = load_pps_data(dirs)

    all_components: dict[str, pd.DataFrame] = {}

    # ── Switzerland (FSO) ─────────────────────────────────────────────
    print("\n--- Loading Switzerland (FSO) ---")
    ppp_factor = load_ch_ppp_factor()
    ch_comp = build_ch_components(ppp_factor)
    if not ch_comp.empty:
        all_components["CH"] = ch_comp

    # ── EU countries (HBS) ────────────────────────────────────────────
    for cc in ["DE", "FR", "AT", "IT"]:
        comp = build_country_components(cc, COUNTRIES[cc], pps_df, n_groups=5)
        if not comp.empty:
            all_components[cc] = comp

    # ── Output ────────────────────────────────────────────────────────
    if all_components:
        plot_expense_comparison(all_components)
        export_excel(all_components)
    else:
        print("ERROR: no data for any country")

    # ── Per-country charts (all 27 HBS countries, parallel) ───────────
    generate_per_country_charts(pps_df)

    print("\nDONE.")


if __name__ == "__main__":
    main()
