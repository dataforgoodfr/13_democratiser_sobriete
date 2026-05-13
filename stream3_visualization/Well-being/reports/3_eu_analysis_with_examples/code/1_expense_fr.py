"""
1_expense_fr.py — Consumption Breakdown Comparison (France report)
==================================================================
France vs. EU neighbours (ES, DE, IT, BE, LU).

All countries use HBS 2020, income quintiles (Q1-Q5), annual PPS per
adult equivalent via the hbs_cluster_comparison / hbs_data_loader pipeline.

Components: Housing · Transport · Food & Beverage · Health · Education · Other.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from hbs_data_loader import setup_directories, load_pps_data, calculate_consumption_in_pps
from hbs_cluster_comparison import (
    load_country_2020,
    assign_simple_deciles,
    assign_consumption_groups,
    calculate_components_by_decile,
    has_valid_components,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "graphs", "HBS_expense")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------
COUNTRIES = {
    "FR": "France",
    "ES": "Spain",
    "DE": "Germany",
    "IT": "Italy",
    "BE": "Belgium",
    "LU": "Luxembourg",
}

# ---------------------------------------------------------------------------
# Display constants (same palette as hbs_cluster_comparison)
# ---------------------------------------------------------------------------
DISPLAY_COMPONENTS = ["Housing", "Transport", "Food & Beverage", "Health", "Education"]
DISPLAY_COLORS = ["#fc8d62", "#b3de69", "#8dd3c7", "#ffffb3", "#bebada"]

plt.rcParams["font.family"] = "Arial"

DECILE_ORDER = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"]


# ═══════════════════════════════════════════════════════════════════════════
# Build country data (deciles, with consumption-based fallback at 10 groups)
# ═══════════════════════════════════════════════════════════════════════════

def build_country_deciles(cc: str, cname: str, pps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Like build_country_components but forces 10-group deciles everywhere.
    The consumption-based fallback uses n_groups=10 (not capped at 5).
    """
    print(f"\n--- {cname} ({cc}) ---")
    df = load_country_2020(cc)
    if df.empty:
        return pd.DataFrame()

    df = calculate_consumption_in_pps(df, pps_df)
    if "EUR_HE00_pps" not in df.columns:
        if "EUR_HE00" in df.columns and "pps_factor" in df.columns:
            df["EUR_HE00_pps"] = (
                pd.to_numeric(df["EUR_HE00"], errors="coerce")
                / pd.to_numeric(df["pps_factor"], errors="coerce")
            )
            print("  INFO: derived EUR_HE00_pps from EUR_HE00 / pps_factor")
        else:
            print("  WARN: EUR_HE00_pps unavailable; skipping country")
            return pd.DataFrame()

    if "EUR_HH099_pps" not in df.columns:
        if "EUR_HH099" in df.columns and "pps_factor" in df.columns:
            df["EUR_HH099_pps"] = (
                pd.to_numeric(df["EUR_HH099"], errors="coerce")
                / pd.to_numeric(df["pps_factor"], errors="coerce")
            )
            print("  INFO: derived EUR_HH099_pps from EUR_HH099 / pps_factor")
        else:
            print("  WARN: EUR_HH099_pps unavailable; income line may be missing")

    # Try income-based deciles first
    df = assign_simple_deciles(df, n_groups=10)

    # Fallback: consumption-based deciles (10 groups, NOT capped at 5)
    if "income_decile" not in df.columns or df["income_decile"].notna().sum() == 0:
        df = assign_consumption_groups(df, n_groups=10)
        if "income_decile" in df.columns and df["income_decile"].notna().sum() > 0:
            print("  INFO: used consumption-based DECILE grouping as fallback")
        else:
            print("  WARN: no grouping possible; skipping country")
            return pd.DataFrame()

    comp = calculate_components_by_decile(df, country_code=cc)
    if has_valid_components(comp):
        print(f"  OK  {len(comp)} groups computed")
        return comp

    print("  WARN: component data has NaN/insufficient values")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_expense_comparison(all_components: dict) -> None:
    """
    3 columns × 2 rows grid of stacked bar charts.
    Row 1: FR | ES | DE
    Row 2: IT | BE | LU
    """
    bar_components = DISPLAY_COMPONENTS + ["Other (Residual)"]
    bar_colors = DISPLAY_COLORS + ["#d3d3d3"]

    country_order = ["FR", "ES", "DE", "IT", "BE", "LU"]
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
        "Consumption Breakdown — France vs. EU Neighbours\n"
        "Housing · Transport · Food · Health · Education · Residual  (annual PPS per adult eq.)",
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

        if cdf is not None and not cdf.empty:
            cdf["decile"] = pd.Categorical(
                cdf["decile"], categories=DECILE_ORDER, ordered=True,
            )
            cdf = cdf.sort_values("decile")

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

        ax.set_xlabel("Income Decile", fontsize=10, fontweight="bold")
        ax.set_ylabel("Annual PPS per adult eq.", fontsize=10, fontweight="bold")
        ax.set_title(
            f"{cname}\n(HBS 2020, per adult eq.)",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylim(0, global_ymax)
        ax.grid(True, alpha=0.3, axis="y")

    # Shared legend
    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="center left", fontsize=10,
            framealpha=0.95, bbox_to_anchor=(1.0, 0.5),
            ncol=1, title="Component", title_fontsize=11,
        )

    out_png = os.path.join(OUTPUT_DIR, "expense_comparison_fr_vs_eu.png")
    out_svg = os.path.join(OUTPUT_DIR, "expense_comparison_fr_vs_eu.svg")
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

    xlsx_path = os.path.join(OUTPUT_DIR, "expense_comparison_fr_vs_eu.xlsx")
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
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("1_EXPENSE_FR: Consumption Breakdown — FR vs EU Neighbours")
    print("=" * 80)

    dirs = setup_directories()
    pps_df = load_pps_data(dirs)

    all_components: dict[str, pd.DataFrame] = {}

    # ── All countries via HBS (deciles) ──────────────────────────────
    for cc, cname in COUNTRIES.items():
        comp = build_country_deciles(cc, cname, pps_df)
        if not comp.empty:
            all_components[cc] = comp

    # ── Output ────────────────────────────────────────────────────────
    if all_components:
        plot_expense_comparison(all_components)
        export_excel(all_components)
    else:
        print("ERROR: no data for any country")

    print("\nDONE.")


if __name__ == "__main__":
    main()
