"""
2_education_scatter.py — Education Investment vs EWBI Education Priority
========================================================================
Scatter plots:
  - X axis: Education public investment per capita (PPS) OR as % of GDP
  - Y axis: EWBI Education priority score (All Deciles, latest available year)

Two panels per figure: public investment (left) and total investment (right).
Points coloured by HBS cluster (same logic as 1_expense.py).

Output: outputs/graphs/education_scatter.png / .svg
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Paths ─────────────────────────────────────────────────────────────────────
CURRENT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR      = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
EXT_DATA_DIR  = os.path.join(BASE_DIR, "external_data")
WELL_BEING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs", "graphs", "education")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Cluster palette (identical to 1_expense.py) ───────────────────────────────
_CL_COLORS = ["#fb8072", "#fdb462", "#8dd3c7", "#80b1d3"]
_CL_NAMES  = {
    0: "Low perf / Low EWBI",  1: "Low perf / High EWBI",
    2: "High perf / Low EWBI", 3: "High perf / High EWBI",
}

# ── Country name → ISO2 mapping (Eurostat full names → EWBI codes) ────────────
_NAME_TO_ISO = {
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR",
    "Cyprus": "CY", "Czechia": "CZ", "Denmark": "DK", "Estonia": "EE",
    "Finland": "FI", "France": "FR", "Germany": "DE", "Greece": "EL",
    "Hungary": "HU", "Iceland": "IS", "Ireland": "IE", "Italy": "IT",
    "Latvia": "LV", "Lithuania": "LT", "Luxembourg": "LU", "Malta": "MT",
    "Netherlands": "NL", "Norway": "NO", "Poland": "PL", "Portugal": "PT",
    "Romania": "RO", "Serbia": "RS", "Slovakia": "SK", "Slovenia": "SI",
    "Spain": "ES", "Sweden": "SE", "Switzerland": "CH",
    "United Kingdom": "UK",
    "European Union - 27 countries (from 2020)": "EU-27",
}

# ISCED levels to sum (ED0–ED5 inclusive, using full label names in the CSV)
_ISCED_LEVELS = {
    "Early childhood education",          # ED0
    "Primary education",                  # ED1
    "Lower secondary education",          # ED2
    "Upper secondary education",          # ED3
    "Post-secondary non-tertiary education",  # ED4
    "Short-cycle tertiary education",     # ED5
}


def _load_cluster_map() -> dict:
    """Load cluster assignments from ewbi_master_aggregated.csv + median income."""
    try:
        data_csv   = os.path.join(WELL_BEING_DIR, "output", "ewbi_master_aggregated.csv")
        income_csv = os.path.join(BASE_DIR, "outputs", "data", "median_income_by_decile.csv")
        if not (os.path.exists(data_csv) and os.path.exists(income_csv)):
            return {}

        df = pd.read_csv(data_csv, low_memory=False)

        _PERF_CUT = 0.006972
        _EWBI_CUT = 0.7
        _M5_STEP  = 5000

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
        inc_df = inc_df.dropna(subset=["country", "year", "decile", "median_equi_disp_inc"])

        merged = ewbi_d.merge(
            inc_df, left_on=["Country", "Year", "Decile"],
            right_on=["country", "year", "decile"], how="inner",
        ).dropna(subset=["median_equi_disp_inc", "Value"])

        last_yr = merged.groupby("Country")["Year"].max().reset_index()
        last_yr.columns = ["Country", "Last_Year"]
        pts = merged.merge(last_yr, on="Country")
        pts = pts[pts["Year"] == pts["Last_Year"]].copy()

        pts["bin"] = np.round(pts["median_equi_disp_inc"] / _M5_STEP) * _M5_STEP
        bench = (pts.groupby("bin", as_index=False)["Value"].mean()
                    .sort_values("bin").rename(columns={"Value": "bench"}))
        pts["ewbi_exp"] = np.interp(
            pts["median_equi_disp_inc"].values.astype(float),
            bench["bin"].values.astype(float),
            bench["bench"].values.astype(float),
        )
        pts["residual"] = pts["Value"] - pts["ewbi_exp"]

        perf  = pts.groupby("Country", as_index=False).agg(
            Performance_Score=("residual", "mean"))
        feats = perf.merge(ewbi_last, on="Country").dropna(
            subset=["Performance_Score", "EWBI_Last"])

        cv = {("Low performer", "Low EWBI"): 0, ("Low performer", "High EWBI"): 1,
              ("High performer", "Low EWBI"): 2, ("High performer", "High EWBI"): 3}
        feats["pg"] = np.where(feats["Performance_Score"] >= _PERF_CUT,
                               "High performer", "Low performer")
        feats["eg"] = np.where(feats["EWBI_Last"] >= _EWBI_CUT, "High EWBI", "Low EWBI")
        feats["Cluster"] = feats.apply(lambda r: cv[(r["pg"], r["eg"])], axis=1)
        return dict(zip(feats["Country"], feats["Cluster"].astype(int)))

    except Exception as exc:
        print(f"  [cluster] Failed: {exc}")
        return {}


def _sum_isced(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to relevant ISCED levels and sum OBS_VALUE per geo × TIME_PERIOD."""
    sub = df[df["isced11"].isin(_ISCED_LEVELS)].copy()
    sub["OBS_VALUE"] = pd.to_numeric(sub["OBS_VALUE"], errors="coerce")
    return (sub.groupby(["geo", "TIME_PERIOD"], as_index=False)["OBS_VALUE"]
               .sum(min_count=1)
               .rename(columns={"OBS_VALUE": "educ_pps"}))


def _load_ewbi_education() -> pd.DataFrame:
    """Return DataFrame with Country (ISO2), Year, ewbi_score (All Deciles)."""
    path = os.path.join(WELL_BEING_DIR, "output", "ewbi_master_aggregated.csv")
    df   = pd.read_csv(path, low_memory=False)
    sub  = df[
        (df["EU priority"] == "Education") &
        (df["Level"] == 2) &
        (df["Decile"] == "All Deciles")
    ][["Country", "Year", "Value"]].copy()
    sub["Year"]  = pd.to_numeric(sub["Year"],  errors="coerce")
    sub["Value"] = pd.to_numeric(sub["Value"], errors="coerce")
    return sub.dropna().rename(columns={"Value": "ewbi_score"})


def _scatter_panel(ax, sub: pd.DataFrame, x_col: str, x_label: str,
                   cluster_map: dict) -> None:
    """Draw one scatter panel onto *ax*."""
    sub = sub.dropna(subset=[x_col, "ewbi_score"])
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        return

    for _, row in sub.iterrows():
        cl    = int(cluster_map.get(row["iso"], -1))
        color = _CL_COLORS[cl] if 0 <= cl <= 3 else "#888888"
        ax.scatter(row[x_col], row["ewbi_score"], color=color,
                   s=80, edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(row["iso"],
                    xy=(row[x_col], row["ewbi_score"]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=7.5, color="#333333", zorder=4)

    x_vals = sub[x_col].values.astype(float)
    y_vals = sub["ewbi_score"].values.astype(float)
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

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("EWBI Education priority score\n(0–1, higher = better well-being)",
                  fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _build_merged(educ_sum: pd.DataFrame, pop: pd.DataFrame,
                  gdp: pd.DataFrame, ewbi: pd.DataFrame,
                  label: str) -> pd.DataFrame:
    """
    For each country use the most recent year where ALL of:
      educ_sum, pop, gdp, ewbi are available.
    Returns one row per country with columns:
      iso, geo, year, educ_pps, pop_n, gdp_pps,
      per_capita (PPS/person), pct_gdp (%), ewbi_score
    """
    educ_sum = educ_sum.copy()
    educ_sum["iso"] = educ_sum["geo"].map(_NAME_TO_ISO)
    educ_sum = educ_sum.dropna(subset=["iso", "educ_pps"])

    pop = pop[pop["age"] == "Total"][pop["sex"] == "Total"][
        ["geo", "TIME_PERIOD", "OBS_VALUE"]
    ].copy()
    pop["OBS_VALUE"] = pd.to_numeric(pop["OBS_VALUE"], errors="coerce")
    pop["iso"] = pop["geo"].map(_NAME_TO_ISO)
    pop = pop.dropna(subset=["iso", "OBS_VALUE"]).rename(
        columns={"OBS_VALUE": "pop_n", "TIME_PERIOD": "year_pop"})

    gdp = gdp[["geo", "TIME_PERIOD", "OBS_VALUE"]].copy()
    gdp["OBS_VALUE"] = pd.to_numeric(gdp["OBS_VALUE"], errors="coerce")
    gdp["iso"] = gdp["geo"].map(_NAME_TO_ISO)
    gdp = gdp.dropna(subset=["iso", "OBS_VALUE"]).rename(
        columns={"OBS_VALUE": "gdp_pps", "TIME_PERIOD": "year_gdp"})

    records = []
    for iso, eg in educ_sum.groupby("iso"):
        ew = ewbi[ewbi["Country"] == iso]
        if ew.empty:
            continue
        pp = pop[pop["iso"] == iso]
        gd = gdp[gdp["iso"] == iso]

        common_years = (
            set(eg["TIME_PERIOD"]) &
            set(pp["year_pop"]) &
            set(gd["year_gdp"]) &
            set(ew["Year"].astype(int))
        )
        if not common_years:
            print(f"  [{iso}] no common year — skipped")
            continue
        yr = max(common_years)

        educ_val = eg[eg["TIME_PERIOD"] == yr]["educ_pps"].values[0]
        pop_val  = pp[pp["year_pop"] == yr]["pop_n"].values[0]
        gdp_val  = gd[gd["year_gdp"] == yr]["gdp_pps"].values[0]
        ewbi_val = ew[ew["Year"] == yr]["ewbi_score"].values[0]

        records.append({
            "iso": iso,
            "year": yr,
            "educ_pps": educ_val,
            "pop_n": pop_val,
            "gdp_pps": gdp_val,
            "per_capita": educ_val * 1e6 / pop_val if pop_val > 0 else np.nan,
            "pct_gdp": educ_val / gdp_val * 100.0 if gdp_val > 0 else np.nan,
            "ewbi_score": ewbi_val,
        })
        print(f"  [{iso}] year={yr}  educ={educ_val:.0f}M PPS"
              f"  per_capita={records[-1]['per_capita']:.0f}"
              f"  pct_gdp={records[-1]['pct_gdp']:.2f}%"
              f"  ewbi={ewbi_val:.3f}")

    return pd.DataFrame(records)


def make_education_scatter() -> None:
    print("\n" + "=" * 70)
    print("EDUCATION INVESTMENT vs EWBI EDUCATION")
    print("=" * 70)

    cluster_map = _load_cluster_map()
    ewbi        = _load_ewbi_education()

    pub_raw = pd.read_csv(os.path.join(EXT_DATA_DIR, "eurostat_educ_public.csv"))
    tot_raw = pd.read_csv(os.path.join(EXT_DATA_DIR, "eurostat_educ_total.csv"))
    pop_raw = pd.read_csv(os.path.join(EXT_DATA_DIR, "eurostat_population.csv"))
    gdp_raw = pd.read_csv(os.path.join(EXT_DATA_DIR, "eurostat_gdp_pps.csv"))

    # filter population to Total/Total
    pop_raw = pop_raw[(pop_raw["age"] == "Total") & (pop_raw["sex"] == "Total")]

    pub_sum = _sum_isced(pub_raw)
    tot_sum = _sum_isced(tot_raw)

    print("\n-- Public investment --")
    pub_df = _build_merged(pub_sum, pop_raw, gdp_raw, ewbi, "public")
    print("\n-- Total investment --")
    tot_df = _build_merged(tot_sum, pop_raw, gdp_raw, ewbi, "total")

    for norm, x_col, x_label, fname_suffix in [
        ("per_capita", "per_capita",
         "Education investment — PPS per capita",
         "per_capita"),
        ("pct_gdp", "pct_gdp",
         "Education investment — % of GDP",
         "pct_gdp"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            f"Education Investment vs. EWBI Education Priority  ({x_label})",
            fontsize=13, fontweight="bold", y=1.01,
        )
        axes[0].set_title("Public investment (government)", fontsize=11, fontweight="bold")
        axes[1].set_title("Total investment (public + private)", fontsize=11, fontweight="bold")

        _scatter_panel(axes[0], pub_df, x_col, x_label, cluster_map)
        _scatter_panel(axes[1], tot_df, x_col, x_label, cluster_map)

        legend_handles = [
            Patch(facecolor=_CL_COLORS[i], label=_CL_NAMES[i])
            for i in range(4)
        ]
        legend_handles.append(
            Patch(facecolor="#888888", label="No cluster data"))
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        for ext in ("png", "svg"):
            path = os.path.join(OUTPUT_DIR, f"education_scatter_{fname_suffix}.{ext}")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"  Saved: {os.path.basename(path)}")
        plt.close(fig)


if __name__ == "__main__":
    make_education_scatter()
