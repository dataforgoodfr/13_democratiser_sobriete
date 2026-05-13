"""
3_health_scatter.py — Health Investment vs EWBI Health Priority
===============================================================
Scatter plots (EWBI Health, All Deciles, latest year) against:
  - Public / government health investment  (icha11_hf = "Government schemes…")
  - Total health investment                (icha11_hf = "All financing schemes")

For each financing scheme, three X-axis variants are produced:
  1. PPS per inhabitant  (direct from file, unit = "Purchasing power standard…")
  2. % of GDP            (direct from file, unit = "Percentage of gross domestic…")
  3. Computed per-capita (MIO_PPS ÷ population × 1e6)   ← cross-check for 1
  4. Computed % GDP      (MIO_PPS ÷ GDP_MIO_PPS × 100)  ← cross-check for 2

Cross-check scatter (computed vs direct) is saved separately to verify consistency.

Output: outputs/graphs/health/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Paths ─────────────────────────────────────────────────────────────────────
CURRENT_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR       = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
EXT_DATA_DIR   = os.path.join(BASE_DIR, "external_data")
WELL_BEING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs", "graphs", "health")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Cluster palette (same as 1_expense.py / 2_education_scatter.py) ──────────
_CL_COLORS = ["#fb8072", "#fdb462", "#8dd3c7", "#80b1d3"]
_CL_NAMES  = {
    0: "Low perf / Low EWBI",  1: "Low perf / High EWBI",
    2: "High perf / Low EWBI", 3: "High perf / High EWBI",
}

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

_UNIT_MIO   = "Million purchasing power standards (PPS)"
_UNIT_HAB   = "Purchasing power standard (PPS) per inhabitant"
_UNIT_PCGDP = "Percentage of gross domestic product (GDP)"
_HF_GOV     = "Government schemes and compulsory contributory health care financing schemes"
_HF_TOT     = "All financing schemes"


# ── Cluster map ───────────────────────────────────────────────────────────────
def _load_cluster_map() -> dict:
    try:
        data_csv   = os.path.join(WELL_BEING_DIR, "output", "ewbi_master_aggregated.csv")
        income_csv = os.path.join(BASE_DIR, "outputs", "data", "median_income_by_decile.csv")
        if not (os.path.exists(data_csv) and os.path.exists(income_csv)):
            return {}

        _PERF_CUT = 0.006972
        _EWBI_CUT = 0.7
        _M5_STEP  = 5000

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
        ewbi_last = (ewbi_all.sort_values("Year").groupby("Country").tail(1)
                             [["Country", "Value"]].rename(columns={"Value": "EWBI_Last"}))

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


# ── EWBI Health ───────────────────────────────────────────────────────────────
def _load_ewbi_health() -> pd.DataFrame:
    path = os.path.join(WELL_BEING_DIR, "output", "ewbi_master_aggregated.csv")
    df   = pd.read_csv(path, low_memory=False)
    sub  = df[
        (df["EU priority"] == "Health") &
        (df["Level"] == 2) &
        (df["Decile"] == "All Deciles")
    ][["Country", "Year", "Value"]].copy()
    sub["Year"]  = pd.to_numeric(sub["Year"],  errors="coerce")
    sub["Value"] = pd.to_numeric(sub["Value"], errors="coerce")
    return sub.dropna().rename(columns={"Value": "ewbi_score"})


# ── Build merged table ────────────────────────────────────────────────────────
def _build_health_df(health_raw: pd.DataFrame, hf_filter: str,
                     pop_raw: pd.DataFrame, gdp_raw: pd.DataFrame,
                     ewbi: pd.DataFrame) -> pd.DataFrame:
    """
    For each country, pick the most recent year common to all four sources.
    Returns one row per country with columns:
      iso, year,
      pps_hab        (direct from file),
      pc_gdp         (direct from file),
      mio_pps        (raw million PPS),
      computed_hab   (mio_pps * 1e6 / population),
      computed_pctgdp(mio_pps / gdp_mio * 100),
      ewbi_score
    """
    sub = health_raw[health_raw["icha11_hf"] == hf_filter].copy()
    sub["OBS_VALUE"] = pd.to_numeric(sub["OBS_VALUE"], errors="coerce")
    sub["iso"] = sub["geo"].map(_NAME_TO_ISO)
    sub = sub.dropna(subset=["iso", "OBS_VALUE"])

    mio  = (sub[sub["unit"] == _UNIT_MIO]
            .rename(columns={"OBS_VALUE": "mio_pps", "TIME_PERIOD": "yr"})
            [["iso", "yr", "mio_pps"]])
    hab  = (sub[sub["unit"] == _UNIT_HAB]
            .rename(columns={"OBS_VALUE": "pps_hab", "TIME_PERIOD": "yr"})
            [["iso", "yr", "pps_hab"]])
    pcgdp= (sub[sub["unit"] == _UNIT_PCGDP]
            .rename(columns={"OBS_VALUE": "pc_gdp", "TIME_PERIOD": "yr"})
            [["iso", "yr", "pc_gdp"]])

    pop = (pop_raw[["geo", "TIME_PERIOD", "OBS_VALUE"]].copy()
           .rename(columns={"OBS_VALUE": "pop_n", "TIME_PERIOD": "yr"}))
    pop["OBS_VALUE_num"] = pd.to_numeric(pop["pop_n"], errors="coerce")
    pop["iso"] = pop["geo"].map(_NAME_TO_ISO)
    pop = pop.dropna(subset=["iso"]).rename(columns={"OBS_VALUE_num": "pop_n2"})
    pop["pop_n2"] = pd.to_numeric(pop["pop_n"], errors="coerce")
    pop = pop[["iso", "yr", "pop_n2"]].dropna()

    gdp = (gdp_raw[["geo", "TIME_PERIOD", "OBS_VALUE"]].copy()
           .rename(columns={"OBS_VALUE": "gdp_mio", "TIME_PERIOD": "yr"}))
    gdp["gdp_mio"] = pd.to_numeric(gdp["gdp_mio"], errors="coerce")
    gdp["iso"] = gdp["geo"].map(_NAME_TO_ISO)
    gdp = gdp[["iso", "yr", "gdp_mio"]].dropna()

    records = []
    for iso in mio["iso"].unique():
        ew  = ewbi[ewbi["Country"] == iso]
        m   = mio[mio["iso"] == iso]
        h   = hab[hab["iso"] == iso]
        pg  = pcgdp[pcgdp["iso"] == iso]
        pp  = pop[pop["iso"] == iso]
        gd  = gdp[gdp["iso"] == iso]
        if ew.empty or m.empty:
            continue
        common = (set(m["yr"]) & set(h["yr"]) & set(pg["yr"]) &
                  set(pp["yr"]) & set(gd["yr"]) & set(ew["Year"].astype(int)))
        if not common:
            print(f"  [{iso}] no common year — skipped")
            continue
        yr = max(common)

        mio_v    = m[m["yr"] == yr]["mio_pps"].values[0]
        hab_v    = h[h["yr"] == yr]["pps_hab"].values[0]
        pcgdp_v  = pg[pg["yr"] == yr]["pc_gdp"].values[0]
        pop_v    = pp[pp["yr"] == yr]["pop_n2"].values[0]
        gdp_v    = gd[gd["yr"] == yr]["gdp_mio"].values[0]
        ewbi_v   = ew[ew["Year"] == yr]["ewbi_score"].values[0]

        comp_hab    = mio_v * 1e6 / pop_v   if pop_v > 0 else np.nan
        comp_pctgdp = mio_v / gdp_v * 100.0 if gdp_v > 0 else np.nan

        records.append({
            "iso": iso, "year": yr,
            "pps_hab": hab_v, "pc_gdp": pcgdp_v,
            "mio_pps": mio_v,
            "computed_hab": comp_hab, "computed_pctgdp": comp_pctgdp,
            "ewbi_score": ewbi_v,
        })
        print(f"  [{iso}] yr={yr}  "
              f"hab_direct={hab_v:.0f}  hab_comp={comp_hab:.0f}  "
              f"pctgdp_direct={pcgdp_v:.2f}%  pctgdp_comp={comp_pctgdp:.2f}%  "
              f"ewbi={ewbi_v:.3f}")

    return pd.DataFrame(records)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _scatter_panel(ax, df: pd.DataFrame, x_col: str, x_label: str,
                   cluster_map: dict) -> None:
    sub = df.dropna(subset=[x_col, "ewbi_score"])
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

    x_v = sub[x_col].values.astype(float)
    y_v = sub["ewbi_score"].values.astype(float)
    if len(x_v) >= 3:
        coeffs = np.polyfit(x_v, y_v, 1)
        y_pred = np.polyval(coeffs, x_v)
        r2 = 1.0 - np.sum((y_v - y_pred) ** 2) / np.sum((y_v - y_v.mean()) ** 2)
        x_fit = np.linspace(x_v.min(), x_v.max(), 200)
        ax.plot(x_fit, np.polyval(coeffs, x_fit),
                color="#555555", linewidth=1.3, linestyle="--", zorder=2)
        ax.text(0.97, 0.05, f"R² = {r2:.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="#555555")

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("EWBI Health priority score\n(0–1, higher = better well-being)",
                  fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _cross_check_panel(ax, df: pd.DataFrame,
                        direct_col: str, comp_col: str, label: str) -> None:
    """Plot computed vs direct values with 45° identity line."""
    sub = df.dropna(subset=[direct_col, comp_col])
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return
    vmin = min(sub[direct_col].min(), sub[comp_col].min())
    vmax = max(sub[direct_col].max(), sub[comp_col].max())
    ax.plot([vmin, vmax], [vmin, vmax], color="#999999",
            linewidth=1, linestyle="--", zorder=1, label="Identity (y=x)")
    for _, row in sub.iterrows():
        ax.scatter(row[direct_col], row[comp_col],
                   color="#4477aa", s=60, edgecolors="white",
                   linewidths=0.5, zorder=3)
        ax.annotate(row["iso"],
                    xy=(row[direct_col], row[comp_col]),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=7, color="#333333")
    ax.set_xlabel(f"Direct from Eurostat ({label})", fontsize=9)
    ax.set_ylabel(f"Computed from MIO_PPS ({label})", fontsize=9)
    ax.set_title(f"Cross-check: {label}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def _legend_handles():
    handles = [Patch(facecolor=_CL_COLORS[i], label=_CL_NAMES[i]) for i in range(4)]
    handles.append(Patch(facecolor="#888888", label="No cluster data"))
    return handles


# ── Main ──────────────────────────────────────────────────────────────────────
def make_health_scatter() -> None:
    print("\n" + "=" * 70)
    print("HEALTH INVESTMENT vs EWBI HEALTH")
    print("=" * 70)

    cluster_map = _load_cluster_map()
    ewbi        = _load_ewbi_health()

    health_raw = pd.read_csv(os.path.join(EXT_DATA_DIR, "eurostat_health_invest.csv"))
    pop_raw    = pd.read_csv(os.path.join(EXT_DATA_DIR, "eurostat_population.csv"))
    pop_raw    = pop_raw[(pop_raw["age"] == "Total") & (pop_raw["sex"] == "Total")]
    gdp_raw    = pd.read_csv(os.path.join(EXT_DATA_DIR, "eurostat_gdp_pps.csv"))

    schemes = [
        ("gov",  _HF_GOV,  "Government / compulsory schemes"),
        ("tot",  _HF_TOT,  "All financing schemes"),
    ]

    dfs = {}
    for key, hf_filter, label in schemes:
        print(f"\n-- {label} --")
        dfs[key] = _build_health_df(health_raw, hf_filter, pop_raw, gdp_raw, ewbi)

    # ── Main scatter: 2 rows × 2 cols  (gov | tot)  ×  (PPS/hab | % GDP) ────
    for x_col, x_label, comp_col, fname in [
        ("pps_hab",  "Health investment — PPS per inhabitant (direct)",
         "computed_hab",     "health_scatter_pps_hab"),
        ("pc_gdp",   "Health investment — % of GDP (direct)",
         "computed_pctgdp",  "health_scatter_pct_gdp"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            f"Health Investment vs. EWBI Health Priority  ({x_label})",
            fontsize=13, fontweight="bold", y=1.01,
        )
        for ax, (key, _, label) in zip(axes, schemes):
            ax.set_title(label, fontsize=11, fontweight="bold")
            _scatter_panel(ax, dfs[key], x_col, x_label, cluster_map)

        fig.legend(handles=_legend_handles(), loc="lower center", ncol=3,
                   fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        for ext in ("png", "svg"):
            path = os.path.join(OUTPUT_DIR, f"{fname}.{ext}")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"  Saved: {os.path.basename(path)}")
        plt.close(fig)

    # ── Cross-check: computed vs direct, 2×2 ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Cross-check: Computed (MIO_PPS ÷ pop/GDP) vs Direct Eurostat values",
        fontsize=13, fontweight="bold",
    )
    checks = [
        (0, 0, "gov", "computed_hab",     "pps_hab",  "PPS/hab — Gov"),
        (0, 1, "tot", "computed_hab",     "pps_hab",  "PPS/hab — Total"),
        (1, 0, "gov", "computed_pctgdp",  "pc_gdp",   "% GDP — Gov"),
        (1, 1, "tot", "computed_pctgdp",  "pc_gdp",   "% GDP — Total"),
    ]
    for r, c, key, comp, direct, lbl in checks:
        _cross_check_panel(axes[r][c], dfs[key], direct, comp, lbl)

    fig.tight_layout()
    for ext in ("png", "svg"):
        path = os.path.join(OUTPUT_DIR, f"health_crosscheck.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {os.path.basename(path)}")
    plt.close(fig)


if __name__ == "__main__":
    make_health_scatter()
