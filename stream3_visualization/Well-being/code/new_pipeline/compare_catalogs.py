"""
Catalog Comparison & Suppression Analysis
==========================================
Compares the LEGACY master_indicator_catalog.csv (old pipeline) with the NEW
pipeline catalog_complete.csv using identical clean-year criteria:

  Clean year = (indicator × country × year) where
    • All 10 decile values are present (no NaN)
    • NOT all values are zero (i.e. at least one decile > 0)
    • (Partial zeros within a year are OK)

For each dataset, per indicator:
  → Count of EU-27 countries with at least one clean year

Then simulates the greedy suppression scenario:
  Starting from all indicators, iteratively drop the one whose removal
  improves EU-27 country coverage the most, until all 27 are fully covered
  (or no further gain is possible).

Output: compare_catalogs.xlsx (written next to this script's output/catalog/)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import xlsxwriter

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import config

LEGACY_CSV = (
    _HERE.parents[1]  # → Well-being/
    / "output"
    / "1_final_df"
    / "master_indicator_catalog.csv"
)
NEW_CSV = _HERE / "output" / "catalog" / "catalog_complete.csv"
OUT_DIR = _HERE / "output" / "catalog"
OUT_XLSX = OUT_DIR / "compare_catalogs.xlsx"

EU27 = config.EU27  # 27 actual EU member states

DECILE_VALS = {str(d) for d in range(1, 11)}  # "1" .. "10"


# ─── Core logic ───────────────────────────────────────────────────────────────

def clean_year_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a long-format DataFrame with columns [code, country, year, decile, value],
    return a DataFrame with one row per (code, country, year) and a boolean
    column 'clean' = True when the year passes the clean-year test.

    Clean = all 10 decile values present AND not all zero.
    """
    # Keep only decile rows D1–D10
    df = df[df["decile"].astype(str).isin(DECILE_VALS)].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    g = df.groupby(["code", "country", "year"])["value"]

    n_present = g.apply(lambda s: s.notna().sum())
    sum_vals   = g.apply(lambda s: s.fillna(0).sum())

    flags = pd.DataFrame({"n_present": n_present, "sum_vals": sum_vals}).reset_index()
    flags["clean"] = (flags["n_present"] == 10) & (flags["sum_vals"] != 0)
    return flags


def country_coverage(flags: pd.DataFrame, eu27: list[str]) -> pd.DataFrame:
    """
    From a flags DataFrame (code, country, year, clean),
    return per-indicator count of EU-27 countries with ≥1 clean year.
    """
    clean = flags[flags["clean"] & flags["country"].isin(eu27)]
    covered = (
        clean.groupby("code")["country"]
        .nunique()
        .reset_index()
        .rename(columns={"country": "n_countries"})
    )
    # Fill missing indicators with 0
    all_codes = flags["code"].unique()
    covered = (
        pd.DataFrame({"code": all_codes})
        .merge(covered, on="code", how="left")
        .fillna({"n_countries": 0})
    )
    covered["n_countries"] = covered["n_countries"].astype(int)
    return covered.sort_values("code").reset_index(drop=True)


def available_matrix(flags: pd.DataFrame, eu27: list[str]) -> pd.DataFrame:
    """
    Return a boolean DataFrame: rows = indicators, columns = EU-27 countries.
    True = country has ≥1 clean year for that indicator.
    """
    clean = flags[flags["clean"] & flags["country"].isin(eu27)]
    avail = (
        clean.groupby(["code", "country"])
        .size()
        .reset_index(name="cnt")
        .assign(has_data=True)
    )
    mat = avail.pivot(index="code", columns="country", values="has_data").reindex(
        columns=eu27
    )
    mat = mat.notna()  # True where data exists, False where NaN (pivot fill)
    return mat


def greedy_impact_table(steps: list[dict]) -> pd.DataFrame:
    """
    Derive a per-indicator impact table from greedy suppression steps.
    Shows for each removed indicator: its step, countries before/after, gain.
    Sorted by gain descending (then step ascending for ties).
    """
    rows = []
    for i, s in enumerate(steps[1:], start=1):           # skip step 0 (baseline)
        prev = steps[i - 1]["n_countries_covered"]
        curr = s["n_countries_covered"]
        rows.append({
            "greedy_step":        s["step"],
            "code":               s["suppressed"],
            "countries_before":   prev,
            "countries_after":    curr,
            "gain":               curr - prev,
            "indicators_remaining": s["n_indicators_remaining"],
        })
    df = pd.DataFrame(rows).sort_values(
        ["gain", "greedy_step"], ascending=[False, True]
    ).reset_index(drop=True)
    return df


def greedy_suppression(mat: pd.DataFrame, eu27: list[str]) -> list[dict]:
    """
    Greedy algorithm: at each step, remove the indicator whose removal maximises
    the number of EU-27 countries with ≥1 clean year for ALL remaining indicators.

    Returns list of steps: {step, suppressed, n_countries_covered, indicators_remaining}.
    """
    remaining = set(mat.index.tolist())
    steps = []

    def coverage(codes):
        if not codes:
            return len(eu27)
        sub = mat.loc[list(codes), eu27]
        # Country is covered if it has data for ALL remaining indicators
        return int(sub.all(axis=0).sum())

    base = coverage(remaining)
    steps.append({
        "step": 0,
        "suppressed": None,
        "n_countries_covered": base,
        "n_indicators_remaining": len(remaining),
        "indicators_remaining": sorted(remaining),
    })

    for step in range(1, len(mat) + 1):
        if not remaining:
            break
        best_gain = -1
        best_code = None
        for code in remaining:
            trial = remaining - {code}
            cov = coverage(trial)
            gain = cov - base
            if gain > best_gain or (gain == best_gain and best_code and code < best_code):
                best_gain = gain
                best_code = code
                best_cov = cov

        remaining.discard(best_code)
        base = best_cov
        steps.append({
            "step": step,
            "suppressed": best_code,
            "n_countries_covered": base,
            "n_indicators_remaining": len(remaining),
            "indicators_remaining": sorted(remaining),
        })
        if base == len(eu27):
            break  # All 27 covered → stop

    return steps


# ─── Excel writer ──────────────────────────────────────────────────────────────

def write_excel(
    legacy_flags: pd.DataFrame,
    new_flags: pd.DataFrame,
    legacy_cov: pd.DataFrame,
    new_cov: pd.DataFrame,
    legacy_steps: list[dict],
    new_steps: list[dict],
    legacy_mat: pd.DataFrame,
    new_mat: pd.DataFrame,
    legacy_impact: pd.DataFrame,
    new_impact: pd.DataFrame,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(OUT_XLSX))

    # ── Formats ────────────────────────────────────────────────────────────────
    hdr  = wb.add_format({"bold": True, "bg_color": "#2F4F8F", "font_color": "#FFFFFF",
                           "border": 1, "text_wrap": True, "valign": "vcenter"})
    bold = wb.add_format({"bold": True})
    norm = wb.add_format({"border": 1})
    num  = wb.add_format({"border": 1, "num_format": "#,##0"})
    pct  = wb.add_format({"border": 1, "num_format": "0.0%"})
    grey = wb.add_format({"border": 1, "bg_color": "#D3D3D3"})

    green  = wb.add_format({"border": 1, "bg_color": "#70AD47", "font_color": "#FFFFFF"})
    yellow = wb.add_format({"border": 1, "bg_color": "#FFD966"})
    orange = wb.add_format({"border": 1, "bg_color": "#F4B942"})
    red    = wb.add_format({"border": 1, "bg_color": "#C00000", "font_color": "#FFFFFF"})

    def country_fmt(n):
        if n == 27:
            return green
        elif n >= 22:
            return yellow
        elif n >= 15:
            return orange
        else:
            return red

    # ── 1. Per-Indicator Comparison ────────────────────────────────────────────
    ws = wb.add_worksheet("Per-Indicator")
    ws.set_column(0, 0, 20)
    ws.set_column(1, 4, 14)

    merged = (
        legacy_cov.rename(columns={"n_countries": "legacy"})
        .merge(new_cov.rename(columns={"n_countries": "new"}), on="code", how="outer")
        .fillna(0)
        .sort_values("code")
        .reset_index(drop=True)
    )
    merged["legacy"] = merged["legacy"].astype(int)
    merged["new"]    = merged["new"].astype(int)
    merged["delta"]  = merged["new"] - merged["legacy"]
    merged["in_both"] = merged[["legacy", "new"]].min(axis=1).astype(int)

    headers = ["Indicator Code", "Legacy (old pipeline)", "New pipeline", "Delta (New−Legacy)", "In both"]
    for c, h in enumerate(headers):
        ws.write(0, c, h, hdr)
    ws.set_row(0, 30)

    for r, row in merged.iterrows():
        ws.write(r + 1, 0, row["code"], norm)
        for c, col in enumerate(["legacy", "new", "in_both"]):
            n = int(row[col])
            ws.write(r + 1, c + 1, n, country_fmt(n))
        d = int(row["delta"])
        fmt = wb.add_format({"border": 1,
                              "font_color": "#006100" if d > 0 else ("#C00000" if d < 0 else "#000000")})
        ws.write(r + 1, 4, d, fmt)

    ws.autofilter(0, 0, len(merged), 4)

    # ── 2. Greedy Suppression — Legacy ────────────────────────────────────────
    for label, steps in [("Suppression Legacy", legacy_steps), ("Suppression New", new_steps)]:
        ws2 = wb.add_worksheet(label)
        ws2.set_column(0, 0, 8)
        ws2.set_column(1, 1, 20)
        ws2.set_column(2, 3, 18)
        ws2.set_column(4, 4, 80)

        hdrs2 = ["Step", "Indicator Suppressed", "Countries Covered", "Indicators Remaining",
                 "Remaining Indicator Set"]
        for c, h in enumerate(hdrs2):
            ws2.write(0, c, h, hdr)
        ws2.set_row(0, 30)

        for r, s in enumerate(steps):
            ws2.write(r + 1, 0, s["step"], norm)
            ws2.write(r + 1, 1, s["suppressed"] or "—", norm)
            n = s["n_countries_covered"]
            ws2.write(r + 1, 2, n, country_fmt(n))
            ws2.write(r + 1, 3, s["n_indicators_remaining"], num)
            ws2.write(r + 1, 4, ", ".join(s["indicators_remaining"]), norm)

    # ── 3. Greedy Impact (ranked by coverage gain) ───────────────────────────
    # Shows each indicator's sequential impact in the greedy suppression order,
    # sorted by gain descending. Gain > 0 = this indicator was the bottleneck
    # unlocking those countries at that step.
    for label, impact in [("Impact Legacy", legacy_impact), ("Impact New", new_impact)]:
        ws_imp = wb.add_worksheet(label)
        ws_imp.set_column(0, 0, 8)   # step
        ws_imp.set_column(1, 1, 18)  # code
        ws_imp.set_column(2, 3, 16)  # before / after
        ws_imp.set_column(4, 4, 8)   # gain
        ws_imp.set_column(5, 5, 8)   # indicators remaining

        imp_hdrs = ["Greedy Step", "Indicator", "Countries Before", "Countries After",
                    "Gain", "Indicators Remaining"]
        for c, h in enumerate(imp_hdrs):
            ws_imp.write(0, c, h, hdr)
        ws_imp.set_row(0, 30)

        for r, row in impact.iterrows():
            ws_imp.write(r + 1, 0, int(row["greedy_step"]), norm)
            ws_imp.write(r + 1, 1, row["code"], norm)
            ws_imp.write(r + 1, 2, int(row["countries_before"]), norm)
            n_after = int(row["countries_after"])
            ws_imp.write(r + 1, 3, n_after, country_fmt(n_after))
            g = int(row["gain"])
            gain_fmt_props = {
                "border": 1,
                "font_color": "#006100" if g > 0 else "#000000",
                "bold": g > 0,
            }
            if g > 4:
                gain_fmt_props["bg_color"] = "#E2EFDA"
            elif g > 1:
                gain_fmt_props["bg_color"] = "#FFEB9C"
            gain_fmt = wb.add_format(gain_fmt_props)
            ws_imp.write(r + 1, 4, g, gain_fmt)
            ws_imp.write(r + 1, 5, int(row["indicators_remaining"]), norm)
        ws_imp.autofilter(0, 0, len(impact), 5)

    # ── 4. Availability Matrix ────────────────────────────────────────────────
    for label, mat in [("Matrix Legacy", legacy_mat), ("Matrix New", new_mat)]:
        ws3 = wb.add_worksheet(label)
        ws3.set_column(0, 0, 20)
        ws3.set_column(1, len(EU27), 5)
        ws3.write(0, 0, "Indicator \\ Country", hdr)
        for c, cc in enumerate(EU27):
            ws3.write(0, c + 1, cc, hdr)
        ws3.set_row(0, 20)

        for r, code in enumerate(mat.index):
            ws3.write(r + 1, 0, code, bold)
            for c, cc in enumerate(EU27):
                val = mat.loc[code, cc] if cc in mat.columns else False
                ws3.write(r + 1, c + 1, "✓" if val else "✗", green if val else red)

    # ── 4. Summary ─────────────────────────────────────────────────────────────
    ws4 = wb.add_worksheet("Summary")
    ws4.set_column(0, 0, 35)
    ws4.set_column(1, 2, 20)

    def write_kv(ws, r, k, v1, v2=None):
        ws.write(r, 0, k, bold)
        ws.write(r, 1, v1, norm)
        if v2 is not None:
            ws.write(r, 2, v2, norm)

    ws4.write(0, 0, "Metric", hdr)
    ws4.write(0, 1, "Legacy (old pipeline)", hdr)
    ws4.write(0, 2, "New pipeline", hdr)

    # Total indicators
    n_leg = len(legacy_cov)
    n_new = len(new_cov)
    write_kv(ws4, 1, "Total indicators", n_leg, n_new)

    # Indicators with ≥1 EU-27 country covered
    leg_any = int((legacy_cov["n_countries"] > 0).sum())
    new_any = int((new_cov["n_countries"] > 0).sum())
    write_kv(ws4, 2, "Indicators with ≥1 EU-27 country covered", leg_any, new_any)

    # Indicators with all 27 EU countries covered
    leg_all = int((legacy_cov["n_countries"] == 27).sum())
    new_all = int((new_cov["n_countries"] == 27).sum())
    write_kv(ws4, 3, "Indicators with all 27 EU-27 countries covered", leg_all, new_all)

    # Countries with data for ALL indicators (baseline)
    leg_base = legacy_steps[0]["n_countries_covered"]
    new_base = new_steps[0]["n_countries_covered"]
    write_kv(ws4, 4, "EU-27 countries covered for ALL indicators (baseline)", leg_base, new_base)

    # Steps to reach 27/27
    leg_reach = next((s["step"] for s in legacy_steps if s["n_countries_covered"] == 27), None)
    new_reach = next((s["step"] for s in new_steps if s["n_countries_covered"] == 27), None)
    write_kv(ws4, 5, "Indicators to suppress to reach 27/27 countries",
             leg_reach if leg_reach else "Not reached",
             new_reach if new_reach else "Not reached")

    wb.close()
    print(f"Saved: {OUT_XLSX}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading legacy catalog …")
    legacy_df = pd.read_csv(LEGACY_CSV, low_memory=False)
    # Filter to EU-27 only
    legacy_df = legacy_df[legacy_df["country"].isin(EU27)]

    print("Loading new pipeline catalog …")
    new_df = pd.read_csv(NEW_CSV, low_memory=False)
    new_df = new_df[new_df["country"].isin(EU27)]

    print("Computing clean-year flags (legacy) …")
    legacy_flags = clean_year_flags(legacy_df)
    print("Computing clean-year flags (new) …")
    new_flags = clean_year_flags(new_df)

    print("Computing per-indicator coverage …")
    legacy_cov = country_coverage(legacy_flags, EU27)
    new_cov    = country_coverage(new_flags, EU27)

    print("Building availability matrices …")
    legacy_mat = available_matrix(legacy_flags, EU27)
    new_mat    = available_matrix(new_flags, EU27)

    print("Running greedy suppression (legacy) …")
    legacy_steps = greedy_suppression(legacy_mat, EU27)

    print("Running greedy suppression (new) …")
    new_steps = greedy_suppression(new_mat, EU27)

    print("Building greedy impact tables …")
    legacy_impact = greedy_impact_table(legacy_steps)
    new_impact    = greedy_impact_table(new_steps)

    print("Writing Excel …")
    write_excel(
        legacy_flags, new_flags,
        legacy_cov, new_cov,
        legacy_steps, new_steps,
        legacy_mat, new_mat,
        legacy_impact, new_impact,
    )

    # ── Print quick summary to console ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Metric':<45} {'Legacy':>8} {'New':>8}")
    print("=" * 60)
    print(f"{'Total indicators':<45} {len(legacy_cov):>8} {len(new_cov):>8}")
    print(f"{'Indicators with ≥1 country covered':<45} {int((legacy_cov['n_countries']>0).sum()):>8} {int((new_cov['n_countries']>0).sum()):>8}")
    print(f"{'Indicators with all 27 covered':<45} {int((legacy_cov['n_countries']==27).sum()):>8} {int((new_cov['n_countries']==27).sum()):>8}")
    print(f"{'Countries covered for ALL indicators':<45} {legacy_steps[0]['n_countries_covered']:>8} {new_steps[0]['n_countries_covered']:>8}")
    leg_reach = next((s["step"] for s in legacy_steps if s["n_countries_covered"] == 27), "N/A")
    new_reach = next((s["step"] for s in new_steps if s["n_countries_covered"] == 27), "N/A")
    print(f"{'Suppressions to reach 27/27':<45} {str(leg_reach):>8} {str(new_reach):>8}")
    print("=" * 60)

    print("\nTop bottleneck indicators — NEW pipeline (greedy sequential gain, sorted by impact):")
    print(f"  {'Step':>4}  {'Indicator':<18}  {'Before':>6}  {'After':>5}  {'Gain':>4}")
    for _, row in new_impact[new_impact["gain"] > 0].iterrows():
        print(f"  {int(row['greedy_step']):>4}  {row['code']:<18}  {int(row['countries_before']):>6}  {int(row['countries_after']):>5}  {int(row['gain']):>4}")

    print("\nTop bottleneck indicators — LEGACY pipeline (greedy sequential gain, sorted by impact):")
    print(f"  {'Step':>4}  {'Indicator':<18}  {'Before':>6}  {'After':>5}  {'Gain':>4}")
    for _, row in legacy_impact[legacy_impact["gain"] > 0].iterrows():
        print(f"  {int(row['greedy_step']):>4}  {row['code']:<18}  {int(row['countries_before']):>6}  {int(row['countries_after']):>5}  {int(row['gain']):>4}")


if __name__ == "__main__":
    main()
