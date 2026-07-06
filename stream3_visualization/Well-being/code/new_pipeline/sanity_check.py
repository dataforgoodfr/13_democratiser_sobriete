"""
sanity_check.py
===============
Sanity checks for all (indicator × country) pairs, using the last available
clean year per (indicator, country).

Three checks
------------
1. Cross-country distribution  — where does each country's D1 / D5 / D10 sit
   in the EU-27 distribution for that indicator?  Percentile rank 0–100.
2. Continuity                  — year-over-year absolute % change in D1/D5/D10;
   flag breaks > CONTINUITY_THRESHOLD (default 50 pp for % indicators).
3. Decile ordering             — Spearman rank-correlation between income decile
   (1…10) and observed value; near 0 means no clear gradient; sign tells
   direction (positive = D10 > D1, negative = D1 > D10).
   Also reports D10/D1 ratio directly.

Outputs
-------
output/quality/sanity_check.xlsx
  • Summary      : one row per (indicator × country), all three checks
  • Flagged      : DE and AT indicators flagged by coworker, highlighted
  • Distribution : cross-country D1/D5/D10 percentile ranks, all indicators
  • Continuity   : year-over-year changes flagged as breaks
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import xlsxwriter

# ── Config ────────────────────────────────────────────────────────────────────
PIPELINE_DIR      = Path(__file__).parent
CATALOG_PATH      = PIPELINE_DIR / "output/catalog/catalog_clean.csv"
OUT_DIR           = PIPELINE_DIR / "output/quality"
OUT_PATH          = OUT_DIR / "sanity_check.xlsx"

CONTINUITY_THRESHOLD = 50.0   # pp absolute change between consecutive years
OUTLIER_PCT_LO       = 10.0   # percentile rank below → flag low outlier
OUTLIER_PCT_HI       = 90.0   # percentile rank above → flag high outlier

# Indicators flagged by coworker
FLAGGED = {
    "DE": ["ED-EHIS-1", "SP-SILC-2", "EC-EHIS-1", "AC-HBS-2"],
    "AT": ["IS-SILC-3", "SP-SILC-2", "ED-EHIS-1", "EC-EHIS-1",
           "IC-HBS-2", "GE-SILC-2"],
}
FLAGGED_SET = {(country, code)
               for country, codes in FLAGGED.items()
               for code in codes}

# ── Helpers ───────────────────────────────────────────────────────────────────

def pct_rank(series: pd.Series) -> pd.Series:
    """Percentile rank within the series (0–100), NaN-safe."""
    return series.rank(pct=True, na_option="keep") * 100


def spearman_gradient(vals: dict) -> float:
    """
    Compute Spearman correlation between decile rank (1–10) and observed value.
    vals: dict {1: v1, 2: v2, ..., 10: v10}
    Returns NaN if fewer than 5 valid points.
    """
    xs, ys = [], []
    for d in range(1, 11):
        v = vals.get(d)
        if v is not None and not np.isnan(v):
            xs.append(d)
            ys.append(v)
    if len(xs) < 5:
        return np.nan
    corr, _ = spearmanr(xs, ys)
    return float(corr)


# ── Load & reshape ─────────────────────────────────────────────────────────────

print("Loading catalog …")
df = pd.read_csv(CATALOG_PATH)
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
df["decile_int"] = pd.to_numeric(df["decile"], errors="coerce")  # NaN for "All"

# Decile-only rows (exclude "All")
dec_df = df[df["decile_int"].notna()].copy()
dec_df["decile_int"] = dec_df["decile_int"].astype(int)

# ── Last year per (code, country) ─────────────────────────────────────────────

last_year = (dec_df.groupby(["code", "country"])["year"]
             .max()
             .reset_index()
             .rename(columns={"year": "last_year"}))

dec_last = dec_df.merge(last_year, on=["code", "country"])
dec_last = dec_last[dec_last["year"] == dec_last["last_year"]]

# Pivot to wide: rows = (code, country, last_year), cols = D1..D10
wide = (dec_last.pivot_table(
            index=["code", "name", "country", "last_year"],
            columns="decile_int",
            values="value",
            aggfunc="first")
        .reset_index())
wide.columns.name = None
d_cols = [c for c in range(1, 11) if c in wide.columns]
wide.columns = [f"D{c}" if isinstance(c, int) else c for c in wide.columns]
d_col_names = [f"D{i}" for i in range(1, 11)]

# Ensure all D1–D10 columns exist
for col in d_col_names:
    if col not in wide.columns:
        wide[col] = np.nan

# ── Check 3: Decile ordering (gradient) ───────────────────────────────────────

print("Computing decile ordering …")

def row_gradient(row):
    vals = {i: row[f"D{i}"] for i in range(1, 11)}
    return spearman_gradient(vals)

wide["spearman_D1_D10"] = wide.apply(row_gradient, axis=1)
wide["D10_D1_ratio"]    = wide["D10"] / wide["D1"].replace(0, np.nan)

# ── Check 1: Cross-country percentile ranks ────────────────────────────────────

print("Computing cross-country percentile ranks …")

for dcol in ["D1", "D5", "D10"]:
    wide[f"pctrank_{dcol}"] = (
        wide.groupby("code")[dcol]
        .transform(pct_rank)
    )
    wide[f"pctrank_{dcol}"] = wide[f"pctrank_{dcol}"].round(1)

# Overall: mean of D1..D10 per row
wide["mean_D1_D10"] = wide[d_col_names].mean(axis=1)
wide["pctrank_overall"] = (
    wide.groupby("code")["mean_D1_D10"]
    .transform(pct_rank)
    .round(1)
)

# Outlier flags
wide["flag_low_D1"]      = wide["pctrank_D1"]      < OUTLIER_PCT_LO
wide["flag_high_D1"]     = wide["pctrank_D1"]      > OUTLIER_PCT_HI
wide["flag_low_D5"]      = wide["pctrank_D5"]      < OUTLIER_PCT_LO
wide["flag_high_D5"]     = wide["pctrank_D5"]      > OUTLIER_PCT_HI
wide["flag_low_D10"]     = wide["pctrank_D10"]     < OUTLIER_PCT_LO
wide["flag_high_D10"]    = wide["pctrank_D10"]     > OUTLIER_PCT_HI
wide["flag_low_overall"] = wide["pctrank_overall"] < OUTLIER_PCT_LO
wide["flag_high_overall"]= wide["pctrank_overall"] > OUTLIER_PCT_HI

wide["any_outlier"] = (
    wide["flag_low_D1"] | wide["flag_high_D1"] |
    wide["flag_low_D5"] | wide["flag_high_D5"] |
    wide["flag_low_D10"]| wide["flag_high_D10"] |
    wide["flag_low_overall"] | wide["flag_high_overall"]
)

# No-gradient flag: |spearman| < 0.3
wide["flag_no_gradient"] = wide["spearman_D1_D10"].abs() < 0.3

# ── Check 2: Continuity ───────────────────────────────────────────────────────

print("Computing continuity (year-over-year changes) …")

yoy_rows = []
for (code, country), grp in dec_df.groupby(["code", "country"]):
    years = sorted(grp["year"].unique())
    if len(years) < 2:
        continue
    name = grp["name"].iloc[0]
    piv = (grp.pivot_table(index="year", columns="decile_int",
                           values="value", aggfunc="first")
           .sort_index())
    # Year gaps
    year_diffs = np.diff(years)
    max_gap = int(max(year_diffs))
    # YoY changes for D1, D5, D10
    for d in [1, 5, 10]:
        if d not in piv.columns:
            continue
        col = piv[d].dropna()
        if len(col) < 2:
            continue
        for i in range(1, len(col)):
            y0, y1 = col.index[i-1], col.index[i]
            v0, v1 = col.iloc[i-1], col.iloc[i]
            if v0 == 0:
                pct = np.nan
            else:
                pct = abs((v1 - v0) / v0 * 100)
            is_break = bool(pct > CONTINUITY_THRESHOLD) if not np.isnan(pct) else False
            yoy_rows.append({
                "code": code, "name": name, "country": country,
                "year_from": y0, "year_to": y1,
                "decile": d,
                "value_from": round(v0, 3), "value_to": round(v1, 3),
                "abs_change_pp": round(v1 - v0, 3),
                "pct_change": round(pct, 1) if not np.isnan(pct) else np.nan,
                "is_break": is_break,
                "max_year_gap": max_gap,
            })

yoy_df = pd.DataFrame(yoy_rows)
breaks_df = yoy_df[yoy_df["is_break"]].copy() if not yoy_df.empty else pd.DataFrame()

# Attach continuity summary to wide
cont_summary = (
    yoy_df.groupby(["code", "country"])
    .agg(
        n_years        = ("year_from", "nunique"),
        max_year_gap   = ("max_year_gap", "max"),
        n_breaks       = ("is_break", "sum"),
        max_pct_change = ("pct_change", "max"),
    )
    .reset_index()
)
wide = wide.merge(cont_summary, on=["code", "country"], how="left")
wide["n_years"]       = wide["n_years"].fillna(1).astype(int)
wide["n_breaks"]      = wide["n_breaks"].fillna(0).astype(int)
wide["flag_break"]    = wide["n_breaks"] > 0
wide["flag_year_gap"] = wide["max_year_gap"].fillna(0) > 3

# Flagged-by-coworker indicator
wide["coworker_flag"] = wide.apply(
    lambda r: (r["country"], r["code"]) in FLAGGED_SET, axis=1
)

# ── Assemble Summary ──────────────────────────────────────────────────────────

summary_cols = [
    "code", "name", "country", "last_year",
    "D1", "D5", "D10",
    "pctrank_D1", "pctrank_D5", "pctrank_D10", "pctrank_overall",
    "D10_D1_ratio", "spearman_D1_D10",
    "n_years", "max_year_gap", "n_breaks", "max_pct_change",
    "flag_no_gradient", "any_outlier", "flag_break", "flag_year_gap",
    "coworker_flag",
]
summary = wide[[c for c in summary_cols if c in wide.columns]].copy()
summary = summary.sort_values(["code", "country"]).reset_index(drop=True)

flagged_sheet = summary[summary["coworker_flag"]].copy()

# Distribution sheet: long format D1/D5/D10 percentile ranks, all indicators
dist_rows = []
for code, grp in wide.groupby("code"):
    for dcol in ["D1", "D5", "D10", "mean_D1_D10"]:
        prcol = f"pctrank_{dcol}" if dcol != "mean_D1_D10" else "pctrank_overall"
        sub = grp[["country", "last_year", dcol, prcol]].copy()
        sub.columns = ["country", "year", "value", "pctrank"]
        sub.insert(0, "decile", dcol)
        sub.insert(0, "code", code)
        dist_rows.append(sub)
dist_df = pd.concat(dist_rows, ignore_index=True).sort_values(
    ["code", "decile", "pctrank"])

# ── Write Excel ───────────────────────────────────────────────────────────────

print(f"Writing Excel → {OUT_PATH} …")
OUT_DIR.mkdir(parents=True, exist_ok=True)

wb = xlsxwriter.Workbook(str(OUT_PATH))

# ── Formats ──
hdr   = wb.add_format({"bold": True, "bg_color": "#2F5496", "font_color": "white",
                        "border": 1, "align": "center", "text_wrap": True})
base  = wb.add_format({"border": 1})
num1  = wb.add_format({"border": 1, "num_format": "0.0"})
num2  = wb.add_format({"border": 1, "num_format": "0.00"})
num3  = wb.add_format({"border": 1, "num_format": "0.000"})
pct_fmt = wb.add_format({"border": 1, "num_format": "0.0"})
red_bg  = wb.add_format({"border": 1, "bg_color": "#FFC7CE", "font_color": "#9C0006"})
yel_bg  = wb.add_format({"border": 1, "bg_color": "#FFEB9C", "font_color": "#9C5700"})
grn_bg  = wb.add_format({"border": 1, "bg_color": "#E2EFDA", "font_color": "#375623"})
flag_bg = wb.add_format({"border": 1, "bg_color": "#FCE4D6", "bold": True})
bool_true  = wb.add_format({"border": 1, "bg_color": "#FFC7CE", "font_color": "#9C0006"})
bool_false = wb.add_format({"border": 1, "bg_color": "#E2EFDA", "font_color": "#375623"})
coworker_fmt = wb.add_format({"border": 2, "bg_color": "#FCE4D6", "bold": True})


def write_header(ws, headers, widths=None):
    for c, h in enumerate(headers):
        ws.write(0, c, h, hdr)
    if widths:
        for c, w in enumerate(widths):
            ws.set_column(c, c, w)


def write_row(ws, r, row_data, formats):
    for c, (val, fmt) in enumerate(zip(row_data, formats)):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            ws.write_blank(r, c, None, fmt)
        else:
            ws.write(r, c, val, fmt)


# ── Sheet 1: Summary ──────────────────────────────────────────────────────────

ws_sum = wb.add_worksheet("Summary")
ws_sum.freeze_panes(1, 0)
ws_sum.set_zoom(85)

headers = [
    "Code", "Name", "Country", "Last Year",
    "D1", "D5", "D10",
    "Pctrank D1", "Pctrank D5", "Pctrank D10", "Pctrank Overall",
    "D10/D1 ratio", "Spearman D1→D10",
    "N years", "Max year gap", "N breaks", "Max % change",
    "No gradient?", "Outlier?", "YoY break?", "Year gap?",
    "Coworker flag",
]
widths = [12, 42, 9, 10, 9, 9, 9, 11, 11, 11, 14, 13, 15, 9, 13, 10, 13, 12, 10, 11, 10, 13]
write_header(ws_sum, headers, widths)

flag_bool_cols = {
    "flag_no_gradient", "any_outlier", "flag_break", "flag_year_gap", "coworker_flag"
}

for r, row in enumerate(summary.itertuples(index=False), start=1):
    is_cw = bool(row.coworker_flag)
    row_base = coworker_fmt if is_cw else base

    def prank_fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return base
        if v < OUTLIER_PCT_LO:
            return red_bg
        if v > OUTLIER_PCT_HI:
            return red_bg
        return num1

    def ratio_fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return base
        if 0.8 <= v <= 1.25:
            return yel_bg   # near 1: little gradient
        return num2

    def spear_fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return base
        if abs(v) < 0.3:
            return red_bg
        if abs(v) < 0.6:
            return yel_bg
        return grn_bg

    def bool_fmt(v):
        return bool_true if v else bool_false

    cells = [
        (getattr(row, "code", ""),          row_base),
        (getattr(row, "name", ""),          row_base),
        (getattr(row, "country", ""),       row_base),
        (getattr(row, "last_year", None),   base),
        (getattr(row, "D1", None),          num2),
        (getattr(row, "D5", None),          num2),
        (getattr(row, "D10", None),         num2),
        (getattr(row, "pctrank_D1", None),  prank_fmt(getattr(row, "pctrank_D1", None))),
        (getattr(row, "pctrank_D5", None),  prank_fmt(getattr(row, "pctrank_D5", None))),
        (getattr(row, "pctrank_D10", None), prank_fmt(getattr(row, "pctrank_D10", None))),
        (getattr(row, "pctrank_overall", None), prank_fmt(getattr(row, "pctrank_overall", None))),
        (getattr(row, "D10_D1_ratio", None),    ratio_fmt(getattr(row, "D10_D1_ratio", None))),
        (getattr(row, "spearman_D1_D10", None), spear_fmt(getattr(row, "spearman_D1_D10", None))),
        (getattr(row, "n_years", None),         base),
        (getattr(row, "max_year_gap", None),    base),
        (getattr(row, "n_breaks", None),        base),
        (getattr(row, "max_pct_change", None),  num1),
        (bool(getattr(row, "flag_no_gradient", False)),  bool_fmt(bool(getattr(row, "flag_no_gradient", False)))),
        (bool(getattr(row, "any_outlier", False)),        bool_fmt(bool(getattr(row, "any_outlier", False)))),
        (bool(getattr(row, "flag_break", False)),         bool_fmt(bool(getattr(row, "flag_break", False)))),
        (bool(getattr(row, "flag_year_gap", False)),      bool_fmt(bool(getattr(row, "flag_year_gap", False)))),
        (bool(getattr(row, "coworker_flag", False)),      bool_fmt(bool(getattr(row, "coworker_flag", False)))),
    ]
    for c, (val, fmt) in enumerate(cells):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            ws_sum.write_blank(r, c, None, fmt)
        else:
            ws_sum.write(r, c, val, fmt)

# ── Sheet 2: Flagged ──────────────────────────────────────────────────────────

ws_flag = wb.add_worksheet("Flagged")
ws_flag.freeze_panes(1, 0)
ws_flag.set_zoom(90)

# Add D2..D10 for the flagged sheet (full decile profile)
flagged_with_deciles = wide[wide["coworker_flag"]].copy()
flag_dcols = ["code", "name", "country", "last_year"] + d_col_names + [
    "pctrank_D1", "pctrank_D5", "pctrank_D10", "pctrank_overall",
    "D10_D1_ratio", "spearman_D1_D10",
    "n_years", "n_breaks", "max_pct_change",
]
flagged_with_deciles = flagged_with_deciles[
    [c for c in flag_dcols if c in flagged_with_deciles.columns]
].sort_values(["code", "country"])

flag_headers = ["Code", "Name", "Country", "Last Year"] + \
               [f"D{i}" for i in range(1, 11)] + \
               ["Pctrank D1", "Pctrank D5", "Pctrank D10", "Pctrank Overall",
                "D10/D1", "Spearman", "N years", "N breaks", "Max % chg"]
flag_widths  = [12, 42, 9, 10] + [9]*10 + [11, 11, 11, 14, 9, 10, 8, 9, 11]
write_header(ws_flag, flag_headers, flag_widths)

for r, row in enumerate(flagged_with_deciles.itertuples(index=False), start=1):
    flds = list(flag_dcols)
    for c, col in enumerate(flds):
        val = getattr(row, col, None)
        if col.startswith("pctrank_"):
            fmt = prank_fmt(val)
        elif col == "D10_D1_ratio":
            fmt = ratio_fmt(val)
        elif col == "spearman_D1_D10":
            fmt = spear_fmt(val)
        elif col in d_col_names:
            fmt = num2
        else:
            fmt = base
        if val is None or (isinstance(val, float) and np.isnan(val)):
            ws_flag.write_blank(r, c, None, fmt)
        else:
            ws_flag.write(r, c, val, fmt)

# Add cross-country context rows below each flagged row (last 3 lines of code group)
ws_flag.write(r + 2, 0, "Colour key:", hdr)
ws_flag.write(r + 2, 1, "RED pctrank = bottom or top 10% among EU27 for that indicator", base)
ws_flag.write(r + 3, 0, "Spearman:", hdr)
ws_flag.write(r + 3, 1, ">0.6 strong gradient (green), 0.3-0.6 moderate (yellow), <0.3 weak/inverted (red)", base)

# ── Sheet 3: Distribution ─────────────────────────────────────────────────────

ws_dist = wb.add_worksheet("Distribution")
ws_dist.freeze_panes(1, 0)
ws_dist.set_zoom(85)

dist_headers = ["Code", "Decile", "Country", "Year", "Value", "Pctrank (EU27)"]
dist_widths  = [12, 8, 9, 8, 12, 16]
write_header(ws_dist, dist_headers, dist_widths)

for r, row in enumerate(dist_df.itertuples(index=False), start=1):
    prank = row.pctrank
    pr_fmt = prank_fmt(prank)
    cells = [
        (row.code,    base),
        (row.decile,  base),
        (row.country, base),
        (row.year,    base),
        (row.value,   num2),
        (prank,       pr_fmt),
    ]
    for c, (val, fmt) in enumerate(cells):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            ws_dist.write_blank(r, c, None, fmt)
        else:
            ws_dist.write(r, c, val, fmt)

# ── Sheet 4: Continuity ───────────────────────────────────────────────────────

ws_cont = wb.add_worksheet("Continuity")
ws_cont.freeze_panes(1, 0)
ws_cont.set_zoom(85)

if not yoy_df.empty:
    cont_headers = ["Code", "Name", "Country", "Year from", "Year to",
                    "Decile", "Value from", "Value to",
                    "Abs change (pp)", "% change", "Break?", "Max year gap"]
    cont_widths  = [12, 42, 9, 10, 8, 8, 11, 11, 15, 11, 8, 13]
    write_header(ws_cont, cont_headers, cont_widths)

    # Sort: breaks first, then by code/country
    yoy_sorted = yoy_df.sort_values(
        ["is_break", "code", "country", "decile", "year_from"],
        ascending=[False, True, True, True, True]
    )
    for r, row in enumerate(yoy_sorted.itertuples(index=False), start=1):
        brk = bool(row.is_break)
        row_fmt = red_bg if brk else base
        cells = [
            (row.code, row_fmt), (row.name, row_fmt), (row.country, row_fmt),
            (row.year_from, base), (row.year_to, base),
            (row.decile, base),
            (row.value_from, num3), (row.value_to, num3),
            (row.abs_change_pp, num3),
            (row.pct_change, num1),
            (brk, bool_fmt(brk)),
            (row.max_year_gap, base),
        ]
        for c, (val, fmt) in enumerate(cells):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                ws_cont.write_blank(r, c, None, fmt)
            else:
                ws_cont.write(r, c, val, fmt)

wb.close()
print(f"Done → {OUT_PATH}")
print(f"  Summary rows      : {len(summary)}")
print(f"  Flagged rows      : {len(flagged_sheet)}")
print(f"  Continuity rows   : {len(yoy_df)}")
print(f"  Breaks (>{CONTINUITY_THRESHOLD}% chg) : {len(breaks_df)}")
