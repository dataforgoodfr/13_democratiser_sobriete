"""
2_ownership_heatmap_eu-silc.py
==============================
Homeownership-rate heatmaps for all EU-27 countries.

Layout (one PNG + SVG per country):
  4 stacked subplots — one per age group (18-30 / 31-45 / 46-60 / 61+)
    Rows    : income deciles  D1 (lowest income) → D10 (highest)
    Columns : survey year (all available 2004-2023)
    Colour  : homeownership rate (%)  ─  red = low, green = high

Speed design
  • Each file (R, H, D) is read exactly ONCE per (country, year).
  • All (country × year) pairs are submitted to a single shared
    ThreadPoolExecutor — maximises IO concurrency on OneDrive.
  • Decile assignment is fully vectorised with numpy.
  • Ownership-rate aggregation uses a single pandas groupby.

Income deciles are based on raw HY020 (total household disposable income)
rather than OECD-equivalised income.  Rank ordering within each
(country, year) is preserved; this is ~5× faster than full equivalisation.

Data: EU-SILC cross-sectional 2004-2023.
"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DATA_PATH = (
    r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr"
    r"/1_WSL/1_EWBI/0_data/EU-SILC"
    r"/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set"
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs', 'graphs', 'EU-SILC', 'ownership_heatmaps')
os.makedirs(OUTPUT_DIR, exist_ok=True)

EU27 = [
    'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES',
    'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT',
    'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK',
]

COUNTRY_NAMES = {
    'AT': 'Austria',    'BE': 'Belgium',     'BG': 'Bulgaria',
    'CY': 'Cyprus',     'CZ': 'Czechia',     'DE': 'Germany',
    'DK': 'Denmark',    'EE': 'Estonia',     'EL': 'Greece',
    'ES': 'Spain',      'FI': 'Finland',     'FR': 'France',
    'HR': 'Croatia',    'HU': 'Hungary',     'IE': 'Ireland',
    'IT': 'Italy',      'LT': 'Lithuania',   'LU': 'Luxembourg',
    'LV': 'Latvia',     'MT': 'Malta',       'NL': 'Netherlands',
    'PL': 'Poland',     'PT': 'Portugal',    'RO': 'Romania',
    'SE': 'Sweden',     'SI': 'Slovenia',    'SK': 'Slovakia',
}

ALL_YEARS   = list(range(2004, 2024))   # 2004-2023
MAX_WORKERS = 16   # tune to your OneDrive / network bandwidth
MIN_COUNT   = 5    # minimum unweighted observations to show a cell
YEAR_STEP   = 2    # pool this many consecutive years per column

AGE_GROUPS = {
    1: '18–30',
    2: '31–45',
    3: '46–60',
    4: '61+',
}

VMIN, VMAX = 0, 100

# Red-Yellow-Green: red = low ownership, green = high ownership
_CMAP = plt.cm.RdYlGn.copy()
_CMAP.set_bad('lightgrey')   # NaN cells → grey

# Tenure codes
_TENURE_OLD = {1: 'owner', 2: 'renter', 3: 'renter', 4: 'renter'}          # HH020 (< 2010)
_TENURE_NEW = {1: 'owner', 2: 'owner',  3: 'renter', 4: 'renter', 5: 'renter'}  # HH021 (≥ 2010)


# ============================================================================
# SMALL HELPERS
# ============================================================================

def _file_code(country: str, year: int) -> str:
    """Greece used file-code 'GR' before 2008."""
    return 'GR' if (country == 'EL' and year <= 2007) else country


def _tenure_col(year: int) -> str:
    return 'HH020' if year < 2010 else 'HH021'


def _tenure_vec(series: pd.Series, year: int) -> pd.Series:
    """Vectorised mapping of tenure code → 'owner' / 'renter' / NaN."""
    return series.map(_TENURE_NEW if year >= 2010 else _TENURE_OLD)


def _age_group_vec(age: pd.Series) -> pd.Series:
    """Vectorised age-group assignment (returns float for NaN compatibility)."""
    out = np.full(len(age), np.nan)
    a = age.to_numpy()
    out[(a >= 18) & (a <= 30)] = 1
    out[(a >= 31) & (a <= 45)] = 2
    out[(a >= 46) & (a <= 60)] = 3
    out[a >= 61]               = 4
    return pd.Series(out, index=age.index)


def _oecd_weight_scalar(age) -> float:
    """OECD modified equivalence scale weight for a single person."""
    if pd.isna(age):
        return 0.5
    try:
        a = int(age)
    except Exception:
        return 0.5
    return 0.3 if a < 14 else 0.5


# ============================================================================
# FAST SINGLE-PASS LOADER  (one call per country/year)
# ============================================================================

def load_year(country: str, year: int) -> pd.DataFrame | None:
    """
    Read R, H, D files once each; return a lean DataFrame:
        age_group (int 1-4), decile (int 1-10), is_owner (float 0/1), weight (float)

    Deciles are based on OECD-equivalised household income, matching
    eu_silc_tenure_by_age_and_decile_analysis_FR.py exactly.
    """
    fc  = _file_code(country, year)
    tag = str(year)[-2:]
    root = f"{BASE_DATA_PATH}/{country}/{year}/UDB_c{fc}{tag}"

    r_path = root + 'R.csv'
    h_path = root + 'H.csv'
    d_path = root + 'D.csv'
    if not (os.path.exists(r_path) and os.path.exists(h_path) and os.path.exists(d_path)):
        return None

    try:
        # ── R FILE: read ALL persons once ─────────────────────────────────
        # Needed for both OECD equivalent sizes (all members) and
        # independent-adult filtering (RB220/RB230) + age groups (RB080).
        r_avail = set(pd.read_csv(r_path, nrows=0).columns)
        r_want  = ['RB010', 'RB020', 'RB030', 'RB080', 'RB100',
                   'RB081', 'RB082', 'RB220', 'RB230']
        r_use   = [c for c in r_want if c in r_avail]
        if 'RB080' not in r_use:
            return None

        r_all = pd.read_csv(r_path, usecols=r_use, on_bad_lines='skip',
                            dtype={'RB010': str, 'RB020': str, 'RB030': str})
        r_all['RB030'] = r_all['RB030'].fillna('0').astype(str)
        r_all['hh_id'] = r_all['RB030'].str[:-2]   # strip last 2 chars → household key

        # Age for OECD scale (RB081 = age in years preferred, else RB082)
        if 'RB081' in r_all.columns and 'RB082' in r_all.columns:
            r_all['age_oecd'] = r_all['RB081'].fillna(r_all['RB082'])
        elif 'RB081' in r_all.columns:
            r_all['age_oecd'] = r_all['RB081']
        elif 'RB082' in r_all.columns:
            r_all['age_oecd'] = r_all['RB082']
        else:
            r_all['age_oecd'] = np.nan

        # ── OECD equivalent sizes ──────────────────────────────────────────
        # Reference person (oldest adult per HH) gets weight 1.0;
        # everyone else gets 0.5 (adult) or 0.3 (child < 14).
        r_all['oecd_w'] = r_all['age_oecd'].apply(_oecd_weight_scalar)
        r_all.sort_values(['RB010', 'RB020', 'hh_id', 'age_oecd'],
                          ascending=[True, True, True, False], inplace=True)
        r_all['_rank'] = r_all.groupby(['RB010', 'RB020', 'hh_id']).cumcount()
        r_all.loc[r_all['_rank'] == 0, 'oecd_w'] = 1.0

        equiv_size = (
            r_all.groupby(['RB010', 'RB020', 'hh_id'])['oecd_w']
                 .sum().reset_index()
                 .rename(columns={'RB010': 'HB010', 'RB020': 'HB020',
                                  'hh_id': 'HB030', 'oecd_w': 'equiv_size'})
        )

        # ── Filter independent adults (RB220=NaN AND RB230=NaN) ───────────
        r_adults = r_all.copy()
        if 'RB220' in r_adults.columns:
            r_adults = r_adults[r_adults['RB220'].isna()]
        if 'RB230' in r_adults.columns:
            r_adults = r_adults[r_adults['RB230'].isna()]
        r_adults = r_adults.dropna(subset=['RB080'])
        r_adults['RB080'] = r_adults['RB080'].astype(int)
        if 'RB100' in r_adults.columns and r_adults['RB100'].notna().any():
            iy = r_adults['RB100'].fillna(year).astype(int)
        else:
            iy = year
        r_adults = r_adults.copy()
        r_adults['age'] = iy - r_adults['RB080']
        r_adults = r_adults[(r_adults['age'] >= 18)].copy()
        r_adults['age_group'] = _age_group_vec(r_adults['age'])
        r_adults.dropna(subset=['age_group'], inplace=True)
        r_adults['age_group'] = r_adults['age_group'].astype(int)
        if r_adults.empty:
            return None

        # ── H FILE: tenure + raw household income ─────────────────────────
        tc    = _tenure_col(year)
        h_use = ['HB010', 'HB020', 'HB030', tc, 'HY020']
        h = pd.read_csv(h_path, usecols=h_use, on_bad_lines='skip',
                        dtype={'HB010': str, 'HB020': str, 'HB030': str})
        h['tenure'] = _tenure_vec(pd.to_numeric(h[tc], errors='coerce'), year)
        h = h[h['tenure'].notna() & h['HY020'].notna()].copy()
        if h.empty:
            return None

        # ── Merge H × equiv_size → equivalised income ─────────────────────
        h = h.merge(equiv_size, on=['HB010', 'HB020', 'HB030'], how='left')
        h.dropna(subset=['equiv_size'], inplace=True)
        h['equi_disp_inc'] = h['HY020'] / h['equiv_size']
        h.dropna(subset=['equi_disp_inc'], inplace=True)
        if h.empty:
            return None

        # ── D FILE: household cross-sectional weights ──────────────────────
        # DB030 is already the household ID — no stripping needed.
        d = pd.read_csv(d_path, usecols=['DB010', 'DB020', 'DB030', 'DB090'],
                        on_bad_lines='skip',
                        dtype={'DB010': str, 'DB020': str, 'DB030': str,
                               'DB090': float})
        d.dropna(subset=['DB090'], inplace=True)
        if d.empty:
            return None

        # ── Merge H × D (straight join on household ID) ───────────────────
        h = h.merge(
            d.rename(columns={'DB010': 'HB010', 'DB020': 'HB020',
                               'DB030': 'HB030', 'DB090': 'weight'}),
            on=['HB010', 'HB020', 'HB030'], how='inner',
        )
        h.dropna(subset=['weight'], inplace=True)
        if h.empty:
            return None

        # ── Decile thresholds (same method as reference script) ────────────
        # Sort by equivalised income, find threshold at each decile boundary
        # using cumulative weight share (idxmin approach).
        h_sorted = h.sort_values('equi_disp_inc').reset_index(drop=True)
        h_sorted['cum_pct'] = h_sorted['weight'].cumsum() / h_sorted['weight'].sum()
        thresholds = {}
        for q in range(1, 10):
            idx = (h_sorted['cum_pct'] - q / 10.0).abs().idxmin()
            thresholds[q] = h_sorted.loc[idx, 'equi_disp_inc']

        # Vectorised assignment: for each household, find lowest threshold it falls under
        inc = h['equi_disp_inc'].to_numpy()
        decile_arr = np.full(len(inc), 10, dtype=np.int8)
        for q in range(9, 0, -1):
            decile_arr[inc <= thresholds[q]] = q
        h['decile'] = decile_arr

        # ── Merge adults → household data ──────────────────────────────────
        m = r_adults[['RB010', 'RB020', 'hh_id', 'age_group']].merge(
            h[['HB010', 'HB020', 'HB030', 'tenure', 'decile', 'weight']],
            left_on=['RB010', 'RB020', 'hh_id'],
            right_on=['HB010', 'HB020', 'HB030'],
            how='inner',
        )
        m.dropna(subset=['tenure', 'decile', 'weight'], inplace=True)
        if m.empty:
            return None

        m['is_owner'] = (m['tenure'] == 'owner').astype(np.float32)
        # 'count' = unweighted number of persons (used for the ≥5 threshold)
        m['count'] = 1
        return m[['age_group', 'decile', 'is_owner', 'weight', 'count']].copy()

    except Exception as exc:
        print(f"  [{country}/{year}] {type(exc).__name__}: {exc}")
        return None


# ============================================================================
# BUILD PER-COUNTRY HEATMAP MATRICES
# ============================================================================

def _pair_label(years: list[int]) -> str:
    """Turn [2004, 2005] → '2004-05', [2006] → '2006'."""
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{str(years[-1])[-2:]}"


def _year_pairs(available_years: list[int]) -> list[tuple[str, list[int]]]:
    """
    Group sorted available years into consecutive YEAR_STEP-sized buckets
    aligned to ALL_YEARS order.

    Returns [(label, [yr1, yr2, ...]), ...] preserving chronological order.
    """
    all_sorted = sorted(ALL_YEARS)
    # Build buckets based on position in ALL_YEARS
    buckets: list[list[int]] = []
    cur: list[int] = []
    for i, yr in enumerate(all_sorted):
        cur.append(yr)
        if len(cur) == YEAR_STEP or i == len(all_sorted) - 1:
            buckets.append(cur)
            cur = []

    available = set(available_years)
    result = []
    for bucket in buckets:
        present = [y for y in bucket if y in available]
        if present:
            result.append((_pair_label(bucket), present))
    return result


def build_matrices(
    country_data: dict[int, pd.DataFrame],
) -> dict[int, pd.DataFrame]:
    """
    Aggregate all years for one country into per-age-group pivot tables.

    Years are pooled in YEAR_STEP-sized consecutive pairs (e.g. 2004-05,
    2006-07 …).  Cells with fewer than MIN_COUNT unweighted observations
    are set to NaN (shown as grey).

    Returns {age_group: DataFrame(index=D1-D10, columns=pair_labels)}
    where each cell is the weighted homeownership rate (%) or NaN.
    """
    frames = []
    for yr, df in country_data.items():
        tmp = df.copy()
        tmp['year']    = yr
        tmp['w_owner'] = tmp['is_owner'] * tmp['weight']
        frames.append(tmp)

    full = pd.concat(frames, ignore_index=True)

    pairs = _year_pairs(sorted(country_data.keys()))
    pair_labels = [lbl for lbl, _ in pairs]

    # Map each year → its pair label
    year_to_label: dict[int, str] = {}
    for lbl, yrs in pairs:
        for y in yrs:
            year_to_label[y] = lbl
    full['pair'] = full['year'].map(year_to_label)

    agg = (
        full.groupby(['pair', 'age_group', 'decile'], sort=False)
            .agg(
                w_owner=('w_owner', 'sum'),
                total_w=('weight', 'sum'),
                n=('count', 'sum'),          # unweighted count
            )
            .reset_index()
    )
    # Apply minimum-count threshold
    agg.loc[agg['n'] < MIN_COUNT, 'w_owner'] = np.nan
    agg.loc[agg['n'] < MIN_COUNT, 'total_w'] = np.nan
    agg['rate'] = agg['w_owner'] / agg['total_w'] * 100

    matrices: dict[int, pd.DataFrame] = {}
    for ag in AGE_GROUPS:
        sub = agg[agg['age_group'] == ag]
        mat = sub.pivot(index='decile', columns='pair', values='rate')
        matrices[ag] = mat.reindex(index=range(1, 11), columns=pair_labels)
    return matrices


# ============================================================================
# PLOTTING
# ============================================================================

def plot_heatmap(country: str, matrices: dict[int, pd.DataFrame]) -> None:
    name = COUNTRY_NAMES.get(country, country)
    # Collect ordered pair labels from the first non-empty matrix
    col_labels: list[str] = []
    for m in matrices.values():
        if m is not None and not m.empty:
            col_labels = list(m.columns)
            break
    n_cols = len(col_labels)
    n_ages = len(AGE_GROUPS)

    fig, axes = plt.subplots(
        n_ages, 1,
        figsize=(max(9, n_cols * 0.85 + 3), 3.5 * n_ages),
        squeeze=False,
    )
    fig.suptitle(
        f'Homeownership Rate (%)  —  {name} ({country})',
        fontsize=14, fontweight='bold',
    )

    for row, (ag, ag_label) in enumerate(AGE_GROUPS.items()):
        ax  = axes[row][0]
        mat = matrices.get(ag)
        if mat is None or mat.empty:
            ax.axis('off')
            continue

        data   = mat.reindex(columns=col_labels).values.astype(float)   # (10, n_cols)
        masked = np.ma.masked_invalid(data)

        im = ax.imshow(masked, aspect='auto', cmap=_CMAP,
                       vmin=VMIN, vmax=VMAX, interpolation='nearest')

        # Annotate each cell with the value
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    txt_col = 'white' if (v < 15 or v > 82) else 'black'
                    ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                            fontsize=6.5, color=txt_col, fontweight='bold')

        # Y axis: decile labels
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels([f'D{d}' for d in range(1, 11)], fontsize=8)
        ax.set_ylabel(f'Age {ag_label}', fontsize=10, fontweight='bold', labelpad=6)

        # X axis: show pair labels only on the bottom subplot
        ax.set_xticks(np.arange(n_cols))
        if row == n_ages - 1:
            ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha='right')
            ax.set_xlabel('Survey period', fontsize=9)
        else:
            ax.set_xticklabels([])

        # Thin grid to separate cells visually
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.5)
        ax.tick_params(which='minor', length=0)

        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
        cb.set_label('%', fontsize=8)
        cb.ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    base = os.path.join(OUTPUT_DIR, f'{country}_ownership_heatmap')
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(base + '.svg', format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [{country}]  → {country}_ownership_heatmap.png / .svg")


# ============================================================================
# EXCEL EXPORT
# ============================================================================

# Colour scale helpers (red → yellow → green, matching the heatmap)
_RED   = openpyxl.styles.Color('FFF03B20')   # ~red
_YELLOW= openpyxl.styles.Color('FFFFEB00')   # ~yellow
_GREEN = openpyxl.styles.Color('FF4CAF50')   # ~green


def _rate_to_hex(rate: float) -> str:
    """Map 0-100 ownership rate to a hex fill colour (red→yellow→green)."""
    t = max(0.0, min(1.0, rate / 100.0))
    if t < 0.5:
        # red → yellow
        r, g, b = 240, int(t * 2 * 235), 32
    else:
        # yellow → green
        s = (t - 0.5) * 2
        r, g, b = int(240 - s * 164), int(235 + s * 20 - s * 55), int(32 + s * 50)
    return f'FF{r:02X}{g:02X}{b:02X}'


def export_excel(
    country_matrices: dict[str, dict[int, pd.DataFrame]],
    out_dir: str,
) -> None:
    """
    One Excel workbook, one sheet per country.

    Layout inside each sheet:
        • 4 blocks, one per age group, stacked vertically with a blank separator.
        • Rows  : D1 (lowest income) → D10 (highest income).
        • Columns: survey years.
        • Cells  : ownership rate in %, colour-coded red→yellow→green.
    """
    xlsx_path = os.path.join(out_dir, 'ownership_heatmaps_all_countries.xlsx')
    wb = openpyxl.Workbook()
    wb.remove(wb.active)   # remove default empty sheet

    hdr_font  = Font(bold=True)
    ctr_align = Alignment(horizontal='center', vertical='center')

    for cc in EU27:
        mats = country_matrices.get(cc)
        if mats is None:
            continue
        name = COUNTRY_NAMES.get(cc, cc)
        ws   = wb.create_sheet(title=f'{cc} – {name}'[:31])   # sheet names ≤ 31 chars

        # Gather ordered pair labels from the first non-empty matrix
        col_labels: list[str] = []
        for m in mats.values():
            if m is not None and not m.empty:
                col_labels = list(m.columns)
                break

        current_row = 1
        for ag, ag_label in AGE_GROUPS.items():
            mat = mats.get(ag)

            # ── Age-group header ─────────────────────────────────────────
            ws.cell(current_row, 1, f'Age group: {ag_label}').font = Font(bold=True, size=11)
            ws.merge_cells(
                start_row=current_row, start_column=1,
                end_row=current_row,   end_column=1 + len(col_labels),
            )
            ws.cell(current_row, 1).alignment = ctr_align
            current_row += 1

            # ── Column headers (period labels) ───────────────────────────
            ws.cell(current_row, 1, 'Decile \ Period').font = hdr_font
            ws.cell(current_row, 1).alignment = ctr_align
            for col_idx, lbl in enumerate(col_labels, start=2):
                c = ws.cell(current_row, col_idx, lbl)
                c.font      = hdr_font
                c.alignment = ctr_align
            current_row += 1

            # ── Data rows (D1 → D10) ────────────────────────────────────
            for decile in range(1, 11):
                ws.cell(current_row, 1, f'D{decile}').font = hdr_font
                ws.cell(current_row, 1).alignment = ctr_align
                for col_idx, lbl in enumerate(col_labels, start=2):
                    if mat is not None and lbl in mat.columns:
                        val = mat.loc[decile, lbl] if decile in mat.index else float('nan')
                    else:
                        val = float('nan')
                    cell = ws.cell(current_row, col_idx)
                    if not (isinstance(val, float) and np.isnan(val)):
                        cell.value     = round(float(val), 1)
                        cell.fill      = PatternFill('solid', fgColor=_rate_to_hex(float(val)))
                        txt_col        = 'FFFFFFFF' if (val < 15 or val > 82) else 'FF000000'
                        cell.font      = Font(color=txt_col)
                    cell.alignment = ctr_align
                    cell.number_format = '0.0'
                current_row += 1

            current_row += 2   # blank separator between age groups

        # ── Column widths ────────────────────────────────────────────────
        ws.column_dimensions['A'].width = 15
        for col_idx in range(2, 2 + len(col_labels)):
            ws.column_dimensions[get_column_letter(col_idx)].width = 9

    wb.save(xlsx_path)
    print(f'  Excel → {os.path.basename(xlsx_path)}')


# ============================================================================
# SNAPSHOT HEATMAP — one row per EU-27 country, metrics from 2022-2023
# ============================================================================

SNAPSHOT_YEARS = (2022, 2023)   # pooled late period

# Cluster thresholds — must match 0_clustering.py exactly
_PERF_CUT = 0.006972
_EWBI_CUT = 0.7
_METHOD5_STEP = 5000
_CLUSTER_NAMES  = {0: 'Low perf / Low EWBI',  1: 'Low perf / High EWBI',
                   2: 'High perf / Low EWBI', 3: 'High perf / High EWBI'}
_CLUSTER_COLORS = ['#fb8072', '#fdb462', '#8dd3c7', '#80b1d3']

# Path to Eurostat tenure-status CSV (same folder used by 2_ownership.py)
_EUROSTAT_TENURE_CSV = os.path.join(BASE_DIR, 'external_data', 'eurostat_tenure status.csv')

# Path to EWBI master aggregated CSV (same as 0_clustering.py / 0_EWBI_priorities.py)
_EWBI_MASTER_CSV = os.path.join(
    os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..')),
    'output', 'ewbi_master_aggregated.csv',
)

# Mapping from EU-SILC ISO codes used in EU27 list → Eurostat full country names
_EUSILC_ISO_TO_ESTAT_NAME = {
    'AT': 'Austria',     'BE': 'Belgium',     'BG': 'Bulgaria',
    'HR': 'Croatia',     'CY': 'Cyprus',      'CZ': 'Czechia',
    'DK': 'Denmark',     'EE': 'Estonia',     'EL': 'Greece',
    'ES': 'Spain',       'FI': 'Finland',     'FR': 'France',
    'DE': 'Germany',     'HU': 'Hungary',     'IE': 'Ireland',
    'IT': 'Italy',       'LT': 'Lithuania',   'LU': 'Luxembourg',
    'LV': 'Latvia',      'MT': 'Malta',       'NL': 'Netherlands',
    'PL': 'Poland',      'PT': 'Portugal',    'RO': 'Romania',
    'SE': 'Sweden',      'SI': 'Slovenia',    'SK': 'Slovakia',
}

# 6 columns: ownership %, D10/D1 ratio, D10-D1 pp diff, 61+/18-30 ratio, 61+-18-30 pp diff,
#             Energy & Housing EWBI (different colormap to distinguish EWBI scale 0-1)
_SNAP_COLS = [
    ('overall',      'Overall\nownership %\n(Eurostat)',  'RdYlGn'),
    ('d10d1_all',    'D10/D1\n(ratio)',                   'RdYlGn_r'),
    ('d10_m_d1',     'D10−D1\n(pp diff)',                 'RdYlGn_r'),
    ('age_ratio',    '61+ / 18–30\n(ratio)',              'RdYlGn_r'),
    ('age_diff',     '61+ − 18–30\n(pp diff)',            'RdYlGn_r'),
    ('eh_ewbi',      'Energy &\nHousing\n(EWBI)',         'RdBu_r'),
]


def _load_eurostat_ownership() -> dict:
    """
    Returns {eu_silc_iso: ownership_pct} from Eurostat tenure-status CSV.
    Filters tenure='Owner', incgrp='Total', hhtyp='Total'.
    Uses each country's most recent year.
    """
    if not os.path.exists(_EUROSTAT_TENURE_CSV):
        print(f'  [snapshot] Eurostat tenure CSV not found: {_EUROSTAT_TENURE_CSV}')
        return {}
    try:
        df = pd.read_csv(_EUROSTAT_TENURE_CSV)
        mask = (df['tenure'] == 'Owner') & (df['incgrp'] == 'Total') & (df['hhtyp'] == 'Total')
        df = df[mask].copy()
        df['value'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
        df = df.dropna(subset=['value'])
        idx = df.groupby('geo')['TIME_PERIOD'].idxmax()
        df = df.loc[idx].copy()
        name_to_pct = dict(zip(df['geo'], df['value']))
        result = {}
        for iso, name in _EUSILC_ISO_TO_ESTAT_NAME.items():
            if name in name_to_pct:
                result[iso] = float(name_to_pct[name])
        return result
    except Exception as exc:
        print(f'  [snapshot] Failed to load Eurostat ownership: {exc}')
        return {}


def _load_ewbi_energy_housing() -> dict:
    """
    Returns {iso_code: ewbi_value} for EU priority 'Energy and Housing' (Level 2,
    All Deciles), using each country's most recent year.
    """
    if not os.path.exists(_EWBI_MASTER_CSV):
        print(f'  [snapshot] EWBI master CSV not found: {_EWBI_MASTER_CSV}')
        return {}
    try:
        df = pd.read_csv(_EWBI_MASTER_CSV, low_memory=False)
        mask = (
            (df['Level'] == 2) &
            (df['Decile'].astype(str) == 'All Deciles') &
            (df['EU priority'] == 'Energy and Housing') &
            (~df['Country'].isin(['EU-27', 'All Countries']))
        )
        df = df[mask].copy()
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Value'])
        idx = df.groupby('Country')['Year'].idxmax()
        df = df.loc[idx].copy()
        return {row['Country']: float(row['Value']) for _, row in df.iterrows()}
    except Exception as exc:
        print(f'  [snapshot] Failed to load EWBI Energy & Housing: {exc}')
        return {}


def _safe_div(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def _load_cluster_map() -> dict:
    """
    Derive cluster id (0-3) per country by replicating 0_clustering.py's
    load_and_prepare_data logic (PERF_CUT + EWBI_CUT thresholds).
    Returns {country_code: cluster_id} or {} on failure.
    """
    try:
        well_being_dir = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))
        data_csv   = os.path.join(well_being_dir, 'output', 'ewbi_master_aggregated.csv')
        income_csv = os.path.join(BASE_DIR, 'outputs', 'data', 'median_income_by_decile.csv')
        if not (os.path.exists(data_csv) and os.path.exists(income_csv)):
            print(f'  [cluster] CSV not found — falling back to ownership sort')
            return {}

        df = pd.read_csv(data_csv, low_memory=False)

        # ── EWBI per decile (Level 1) ──────────────────────────────────────
        ewbi_d = df[
            (df['Level'] == 1) &
            (df['Decile'] != 'All Deciles') &
            (~df['Country'].isin(['EU-27', 'All Countries']))
        ].copy()
        ewbi_d['Decile'] = pd.to_numeric(ewbi_d['Decile'], errors='coerce')
        ewbi_d['Year']   = pd.to_numeric(ewbi_d['Year'],   errors='coerce')
        ewbi_d['Value']  = pd.to_numeric(ewbi_d['Value'],  errors='coerce')
        ewbi_d = ewbi_d.dropna(subset=['Country', 'Year', 'Decile', 'Value'])

        # ── EWBI overall (Level 1, All Deciles) ───────────────────────────
        ewbi_all = df[
            (df['Level'] == 1) &
            (df['Decile'] == 'All Deciles') &
            (~df['Country'].isin(['EU-27', 'All Countries']))
        ].copy()
        ewbi_all['Year']  = pd.to_numeric(ewbi_all['Year'],  errors='coerce')
        ewbi_all['Value'] = pd.to_numeric(ewbi_all['Value'], errors='coerce')
        ewbi_all = ewbi_all.dropna(subset=['Country', 'Year', 'Value'])
        ewbi_last = (ewbi_all.sort_values('Year')
                             .groupby('Country').tail(1)[['Country', 'Value']]
                             .rename(columns={'Value': 'EWBI_Last'}))

        # ── Median income ─────────────────────────────────────────────────
        inc_df = pd.read_csv(income_csv)
        inc_df['year']   = pd.to_numeric(inc_df['year'],   errors='coerce')
        inc_df['decile'] = pd.to_numeric(inc_df['decile'], errors='coerce')
        inc_df['median_equi_disp_inc'] = pd.to_numeric(
            inc_df['median_equi_disp_inc'], errors='coerce')
        inc_df = inc_df.dropna(subset=['country', 'year', 'decile', 'median_equi_disp_inc'])

        merged = ewbi_d.merge(
            inc_df, left_on=['Country', 'Year', 'Decile'],
            right_on=['country', 'year', 'decile'], how='inner',
        ).dropna(subset=['median_equi_disp_inc', 'Value'])

        last_yr = merged.groupby('Country')['Year'].max().reset_index()
        last_yr.columns = ['Country', 'Last_Year']
        pts = merged.merge(last_yr, on='Country')
        pts = pts[pts['Year'] == pts['Last_Year']].copy()

        pts['bin'] = (np.round(pts['median_equi_disp_inc'] / _METHOD5_STEP) * _METHOD5_STEP)
        bench = (pts.groupby('bin', as_index=False)['Value'].mean()
                    .sort_values('bin')
                    .rename(columns={'Value': 'bench'}))
        pts['ewbi_exp'] = np.interp(
            pts['median_equi_disp_inc'].values.astype(float),
            bench['bin'].values.astype(float),
            bench['bench'].values.astype(float),
        )
        pts['residual'] = pts['Value'] - pts['ewbi_exp']

        perf = (pts.groupby('Country', as_index=False)
                   .agg(Performance_Score=('residual', 'mean')))

        features = perf.merge(ewbi_last, on='Country', how='inner')
        features = features.dropna(subset=['Performance_Score', 'EWBI_Last'])

        cluster_map_vals = {
            ('Low performer',  'Low EWBI'):  0,
            ('Low performer',  'High EWBI'): 1,
            ('High performer', 'Low EWBI'):  2,
            ('High performer', 'High EWBI'): 3,
        }
        features['pg'] = np.where(features['Performance_Score'] >= _PERF_CUT,
                                   'High performer', 'Low performer')
        features['eg'] = np.where(features['EWBI_Last'] >= _EWBI_CUT,
                                   'High EWBI', 'Low EWBI')
        features['Cluster'] = features.apply(
            lambda r: cluster_map_vals[(r['pg'], r['eg'])], axis=1)

        return dict(zip(features['Country'], features['Cluster'].astype(int)))

    except Exception as exc:
        print(f'  [cluster] failed ({exc}) — falling back to ownership sort')
        return {}


def _snapshot_rates(country_store: dict, cc: str) -> dict | None:
    """
    Pool SNAPSHOT_YEARS frames for country *cc*.
    Returns {age_group: {decile: weighted_ownership_%}} or None.
    """
    frames = [
        country_store[cc][yr]
        for yr in SNAPSHOT_YEARS
        if cc in country_store and yr in country_store[cc]
    ]
    if not frames:
        return None
    full = pd.concat(frames, ignore_index=True)
    full['w_owner'] = full['is_owner'] * full['weight']
    agg = (
        full.groupby(['age_group', 'decile'])
            .agg(w_owner=('w_owner', 'sum'),
                 total_w=('weight', 'sum'),
                 n=('count', 'sum'))
            .reset_index()
    )
    agg.loc[agg['n'] < MIN_COUNT, ['w_owner', 'total_w']] = np.nan
    agg['rate'] = agg['w_owner'] / agg['total_w'] * 100

    rates: dict[int, dict[int, float]] = {}
    for _, row in agg.iterrows():
        rates.setdefault(int(row['age_group']), {})[int(row['decile'])] = (
            float(row['rate']) if not np.isnan(row['rate']) else np.nan
        )
    return rates or None


def _snapshot_metrics(rates: dict) -> dict:
    """Compute 5 scalar metrics from an {age_group: {decile: %}} dict."""
    all_vals = [
        v for ag in AGE_GROUPS for d in range(1, 11)
        if not np.isnan(v := rates.get(ag, {}).get(d, np.nan))
    ]
    overall = float(np.nanmean(all_vals)) if all_vals else np.nan

    d1_vals  = [rates.get(ag, {}).get(1,  np.nan) for ag in AGE_GROUPS]
    d10_vals = [rates.get(ag, {}).get(10, np.nan) for ag in AGE_GROUPS]
    d1_mean  = float(np.nanmean([v for v in d1_vals  if not np.isnan(v)])) if any(not np.isnan(v) for v in d1_vals)  else np.nan
    d10_mean = float(np.nanmean([v for v in d10_vals if not np.isnan(v)])) if any(not np.isnan(v) for v in d10_vals) else np.nan

    ag4_vals = [rates.get(4, {}).get(d, np.nan) for d in range(1, 11)]
    ag1_vals = [rates.get(1, {}).get(d, np.nan) for d in range(1, 11)]
    ag4_mean = float(np.nanmean([v for v in ag4_vals if not np.isnan(v)])) if any(not np.isnan(v) for v in ag4_vals) else np.nan
    ag1_mean = float(np.nanmean([v for v in ag1_vals if not np.isnan(v)])) if any(not np.isnan(v) for v in ag1_vals) else np.nan

    return {
        'overall':   overall,
        'd10d1_all': _safe_div(d10_mean, d1_mean),
        'd10_m_d1':  (d10_mean - d1_mean) if not (np.isnan(d10_mean) or np.isnan(d1_mean)) else np.nan,
        'age_ratio': _safe_div(ag4_mean, ag1_mean),
        'age_diff':  (ag4_mean - ag1_mean) if not (np.isnan(ag4_mean) or np.isnan(ag1_mean)) else np.nan,
    }


def plot_snapshot_heatmap(country_store: dict) -> None:
    """
    One-row-per-country heatmap with 3 columns:
      Overall ownership %  |  D10/D1  |  61+/18-30
    Rows grouped by cluster (0→3), sorted by country name within each cluster.
    Saved to OUTPUT_DIR as eu27_ownership_snapshot_heatmap.png/.svg/.xlsx
    """
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm

    cluster_map = _load_cluster_map()
    eurostat_ownership = _load_eurostat_ownership()
    ewbi_eh = _load_ewbi_energy_housing()

    rows = []
    for cc in EU27:
        rates = _snapshot_rates(country_store, cc)
        if rates is None:
            print(f'  [snapshot] {cc} — no data, skipped')
            continue
        metrics = _snapshot_metrics(rates)
        # Override overall ownership with Eurostat value when available
        if cc in eurostat_ownership:
            metrics['overall'] = eurostat_ownership[cc]
        # Energy & Housing EWBI score
        metrics['eh_ewbi'] = ewbi_eh.get(cc, np.nan)
        rows.append({
            'Country':      cc,
            'Country_Name': COUNTRY_NAMES.get(cc, cc),
            'Cluster':      cluster_map.get(cc, -1),
            **metrics,
        })

    if not rows:
        print('  [snapshot] no data — skipped')
        return

    df = pd.DataFrame(rows)

    # Sort: cluster (0→3), then country name alphabetically within cluster
    # Countries without a cluster assignment (-1) go last, sorted by ownership
    has_cluster  = df[df['Cluster'] >= 0].sort_values(['Cluster', 'Country_Name'])
    no_cluster   = df[df['Cluster'] < 0].sort_values('overall', ascending=False)
    df = pd.concat([has_cluster, no_cluster], ignore_index=True)

    n_rows = len(df)
    col_keys   = [k   for k, _,  _  in _SNAP_COLS]
    col_labels = [lbl for _, lbl, _  in _SNAP_COLS]
    col_cmaps  = [cm  for _, _,  cm in _SNAP_COLS]
    n_cols = len(col_keys)

    col_min = {k: df[k].min() for k in col_keys}
    col_max = {k: df[k].max() for k in col_keys}

    cell_w, cell_h = 1.0, 1.0
    label_margin = 3.0    # space to the left of the grid for country names + cluster label
    fig_w = n_cols * cell_w + label_margin + 0.5
    fig_h = n_rows * 0.44 + 1.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-label_margin, n_cols * cell_w + 0.2)
    ax.set_ylim(n_rows * cell_h - 0.5, -1.3)
    ax.set_axis_off()

    # Track cluster group boundaries for separators + labels
    prev_cluster = None
    cluster_start_rows: list[tuple[int, int]] = []   # (cluster_id, first_row_idx)

    for row_idx, row in df.iterrows():
        cl = row['Cluster']
        if cl != prev_cluster:
            cluster_start_rows.append((int(cl), row_idx))
            prev_cluster = cl

        # Cluster colour for the country name (grey if unassigned)
        name_color = _CLUSTER_COLORS[cl] if 0 <= cl <= 3 else '#888888'

        for col_idx, (key, cmap_name) in enumerate(zip(col_keys, col_cmaps)):
            val  = row[key]
            cmap = mcm.get_cmap(cmap_name)
            vmin, vmax = col_min[key], col_max[key]

            if pd.isna(val) or vmax == vmin:
                face, txt = '#e0e0e0', '—'
            else:
                norm_val = (val - vmin) / (vmax - vmin)
                face = mcolors.to_hex(cmap(norm_val))
                if key == 'overall':
                    txt = f'{val:.1f}%'
                elif key in ('d10_m_d1', 'age_diff'):
                    txt = f'{val:+.1f}pp'
                elif key == 'eh_ewbi':
                    txt = f'{val:.3f}'
                else:
                    txt = f'{val:.2f}'

            ax.add_patch(plt.Rectangle(
                (col_idx * cell_w, row_idx * cell_h - 0.5), cell_w, cell_h,
                facecolor=face, edgecolor='white', linewidth=0.7,
            ))
            r, g, b, _ = mcolors.to_rgba(face)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(col_idx * cell_w + 0.5, row_idx * cell_h, txt,
                    ha='center', va='center', fontsize=8,
                    color='white' if lum < 0.45 else 'black')

        # Country name, coloured by cluster
        ax.text(-0.12, row_idx * cell_h, row['Country_Name'],
                ha='right', va='center', fontsize=8, fontweight='bold',
                color=name_color)

    # Cluster labels + horizontal separator lines
    for i, (cl, first_row) in enumerate(cluster_start_rows):
        # Separator line above this cluster (skip very first)
        if i > 0:
            y_sep = first_row * cell_h - 0.5
            ax.axhline(y_sep, xmin=0, xmax=1, color='#555555',
                       linewidth=1.2, linestyle='--')

        # Cluster label in left margin
        if 0 <= cl <= 3:
            # Determine last row of this cluster
            if i + 1 < len(cluster_start_rows):
                last_row = cluster_start_rows[i + 1][1] - 1
            else:
                last_row = n_rows - 1
            mid_y = (first_row + last_row) / 2.0 * cell_h
            ax.text(
                -(label_margin - 0.1), mid_y,
                f'C{cl} – {_CLUSTER_NAMES[cl]}',
                ha='left', va='center', fontsize=7, fontstyle='italic',
                color=_CLUSTER_COLORS[cl], fontweight='bold',
                rotation=90, rotation_mode='anchor',
            )

    # Column headers
    for col_idx, label in enumerate(col_labels):
        ax.text(col_idx * cell_w + 0.5, -0.85, label,
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.suptitle(
        f'EU-27 Homeownership Snapshot — {SNAPSHOT_YEARS[0]}–{SNAPSHOT_YEARS[-1]}\n'
        'Grouped by cluster · sorted alphabetically within cluster',
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout()

    base = os.path.join(OUTPUT_DIR, 'eu27_ownership_snapshot_heatmap')
    fig.savefig(base + '.png', dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(base + '.svg', format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  → eu27_ownership_snapshot_heatmap.png / .svg')

    # Excel
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Snapshot'
    ws.append(['Country', 'Country_Name', 'Cluster', 'Cluster_Name']
              + [lbl.replace('\n', ' ') for lbl in col_labels])
    for _, row in df.iterrows():
        cl = int(row['Cluster'])
        ws.append([
            row['Country'], row['Country_Name'],
            cl if cl >= 0 else None,
            _CLUSTER_NAMES.get(cl, '') if cl >= 0 else None,
            *[round(row[k], 4) if not pd.isna(row[k]) else None for k in col_keys],
        ])
    wb.save(base + '.xlsx')
    print(f'  → eu27_ownership_snapshot_heatmap.xlsx')


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print(f'Years : {ALL_YEARS[0]}–{ALL_YEARS[-1]}')
    print(f'Workers: {MAX_WORKERS}   (ThreadPoolExecutor)')
    print(f'Output : {OUTPUT_DIR}')
    print('=' * 65)

    # ── Step 1: load ALL (country × year) files in parallel ────────────────
    all_tasks = [(cc, yr) for cc in EU27 for yr in ALL_YEARS]
    country_store: dict[str, dict[int, pd.DataFrame]] = {}

    print(f'\nLoading {len(all_tasks)} (country, year) file-sets …')
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {
            pool.submit(load_year, cc, yr): (cc, yr)
            for cc, yr in all_tasks
        }
        for fut in as_completed(future_map):
            cc, yr = future_map[fut]
            df = fut.result()
            done += 1
            if df is not None and not df.empty:
                country_store.setdefault(cc, {})[yr] = df
            if done % 50 == 0 or done == len(all_tasks):
                print(f'  {done}/{len(all_tasks)} done …')

    # ── Step 2: aggregate + plot per country ───────────────────────────────
    print('\nBuilding matrices and plotting …')
    all_matrices: dict[str, dict[int, pd.DataFrame]] = {}
    for cc in EU27:
        name = COUNTRY_NAMES.get(cc, cc)
        if cc not in country_store:
            print(f'  [{cc}] {name}  — no data found, skipped.')
            continue
        print(f'  [{cc}] {name}  ({len(country_store[cc])} years)')
        matrices = build_matrices(country_store[cc])
        all_matrices[cc] = matrices
        plot_heatmap(cc, matrices)

    # ── Step 3: export single Excel workbook ─────────────────────────────
    print('\nExporting Excel …')
    export_excel(all_matrices, OUTPUT_DIR)

    # ── Step 4: EU-27 snapshot heatmap (2022-2023) ────────────────────────
    print('\nBuilding EU-27 snapshot heatmap …')
    plot_snapshot_heatmap(country_store)

    print('\n' + '=' * 65)
    print('DONE')
    print('=' * 65)


if __name__ == '__main__':
    main()
