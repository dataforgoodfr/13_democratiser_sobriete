"""
ewbi_clustering.py - Country clustering using Method 5 (constant-income performance)

Method:
    1. Build EWBI benchmark curve in EWBI x income space (step = 5,000 EUR)
    2. Compute country performance score = average decile deviation from benchmark
    3. Split countries low/high performer based on performance score
    4. Final split by EWBI value (low/high) to get 4 clusters

This file now uses this method only.

Produces:
  - Radar chart of cluster profiles
  - EWBI vs Median Income scatter per cluster (with time trend)
  - Combined EWBI vs Median Income scatter with highlighted countries
  - Excel export of all values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
report_dir = os.path.abspath(os.path.join(current_dir, '..'))
well_being_dir = os.path.abspath(os.path.join(report_dir, '..', '..'))
data_path = os.path.join(well_being_dir, 'output', 'ewbi_master_aggregated.csv')

output_dir = os.path.join(report_dir, 'outputs', 'graphs', 'EWBI', 'Clustering')
os.makedirs(output_dir, exist_ok=True)

# Shapefile for matplotlib-based map exports (NUTS level 0 = countries)
_NUTS_GPKG = os.path.join(
    report_dir, '..', '1_switzerland_vs_eu27_housing_energy',
    'external_data', '0_shapefile', 'NUTS_RG_10M_2024_3035.gpkg',
)
_WORLD_SHP = os.path.join(
    report_dir, '..', '1_switzerland_vs_eu27_housing_energy',
    'external_data', '0_shapefile', 'ne_50m_admin_0_countries',
    'ne_50m_admin_0_countries.shp',
)

# ---------------------------------------------------------------------------
# Country code -> name mapping
# ---------------------------------------------------------------------------
COUNTRY_NAME_MAP = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus',
    'CZ': 'Czechia', 'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia',
    'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland', 'FR': 'France',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta',
    'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania',
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia',
}

ISO2_TO_ISO3 = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'CH': 'CHE', 'CY': 'CYP',
    'CZ': 'CZE', 'DE': 'DEU', 'DK': 'DNK', 'EE': 'EST', 'EL': 'GRC',
    'ES': 'ESP', 'FI': 'FIN', 'FR': 'FRA', 'HR': 'HRV', 'HU': 'HUN',
    'IE': 'IRL', 'IS': 'ISL', 'IT': 'ITA', 'LT': 'LTU', 'LU': 'LUX',
    'LV': 'LVA', 'MT': 'MLT', 'NL': 'NLD', 'NO': 'NOR', 'PL': 'POL',
    'PT': 'PRT', 'RO': 'ROU', 'RS': 'SRB', 'SE': 'SWE', 'SI': 'SVN',
    'SK': 'SVK', 'UK': 'GBR'
}

# ---------------------------------------------------------------------------
# 4-cluster definitions (Method 5: performance split + EWBI split)
# ---------------------------------------------------------------------------
CLUSTER_NAMES = {
    0: 'Low performer / Low EWBI',
    1: 'Low performer / High EWBI',
    2: 'High performer / Low EWBI',
    3: 'High performer / High EWBI',
}
CLUSTER_COLORS = ['#fb8072', '#fdb462', '#8dd3c7', '#80b1d3']

HIGHLIGHT_COUNTRIES = {
    'EL': 'Greece',
    'ES': 'Spain',
    'FR': 'France',
    'PL': 'Poland',
    'DE': 'Germany',
    'RO': 'Romania',
    'SE': 'Sweden',
}

METHOD5_STEP_EUR = 5000

EU_PRIORITY_DISPLAY_MAP = {
    'Health and Animal Welfare': 'Health',
    'Intergenerational Fairness, Youth, Culture and Sport': 'Education',
    'Social Rights and Skills, Quality Jobs and Preparedness': 'Quality of Jobs',
}


def _priority_label(priority_name):
    """Return display label for an EU priority name."""
    return EU_PRIORITY_DISPLAY_MAP.get(priority_name, priority_name)


def _save_fig_png_svg(fig, png_path, dpi=150):
    """Save a matplotlib figure to PNG and SVG side-by-side."""
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    svg_path = os.path.splitext(png_path)[0] + '.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved: {png_path}")
    print(f"  Saved: {svg_path}")


def load_and_prepare_data():
    """Load data and compute Method-5 features and final 4 clusters."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    median_income_path = os.path.join(report_dir, 'outputs', 'data', 'median_income_by_decile.csv')
    median_income_df = pd.read_csv(median_income_path)

    # EWBI per decile (Level 1), excluding aggregates.
    ewbi_decile = df[
        (df['Level'] == 1) &
        (df['Decile'] != 'All Deciles') &
        (~df['Country'].isin(['EU-27', 'All Countries']))
    ].copy()
    ewbi_decile['Decile'] = pd.to_numeric(ewbi_decile['Decile'], errors='coerce').astype('Int64')
    ewbi_decile['Year'] = pd.to_numeric(ewbi_decile['Year'], errors='coerce').astype('Int64')
    ewbi_decile['Value'] = pd.to_numeric(ewbi_decile['Value'], errors='coerce')
    ewbi_decile = ewbi_decile.dropna(subset=['Country', 'Year', 'Decile', 'Value'])

    # EWBI all-deciles (country-level).
    ewbi_all = df[
        (df['Level'] == 1) &
        (df['Decile'] == 'All Deciles') &
        (~df['Country'].isin(['EU-27', 'All Countries']))
    ].copy()
    ewbi_all['Year'] = pd.to_numeric(ewbi_all['Year'], errors='coerce').astype('Int64')
    ewbi_all['Value'] = pd.to_numeric(ewbi_all['Value'], errors='coerce')
    ewbi_all = ewbi_all.dropna(subset=['Country', 'Year', 'Value'])

    median_income_df['year'] = pd.to_numeric(median_income_df['year'], errors='coerce').astype('Int64')
    median_income_df['decile'] = pd.to_numeric(median_income_df['decile'], errors='coerce').astype('Int64')
    median_income_df['median_equi_disp_inc'] = pd.to_numeric(median_income_df['median_equi_disp_inc'], errors='coerce')
    median_income_df = median_income_df.dropna(subset=['country', 'year', 'decile', 'median_equi_disp_inc'])

    # Merge EWBI deciles with income deciles.
    merged = ewbi_decile.merge(
        median_income_df,
        left_on=['Country', 'Year', 'Decile'],
        right_on=['country', 'year', 'decile'],
        how='inner',
    )
    merged = merged.dropna(subset=['median_equi_disp_inc', 'Value'])

    # Keep last year per country for benchmark-based comparison.
    last_year = merged.groupby('Country')['Year'].max().reset_index()
    last_year.columns = ['Country', 'Last_Year']
    points = merged.merge(last_year, on='Country')
    points = points[points['Year'] == points['Last_Year']].copy()

    # Build benchmark from income bins and interpolate expected EWBI.
    points['income_bin_center'] = (
        np.round(points['median_equi_disp_inc'] / METHOD5_STEP_EUR) * METHOD5_STEP_EUR
    )
    benchmark = (
        points.groupby('income_bin_center', as_index=False)['Value']
        .mean()
        .sort_values('income_bin_center')
        .rename(columns={'Value': 'benchmark_ewbi'})
    )
    x = benchmark['income_bin_center'].values.astype(float)
    y = benchmark['benchmark_ewbi'].values.astype(float)
    points['ewbi_expected'] = np.interp(points['median_equi_disp_inc'].values.astype(float), x, y)
    points['ewbi_residual'] = points['Value'] - points['ewbi_expected']

    # Average deviation from baseline across deciles.
    decile_resid = (
        points.groupby(['Country', 'Decile'], as_index=False)
        .agg(decile_residual=('ewbi_residual', 'mean'))
    )
    perf_score = decile_resid.groupby('Country', as_index=False).agg(
        Performance_Score=('decile_residual', 'mean'),
        n_deciles=('decile_residual', 'size'),
    )

    # Country-level income metrics from the last available year.
    income_last = points.groupby('Country', as_index=False).agg(
        Mean_Income_Last=('median_equi_disp_inc', 'mean'),
    )

    income_d1 = points[points['Decile'] == 1][['Country', 'median_equi_disp_inc']].rename(
        columns={'median_equi_disp_inc': 'Income_D1'}
    )
    income_d10 = points[points['Decile'] == 10][['Country', 'median_equi_disp_inc']].rename(
        columns={'median_equi_disp_inc': 'Income_D10'}
    )
    income_inter = income_d10.merge(income_d1, on='Country', how='inner')
    income_inter['Income_Interdecile_Last'] = income_inter['Income_D10'] / income_inter['Income_D1']
    income_inter = income_inter.replace([np.inf, -np.inf], np.nan).dropna(subset=['Income_Interdecile_Last'])

    # Country-level EWBI last/first and growth.
    ewbi_last = ewbi_all.sort_values('Year').groupby('Country').tail(1)[['Country', 'Year', 'Value']].copy()
    ewbi_last.columns = ['Country', 'Last_Year_all', 'EWBI_Last']

    ewbi_first = ewbi_all.sort_values('Year').groupby('Country').head(1)[['Country', 'Year', 'Value']].copy()
    ewbi_first.columns = ['Country', 'First_Year', 'EWBI_First']

    # Interdecile ratios first/last year from decile EWBI table.
    d1 = ewbi_decile[ewbi_decile['Decile'] == 1][['Country', 'Year', 'Value']].rename(columns={'Value': 'D1'})
    d10 = ewbi_decile[ewbi_decile['Decile'] == 10][['Country', 'Year', 'Value']].rename(columns={'Value': 'D10'})
    inter = d10.merge(d1, on=['Country', 'Year'], how='inner')
    inter['Interdecile'] = inter['D10'] / inter['D1']
    inter = inter.replace([np.inf, -np.inf], np.nan).dropna(subset=['Interdecile'])

    inter_last = inter.sort_values('Year').groupby('Country').tail(1)[['Country', 'Interdecile']].copy()
    inter_last.columns = ['Country', 'Interdecile_Last']
    inter_first = inter.sort_values('Year').groupby('Country').head(1)[['Country', 'Interdecile']].copy()
    inter_first.columns = ['Country', 'Interdecile_First']

    # Build final feature table.
    features_df = (
        ewbi_last
        .merge(ewbi_first, on='Country', how='inner')
        .merge(inter_last, on='Country', how='left')
        .merge(inter_first, on='Country', how='left')
        .merge(income_last, on='Country', how='left')
        .merge(income_inter[['Country', 'Income_Interdecile_Last']], on='Country', how='left')
        .merge(perf_score, on='Country', how='left')
    )
    features_df['Country_Name'] = features_df['Country'].map(COUNTRY_NAME_MAP).fillna(features_df['Country'])
    features_df['Last_Year'] = features_df['Last_Year_all'].astype(int)
    features_df.drop(columns=['Last_Year_all'], inplace=True)

    n_years = (features_df['Last_Year'] - features_df['First_Year']).replace(0, np.nan)
    features_df['Annual_Growth'] = ((features_df['EWBI_Last'] / features_df['EWBI_First']) ** (1 / n_years) - 1)
    features_df['Annual_Growth'] = features_df['Annual_Growth'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    features_df = features_df.dropna(
        subset=['EWBI_Last', 'EWBI_First', 'Interdecile_Last', 'Mean_Income_Last', 'Income_Interdecile_Last', 'Performance_Score']
    ).copy()

    # Step 1: performance split.
    perf_cut = 0.01
    print(f"Performance threshold: {perf_cut:.6f}")
    features_df['Performance_Group'] = np.where(
        features_df['Performance_Score'] >= perf_cut,
        'High performer',
        'Low performer'
    )

    # Step 2: final separation by EWBI value.
    ewbi_cut = 0.7
    features_df['EWBI_Group'] = np.where(features_df['EWBI_Last'] >= ewbi_cut, 'High EWBI', 'Low EWBI')

    cluster_map = {
        ('Low performer', 'Low EWBI'): 0,
        ('Low performer', 'High EWBI'): 1,
        ('High performer', 'Low EWBI'): 2,
        ('High performer', 'High EWBI'): 3,
    }
    features_df['Cluster'] = features_df.apply(
        lambda r: cluster_map[(r['Performance_Group'], r['EWBI_Group'])],
        axis=1,
    ).astype(int)

    print(f"Method 5 EWBI split threshold (median): {ewbi_cut:.4f}")
    print(f"Countries with complete data: {len(features_df)}")
    return features_df


def cluster_summary(result_df):
    """Produce a summary DataFrame characterizing each cluster."""
    feature_cols = [
        'EWBI_Last',
        'EWBI_First',
        'Annual_Growth',
        'Interdecile_Last',
        'Interdecile_First',
        'Mean_Income_Last',
        'Income_Interdecile_Last',
    ]
    rows = []
    for cl in sorted(result_df['Cluster'].unique()):
        cdf = result_df[result_df['Cluster'] == cl]
        row = {'Cluster': cl, 'Cluster_Name': CLUSTER_NAMES[cl], 'N_Countries': len(cdf)}
        row['Countries'] = ', '.join(cdf['Country_Name'].tolist())
        for col in feature_cols:
            row[f'{col}_mean'] = cdf[col].mean()
            row[f'{col}_std'] = cdf[col].std()
        rows.append(row)
    return pd.DataFrame(rows)


# ===================================================================
# PLOT FUNCTIONS
# ===================================================================

def _load_ewbi_income_merged():
    """Load and merge EWBI per-decile with median income. Returns all years."""
    median_income_path = os.path.join(report_dir, 'outputs', 'data', 'median_income_by_decile.csv')
    if not os.path.exists(median_income_path):
        print(f"  WARNING: median income file not found at {median_income_path}")
        return None

    median_income_df = pd.read_csv(median_income_path)
    ewbi_df = pd.read_csv(data_path, low_memory=False)

    ewbi_decile = ewbi_df[
        (ewbi_df['Level'] == 1) &
        (ewbi_df['Decile'] != 'All Deciles') &
        (~ewbi_df['Country'].isin(['EU-27', 'All Countries']))
    ].copy()
    ewbi_decile['Decile'] = pd.to_numeric(ewbi_decile['Decile'], errors='coerce').astype('Int64')
    ewbi_decile = ewbi_decile.dropna(subset=['Decile', 'Value'])

    merged = ewbi_decile.merge(
        median_income_df,
        left_on=['Country', 'Year', 'Decile'],
        right_on=['country', 'year', 'decile'],
        how='inner',
    )
    merged = merged.dropna(subset=['Value', 'median_equi_disp_inc'])
    return merged if not merged.empty else None


def _load_priority_income_merged():
    """Load and merge Level-2 EU priorities by decile with median income."""
    median_income_path = os.path.join(report_dir, 'outputs', 'data', 'median_income_by_decile.csv')
    if not os.path.exists(median_income_path):
        print(f"  WARNING: median income file not found at {median_income_path}")
        return None

    median_income_df = pd.read_csv(median_income_path)
    ewbi_df = pd.read_csv(data_path, low_memory=False)

    priorities_decile = ewbi_df[
        (ewbi_df['Level'] == 2) &
        (ewbi_df['Decile'] != 'All Deciles') &
        (~ewbi_df['Country'].isin(['EU-27', 'All Countries']))
    ].copy()

    priorities_decile['Decile'] = pd.to_numeric(priorities_decile['Decile'], errors='coerce').astype('Int64')
    priorities_decile = priorities_decile.dropna(subset=['Decile', 'Value', 'EU priority'])

    merged = priorities_decile.merge(
        median_income_df,
        left_on=['Country', 'Year', 'Decile'],
        right_on=['country', 'year', 'decile'],
        how='inner',
    )
    merged = merged.dropna(subset=['Value', 'median_equi_disp_inc', 'EU priority'])
    return merged if not merged.empty else None


def plot_radar_clusters(result_df, output_dir):
    """Radar chart showing mean feature values per cluster (on normalized scale)."""
    feature_cols = [
        'EWBI_Last',
        'Annual_Growth',
        'Interdecile_Last',
        'Interdecile_First',
        'Mean_Income_Last',
        'Income_Interdecile_Last',
    ]
    display_labels = [
        'EWBI\n(Last Year)',
        'Annual\nGrowth',
        'EWBI Interdecile\n(Last Year)',
        'EWBI Interdecile\n(First Year)',
        'Mean Income\n(Last Year)',
        'Income Interdecile\n(Last Year)',
    ]

    df_norm = result_df.copy()
    for col in feature_cols:
        cmin, cmax = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - cmin) / max(cmax - cmin, 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for cl in sorted(result_df['Cluster'].unique()):
        values = df_norm[df_norm['Cluster'] == cl][feature_cols].mean().tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', color=CLUSTER_COLORS[cl], linewidth=2,
                label=f'Cluster {cl} – {CLUSTER_NAMES[cl]}')
        ax.fill(angles, values, alpha=0.15, color=CLUSTER_COLORS[cl])

    ax.set_thetagrids(np.degrees(angles[:-1]), display_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Cluster Profiles', fontsize=15, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    plt.tight_layout()

    path = os.path.join(output_dir, 'ewbi_cluster_radar.png')
    _save_fig_png_svg(fig, path, dpi=150)
    plt.close(fig)

    radar_html = go.Figure()
    for cl in sorted(result_df['Cluster'].unique()):
        values = df_norm[df_norm['Cluster'] == cl][feature_cols].mean().tolist()
        radar_html.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=display_labels + display_labels[:1],
                fill='toself',
                line=dict(color=CLUSTER_COLORS[cl], width=2),
                name=f'Cluster {cl} - {CLUSTER_NAMES[cl]}',
                opacity=0.7,
            )
        )

    radar_html.update_layout(
        title='Cluster Profiles',
        template='plotly_white',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
    )

    html_path = os.path.join(output_dir, 'ewbi_cluster_radar.html')
    radar_html.write_html(html_path)
    print(f"  Saved: {html_path}")


def plot_priority_radar_clusters(result_df, output_dir):
    """Radar chart of mean EWBI per cluster across Level-2 EU priorities."""
    ewbi_df = pd.read_csv(data_path, low_memory=False)

    # Use Level-2 all-deciles values for clean priority-level comparison.
    priorities_all = ewbi_df[
        (ewbi_df['Level'] == 2) &
        (ewbi_df['Decile'] == 'All Deciles') &
        (~ewbi_df['Country'].isin(['EU-27', 'All Countries']))
    ].copy()

    priorities_all['Year'] = pd.to_numeric(priorities_all['Year'], errors='coerce').astype('Int64')
    priorities_all['Value'] = pd.to_numeric(priorities_all['Value'], errors='coerce')
    priorities_all = priorities_all.dropna(subset=['Country', 'Year', 'Value', 'EU priority'])

    # Keep last year per country and priority.
    last_year_cp = priorities_all.groupby(['Country', 'EU priority'])['Year'].max().reset_index()
    last_year_cp.columns = ['Country', 'EU priority', 'last_year']
    plot_data = priorities_all.merge(last_year_cp, on=['Country', 'EU priority'], how='inner')
    plot_data = plot_data[plot_data['Year'] == plot_data['last_year']].copy()

    cluster_map = dict(zip(result_df['Country'], result_df['Cluster']))
    plot_data['Cluster'] = plot_data['Country'].map(cluster_map)
    plot_data = plot_data.dropna(subset=['Cluster']).copy()
    plot_data['Cluster'] = plot_data['Cluster'].astype(int)

    if plot_data.empty:
        print("  WARNING: No data available for priority radar plot")
        return

    priorities = sorted(plot_data['EU priority'].unique().tolist())
    priority_labels = [_priority_label(p) for p in priorities]
    priority_means = (
        plot_data.groupby(['Cluster', 'EU priority'], as_index=False)['Value']
        .mean()
    )

    # Build wide table with raw means (no normalization).
    wide = priority_means.pivot(index='Cluster', columns='EU priority', values='Value')
    wide = wide.reindex(index=sorted(CLUSTER_NAMES.keys()), columns=priorities)

    raw_min = float(np.nanmin(wide.values))
    raw_max = float(np.nanmax(wide.values))
    radial_min = max(0.0, raw_min - 0.03)
    radial_max = min(1.0, raw_max + 0.03)

    angles = np.linspace(0, 2 * np.pi, len(priorities), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for cl in sorted(CLUSTER_NAMES.keys()):
        if cl not in wide.index:
            continue
        values = wide.loc[cl, priorities].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', color=CLUSTER_COLORS[cl], linewidth=2,
                label=f'Cluster {cl} - {CLUSTER_NAMES[cl]}')
        ax.fill(angles, values, alpha=0.15, color=CLUSTER_COLORS[cl])

    ax.set_thetagrids(np.degrees(angles[:-1]), priority_labels, fontsize=10)
    ax.set_ylim(radial_min, radial_max)
    ax.set_title('EU Priorities Radar by Cluster (mean EWBI)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1), fontsize=9)
    plt.tight_layout()

    png_path = os.path.join(output_dir, 'ewbi_cluster_priority_radar.png')
    _save_fig_png_svg(fig, png_path, dpi=160)
    plt.close(fig)

    fig_html = go.Figure()
    for cl in sorted(CLUSTER_NAMES.keys()):
        if cl not in wide.index:
            continue
        values = wide.loc[cl, priorities].tolist()
        fig_html.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=priority_labels + priority_labels[:1],
                fill='toself',
                line=dict(color=CLUSTER_COLORS[cl], width=2),
                name=f'Cluster {cl} - {CLUSTER_NAMES[cl]}',
                opacity=0.7,
                hovertemplate='Priority: %{theta}<br>Mean EWBI: %{r:.3f}<extra></extra>',
            )
        )

    fig_html.update_layout(
        title='EU Priorities Radar by Cluster (mean EWBI)',
        template='plotly_white',
        polar=dict(radialaxis=dict(visible=True, range=[radial_min, radial_max])),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
    )

    html_path = os.path.join(output_dir, 'ewbi_cluster_priority_radar.html')
    fig_html.write_html(html_path)
    print(f"  Saved: {html_path}")


def plot_ewbi_vs_income_per_cluster(result_df, output_dir):
    """
    One EWBI vs Median Income scatter per cluster showing ALL available years
    as a time trend (light→dark color gradient), styled like the reference scatter.
    """
    import matplotlib.cm as cm

    merged = _load_ewbi_income_merged()
    if merged is None:
        return

    cluster_map = dict(zip(result_df['Country'], result_df['Cluster']))

    for cl in sorted(result_df['Cluster'].unique()):
        cluster_countries = result_df[result_df['Cluster'] == cl]['Country'].tolist()
        cdata = merged[merged['Country'].isin(cluster_countries)].copy()
        if cdata.empty:
            continue

        years = sorted(cdata['Year'].unique())
        n_years = len(years)

        # Use a color gradient from the cluster base color
        base_color = CLUSTER_COLORS[cl]
        # Build a custom colormap: white → base color
        from matplotlib.colors import LinearSegmentedColormap, to_rgb
        rgb = to_rgb(base_color)
        cmap = LinearSegmentedColormap.from_list('custom', [(1, 1, 1), rgb], N=n_years + 2)

        fig, ax = plt.subplots(figsize=(16, 10))

        for i, year in enumerate(years):
            ydata = cdata[cdata['Year'] == year]
            color = cmap(i + 2)
            alpha = 0.3 + 0.7 * (i / max(n_years - 1, 1))
            lw = 1.0 + 1.5 * (i / max(n_years - 1, 1))

            for country in cluster_countries:
                cd = ydata[ydata['Country'] == country].sort_values('Decile')
                if cd.empty:
                    continue
                ax.plot(cd['median_equi_disp_inc'], cd['Value'],
                        color=color, alpha=alpha, linewidth=lw, zorder=2 + i)
                ax.scatter(cd['median_equi_disp_inc'], cd['Value'],
                           c=[color], s=25 + 25 * (i / max(n_years - 1, 1)),
                           alpha=alpha, edgecolors='white', linewidths=0.3, zorder=2 + i)

        # Label highlighted countries at last year
        last_year = years[-1]
        for code, name in HIGHLIGHT_COUNTRIES.items():
            if code not in cluster_countries:
                continue
            ld = cdata[(cdata['Country'] == code) & (cdata['Year'] == last_year)].sort_values('Decile')
            if ld.empty:
                continue
            row = ld.iloc[-1]
            ax.annotate(name, (row['median_equi_disp_inc'], row['Value']),
                        fontsize=10, fontweight='bold', color=to_rgb(base_color),
                        textcoords='offset points', xytext=(6, 4), zorder=100)

        # Year labels
        for year, ha, offset in [(years[0], 'right', (-8, -8)), (years[-1], 'left', (8, 4))]:
            yr_data = cdata[cdata['Year'] == year]
            if not yr_data.empty:
                mid_country = cluster_countries[len(cluster_countries) // 2]
                pts = yr_data[yr_data['Country'] == mid_country].sort_values('Decile')
                if not pts.empty:
                    row = pts.iloc[-1]
                    ax.annotate(str(int(year)), (row['median_equi_disp_inc'], row['Value']),
                                fontsize=10, fontweight='bold',
                                textcoords='offset points', xytext=offset, ha=ha, zorder=100)

        # Colorbar for years
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=int(years[0]), vmax=int(years[-1])))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cbar.set_label('Year', fontsize=12)

        ax.set_xlabel('Median Equivalized Disposable Income (€)', fontsize=14)
        ax.set_ylabel('EWBI Score', fontsize=14)
        ax.set_title(
            f'Cluster {cl} – {CLUSTER_NAMES[cl]}: EWBI vs Income Over Time\n'
            f'({int(years[0])}–{int(years[-1])})',
            fontsize=16, fontweight='bold', pad=20,
        )
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        plt.tight_layout()

        path = os.path.join(output_dir, f'ewbi_vs_income_cluster_{cl}_trend.png')
        _save_fig_png_svg(fig, path, dpi=150)
        plt.close(fig)


def plot_ewbi_vs_income_all_clusters(result_df, output_dir):
    """
    One visual with one panel per cluster (last year), with identical axis scales.
    In each panel, non-target clusters are shown in grey in the background.
    Highlights at least 2 countries per cluster and always includes
    France, Germany, and Poland.
    """
    merged = _load_ewbi_income_merged()
    if merged is None:
        return

    # Keep last year per country
    last_year = merged.groupby('Country')['Year'].max().reset_index()
    last_year.columns = ['Country', 'last_year']
    plot_data = merged.merge(last_year, on='Country')
    plot_data = plot_data[plot_data['Year'] == plot_data['last_year']]

    cluster_map = dict(zip(result_df['Country'], result_df['Cluster']))
    countries = sorted(plot_data['Country'].unique())

    # EU-27 countries (exclude CH, NO, etc.)
    eu27_countries = {'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GR',
                      'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO',
                      'SE', 'SI', 'SK'}

    # Force these countries to always be highlighted when available.
    forced_highlights = {'FR': 'France', 'DE': 'Germany', 'PL': 'Poland'}
    highlight_codes = {c for c in forced_highlights if c in countries}

    # Ensure at least two highlighted countries per cluster category (prefer EU-27).
    country_score = plot_data.groupby('Country')['Value'].mean()
    for cl in sorted(CLUSTER_NAMES.keys()):
        cluster_countries = [c for c in countries if cluster_map.get(c) == cl]
        if not cluster_countries:
            continue
        already = [c for c in highlight_codes if cluster_map.get(c) == cl]
        needed = max(0, 2 - len(already))
        if needed == 0:
            continue
        # Prioritize EU-27 candidates
        candidates = [c for c in cluster_countries if c not in highlight_codes and c in eu27_countries]
        candidates = sorted(candidates, key=lambda c: country_score.get(c, -np.inf), reverse=True)
        highlight_codes.update(candidates[:needed])

    years_used = plot_data.groupby('Country')['Year'].first()
    year_min, year_max = int(years_used.min()), int(years_used.max())
    year_label = f"{year_min}" if year_min == year_max else f"{year_min}-{year_max}"

    x_min = plot_data['median_equi_disp_inc'].min()
    x_max = plot_data['median_equi_disp_inc'].max()
    y_min = plot_data['Value'].min()
    y_max = plot_data['Value'].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.07

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, cl in enumerate(sorted(CLUSTER_NAMES.keys())):
        ax = axes[i]
        target_countries = [c for c in countries if cluster_map.get(c) == cl]
        other_countries = [c for c in countries if cluster_map.get(c) != cl]

        # Background: all other clusters in grey
        for country in other_countries:
            cdata = plot_data[plot_data['Country'] == country].sort_values('Decile')
            if cdata.empty:
                continue
            ax.plot(
                cdata['median_equi_disp_inc'], cdata['Value'],
                color='#bdbdbd', alpha=0.35, linewidth=0.9, zorder=1,
            )
            ax.scatter(
                cdata['median_equi_disp_inc'], cdata['Value'],
                c='#bdbdbd', s=30, alpha=0.35,
                edgecolors='white', linewidths=0.2, zorder=1,
            )

        # Foreground: target cluster in color
        for country in target_countries:
            cdata = plot_data[plot_data['Country'] == country].sort_values('Decile')
            if cdata.empty:
                continue
            lw = 2.4 if country in highlight_codes else 1.4
            ms = 90 if country in highlight_codes else 55
            alpha = 0.95 if country in highlight_codes else 0.78
            ax.plot(
                cdata['median_equi_disp_inc'], cdata['Value'],
                color=CLUSTER_COLORS[cl], alpha=0.75, linewidth=lw, zorder=2,
            )
            ax.scatter(
                cdata['median_equi_disp_inc'], cdata['Value'],
                c=CLUSTER_COLORS[cl], s=ms, alpha=alpha,
                edgecolors='black' if country in highlight_codes else 'white',
                linewidths=0.6 if country in highlight_codes else 0.3,
                zorder=3,
            )

            if country in highlight_codes:
                label = forced_highlights.get(country, COUNTRY_NAME_MAP.get(country, country))
                row_last = cdata.iloc[-1]
                ax.annotate(
                    label,
                    (row_last['median_equi_disp_inc'], row_last['Value']),
                    fontsize=8.5,
                    fontweight='bold',
                    color=CLUSTER_COLORS[cl],
                    textcoords='offset points',
                    xytext=(5, 2),
                    zorder=4,
                )

        # Draw benchmark interpolation line in dark
        all_points_cl = plot_data[plot_data['Cluster'] == cl].copy() if 'Cluster' in plot_data.columns else None
        if all_points_cl is None or all_points_cl.empty:
            all_data_for_bench = plot_data  # Use all data if cluster not yet in plot_data
        else:
            all_data_for_bench = all_points_cl
        
        points_bench = all_data_for_bench.dropna(subset=['median_equi_disp_inc', 'Value'])
        if not points_bench.empty:
            points_bench['income_bin_center'] = (
                np.round(points_bench['median_equi_disp_inc'] / 5000) * 5000
            )
            bench_df = (
                points_bench.groupby('income_bin_center', as_index=False)['Value'].mean()
                .sort_values('income_bin_center')
            )
            if len(bench_df) > 1:
                x_bench = bench_df['income_bin_center'].values.astype(float)
                y_bench = bench_df['Value'].values.astype(float)
                x_interp = np.linspace(x_bench.min(), x_bench.max(), 100)
                y_interp = np.interp(x_interp, x_bench, y_bench)
                ax.plot(x_interp, y_interp, color='#2c2c2c', linewidth=2.8, linestyle='-',
                        alpha=0.85, zorder=2.5, label='Benchmark curve')

        ax.set_title(f'Cluster {cl} - {CLUSTER_NAMES[cl]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_facecolor('white')
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    for i in [2, 3]:
        axes[i].set_xlabel('Median Equivalized Disposable Income (€)', fontsize=12)
    for i in [0, 2]:
        axes[i].set_ylabel('EWBI Score', fontsize=12)

    fig.suptitle(
        'EWBI vs Median Income by Country and Decile - One panel per cluster\n'
        f'(Last available year per country: {year_label}; other clusters shown in grey)',
        fontsize=16,
        fontweight='bold',
        y=0.98,
    )

    legend_handles = [
        Line2D([0], [0], color='#bdbdbd', lw=2, label='Other clusters (background)'),
    ] + [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CLUSTER_COLORS[cl], markeredgecolor='black',
               markeredgewidth=0.7, markersize=9,
               label=f'Cluster {cl} - {CLUSTER_NAMES[cl]}')
        for cl in sorted(CLUSTER_NAMES.keys())
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.01), frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    path = os.path.join(output_dir, 'ewbi_vs_income_by_cluster.png')
    _save_fig_png_svg(fig, path, dpi=150)
    plt.close(fig)

    # --- Export scatter plot data as CSV ---
    plot_data['Cluster'] = plot_data['Country'].map(cluster_map)
    plot_data['Cluster_Name'] = plot_data['Cluster'].map(CLUSTER_NAMES)
    plot_data['Country_Name'] = plot_data['Country'].map(COUNTRY_NAME_MAP).fillna(plot_data['Country'])
    export_cols = ['Country', 'Country_Name', 'Cluster', 'Cluster_Name', 'Year', 'Decile', 'Value', 'median_equi_disp_inc']
    csv_path = os.path.join(output_dir, 'ewbi_vs_income_by_cluster_data.csv')
    plot_data[export_cols].sort_values(['Cluster', 'Country', 'Decile']).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def plot_ewbi_vs_income_france(result_df, output_dir):
    """
    EWBI vs Median Income scatter (last year) with France highlighted in yellow
    and all other countries in dark blue background.
    """
    merged = _load_ewbi_income_merged()
    if merged is None:
        return

    # Keep last year per country
    last_year = merged.groupby('Country')['Year'].max().reset_index()
    last_year.columns = ['Country', 'last_year']
    plot_data = merged.merge(last_year, on='Country')
    plot_data = plot_data[plot_data['Year'] == plot_data['last_year']]

    countries = sorted(plot_data['Country'].unique())
    years_used = plot_data.groupby('Country')['Year'].first()
    year_min, year_max = int(years_used.min()), int(years_used.max())
    year_label = f"{year_min}" if year_min == year_max else f"{year_min}-{year_max}"

    DARK_BLUE = '#80b1d3'
    FRANCE_YELLOW = '#ffd558'

    fig, ax = plt.subplots(figsize=(16, 10))

    # --- All other countries in dark blue ---
    for country in countries:
        if country == 'FR':
            continue
        cdata = plot_data[plot_data['Country'] == country].sort_values('Decile')
        ax.scatter(
            cdata['median_equi_disp_inc'], cdata['Value'],
            c=DARK_BLUE, s=40, alpha=0.4,
            edgecolors='white', linewidths=0.2, zorder=1,
        )
        ax.plot(
            cdata['median_equi_disp_inc'], cdata['Value'],
            color=DARK_BLUE, alpha=0.25, linewidth=0.9, zorder=1,
        )

    # --- France highlighted in yellow ---
    fr_data = plot_data[plot_data['Country'] == 'FR'].sort_values('Decile')
    if not fr_data.empty:
        ax.scatter(
            fr_data['median_equi_disp_inc'], fr_data['Value'],
            c=FRANCE_YELLOW, s=120, alpha=0.95,
            edgecolors='black', linewidths=0.8, zorder=3,
        )
        ax.plot(
            fr_data['median_equi_disp_inc'], fr_data['Value'],
            color=FRANCE_YELLOW, alpha=0.8, linewidth=3, zorder=2,
        )
        row_last = fr_data.iloc[-1]
        ax.annotate(
            'France', (row_last['median_equi_disp_inc'], row_last['Value']),
            fontsize=11, fontweight='bold', color=FRANCE_YELLOW,
            textcoords='offset points', xytext=(8, 4), zorder=4,
        )
        # Label D1 and D10
        row_first = fr_data.iloc[0]
        ax.annotate(
            'D1', (row_first['median_equi_disp_inc'], row_first['Value']),
            fontsize=9, fontweight='bold', color='#b39740',
            textcoords='offset points', xytext=(-12, -10), zorder=4,
        )
        ax.annotate(
            'D10', (row_last['median_equi_disp_inc'], row_last['Value']),
            fontsize=9, fontweight='bold', color='#b39740',
            textcoords='offset points', xytext=(8, -10), zorder=4,
        )

    ax.set_xlabel('Median Equivalized Disposable Income (€)', fontsize=14)
    ax.set_ylabel('EWBI Score', fontsize=14)
    ax.set_title(
        f'EWBI vs Median Income by Country and Decile — France\n'
        f'(Last available year per country: {year_label})',
        fontsize=16, fontweight='bold', pad=20,
    )
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')

    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=DARK_BLUE, markersize=8, alpha=0.5,
               label='Other EU countries'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=FRANCE_YELLOW, markeredgecolor='black',
               markeredgewidth=0.7, markersize=10, label='France'),
    ]
    ax.legend(handles=legend_handles, title='Country', loc='lower right',
              fontsize=10, title_fontsize=11)
    plt.tight_layout()

    path = os.path.join(output_dir, 'ewbi_vs_income_france.png')
    _save_fig_png_svg(fig, path, dpi=150)
    plt.close(fig)


def plot_ewbi_vs_income_cluster_priority_grid(result_df, output_dir):
    """
    Create one visual with rows as EU priorities and columns as clusters.
    In each panel, the target cluster/priority series are highlighted while
    all other clusters appear in grey background.
    """
    merged = _load_priority_income_merged()
    if merged is None:
        return

    cluster_map = dict(zip(result_df['Country'], result_df['Cluster']))
    merged['Cluster'] = merged['Country'].map(cluster_map)
    merged = merged.dropna(subset=['Cluster']).copy()
    merged['Cluster'] = merged['Cluster'].astype(int)

    # Keep latest available year per country and EU priority for cleaner visual comparison.
    last_year_cp = merged.groupby(['Country', 'EU priority'])['Year'].max().reset_index()
    last_year_cp.columns = ['Country', 'EU priority', 'last_year']
    plot_data = merged.merge(last_year_cp, on=['Country', 'EU priority'])
    plot_data = plot_data[plot_data['Year'] == plot_data['last_year']].copy()

    if plot_data.empty:
        print("  WARNING: No data available for cluster-priority grid plot")
        return

    cluster_ids = sorted(CLUSTER_NAMES.keys())
    priorities = sorted(plot_data['EU priority'].unique().tolist())

    if len(priorities) != 5:
        print(f"  WARNING: Expected 5 EU priorities, found {len(priorities)}")

    x_min, x_max = plot_data['median_equi_disp_inc'].min(), plot_data['median_equi_disp_inc'].max()
    y_min, y_max = plot_data['Value'].min(), plot_data['Value'].max()
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.01

    n_rows = len(priorities)
    n_cols = len(cluster_ids)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.8 * n_rows),
        sharex=True,
        sharey=True,
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for r, priority in enumerate(priorities):
        for c, cl in enumerate(cluster_ids):
            ax = axes[r, c]
            cell = plot_data[plot_data['EU priority'] == priority].copy()

            # Background: all other clusters in grey.
            bg = cell[cell['Cluster'] != cl]
            for country in sorted(bg['Country'].unique()):
                cdata = bg[bg['Country'] == country].sort_values('Decile')
                ax.plot(
                    cdata['median_equi_disp_inc'],
                    cdata['Value'],
                    color='#bfbfbf',
                    alpha=0.35,
                    linewidth=0.9,
                    zorder=1,
                )
                ax.scatter(
                    cdata['median_equi_disp_inc'],
                    cdata['Value'],
                    c='#bfbfbf',
                    s=16,
                    alpha=0.35,
                    edgecolors='none',
                    zorder=1,
                )

            # Foreground: selected cluster.
            fg = cell[cell['Cluster'] == cl]
            for country in sorted(fg['Country'].unique()):
                cdata = fg[fg['Country'] == country].sort_values('Decile')
                ax.plot(
                    cdata['median_equi_disp_inc'],
                    cdata['Value'],
                    color=CLUSTER_COLORS[cl],
                    alpha=0.85,
                    linewidth=1.6,
                    zorder=2,
                )
                ax.scatter(
                    cdata['median_equi_disp_inc'],
                    cdata['Value'],
                    c=CLUSTER_COLORS[cl],
                    s=24,
                    alpha=0.9,
                    edgecolors='white',
                    linewidths=0.25,
                    zorder=2,
                )

            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.grid(True, alpha=0.25)
            ax.set_facecolor('white')

            if r == 0:
                ax.set_title(f"Cluster {cl}", fontsize=11, fontweight='bold')
            if c == 0:
                ax.set_ylabel(f"{_priority_label(priority)}\nEWBI", fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel('Median Equivalized Disposable Income (€)', fontsize=10)

    fig.suptitle(
        'EWBI vs Median Income by EU Priority (rows) and Cluster (columns)\n'
        'Foreground = EU-priority series in selected cluster, background = other clusters (grey)',
        fontsize=16,
        fontweight='bold',
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])

    path = os.path.join(output_dir, 'ewbi_vs_income_cluster_priority_grid.png')
    _save_fig_png_svg(fig, path, dpi=180)
    plt.close(fig)


def _load_nuts0_and_background():
    """Load NUTS level-0 (countries) and background world geometries."""
    nuts = gpd.read_file(_NUTS_GPKG)
    nuts0 = nuts[nuts['LEVL_CODE'] == 0].copy()
    if nuts0.crs != 'EPSG:3035':
        nuts0 = nuts0.to_crs(epsg=3035)

    world = gpd.read_file(_WORLD_SHP)
    bg = world[(world['CONTINENT'] == 'Europe') | (world['ISO_A2'] == 'TR')].copy()
    bg = bg.to_crs(epsg=3035)
    return nuts0, bg


def plot_ewbi_country_map(result_df, output_dir):
    """Create choropleth map of EWBI last-year values."""
    map_df = result_df.copy()
    map_df['iso3'] = map_df['Country'].map(ISO2_TO_ISO3)
    map_df = map_df.dropna(subset=['iso3']).copy()

    fig = px.choropleth(
        map_df,
        locations='iso3',
        color='EWBI_Last',
        hover_name='Country_Name',
        hover_data={'Country': True, 'Cluster': True, 'EWBI_Last': ':.3f', 'iso3': False},
        color_continuous_scale='Viridis',
        scope='europe',
        title='EWBI Country Map (Last Available Year)'
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        coloraxis_colorbar=dict(title='EWBI')
    )

    html_path = os.path.join(output_dir, 'ewbi_country_map.html')
    fig.write_html(html_path)
    print(f"  Saved: {html_path}")

    # --- Matplotlib SVG export ---
    nuts0, bg = _load_nuts0_and_background()
    merged_geo = nuts0.merge(map_df[['Country', 'EWBI_Last']], left_on='NUTS_ID', right_on='Country', how='left')
    # Determine which NUTS_IDs are study countries
    study_ids = set(map_df['Country'].tolist())

    fig_mpl, ax = plt.subplots(figsize=(12, 10))
    bg.plot(ax=ax, color='white', edgecolor='black', linewidth=0.3, alpha=0.35, hatch='///')
    no_data = merged_geo[merged_geo['EWBI_Last'].isna() & merged_geo['NUTS_ID'].isin(study_ids)]
    if not no_data.empty:
        no_data.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.3)
    has_data = merged_geo[merged_geo['EWBI_Last'].notna()]
    if not has_data.empty:
        has_data.plot(column='EWBI_Last', cmap='viridis', edgecolor='black', linewidth=0.4,
                      ax=ax, legend=True, legend_kwds={'label': 'EWBI', 'shrink': 0.6})
    ax.set_xlim(2.5e6, 6.5e6)
    ax.set_ylim(1.3e6, 5.5e6)
    ax.set_axis_off()
    ax.set_title('EWBI Country Map (Last Available Year)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    svg_path = os.path.join(output_dir, 'ewbi_country_map.svg')
    _save_fig_png_svg(fig_mpl, os.path.join(output_dir, 'ewbi_country_map_mpl.png'), dpi=150)
    plt.close(fig_mpl)


def plot_cluster_map(result_df, output_dir):
    """Create choropleth map of EWBI cluster assignment."""
    map_df = result_df.copy()
    map_df['iso3'] = map_df['Country'].map(ISO2_TO_ISO3)
    map_df = map_df.dropna(subset=['iso3']).copy()
    map_df['Cluster_Label'] = map_df['Cluster'].map(lambda c: f"Cluster {int(c)}")

    cluster_order = [f"Cluster {i}" for i in sorted(CLUSTER_NAMES.keys())]
    color_map = {
        f"Cluster {i}": CLUSTER_COLORS[i]
        for i in sorted(CLUSTER_NAMES.keys())
    }

    fig = px.choropleth(
        map_df,
        locations='iso3',
        color='Cluster_Label',
        category_orders={'Cluster_Label': cluster_order},
        color_discrete_map=color_map,
        hover_name='Country_Name',
        hover_data={
            'Country': True,
            'Cluster_Label': True,
            'EWBI_Last': ':.3f',
            'Interdecile_Last': ':.3f',
            'iso3': False,
        },
        scope='europe',
        title='EWBI Cluster Map (Manual Threshold Clusters)'
    )

    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))

    html_path = os.path.join(output_dir, 'ewbi_cluster_map.html')
    fig.write_html(html_path)
    print(f"  Saved: {html_path}")

    # --- Matplotlib SVG export ---
    nuts0, bg = _load_nuts0_and_background()
    merged_geo = nuts0.merge(map_df[['Country', 'Cluster']], left_on='NUTS_ID', right_on='Country', how='left')
    study_ids = set(map_df['Country'].tolist())

    fig_mpl, ax = plt.subplots(figsize=(12, 10))
    bg.plot(ax=ax, color='white', edgecolor='black', linewidth=0.3, alpha=0.35, hatch='///')
    no_data = merged_geo[merged_geo['Cluster'].isna() & merged_geo['NUTS_ID'].isin(study_ids)]
    if not no_data.empty:
        no_data.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.3)
    for cl in sorted(CLUSTER_NAMES.keys()):
        cl_geo = merged_geo[merged_geo['Cluster'] == cl]
        if not cl_geo.empty:
            cl_geo.plot(ax=ax, color=CLUSTER_COLORS[cl], edgecolor='black', linewidth=0.4)
    # Legend
    legend_handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=CLUSTER_COLORS[cl],
               markersize=10, label=f'Cluster {cl} - {CLUSTER_NAMES[cl]}')
        for cl in sorted(CLUSTER_NAMES.keys())
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=9, frameon=True)
    ax.set_xlim(2.5e6, 6.5e6)
    ax.set_ylim(1.3e6, 5.5e6)
    ax.set_axis_off()
    ax.set_title('EWBI Cluster Map (Manual Threshold Clusters)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig_png_svg(fig_mpl, os.path.join(output_dir, 'ewbi_cluster_map_mpl.png'), dpi=150)
    plt.close(fig_mpl)


def plot_method5_points_all_clustered(result_df, output_dir):
    """
    Plot all country-decile points (last year, step=5000 benchmark context)
    with country lines and cluster-based colors.
    """
    merged = _load_ewbi_income_merged()
    if merged is None:
        return

    # Last available year per country.
    last_year = merged.groupby('Country')['Year'].max().reset_index()
    last_year.columns = ['Country', 'last_year']
    points = merged.merge(last_year, on='Country')
    points = points[points['Year'] == points['last_year']].copy()

    # Method-5 benchmark for reference fields.
    points['income_bin_center'] = (np.round(points['median_equi_disp_inc'] / METHOD5_STEP_EUR) * METHOD5_STEP_EUR)
    benchmark = (
        points.groupby('income_bin_center', as_index=False)['Value']
        .mean()
        .sort_values('income_bin_center')
        .rename(columns={'Value': 'benchmark_ewbi'})
    )
    if len(benchmark) >= 2:
        x = benchmark['income_bin_center'].values.astype(float)
        y = benchmark['benchmark_ewbi'].values.astype(float)
        points['ewbi_expected'] = np.interp(points['median_equi_disp_inc'].values.astype(float), x, y)
        points['ewbi_residual'] = points['Value'] - points['ewbi_expected']
    else:
        points['ewbi_expected'] = np.nan
        points['ewbi_residual'] = np.nan

    cluster_map = dict(zip(result_df['Country'], result_df['Cluster']))
    points['Cluster'] = points['Country'].map(cluster_map)
    points = points.dropna(subset=['Cluster']).copy()
    points['Cluster'] = points['Cluster'].astype(int)
    points['Cluster_Label'] = points['Cluster'].map(lambda c: f"Cluster {c}: {CLUSTER_NAMES[c]}")

    color_map = {
        f"Cluster {i}: {CLUSTER_NAMES[i]}": CLUSTER_COLORS[i]
        for i in sorted(CLUSTER_NAMES.keys())
    }

    fig = px.line(
        points.sort_values(['Country', 'Decile']),
        x='median_equi_disp_inc',
        y='Value',
        color='Cluster_Label',
        line_group='Country',
        markers=True,
        hover_name='Country',
        hover_data={
            'Decile': True,
            'Year': True,
            'ewbi_expected': ':.3f',
            'ewbi_residual': ':.3f',
            'Cluster_Label': True,
            'median_equi_disp_inc': ':.0f',
            'Value': ':.3f',
        },
        color_discrete_map=color_map,
        title='Method 5 (step 5000): All country-decile points colored by cluster'
    )

    fig.update_layout(
        xaxis_title='Median Equivalized Disposable Income (€)',
        yaxis_title='EWBI Score',
        legend_title='Cluster',
        template='plotly_white'
    )

    html_path = os.path.join(output_dir, 'method5_points_all_step_5000_clustered.html')
    fig.write_html(html_path)
    print(f"  Saved: {html_path}")




def plot_performance_vs_ewbi(result_df, output_dir):
    """
    Scatter plot: EWBI (x-axis) vs Performance Score (y-axis)
    with threshold lines showing cluster allocation boundaries.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    ewbi_cut = 0.7
    perf_cut = 0.01

    # Axis range
    x_min = result_df['EWBI_Last'].min() - 0.02
    x_max = result_df['EWBI_Last'].max() + 0.02
    y_min = result_df['Performance_Score'].min() - 0.01
    y_max = result_df['Performance_Score'].max() + 0.01

    # Shade quadrants
    ax.axvspan(x_min, ewbi_cut, ymin=0,
               ymax=(perf_cut - y_min) / (y_max - y_min),
               alpha=0.06, color=CLUSTER_COLORS[0])
    ax.axvspan(ewbi_cut, x_max, ymin=0,
               ymax=(perf_cut - y_min) / (y_max - y_min),
               alpha=0.06, color=CLUSTER_COLORS[1])
    ax.axvspan(x_min, ewbi_cut,
               ymin=(perf_cut - y_min) / (y_max - y_min), ymax=1,
               alpha=0.06, color=CLUSTER_COLORS[2])
    ax.axvspan(ewbi_cut, x_max,
               ymin=(perf_cut - y_min) / (y_max - y_min), ymax=1,
               alpha=0.06, color=CLUSTER_COLORS[3])

    # Threshold lines
    ax.axvline(ewbi_cut, color='#333', linewidth=1.5, linestyle='--', alpha=0.7,
               label=f'EWBI threshold = {ewbi_cut}')
    ax.axhline(perf_cut, color='#333', linewidth=1.5, linestyle='--', alpha=0.7,
               label=f'Performance threshold = {perf_cut}')

    # Plot dots per cluster
    for cl in sorted(result_df['Cluster'].unique()):
        cdf = result_df[result_df['Cluster'] == cl]
        ax.scatter(cdf['EWBI_Last'], cdf['Performance_Score'],
                   c=CLUSTER_COLORS[cl], s=120, edgecolors='black', linewidths=0.6,
                   zorder=3, label=f'Cluster {cl} \u2013 {CLUSTER_NAMES[cl]}')

    # Country labels
    for _, row in result_df.iterrows():
        label = row['Country_Name'] if pd.notna(row.get('Country_Name')) else row['Country']
        ax.annotate(label, (row['EWBI_Last'], row['Performance_Score']),
                    fontsize=8, fontweight='bold',
                    textcoords='offset points', xytext=(6, 4), zorder=4,
                    color=CLUSTER_COLORS[int(row['Cluster'])])

    # Quadrant labels
    ax.text(x_min + 0.005, y_max - 0.003, 'High performer / Low EWBI',
            fontsize=9, fontstyle='italic', color=CLUSTER_COLORS[2], alpha=0.7, va='top')
    ax.text(ewbi_cut + 0.005, y_max - 0.003, 'High performer / High EWBI',
            fontsize=9, fontstyle='italic', color=CLUSTER_COLORS[3], alpha=0.7, va='top')
    ax.text(x_min + 0.005, y_min + 0.003, 'Low performer / Low EWBI',
            fontsize=9, fontstyle='italic', color=CLUSTER_COLORS[0], alpha=0.7, va='bottom')
    ax.text(ewbi_cut + 0.005, y_min + 0.003, 'Low performer / High EWBI',
            fontsize=9, fontstyle='italic', color=CLUSTER_COLORS[1], alpha=0.7, va='bottom')

    ax.set_xlabel('EWBI (Last Available Year)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance Score (deviation from benchmark)', fontsize=13, fontweight='bold')
    ax.set_title('Cluster Allocation: Performance Score vs EWBI',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    plt.tight_layout()

    path = os.path.join(output_dir, 'ewbi_performance_vs_ewbi_clusters.png')
    _save_fig_png_svg(fig, path, dpi=150)
    plt.close(fig)


def export_ewbi_by_decile_clusters(result_df, output_dir):
    """
    Export EWBI values by decile with structure:
    - Columns: Deciles 1-10
    - Rows: Cluster / Country pairs
    - Last available year per country
    """
    merged = _load_ewbi_income_merged()
    if merged is None:
        return

    # Keep last year per country
    last_year = merged.groupby('Country')['Year'].max().reset_index()
    last_year.columns = ['Country', 'last_year']
    plot_data = merged.merge(last_year, on='Country')
    plot_data = plot_data[plot_data['Year'] == plot_data['last_year']].copy()

    cluster_map = dict(zip(result_df['Country'], result_df['Cluster']))
    plot_data['Cluster'] = plot_data['Country'].map(cluster_map)
    plot_data['Country_Name'] = plot_data['Country'].map(COUNTRY_NAME_MAP).fillna(plot_data['Country'])

    # Build pivot table: rows = Cluster/Country, columns = Deciles
    pivot_data = plot_data[['Cluster', 'Country', 'Country_Name', 'Decile', 'Value']].copy()
    pivot_data = pivot_data.dropna(subset=['Cluster', 'Decile', 'Value'])
    pivot_data['Cluster'] = pivot_data['Cluster'].astype(int)
    pivot_data['Decile'] = pivot_data['Decile'].astype(int)

    # Create row labels with cluster name and country
    cluster_names_map = CLUSTER_NAMES
    pivot_data['Cluster_Name'] = pivot_data['Cluster'].map(cluster_names_map)
    pivot_data['Row_Label'] = (
        'Cluster ' + pivot_data['Cluster'].astype(str) + ' - ' +
        pivot_data['Cluster_Name'] + ' / ' + pivot_data['Country_Name']
    )

    # Pivot to wide format
    wide = pivot_data.pivot_table(
        index='Row_Label',
        columns='Decile',
        values='Value',
        aggfunc='first'
    )
    wide.columns = [f'Decile_{int(d)}' for d in wide.columns]
    wide = wide.sort_index()

    # Export to Excel
    excel_path = os.path.join(output_dir, 'ewbi_by_decile_clusters.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        wide.to_excel(writer, sheet_name='EWBI_by_Decile')
    
    print(f"  Saved: {excel_path}")


def export_excel(result_df, summary_df, output_dir):
    """Export all data to a single Excel workbook with multiple sheets."""
    merged = _load_ewbi_income_merged()

    excel_path = os.path.join(output_dir, 'ewbi_clustering_data.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: cluster assignments + features
        result_df.to_excel(writer, sheet_name='Cluster_Assignments', index=False)

        # Sheet 2: cluster summary
        summary_df.to_excel(writer, sheet_name='Cluster_Summary', index=False)

        # Sheet 3: EWBI per-decile + income (all years)
        if merged is not None:
            cluster_map = dict(zip(result_df['Country'], result_df['Cluster']))
            merged['Cluster'] = merged['Country'].map(cluster_map)
            merged['Country_Name'] = merged['Country'].map(COUNTRY_NAME_MAP).fillna(merged['Country'])
            export_cols = ['Country', 'Country_Name', 'Cluster', 'Year', 'Decile', 'Value', 'median_equi_disp_inc']
            export_df = merged[export_cols].sort_values(['Cluster', 'Country', 'Year', 'Decile'])
            export_df.to_excel(writer, sheet_name='EWBI_Income_All_Years', index=False)

    print(f"  Saved: {excel_path}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    features_df = load_and_prepare_data()

    # Save features CSV
    features_path = os.path.join(output_dir, 'ewbi_clustering_features.csv')
    features_df.to_csv(features_path, index=False)
    print(f"  Saved features: {features_path}")

    # Summary
    summary = cluster_summary(features_df)
    summary_path = os.path.join(output_dir, 'ewbi_cluster_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"  Saved summary: {summary_path}")

    print("\n  Cluster summary:")
    for _, row in summary.iterrows():
        print(f"    Cluster {int(row['Cluster'])} ({row['Cluster_Name']}): {int(row['N_Countries'])} countries")
        print(f"      Countries: {row['Countries']}")
        print(f"      EWBI (last): {row['EWBI_Last_mean']:.3f} ± {row['EWBI_Last_std']:.3f}")
        print(f"      Growth: {row['Annual_Growth_mean']*100:.2f}% ± {row['Annual_Growth_std']*100:.2f}%")
        print(f"      Interdecile (last): {row['Interdecile_Last_mean']:.3f} ± {row['Interdecile_Last_std']:.3f}")
        print(f"      Interdecile (first): {row['Interdecile_First_mean']:.3f} ± {row['Interdecile_First_std']:.3f}")

    # Plots
    plot_radar_clusters(features_df, output_dir)
    plot_priority_radar_clusters(features_df, output_dir)
    plot_ewbi_vs_income_per_cluster(features_df, output_dir)
    plot_ewbi_vs_income_all_clusters(features_df, output_dir)
    plot_ewbi_vs_income_france(features_df, output_dir)
    plot_ewbi_vs_income_cluster_priority_grid(features_df, output_dir)
    plot_ewbi_country_map(features_df, output_dir)
    plot_cluster_map(features_df, output_dir)
    plot_method5_points_all_clustered(features_df, output_dir)
    plot_performance_vs_ewbi(features_df, output_dir)

    # Excel export
    export_excel(features_df, summary, output_dir)
    export_ewbi_by_decile_clusters(features_df, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
