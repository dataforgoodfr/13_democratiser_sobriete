"""
ewbi_clustering_methods.py

Test two alternative clustering methods on country-level EWBI and income features:
1) Density-based clustering in EWBI x Median Income space (DBSCAN with parameter search)
2) 4-variable clustering on:
   - EWBI
   - Interdecile EWBI (D10/D1)
   - Median Income
   - Interdecile Median Income (D10/D1)

Exports:
- Country-level feature table
- Cluster assignments
- Diagnostic plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
WELL_BEING_DIR = os.path.abspath(os.path.join(REPORT_DIR, '..', '..'))

EWBI_MASTER_PATH = os.path.join(WELL_BEING_DIR, 'output', 'ewbi_master_aggregated.csv')
MEDIAN_INCOME_PATH = os.path.join(REPORT_DIR, 'outputs', 'data', 'median_income_by_decile.csv')

OUTPUT_DIR = os.path.join(REPORT_DIR, 'outputs', 'graphs', 'EWBI', 'Clustering_methods')
os.makedirs(OUTPUT_DIR, exist_ok=True)


COUNTRY_NAME_MAP = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus',
    'CZ': 'Czechia', 'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia',
    'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland', 'FR': 'France',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta',
    'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania',
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia', 'IS': 'Iceland',
    'NO': 'Norway', 'CH': 'Switzerland', 'UK': 'United Kingdom', 'RS': 'Serbia'
}

HIGHLIGHT_COUNTRIES = {
    'EL': 'Greece',
    'ES': 'Spain',
    'FR': 'France',
    'PL': 'Poland',
    'DE': 'Germany',
    'RO': 'Romania',
    'SE': 'Sweden',
}

CLUSTER_COLORS = ['#fb8072', '#8dd3c7', '#bebada', '#80b1d3', '#fdb462', '#b3de69']


def _safe_interdecile(df, value_col, d1='1.0', d10='10.0'):
    v10 = df[df['Decile'] == d10][value_col].values
    v1 = df[df['Decile'] == d1][value_col].values
    if len(v10) == 0 or len(v1) == 0 or pd.isna(v10[0]) or pd.isna(v1[0]) or v1[0] == 0:
        return np.nan
    return float(v10[0]) / float(v1[0])


def load_country_features():
    print(f"Loading EWBI master: {EWBI_MASTER_PATH}")
    print(f"Loading median income: {MEDIAN_INCOME_PATH}")

    ewbi = pd.read_csv(EWBI_MASTER_PATH, low_memory=False)
    income = pd.read_csv(MEDIAN_INCOME_PATH)

    ewbi = ewbi[
        (ewbi['Level'] == 1) &
        (~ewbi['Country'].isin(['EU-27', 'All Countries']))
    ].copy()

    ewbi['Year'] = pd.to_numeric(ewbi['Year'], errors='coerce')
    ewbi['Value'] = pd.to_numeric(ewbi['Value'], errors='coerce')
    ewbi['Decile'] = ewbi['Decile'].astype(str).str.strip()
    ewbi = ewbi.dropna(subset=['Year', 'Value', 'Country'])
    ewbi['Year'] = ewbi['Year'].astype(int)

    income['year'] = pd.to_numeric(income['year'], errors='coerce')
    income['decile'] = pd.to_numeric(income['decile'], errors='coerce')
    income['median_equi_disp_inc'] = pd.to_numeric(income['median_equi_disp_inc'], errors='coerce')
    income = income.dropna(subset=['country', 'year', 'decile', 'median_equi_disp_inc'])
    income['year'] = income['year'].astype(int)
    income['decile'] = income['decile'].astype(int)

    records = []
    for country in sorted(ewbi['Country'].dropna().unique()):
        e_country = ewbi[ewbi['Country'] == country].copy()
        i_country = income[income['country'] == country].copy()

        if e_country.empty or i_country.empty:
            continue

        ewbi_years = set(e_country['Year'].unique())
        income_years = set(i_country['year'].unique())
        common_years = sorted(ewbi_years.intersection(income_years))
        if not common_years:
            continue

        # Use latest common year to compare consistent variables.
        year = common_years[-1]
        e_year = e_country[e_country['Year'] == year].copy()
        i_year = i_country[i_country['year'] == year].copy()

        ewbi_all = e_year[e_year['Decile'] == 'All Deciles']['Value'].values
        if len(ewbi_all) == 0:
            continue

        inter_ewbi = _safe_interdecile(e_year, 'Value', d1='1.0', d10='10.0')

        i_d1 = i_year[i_year['decile'] == 1]['median_equi_disp_inc'].values
        i_d10 = i_year[i_year['decile'] == 10]['median_equi_disp_inc'].values
        if len(i_d1) == 0 or len(i_d10) == 0 or i_d1[0] == 0:
            continue

        # Prefer decile 5 as central median-income proxy; fallback to overall median of deciles.
        i_d5 = i_year[i_year['decile'] == 5]['median_equi_disp_inc'].values
        if len(i_d5) > 0 and not pd.isna(i_d5[0]):
            median_income = float(i_d5[0])
        else:
            median_income = float(i_year['median_equi_disp_inc'].median())

        inter_income = float(i_d10[0]) / float(i_d1[0])

        records.append({
            'Country': country,
            'Country_Name': COUNTRY_NAME_MAP.get(country, country),
            'Year': int(year),
            'EWBI': float(ewbi_all[0]),
            'Interdecile_EWBI': inter_ewbi,
            'Median_Income': median_income,
            'Interdecile_Median_Income': inter_income,
        })

    features = pd.DataFrame(records)
    features = features.dropna(subset=['EWBI', 'Interdecile_EWBI', 'Median_Income', 'Interdecile_Median_Income'])

    print(f"Countries with complete features: {len(features)}")
    return features


def _plot_cluster_scatter(df, label_col, title, output_path):
    plt.figure(figsize=(12, 8))

    labels = sorted(df[label_col].dropna().unique())
    cmap = plt.get_cmap('tab10')

    for i, lab in enumerate(labels):
        subset = df[df[label_col] == lab]
        color = '#bdbdbd' if lab == -1 else cmap(i % 10)
        alpha = 0.65 if lab == -1 else 0.85
        plt.scatter(
            subset['Median_Income'],
            subset['EWBI'],
            color=color,
            s=90,
            alpha=alpha,
            edgecolors='white',
            linewidths=0.4,
            label=f'Cluster {lab}' if lab != -1 else 'Noise'
        )
        for _, row in subset.iterrows():
            plt.text(row['Median_Income'], row['EWBI'], row['Country'], fontsize=8, alpha=0.8)

    plt.xlabel('Median Income (Decile 5 proxy, EUR)')
    plt.ylabel('EWBI (All Deciles)')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def _load_ewbi_income_merged():
    """Load and merge EWBI deciles with median income by decile."""
    median_income_df = pd.read_csv(MEDIAN_INCOME_PATH)
    ewbi_df = pd.read_csv(EWBI_MASTER_PATH, low_memory=False)

    ewbi_decile = ewbi_df[
        (ewbi_df['Level'] == 1) &
        (ewbi_df['Decile'] != 'All Deciles') &
        (~ewbi_df['Country'].isin(['EU-27', 'All Countries']))
    ].copy()

    ewbi_decile['Decile'] = pd.to_numeric(ewbi_decile['Decile'], errors='coerce').astype('Int64')
    ewbi_decile['Value'] = pd.to_numeric(ewbi_decile['Value'], errors='coerce')
    ewbi_decile = ewbi_decile.dropna(subset=['Decile', 'Value', 'Country', 'Year'])
    ewbi_decile['Year'] = pd.to_numeric(ewbi_decile['Year'], errors='coerce').astype('Int64')

    median_income_df['year'] = pd.to_numeric(median_income_df['year'], errors='coerce').astype('Int64')
    median_income_df['decile'] = pd.to_numeric(median_income_df['decile'], errors='coerce').astype('Int64')
    median_income_df['median_equi_disp_inc'] = pd.to_numeric(median_income_df['median_equi_disp_inc'], errors='coerce')
    median_income_df = median_income_df.dropna(subset=['country', 'year', 'decile', 'median_equi_disp_inc'])

    merged = ewbi_decile.merge(
        median_income_df,
        left_on=['Country', 'Year', 'Decile'],
        right_on=['country', 'year', 'decile'],
        how='inner',
    )
    merged = merged.dropna(subset=['Value', 'median_equi_disp_inc'])
    return merged if not merged.empty else None


def _plot_reference_style_by_cluster(assignments, label_col, method_title, output_path):
    """
    Plot one panel per cluster in the style of ewbi_vs_income_by_cluster.png:
    decile curves, same axis scales, and other clusters in grey.
    """
    merged = _load_ewbi_income_merged()
    if merged is None:
        print("  WARNING: No merged EWBI/income decile data for reference-style plot")
        return

    country_to_cluster = dict(zip(assignments['Country'], assignments[label_col]))
    plot_data = merged[merged['Country'].isin(country_to_cluster.keys())].copy()

    # Keep last year per country for comparability with the reference graph.
    last_year = plot_data.groupby('Country')['Year'].max().reset_index()
    last_year.columns = ['Country', 'last_year']
    plot_data = plot_data.merge(last_year, on='Country')
    plot_data = plot_data[plot_data['Year'] == plot_data['last_year']].copy()
    plot_data['cluster'] = plot_data['Country'].map(country_to_cluster)

    cluster_ids = sorted([c for c in plot_data['cluster'].dropna().unique() if c != -1])
    if not cluster_ids:
        print("  WARNING: No non-noise clusters to plot")
        return

    n = len(cluster_ids)
    n_cols = 2 if n > 1 else 1
    n_rows = int(np.ceil(n / n_cols))

    x_min, x_max = plot_data['median_equi_disp_inc'].min(), plot_data['median_equi_disp_inc'].max()
    y_min, y_max = plot_data['Value'].min(), plot_data['Value'].max()
    x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.07 if y_max > y_min else 0.01

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 5.5 * n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(-1)

    countries = sorted(plot_data['Country'].unique())

    for i, cl in enumerate(cluster_ids):
        ax = axes[i]
        target_countries = [c for c in countries if country_to_cluster.get(c) == cl]
        other_countries = [c for c in countries if country_to_cluster.get(c) != cl]

        # Background: all other clusters + optional noise in grey.
        for country in other_countries:
            cdata = plot_data[plot_data['Country'] == country].sort_values('Decile')
            if cdata.empty:
                continue
            ax.plot(cdata['median_equi_disp_inc'], cdata['Value'], color='#bdbdbd', alpha=0.35, linewidth=0.9, zorder=1)
            ax.scatter(cdata['median_equi_disp_inc'], cdata['Value'], c='#bdbdbd', s=30, alpha=0.35,
                       edgecolors='white', linewidths=0.2, zorder=1)

        # Foreground: target cluster in color.
        cluster_color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        for country in target_countries:
            cdata = plot_data[plot_data['Country'] == country].sort_values('Decile')
            if cdata.empty:
                continue
            is_highlight = country in HIGHLIGHT_COUNTRIES
            ax.plot(
                cdata['median_equi_disp_inc'], cdata['Value'],
                color=cluster_color, alpha=0.75, linewidth=2.3 if is_highlight else 1.3, zorder=2,
            )
            ax.scatter(
                cdata['median_equi_disp_inc'], cdata['Value'],
                c=cluster_color, s=88 if is_highlight else 52, alpha=0.95 if is_highlight else 0.78,
                edgecolors='black' if is_highlight else 'white',
                linewidths=0.6 if is_highlight else 0.3,
                zorder=3,
            )

            if is_highlight:
                row_last = cdata.iloc[-1]
                ax.annotate(
                    HIGHLIGHT_COUNTRIES[country],
                    (row_last['median_equi_disp_inc'], row_last['Value']),
                    fontsize=8.5, fontweight='bold', color=cluster_color,
                    textcoords='offset points', xytext=(5, 2), zorder=4,
                )

        ax.set_title(f'Cluster {cl}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.set_facecolor('white')
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Hide unused panels.
    for j in range(len(cluster_ids), len(axes)):
        axes[j].axis('off')

    for idx, ax in enumerate(axes[:len(cluster_ids)]):
        if idx >= len(cluster_ids) - n_cols:
            ax.set_xlabel('Median Equivalized Disposable Income (€)', fontsize=11)
        if idx % n_cols == 0:
            ax.set_ylabel('EWBI Score', fontsize=11)

    legend_handles = [
        Line2D([0], [0], color='#bdbdbd', lw=2, label='Other clusters (background)')
    ]
    for i, cl in enumerate(cluster_ids):
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                   markeredgecolor='black', markeredgewidth=0.7, markersize=9, label=f'Cluster {cl}')
        )

    fig.suptitle(method_title, fontsize=16, fontweight='bold', y=0.98)
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(4, len(legend_handles)),
               fontsize=10, bbox_to_anchor=(0.5, -0.01), frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def run_density_method(features):
    print("\n=== Method 1: Density-based clustering in EWBI x Median Income space ===")

    X = features[['EWBI', 'Median_Income']].values
    Xs = StandardScaler().fit_transform(X)

    best = None
    for min_samples in [2, 3, 4]:
        for eps in np.linspace(0.2, 2.0, 60):
            model = DBSCAN(eps=float(eps), min_samples=min_samples)
            labels = model.fit_predict(Xs)
            non_noise = labels[labels != -1]
            n_clusters = len(set(non_noise)) if len(non_noise) else 0
            n_noise = int((labels == -1).sum())

            if n_clusters in (3, 4):
                score = (n_clusters, len(non_noise), -n_noise)
                if best is None or score > best['score']:
                    best = {
                        'labels': labels,
                        'eps': float(eps),
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'score': score,
                    }

    # Fallback if 3/4 clusters not found in scan.
    if best is None:
        fallback = DBSCAN(eps=0.6, min_samples=3)
        labels = fallback.fit_predict(Xs)
        non_noise = labels[labels != -1]
        n_clusters = len(set(non_noise)) if len(non_noise) else 0
        n_noise = int((labels == -1).sum())
        best = {
            'labels': labels,
            'eps': 0.6,
            'min_samples': 3,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'score': (n_clusters, len(non_noise), -n_noise),
        }

    out = features.copy()
    out['density_cluster'] = best['labels']

    print(
        f"Selected DBSCAN params: eps={best['eps']:.3f}, min_samples={best['min_samples']}, "
        f"clusters={best['n_clusters']}, noise={best['n_noise']}"
    )

    csv_path = os.path.join(OUTPUT_DIR, 'method1_density_clusters.csv')
    out.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    plot_path = os.path.join(OUTPUT_DIR, 'method1_density_ewbi_income_space.png')
    _plot_cluster_scatter(
        out,
        'density_cluster',
        'Method 1 (Density): EWBI x Median Income Space',
        plot_path,
    )

    ref_plot_path = os.path.join(OUTPUT_DIR, 'method1_density_reference_style_by_cluster.png')
    _plot_reference_style_by_cluster(
        out,
        'density_cluster',
        'Method 1 (Density): EWBI vs Income by Cluster (reference style)',
        ref_plot_path,
    )


def run_four_variable_method(features):
    print("\n=== Method 2: 4-variable clustering ===")
    vars4 = ['EWBI', 'Interdecile_EWBI', 'Median_Income', 'Interdecile_Median_Income']

    X = features[vars4].values
    Xs = StandardScaler().fit_transform(X)

    out = features.copy()
    summary_rows = []

    for k in [3, 4]:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(Xs)
        out[f'kmeans_{k}'] = labels

        sil = silhouette_score(Xs, labels) if len(set(labels)) > 1 else np.nan
        summary_rows.append({'method': f'kmeans_{k}', 'n_clusters': k, 'silhouette': sil})
        print(f"k={k} silhouette={sil:.3f}")

        plot_path = os.path.join(OUTPUT_DIR, f'method2_kmeans_{k}_ewbi_income_space.png')
        _plot_cluster_scatter(
            out,
            f'kmeans_{k}',
            f'Method 2 (4 vars, KMeans k={k}): projected on EWBI x Median Income',
            plot_path,
        )

        ref_plot_path = os.path.join(OUTPUT_DIR, f'method2_kmeans_{k}_reference_style_by_cluster.png')
        _plot_reference_style_by_cluster(
            out,
            f'kmeans_{k}',
            f'Method 2 (4 vars, KMeans k={k}): EWBI vs Income by Cluster (reference style)',
            ref_plot_path,
        )

    csv_path = os.path.join(OUTPUT_DIR, 'method2_fourvar_clusters.csv')
    out.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    summary_path = os.path.join(OUTPUT_DIR, 'method2_fourvar_quality_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")


def run_manual_threshold_methods(features):
    print("\n=== Method 3: Manual threshold clustering variants ===")

    out = features.copy()

    # Variant A (3 clusters):
    # 0: EWBI < 0.68
    # 1: EWBI >= 0.68 and Interdecile_Median_Income < 5
    # 2: rest
    cond0 = out['EWBI'] < 0.68
    cond1 = (out['EWBI'] >= 0.68) & (out['Interdecile_Median_Income'] < 5)
    out['manual_3c'] = np.select([cond0, cond1], [0, 1], default=2).astype(int)

    print("Variant A counts (manual_3c):")
    print(out['manual_3c'].value_counts().sort_index().to_string())

    plot_path_3c = os.path.join(OUTPUT_DIR, 'method3_manual_3c_ewbi_income_space.png')
    _plot_cluster_scatter(
        out,
        'manual_3c',
        'Method 3A (Manual): EWBI < 0.68; then interdecile income < 5',
        plot_path_3c,
    )

    ref_plot_path_3c = os.path.join(OUTPUT_DIR, 'method3_manual_3c_reference_style_by_cluster.png')
    _plot_reference_style_by_cluster(
        out,
        'manual_3c',
        'Method 3A (Manual): EWBI vs Income by Cluster (reference style)',
        ref_plot_path_3c,
    )

    # Variant B (4 clusters):
    # Split by EWBI at 0.7, then by Interdecile_Median_Income:
    # - EWBI < 0.7: threshold 7.5
    # - EWBI >= 0.7: threshold 5
    cond_b0 = (out['EWBI'] < 0.7) & (out['Interdecile_Median_Income'] < 7.5)
    cond_b1 = (out['EWBI'] < 0.7) & (out['Interdecile_Median_Income'] >= 7.5)
    cond_b2 = (out['EWBI'] >= 0.7) & (out['Interdecile_Median_Income'] < 5)
    out['manual_4c'] = np.select([cond_b0, cond_b1, cond_b2], [0, 1, 2], default=3).astype(int)

    print("Variant B counts (manual_4c):")
    print(out['manual_4c'].value_counts().sort_index().to_string())

    plot_path_4c = os.path.join(OUTPUT_DIR, 'method3_manual_4c_ewbi_income_space.png')
    _plot_cluster_scatter(
        out,
        'manual_4c',
        'Method 3B (Manual): EWBI split 0.7; interdecile income split 7.5/5',
        plot_path_4c,
    )

    ref_plot_path_4c = os.path.join(OUTPUT_DIR, 'method3_manual_4c_reference_style_by_cluster.png')
    _plot_reference_style_by_cluster(
        out,
        'manual_4c',
        'Method 3B (Manual): EWBI vs Income by Cluster (reference style)',
        ref_plot_path_4c,
    )

    csv_path = os.path.join(OUTPUT_DIR, 'method3_manual_threshold_clusters.csv')
    out.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def run_anchor_profile_method(features):
    """
    Method 4: rule-based clustering from anchor profiles.

    Build 4 prototype profiles from requested country pairs using:
      - EWBI
      - Interdecile_EWBI
      - Median_Income
      - Interdecile_Median_Income

    Then assign each country to the nearest anchor profile in standardized space.
    This keeps the method variable-driven while forcing the anchor countries into
    the intended groups.
    """
    print("\n=== Method 4: Anchor-profile clustering (4 groups) ===")

    vars4 = ['EWBI', 'Interdecile_EWBI', 'Median_Income', 'Interdecile_Median_Income']
    anchors = {
        0: ['PT', 'EL'],  # Portugal + Greece
        1: ['FR', 'ES'],  # France + Spain
        2: ['RO', 'PL'],  # Romania + Poland
        3: ['LU', 'IS'],  # Luxembourg + Iceland
    }

    out = features.copy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(out[vars4])
    Z = pd.DataFrame(Xs, columns=vars4, index=out.index)

    # Build standardized anchor centroids from requested country pairs.
    centroid_rows = []
    centroids = {}
    for cluster_id, countries in anchors.items():
        idx = out[out['Country'].isin(countries)].index
        if len(idx) == 0:
            raise ValueError(f"Anchor countries missing for cluster {cluster_id}: {countries}")

        centroid = Z.loc[idx, vars4].mean().values
        centroids[cluster_id] = centroid

        centroid_rows.append({
            'cluster': cluster_id,
            'anchor_countries': ','.join(countries),
            'EWBI_z': float(centroid[0]),
            'Interdecile_EWBI_z': float(centroid[1]),
            'Median_Income_z': float(centroid[2]),
            'Interdecile_Median_Income_z': float(centroid[3]),
        })

    # Distance-to-anchor rule in standardized variable space.
    dist_cols = []
    for cluster_id, centroid in centroids.items():
        d = np.sqrt(((Z[vars4].values - centroid) ** 2).sum(axis=1))
        col = f'dist_anchor_{cluster_id}'
        out[col] = d
        dist_cols.append(col)

    out['anchor_4c'] = out[dist_cols].values.argmin(axis=1).astype(int)
    out['anchor_group_name'] = out['anchor_4c'].map({
        0: 'PT-EL profile',
        1: 'FR-ES profile',
        2: 'RO-PL profile',
        3: 'LU-IS profile',
    })

    print("Anchor-profile counts (anchor_4c):")
    print(out['anchor_4c'].value_counts().sort_index().to_string())

    # Save assignments and anchor definitions.
    csv_path = os.path.join(OUTPUT_DIR, 'method4_anchor_profile_clusters.csv')
    out.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    anchor_path = os.path.join(OUTPUT_DIR, 'method4_anchor_profiles_zspace.csv')
    pd.DataFrame(centroid_rows).to_csv(anchor_path, index=False)
    print(f"  Saved: {anchor_path}")

    plot_path = os.path.join(OUTPUT_DIR, 'method4_anchor_profile_ewbi_income_space.png')
    _plot_cluster_scatter(
        out,
        'anchor_4c',
        'Method 4 (Anchor profiles): projected on EWBI x Median Income',
        plot_path,
    )

    ref_plot_path = os.path.join(OUTPUT_DIR, 'method4_anchor_profile_reference_style_by_cluster.png')
    _plot_reference_style_by_cluster(
        out,
        'anchor_4c',
        'Method 4 (Anchor profiles): EWBI vs Income by Cluster (reference style)',
        ref_plot_path,
    )


def _compute_income_benchmark_residuals(step_eur):
    """
    Compute point-level EWBI residuals relative to an interpolated benchmark curve
    built from income bins of size `step_eur`.
    """
    merged = _load_ewbi_income_merged()
    if merged is None or merged.empty:
        return None, None

    # Keep last year per country for cross-country comparability.
    last_year = merged.groupby('Country')['Year'].max().reset_index()
    last_year.columns = ['Country', 'last_year']
    points = merged.merge(last_year, on='Country')
    points = points[points['Year'] == points['last_year']].copy()

    if points.empty:
        return None, None

    # Bin median incomes and compute average EWBI per bin center.
    points['income_bin_center'] = (
        np.round(points['median_equi_disp_inc'] / step_eur) * step_eur
    )

    benchmark = (
        points.groupby('income_bin_center', as_index=False)['Value']
        .mean()
        .sort_values('income_bin_center')
        .rename(columns={'Value': 'benchmark_ewbi'})
    )

    if len(benchmark) < 2:
        return None, None

    # Interpolate expected EWBI at each observed income.
    x = benchmark['income_bin_center'].values.astype(float)
    y = benchmark['benchmark_ewbi'].values.astype(float)
    points['ewbi_expected'] = np.interp(points['median_equi_disp_inc'].values.astype(float), x, y)
    points['ewbi_residual'] = points['Value'] - points['ewbi_expected']

    return points, benchmark


def run_performance_method():
    """
    Method 5: performance at constant income.

    Uses 5,000 EUR income bins.
    1) Compute average EWBI by income bins
    2) Interpolate benchmark EWBI(income)
    3) Compute residual EWBI - benchmark for each country/decile point
    4) Aggregate performance as average decile deviation from baseline
    5) Split into Low vs High performers
    6) Also test EWBI-based clustering for k=3 and k=4
    """
    print("\n=== Method 5: Performance at constant income (residual-based) ===")

    step = 5000
    print(f"\nUsing income step: {step} EUR")
    points, benchmark = _compute_income_benchmark_residuals(step)
    if points is None or benchmark is None:
        print(f"  Skipping step {step}: insufficient data")
        return

    # Average deviation from baseline across deciles (equal weight per decile).
    decile_resid = (
        points.groupby(['Country', 'Decile'], as_index=False)
        .agg(decile_residual=('ewbi_residual', 'mean'))
    )

    perf = (
        points.groupby('Country', as_index=False)
        .agg(
            Country_Name=('Country', lambda s: COUNTRY_NAME_MAP.get(s.iloc[0], s.iloc[0])),
            Year=('Year', 'max'),
            EWBI_Mean=('Value', 'mean'),
            Income_Mean=('median_equi_disp_inc', 'mean'),
        )
        .merge(
            decile_resid.groupby('Country', as_index=False).agg(
                Performance_Score=('decile_residual', 'mean'),
                Performance_Std=('decile_residual', 'std'),
                n_deciles=('decile_residual', 'size'),
            ),
            on='Country',
            how='left',
        )
    )

    # Step 1: positive residual => above benchmark at constant income.
    perf['Performance_Group'] = np.where(
        perf['Performance_Score'] >= 0,
        'High performer',
        'Low performer'
    )

    # Step 2a: split by income (low/high) and combine with performance split.
    income_cut = perf['Income_Mean'].median()
    perf['Income_Group'] = np.where(perf['Income_Mean'] >= income_cut, 'High income', 'Low income')

    # Step 2b: split by EWBI (low/high) and combine with performance split.
    ewbi_cut = perf['EWBI_Mean'].median()
    perf['EWBI_Group'] = np.where(perf['EWBI_Mean'] >= ewbi_cut, 'High EWBI', 'Low EWBI')

    # 4-group segmentation: performance x income.
    perf_income_map = {
        ('Low performer', 'Low income'): 0,
        ('Low performer', 'High income'): 1,
        ('High performer', 'Low income'): 2,
        ('High performer', 'High income'): 3,
    }
    perf['PerfIncome_4c'] = perf.apply(
        lambda r: perf_income_map[(r['Performance_Group'], r['Income_Group'])],
        axis=1,
    ).astype(int)
    perf['PerfIncome_4c_Name'] = perf.apply(
        lambda r: f"{r['Performance_Group']} / {r['Income_Group']}",
        axis=1,
    )

    # 4-group segmentation: performance x EWBI.
    perf_ewbi_map = {
        ('Low performer', 'Low EWBI'): 0,
        ('Low performer', 'High EWBI'): 1,
        ('High performer', 'Low EWBI'): 2,
        ('High performer', 'High EWBI'): 3,
    }
    perf['PerfEWBI_4c'] = perf.apply(
        lambda r: perf_ewbi_map[(r['Performance_Group'], r['EWBI_Group'])],
        axis=1,
    ).astype(int)
    perf['PerfEWBI_4c_Name'] = perf.apply(
        lambda r: f"{r['Performance_Group']} / {r['EWBI_Group']}",
        axis=1,
    )

    # EWBI-based clusters to test (k=3 and k=4).
    perf['EWBI_cluster_3'] = pd.qcut(perf['EWBI_Mean'], q=3, labels=False, duplicates='drop').astype(int)
    perf['EWBI_cluster_4'] = pd.qcut(perf['EWBI_Mean'], q=4, labels=False, duplicates='drop').astype(int)

    # Export point-level and country-level tables.
    points_path = os.path.join(OUTPUT_DIR, f'method5_points_residuals_step_{step}.csv')
    points.to_csv(points_path, index=False)
    print(f"  Saved: {points_path}")

    decile_path = os.path.join(OUTPUT_DIR, f'method5_decile_residuals_step_{step}.csv')
    decile_resid.to_csv(decile_path, index=False)
    print(f"  Saved: {decile_path}")

    bench_path = os.path.join(OUTPUT_DIR, f'method5_benchmark_curve_step_{step}.csv')
    benchmark.to_csv(bench_path, index=False)
    print(f"  Saved: {bench_path}")

    perf_path = os.path.join(OUTPUT_DIR, f'method5_country_performance_step_{step}.csv')
    perf.sort_values('Performance_Score', ascending=False).to_csv(perf_path, index=False)
    print(f"  Saved: {perf_path}")

    print(f"  Income split median: {income_cut:.1f}")
    print(f"  EWBI split median: {ewbi_cut:.4f}")
    print("  PerfIncome_4c counts:")
    print(perf['PerfIncome_4c'].value_counts().sort_index().to_string())
    print("  PerfEWBI_4c counts:")
    print(perf['PerfEWBI_4c'].value_counts().sort_index().to_string())

    # Diagnostic print for requested examples.
    for c in ['FR', 'DE', 'PL']:
        row = perf[perf['Country'] == c]
        if not row.empty:
            r = row.iloc[0]
            print(
                f"  {c}: score={r['Performance_Score']:.4f}, "
                f"group={r['Performance_Group']}, income_mean={r['Income_Mean']:.0f}, ewbi_mean={r['EWBI_Mean']:.3f}"
            )

    # Plot 1: benchmark curve with points.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(
        points['median_equi_disp_inc'],
        points['Value'],
        c='#bdbdbd',
        alpha=0.45,
        s=26,
        edgecolors='none',
        label='Country-decile points'
    )
    ax.plot(
        benchmark['income_bin_center'],
        benchmark['benchmark_ewbi'],
        color='#1f78b4',
        linewidth=2.4,
        label=f'Interpolated benchmark (step={step} EUR)'
    )
    ax.set_xlabel('Median Equivalized Disposable Income (€)')
    ax.set_ylabel('EWBI Score')
    ax.set_title(
        f'Method 5: EWBI benchmark at constant income (step {step} EUR)',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plot1 = os.path.join(OUTPUT_DIR, f'method5_benchmark_plot_step_{step}.png')
    fig.savefig(plot1, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {plot1}")

    # Plot 2: country means colored by performance group + benchmark curve.
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = perf['Performance_Group'].map({'High performer': '#1b9e77', 'Low performer': '#d95f02'})
    ax.scatter(
        perf['Income_Mean'],
        perf['EWBI_Mean'],
        c=colors,
        s=90,
        alpha=0.88,
        edgecolors='white',
        linewidths=0.5,
        zorder=2,
    )

    for _, r in perf.iterrows():
        if r['Country'] in ['FR', 'DE', 'PL']:
            ax.annotate(
                f"{r['Country']} ({r['Performance_Group'].split()[0]})",
                (r['Income_Mean'], r['EWBI_Mean']),
                fontsize=9,
                fontweight='bold',
                textcoords='offset points',
                xytext=(6, 3),
                zorder=3,
            )

    ax.plot(
        benchmark['income_bin_center'],
        benchmark['benchmark_ewbi'],
        color='#377eb8',
        linewidth=2.0,
        alpha=0.9,
        label='Benchmark line'
    )
    ax.set_xlabel('Mean Median Income (EUR)')
    ax.set_ylabel('Mean EWBI')
    ax.set_title(
        f'Method 5: Low vs High performers at constant income (step {step} EUR)',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plot2 = os.path.join(OUTPUT_DIR, f'method5_performance_groups_step_{step}.png')
    fig.savefig(plot2, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {plot2}")

    # Reference-style panels using EWBI-based clusters (k=3 and k=4).
    perf_assign = perf[['Country', 'EWBI_cluster_3', 'EWBI_cluster_4']].copy()

    ref3 = os.path.join(OUTPUT_DIR, f'method5_ewbi_cluster3_reference_style_step_{step}.png')
    _plot_reference_style_by_cluster(
        perf_assign,
        'EWBI_cluster_3',
        'Method 5: EWBI-based clustering (k=3) - reference style',
        ref3,
    )

    ref4 = os.path.join(OUTPUT_DIR, f'method5_ewbi_cluster4_reference_style_step_{step}.png')
    _plot_reference_style_by_cluster(
        perf_assign,
        'EWBI_cluster_4',
        'Method 5: EWBI-based clustering (k=4) - reference style',
        ref4,
    )

    # Reference-style panels for requested two-step segmentations.
    perf_assign2 = perf[['Country', 'PerfIncome_4c', 'PerfEWBI_4c']].copy()

    ref_perf_income = os.path.join(OUTPUT_DIR, f'method5_perf_income_4c_reference_style_step_{step}.png')
    _plot_reference_style_by_cluster(
        perf_assign2,
        'PerfIncome_4c',
        'Method 5: Performance first, then Income split (4 groups)',
        ref_perf_income,
    )

    ref_perf_ewbi = os.path.join(OUTPUT_DIR, f'method5_perf_ewbi_4c_reference_style_step_{step}.png')
    _plot_reference_style_by_cluster(
        perf_assign2,
        'PerfEWBI_4c',
        'Method 5: Performance first, then EWBI split (4 groups)',
        ref_perf_ewbi,
    )


# ---------------------------------------------------------------------------
# Method 6: EWBI × Income quadrant clustering
# ---------------------------------------------------------------------------
# Each country gets a 2×2 quadrant label based on simultaneous thresholds:
#   0 = Low EWBI / Low Income
#   1 = Low EWBI / High Income
#   2 = High EWBI / Low Income
#   3 = High EWBI / High Income
#
# Seven threshold variants are tested so the sensitivity of cluster membership
# to the choice of cut-points can be inspected visually and tabularly.
# ---------------------------------------------------------------------------
_EWBI_INCOME_THRESHOLD_VARIANTS = [
    # (label,         ewbi_cut,   income_cut,   description)
    ('T1_data_median',  None,    None,   'EWBI = sample median  /  Income = sample median'),
    ('T2_ewbi70_med',   0.70,    None,   'EWBI = 0.70  /  Income = sample median'),
    ('T3_ewbi70_25k',   0.70,    25_000, 'EWBI = 0.70  /  Income = 25 000 €'),
    ('T4_ewbi70_20k',   0.70,    20_000, 'EWBI = 0.70  /  Income = 20 000 €'),
    ('T5_ewbi68_med',   0.68,    None,   'EWBI = 0.68  /  Income = sample median'),
    ('T6_ewbi68_22k',   0.68,    22_000, 'EWBI = 0.68  /  Income = 22 000 €'),
    ('T7_pct33',        None,    None,   'EWBI = 33rd pct  /  Income = 33rd pct',),   # handled specially
]

_QUADRANT_NAMES = {
    0: 'Low EWBI / Low Income',
    1: 'Low EWBI / High Income',
    2: 'High EWBI / Low Income',
    3: 'High EWBI / High Income',
}
_QUADRANT_COLORS = ['#fb8072', '#fdb462', '#8dd3c7', '#80b1d3']


def _assign_quadrants(df, ewbi_cut, income_cut):
    """Return integer array 0-3 for each row in df."""
    hi_ewbi   = df['EWBI']          >= ewbi_cut
    hi_income = df['Median_Income'] >= income_cut
    return np.where(
        ~hi_ewbi & ~hi_income, 0,
        np.where(~hi_ewbi & hi_income, 1,
                 np.where(hi_ewbi & ~hi_income, 2, 3))
    ).astype(int)


def _plot_quadrant_scatter(df, label_col, ewbi_cut, income_cut, title, output_path):
    """Scatter of country means, coloured by quadrant, with threshold cross-hairs."""
    fig, ax = plt.subplots(figsize=(11, 7))

    for q in range(4):
        sub = df[df[label_col] == q]
        ax.scatter(
            sub['Median_Income'], sub['EWBI'],
            color=_QUADRANT_COLORS[q], s=95, alpha=0.88,
            edgecolors='white', linewidths=0.5,
            label=_QUADRANT_NAMES[q], zorder=3,
        )
        for _, row in sub.iterrows():
            ax.text(row['Median_Income'] + 200, row['EWBI'], row['Country'],
                    fontsize=7.5, alpha=0.85, zorder=4)

    # Threshold lines
    ax.axvline(income_cut, color='#444444', linewidth=1.2, linestyle='--', alpha=0.7,
               label=f'Income cut = {income_cut:,.0f} €')
    ax.axhline(ewbi_cut, color='#444444', linewidth=1.2, linestyle=':',  alpha=0.7,
               label=f'EWBI cut = {ewbi_cut:.4f}')

    ax.set_xlabel('Median Income – decile 5 proxy (€)', fontsize=11)
    ax.set_ylabel('EWBI (All Deciles)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {output_path}')


def run_ewbi_income_quadrant_method(features):
    """
    Method 6: High/Low EWBI × High/Low Income – 2×2 quadrant clustering.

    Tests 7 threshold variants (see _EWBI_INCOME_THRESHOLD_VARIANTS).
    Outputs per variant:
      - scatter plot with threshold lines
      - reference-style panel plot
    Aggregated outputs:
      - CSV with all assignments side by side
      - stability heatmap: rows=countries, columns=variants, cells=quadrant (0-3)
      - silhouette scores for each variant
    """
    print('\n=== Method 6: EWBI × Income quadrant clustering (7 threshold variants) ===')

    out = features[['Country', 'Country_Name', 'Year', 'EWBI', 'Median_Income',
                     'Interdecile_EWBI', 'Interdecile_Median_Income']].copy()

    ewbi_median   = float(out['EWBI'].median())
    income_median = float(out['Median_Income'].median())
    ewbi_p33      = float(out['EWBI'].quantile(0.33))
    income_p33    = float(out['Median_Income'].quantile(0.33))

    print(f'  Data-driven values — EWBI median={ewbi_median:.4f}, '
          f'Income median={income_median:,.0f} €')
    print(f'  Data-driven values — EWBI 33rd pct={ewbi_p33:.4f}, '
          f'Income 33rd pct={income_p33:,.0f} €')

    Xs = StandardScaler().fit_transform(out[['EWBI', 'Median_Income']].values)

    summary_rows = []

    for entry in _EWBI_INCOME_THRESHOLD_VARIANTS:
        tag, ewbi_cut, income_cut, desc = entry[0], entry[1], entry[2], entry[3]

        # Resolve data-driven cuts
        if tag == 'T7_pct33':
            ewbi_cut_   = ewbi_p33
            income_cut_ = income_p33
        else:
            ewbi_cut_   = ewbi_cut   if ewbi_cut   is not None else ewbi_median
            income_cut_ = income_cut if income_cut is not None else income_median

        out[tag] = _assign_quadrants(out, ewbi_cut_, income_cut_)

        counts = out[tag].value_counts().sort_index()
        n_labels = len(counts)
        sil = float('nan')
        if n_labels > 1:
            try:
                sil = float(silhouette_score(Xs, out[tag].values))
            except Exception:
                pass

        summary_rows.append({
            'variant':     tag,
            'description': desc,
            'ewbi_cut':    round(ewbi_cut_, 4),
            'income_cut':  round(income_cut_, 1),
            'n_quadrants': n_labels,
            'silhouette':  round(sil, 4) if not np.isnan(sil) else None,
            **{f'Q{q}_count': int(counts.get(q, 0)) for q in range(4)},
        })

        print(f'  {tag}: cuts EWBI≥{ewbi_cut_:.4f} / Income≥{income_cut_:,.0f}  '
              f'| counts {dict(counts.to_dict())}  | sil={sil:.3f}')

        scatter_path = os.path.join(OUTPUT_DIR, f'method6_{tag}_scatter.png')
        _plot_quadrant_scatter(
            out, tag, ewbi_cut_, income_cut_,
            f'Method 6 ({tag}): EWBI × Income quadrants\n{desc}',
            scatter_path,
        )

        ref_path = os.path.join(OUTPUT_DIR, f'method6_{tag}_reference_style.png')
        _plot_reference_style_by_cluster(
            out[['Country', tag]],
            tag,
            f'Method 6 ({tag}): {desc}',
            ref_path,
        )

    # ── Stability heatmap ──────────────────────────────────────────────────
    variant_cols = [e[0] for e in _EWBI_INCOME_THRESHOLD_VARIANTS]
    out_sorted = out.sort_values('Country_Name').reset_index(drop=True)

    n_c = len(out_sorted)
    n_v = len(variant_cols)
    cell_w, cell_h = 1.0, 0.45
    label_w = 3.0
    fig_w = n_v * cell_w + label_w + 0.5
    fig_h = n_c * cell_h + 1.4

    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-label_w, n_v * cell_w + 0.2)
    ax.set_ylim(n_c * cell_h - 0.3, -1.0)
    ax.set_axis_off()

    for ri, row in out_sorted.iterrows():
        ax.text(-0.12, ri * cell_h, row['Country_Name'],
                ha='right', va='center', fontsize=7.5, fontweight='bold')
        for ci, vcol in enumerate(variant_cols):
            q = int(row[vcol])
            face = _QUADRANT_COLORS[q]
            ax.add_patch(plt.Rectangle(
                (ci * cell_w, ri * cell_h - 0.22), cell_w, cell_h,
                facecolor=face, edgecolor='white', linewidth=0.6,
            ))
            r_, g_, b_, _ = mcolors.to_rgba(face)
            lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
            ax.text(ci * cell_w + 0.5, ri * cell_h + 0.005, str(q),
                    ha='center', va='center', fontsize=7,
                    color='white' if lum < 0.45 else 'black')

    # Column headers (rotated)
    for ci, vcol in enumerate(variant_cols):
        ax.text(ci * cell_w + 0.5, -0.55, vcol.replace('_', '\n'),
                ha='center', va='bottom', fontsize=7, fontweight='bold', rotation=0)

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_QUADRANT_COLORS[q],
                       edgecolor='grey', linewidth=0.5)
        for q in range(4)
    ]
    fig.legend(legend_patches, [_QUADRANT_NAMES[q] for q in range(4)],
               loc='lower center', ncol=2, fontsize=8,
               bbox_to_anchor=(0.55, -0.03), frameon=True)

    fig.suptitle(
        'Method 6: Cluster assignment stability across EWBI × Income threshold variants\n'
        '(quadrant 0–3, sorted alphabetically)',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    stab_path = os.path.join(OUTPUT_DIR, 'method6_stability_heatmap.png')
    fig.savefig(stab_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {stab_path}')

    # Also flag "stable" countries (same quadrant in all 7 variants)
    out_sorted['stable'] = out_sorted[variant_cols].nunique(axis=1) == 1
    n_stable = int(out_sorted['stable'].sum())
    print(f'  Countries with identical quadrant across all variants: {n_stable}/{n_c}')
    stable_names = out_sorted.loc[out_sorted['stable'], 'Country_Name'].tolist()
    print(f'  Stable: {", ".join(stable_names)}')

    # ── CSV exports ────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, 'method6_ewbi_income_quadrant_all_variants.csv')
    out_sorted.to_csv(csv_path, index=False)
    print(f'  Saved: {csv_path}')

    summary_path = os.path.join(OUTPUT_DIR, 'method6_ewbi_income_quadrant_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f'  Saved: {summary_path}')

    # ── Combined grid: all 7 variants in one figure ────────────────────────
    n_variants = len(variant_cols)
    n_cols_grid = 4
    n_rows_grid = int(np.ceil(n_variants / n_cols_grid))

    # Shared axis limits
    x_all = out_sorted['Median_Income'].values
    y_all = out_sorted['EWBI'].values
    x_pad = (x_all.max() - x_all.min()) * 0.07
    y_pad = (y_all.max() - y_all.min()) * 0.07
    xlim = (x_all.min() - x_pad, x_all.max() + x_pad)
    ylim = (y_all.min() - y_pad, y_all.max() + y_pad)

    fig, axes = plt.subplots(
        n_rows_grid, n_cols_grid,
        figsize=(5.5 * n_cols_grid, 4.5 * n_rows_grid),
        sharex=True, sharey=True,
    )
    axes_flat = np.array(axes).reshape(-1)

    # Rebuild cuts dict from summary_rows for easy lookup
    cuts_by_tag = {r['variant']: (r['ewbi_cut'], r['income_cut']) for r in summary_rows}

    for vi, vcol in enumerate(variant_cols):
        ax = axes_flat[vi]
        ewbi_c, income_c = cuts_by_tag[vcol]
        entry = next(e for e in _EWBI_INCOME_THRESHOLD_VARIANTS if e[0] == vcol)
        desc = entry[3]

        for q in range(4):
            sub = out_sorted[out_sorted[vcol] == q]
            ax.scatter(
                sub['Median_Income'], sub['EWBI'],
                color=_QUADRANT_COLORS[q], s=55, alpha=0.88,
                edgecolors='white', linewidths=0.4,
                label=_QUADRANT_NAMES[q], zorder=3,
            )
            for _, row in sub.iterrows():
                ax.text(row['Median_Income'] + 150, row['EWBI'], row['Country'],
                        fontsize=6.5, alpha=0.80, zorder=4)

        ax.axvline(income_c, color='#333333', linewidth=1.0, linestyle='--', alpha=0.6)
        ax.axhline(ewbi_c,   color='#333333', linewidth=1.0, linestyle=':',  alpha=0.6)

        # Counts per quadrant in title
        counts_str = '  '.join(
            f'Q{q}={int((out_sorted[vcol] == q).sum())}' for q in range(4)
        )
        ax.set_title(f'{vcol}\n{desc}\n{counts_str}', fontsize=7.5, fontweight='bold')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.20)
        ax.tick_params(labelsize=7)

    # Axis labels on border panels only
    for ax in axes_flat[n_cols_grid * (n_rows_grid - 1):]:
        ax.set_xlabel('Median Income (€)', fontsize=9)
    for ri in range(n_rows_grid):
        axes_flat[ri * n_cols_grid].set_ylabel('EWBI', fontsize=9)

    # Hide unused panels
    for vi in range(n_variants, len(axes_flat)):
        axes_flat[vi].set_axis_off()

    # Shared legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_QUADRANT_COLORS[q],
                       edgecolor='grey', linewidth=0.5, label=_QUADRANT_NAMES[q])
        for q in range(4)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle(
        'Method 6: EWBI × Income quadrant clustering — all threshold variants',
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, 'method6_all_variants_grid.png')
    fig.savefig(grid_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {grid_path}')


def main():
    features = load_country_features()

    if features.empty:
        print('No complete features available. Exiting.')
        return

    features_path = os.path.join(OUTPUT_DIR, 'country_features_for_clustering.csv')
    features.to_csv(features_path, index=False)
    print(f"Saved features: {features_path}")

    run_density_method(features)
    run_four_variable_method(features)
    run_manual_threshold_methods(features)
    run_anchor_profile_method(features)
    run_performance_method()
    run_ewbi_income_quadrant_method(features)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
