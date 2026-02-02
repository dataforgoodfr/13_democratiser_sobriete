#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multivariate Analysis Script with PCA and Visualizations

This script performs multivariate analysis on primary indicators:
1. Loads imputed data from Stage 1 (1_missing_data.py)
2. Performs Global PCA across all indicators
3. Performs PCA for each EU Priority separately
4. Normalizes indicators per indicator before PCA
5. Generates comprehensive visualizations:
   - Scree plots (variance explained)
   - Loading plots (biplot)
   - Correlation matrices
   - PC score distributions
   - Summary statistics tables

Author: Data Processing Pipeline
Date: 2024
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ===============================
# CONFIGURATION
# ===============================
OUTPUT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output')))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MISSING_DATA_OUTPUT = OUTPUT_DIR / "1_missing_data_output"
MULTIVARIATE_OUTPUT = OUTPUT_DIR / "2_multivariate_analysis_output"
MULTIVARIATE_OUTPUT.mkdir(parents=True, exist_ok=True)

# EU Priority to Indicator Mapping
# Based on EWBI hierarchical structure
EU_PRIORITY_MAPPING = {
    'Energy and Housing': [
        'HE-SILC-2', 'HQ-SILC-1', 'HQ-SILC-2', 'HQ-SILC-3', 
        'HQ-SILC-4', 'HQ-SILC-5', 'HQ-SILC-6', 'HQ-SILC-7', 'HQ-SILC-8'
    ],
    'Equality': [
        'EL-SILC-2', 'ES-SILC-1', 'ES-SILC-2', 
        'EC-SILC-2', 'EC-SILC-3', 'EC-SILC-4'
    ],
    'Health and Animal Welfare': [
        'AH-SILC-2', 'AH-SILC-3', 'AH-SILC-4', 
        'AC-SILC-3', 'AC-SILC-4'
    ],
    'Intergenerational Fairness, Youth, Culture and Sport': [
        'IS-SILC-1', 'IS-SILC-2', 'IS-SILC-3', 
        'IS-SILC-4', 'IS-SILC-5'
    ],
    'Social Rights and Skills, Quality Jobs and Preparedness': [
        'RT-SILC-1', 'RT-SILC-2', 'RT-LFS-1', 'RT-LFS-2', 
        'RT-LFS-3', 'RT-LFS-4', 'RT-LFS-5', 'RT-LFS-6', 
        'RT-LFS-7', 'RT-LFS-8', 'RU-SILC-1', 'RU-LFS-1'
    ],
    'Sustainable Transport and Tourism': [
        'TS-SILC-1', 'IC-SILC-1', 'IC-SILC-2'
    ]
}

# ===============================
# HELPER FUNCTIONS
# ===============================

def load_imputed_data(missing_data_dir):
    """Load the forward-filled raw data from Stage 3."""
    print("📂 Loading forward-filled raw data from Stage 3...")
    
    input_path = missing_data_dir / 'raw_data_forward_filled.csv'
    
    if not input_path.exists():
        print(f"[ERROR] Forward-filled data not found at {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    print(f"[OK] Loaded {len(df):,} records from Stage 3")
    
    return df


def prepare_pca_data(df, indicators=None, filter_country=None, filter_eu_priority=None):
    """
    Prepare data for PCA by pivoting into indicator columns.
    
    Normalization: Each indicator is normalized (standardized) independently
    before PCA to ensure equal weight regardless of scale differences.
    
    Args:
        df: Input DataFrame
        indicators: List of indicators to include (None = all)
        filter_country: Filter to specific country (None = all)
        filter_eu_priority: Filter to specific EU priority (None = all)
    
    Returns:
        Tuple of (pivot_data, scaler, indicators_used, metadata)
    """
    # Filter data
    filtered = df[
        (df['Country'] != 'All Countries') &
        (df['Decile'] != 'All')
    ].copy()
    
    if filter_country:
        filtered = filtered[filtered['Country'] == filter_country]
    
    if filter_eu_priority:
        filtered = filtered[filtered['EU priority'] == filter_eu_priority]
    
    if indicators:
        filtered = filtered[filtered['Primary and raw data'].isin(indicators)]
    
    if len(filtered) == 0:
        return None, None, [], {}
    
    # Pivot to get indicators as columns
    # Rows = (Country, Year, Decile), Columns = indicators
    pivot_data = filtered.pivot_table(
        index=['Country', 'Year', 'Decile'],
        columns='Primary and raw data',
        values='Value',
        aggfunc='first'
    )
    
    # Remove columns with all NaN
    pivot_data = pivot_data.dropna(axis=1, how='all')
    
    # Remove rows with all NaN
    pivot_data = pivot_data.dropna(axis=0, how='all')
    
    if pivot_data.shape[0] < 2 or pivot_data.shape[1] < 2:
        return None, None, [], {}
    
    # For PCA: remove rows with ANY NaN (complete cases only)
    pivot_data_clean = pivot_data.dropna()
    
    if len(pivot_data_clean) < 2:
        return None, None, [], {}
    
    # Normalize each indicator independently (standardize per indicator)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(pivot_data_clean)
    
    metadata = {
        'n_observations': data_scaled.shape[0],
        'n_indicators': data_scaled.shape[1],
        'indicators': pivot_data_clean.columns.tolist(),
        'index': pivot_data_clean.index.tolist()
    }
    
    return data_scaled, scaler, pivot_data_clean.columns.tolist(), metadata


def perform_pca_analysis(data_scaled, n_components=None):
    """
    Perform PCA analysis on scaled data.
    
    Args:
        data_scaled: Normalized data
        n_components: Number of components to retain (None = all)
    
    Returns:
        Dictionary with PCA results
    """
    if data_scaled is None or len(data_scaled) < 2:
        return None
    
    # Fit full PCA first to determine optimal components
    pca_full = PCA()
    pca_full.fit(data_scaled)
    
    # Calculate cumulative explained variance
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Determine number of components to retain 70% variance or use all if less than 70%
    if n_components is None:
        threshold = 0.70
        n_components_optimal = np.argmax(cumsum_var >= threshold) + 1
        n_components_optimal = min(n_components_optimal, len(pca_full.explained_variance_ratio_))
    else:
        n_components_optimal = n_components
    
    # Refit with optimal number of components
    pca = PCA(n_components=n_components_optimal)
    scores = pca.fit_transform(data_scaled)
    
    results = {
        'pca': pca,
        'scores': scores,
        'n_components': n_components_optimal,
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': cumsum_var[:n_components_optimal],
        'total_variance_explained': np.sum(pca.explained_variance_ratio_),
        'loadings': pca.components_,
        'full_variance': pca_full.explained_variance_ratio_,
        'full_cumulative': cumsum_var
    }
    
    return results


def create_scree_plot(pca_results, indicators, title, output_path):
    """Create interactive scree plot using Plotly."""
    if pca_results is None:
        return
    
    n_components = len(pca_results['explained_variance'])
    cumulative = np.cumsum(pca_results['explained_variance'])
    
    fig = go.Figure()
    
    # Individual variance explained
    fig.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(n_components)],
        y=pca_results['explained_variance'] * 100,
        name='Individual Variance',
        marker_color='#1f77b4',
        yaxis='y1'
    ))
    
    # Cumulative variance explained
    fig.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(n_components)],
        y=cumulative * 100,
        name='Cumulative Variance',
        marker=dict(size=8, color='#ff7f0e'),
        yaxis='y2',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Principal Component',
        yaxis=dict(title='Individual Variance Explained (%)', side='left'),
        yaxis2=dict(title='Cumulative Variance (%)', overlaying='y', side='right', range=[0, 105]),
        hovermode='x unified',
        height=600,
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0.6, y=1)
    )
    
    fig.write_image(str(output_path.with_suffix('.png')), width=1200, height=600)
    print(f"[OK] Saved scree plot: {output_path.with_suffix('.png').name}")


def create_loadings_heatmap(pca_results, indicators, title, output_path):
    """Create interactive loadings heatmap using Plotly."""
    if pca_results is None or len(indicators) == 0:
        return
    
    loadings = pca_results['loadings'].T
    n_pc = loadings.shape[1]
    
    fig = go.Figure(data=go.Heatmap(
        z=loadings,
        x=[f'PC{i+1}' for i in range(n_pc)],
        y=indicators,
        colorscale='RdBu',
        zmid=0,
        text=np.round(loadings, 3),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Loading")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Principal Component',
        yaxis_title='Indicator',
        height=max(400, len(indicators) * 20),
        template='plotly_white'
    )
    
    fig.write_image(str(output_path.with_suffix('.png')), width=1200, height=max(400, len(indicators) * 20))
    print(f"[OK] Saved loadings heatmap: {output_path.with_suffix('.png').name}")


def create_biplot(pca_results, indicators, title, output_path):
    """Create interactive biplot (PC1 vs PC2) with loadings."""
    if pca_results is None or pca_results['n_components'] < 2:
        return
    
    scores = pca_results['scores']
    loadings = pca_results['loadings']
    
    # Scatter plot of scores
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=scores[:, 0],
        y=scores[:, 1],
        mode='markers',
        marker=dict(size=6, color='#1f77b4', opacity=0.6),
        name='Observations',
        text=[f"PC1: {x:.2f}<br>PC2: {y:.2f}" for x, y in zip(scores[:, 0], scores[:, 1])],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Add loading vectors
    scale_factor = max(scores.std(axis=0)[:2]) * 1.5
    
    for i, indicator in enumerate(indicators[:min(20, len(indicators))]):  # Limit labels for clarity
        fig.add_trace(go.Scatter(
            x=[0, loadings[0, i] * scale_factor],
            y=[0, loadings[1, i] * scale_factor],
            mode='lines+text',
            line=dict(color='red', width=2),
            name=indicator,
            text=['', indicator],
            textposition='top center',
            hoverinfo='skip',
            showlegend=False
        ))
    
    fig.update_xaxes(title_text=f"PC1 ({pca_results['explained_variance'][0]*100:.1f}%)")
    fig.update_yaxes(title_text=f"PC2 ({pca_results['explained_variance'][1]*100:.1f}%)")
    
    fig.update_layout(
        title=title,
        hovermode='closest',
        height=600,
        template='plotly_white'
    )
    
    fig.write_image(str(output_path.with_suffix('.png')), width=1200, height=600)
    print(f"[OK] Saved biplot: {output_path.with_suffix('.png').name}")


def create_correlation_heatmap(df_pivot, title, output_path):
    """Create correlation matrix heatmap."""
    if df_pivot is None or df_pivot.shape[1] < 2:
        return
    
    corr_matrix = df_pivot.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=title,
        height=max(600, len(corr_matrix) * 25),
        xaxis_title='Indicator',
        yaxis_title='Indicator',
        template='plotly_white'
    )
    
    fig.write_image(str(output_path.with_suffix('.png')), width=1200, height=max(600, len(corr_matrix) * 25))
    print(f"[OK] Saved correlation heatmap: {output_path.with_suffix('.png').name}")


def create_pc_distributions(pca_results, title, output_path):
    """Create PC score distributions."""
    if pca_results is None:
        return
    
    scores = pca_results['scores']
    n_pc = min(4, scores.shape[1])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'PC{i+1} ({pca_results["explained_variance"][i]*100:.1f}%)' 
                        for i in range(n_pc)]
    )
    
    for i in range(n_pc):
        row, col = i // 2 + 1, i % 2 + 1
        fig.add_trace(
            go.Histogram(x=scores[:, i], name=f'PC{i+1}', nbinsx=30),
            row=row, col=col
        )
    
    fig.update_xaxes(title_text="Score")
    fig.update_yaxes(title_text="Frequency")
    
    fig.update_layout(
        title_text=title,
        height=600,
        template='plotly_white',
        showlegend=False
    )
    
    fig.write_image(str(output_path.with_suffix('.png')), width=1200, height=600)
    print(f"[OK] Saved PC distributions: {output_path.with_suffix('.png').name}")


def generate_summary_table(pca_results, indicators, output_path):
    """Generate summary statistics table."""
    if pca_results is None:
        return None
    
    summary_data = {
        'Component': [f'PC{i+1}' for i in range(len(pca_results['explained_variance']))],
        'Variance Explained (%)': [f"{v*100:.2f}" for v in pca_results['explained_variance']],
        'Cumulative (%)': [f"{c*100:.2f}" for c in pca_results['cumulative_variance']],
        'Eigenvalue': [f"{e:.4f}" for e in pca_results['pca'].explained_variance_]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    summary_df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    # Also create HTML table
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>PCA Summary Statistics</h1>
        {summary_df.to_html(index=False)}
    </body>
    </html>
    """
    
    with open(output_path.with_suffix('.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[OK] Saved summary table: {output_path.name}")
    
    return summary_df


# ===============================
# MAIN ANALYSIS FUNCTIONS
# ===============================

def perform_global_pca(df):
    """Perform PCA across all indicators."""
    print("\n" + "="*70)
    print("GLOBAL PCA: All Indicators")
    print("="*70)
    
    # Prepare data
    data_scaled, scaler, indicators, metadata = prepare_pca_data(df)
    
    if data_scaled is None:
        print("[ERROR] Could not prepare data for global PCA")
        return None
    
    print(f"\n[INFO] Data prepared: {metadata['n_observations']} observations x {metadata['n_indicators']} indicators")
    print(f"       Indicators: {', '.join(indicators)}")
    
    # Perform PCA
    pca_results = perform_pca_analysis(data_scaled)
    
    if pca_results is None:
        print("[ERROR] PCA analysis failed")
        return None
    
    print(f"\n[OK] PCA Analysis Complete:")
    print(f"     Components retained: {pca_results['n_components']}")
    print(f"     Total variance explained: {pca_results['total_variance_explained']*100:.2f}%")
    
    # Create visualizations
    print("\n[GENERATING VISUALIZATIONS]")
    
    subdir = MULTIVARIATE_OUTPUT / "Global_PCA"
    subdir.mkdir(exist_ok=True)
    
    # Scree plot
    create_scree_plot(
        pca_results, indicators,
        "Global PCA: Scree Plot (All Indicators)",
        subdir / "01_scree_plot"
    )
    
    # Loadings heatmap
    create_loadings_heatmap(
        pca_results, indicators,
        "Global PCA: Loading Matrix",
        subdir / "02_loadings_heatmap"
    )
    
    # Biplot
    create_biplot(
        pca_results, indicators,
        "Global PCA: Biplot (PC1 vs PC2)",
        subdir / "03_biplot"
    )
    
    # PC distributions
    create_pc_distributions(
        pca_results, 
        "Global PCA: Principal Component Distributions",
        subdir / "04_pc_distributions"
    )
    
    # Summary table
    generate_summary_table(
        pca_results, indicators,
        subdir / "05_summary_statistics"
    )
    
    # Save full results to JSON
    results_json = {
        'metadata': {
            'n_observations': int(metadata['n_observations']),
            'n_indicators': int(metadata['n_indicators']),
            'indicators': metadata['indicators']
        },
        'n_components': int(pca_results['n_components']),
        'variance_explained': [float(v) for v in pca_results['explained_variance']],
        'cumulative_variance': [float(c) for c in pca_results['cumulative_variance']],
        'total_variance': float(pca_results['total_variance_explained']),
        'loadings_shape': [int(s) for s in pca_results['loadings'].shape],
        'indicators': indicators
    }
    
    with open(subdir / "pca_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Global PCA complete. Output: {subdir}")
    
    # Add indicators and metadata to pca_results for use by main()
    pca_results['indicators'] = indicators
    pca_results['metadata'] = metadata
    
    return pca_results


def perform_priority_pca(df):
    """Perform separate PCA for each EU Priority."""
    print("\n" + "="*70)
    print("PRIORITY-SPECIFIC PCA: Separate Analysis per EU Priority")
    print("="*70)
    
    priority_results = {}
    
    for priority, indicators in EU_PRIORITY_MAPPING.items():
        print(f"\n{'='*70}")
        print(f"Processing: {priority}")
        print(f"{'='*70}")
        
        # Prepare data for this priority
        data_scaled, scaler, indicators_available, metadata = prepare_pca_data(
            df, indicators=indicators
        )
        
        if data_scaled is None:
            print(f"[WARN] Insufficient data for {priority}, skipping...")
            continue
        
        print(f"\n[INFO] Data prepared: {metadata['n_observations']} observations x {metadata['n_indicators']} indicators")
        print(f"       Using indicators: {', '.join(indicators_available)}")
        
        # Perform PCA
        pca_results = perform_pca_analysis(data_scaled)
        
        if pca_results is None:
            print(f"[WARN] PCA failed for {priority}, skipping...")
            continue
        
        print(f"\n[OK] PCA Analysis Complete:")
        print(f"     Components retained: {pca_results['n_components']}")
        print(f"     Total variance explained: {pca_results['total_variance_explained']*100:.2f}%")
        
        # Create visualizations
        print("\n[GENERATING VISUALIZATIONS]")
        
        safe_priority_name = priority.replace(' ', '_').replace(',', '').lower()
        subdir = MULTIVARIATE_OUTPUT / f"Priority_PCA_{safe_priority_name}"
        subdir.mkdir(exist_ok=True)
        
        # Scree plot
        create_scree_plot(
            pca_results, indicators_available,
            f"PCA: {priority} - Scree Plot",
            subdir / "01_scree_plot"
        )
        
        # Loadings heatmap
        create_loadings_heatmap(
            pca_results, indicators_available,
            f"PCA: {priority} - Loading Matrix",
            subdir / "02_loadings_heatmap"
        )
        
        # Biplot
        if pca_results['n_components'] >= 2:
            create_biplot(
                pca_results, indicators_available,
                f"PCA: {priority} - Biplot (PC1 vs PC2)",
                subdir / "03_biplot"
            )
        
        # PC distributions
        create_pc_distributions(
            pca_results,
            f"PCA: {priority} - Principal Component Distributions",
            subdir / "04_pc_distributions"
        )
        
        # Summary table
        generate_summary_table(
            pca_results, indicators_available,
            subdir / "05_summary_statistics"
        )
        
        # Save results
        results_json = {
            'priority': priority,
            'metadata': {
                'n_observations': int(metadata['n_observations']),
                'n_indicators': int(metadata['n_indicators']),
                'indicators': metadata['indicators']
            },
            'n_components': int(pca_results['n_components']),
            'variance_explained': [float(v) for v in pca_results['explained_variance']],
            'cumulative_variance': [float(c) for c in pca_results['cumulative_variance']],
            'total_variance': float(pca_results['total_variance_explained']),
            'indicators': indicators_available
        }
        
        with open(subdir / "pca_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2)
        
        priority_results[priority] = {
            'results': pca_results,
            'indicators': indicators_available,
            'output_dir': str(subdir)
        }
        
        print(f"\n✅ {priority} analysis complete. Output: {subdir}")
    
    return priority_results


def create_comparison_summary(global_results, priority_results):
    """Create a summary comparing global vs priority-specific PCA."""
    print("\n" + "="*70)
    print("CREATING COMPARISON SUMMARY")
    print("="*70)
    
    summary_html = """
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            h1, h2 {{ color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .stat {{ font-weight: bold; color: #1f77b4; }}
            .section {{ margin: 30px 0; }}
            hr {{ border: none; border-top: 2px solid #4CAF50; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Multivariate Analysis Summary - PCA Results</h1>
            <p><em>Analysis of underlying indicator structure using Principal Component Analysis</em></p>
            
            <div class="section">
                <h2>Global PCA Results</h2>
                <p>PCA performed on all <span class="stat">{n_global_indicators}</span> indicators across all observations.</p>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Number of Observations</td>
                        <td><span class="stat">{global_n_obs}</span></td>
                    </tr>
                    <tr>
                        <td>Number of Indicators</td>
                        <td><span class="stat">{n_global_indicators}</span></td>
                    </tr>
                    <tr>
                        <td>Principal Components Retained</td>
                        <td><span class="stat">{global_n_pc}</span></td>
                    </tr>
                    <tr>
                        <td>Total Variance Explained</td>
                        <td><span class="stat">{global_variance:.2f}%</span></td>
                    </tr>
                    <tr>
                        <td>PC1 Variance</td>
                        <td><span class="stat">{global_pc1:.2f}%</span></td>
                    </tr>
                    <tr>
                        <td>PC1 + PC2 Variance</td>
                        <td><span class="stat">{global_pc12:.2f}%</span></td>
                    </tr>
                </table>
            </div>
            
            <hr>
            
            <div class="section">
                <h2>Priority-Specific PCA Results</h2>
                <p>Separate PCA analysis for each EU Priority.</p>
                <table>
                    <tr>
                        <th>EU Priority</th>
                        <th>Indicators</th>
                        <th>Components</th>
                        <th>Variance Explained</th>
                        <th>PC1 Variance</th>
                    </tr>
                    {priority_rows}
                </table>
            </div>
            
            <hr>
            
            <div class="section">
                <h2>📊 Key Insights</h2>
                <ul>
                    <li><strong>Global Structure:</strong> The global PCA explains the underlying correlation structure among all {n_global_indicators} indicators.</li>
                    <li><strong>Dimensionality:</strong> Retaining {global_n_pc} components explains {global_variance:.1f}% of variance, suggesting moderate dimensionality reduction opportunity.</li>
                    <li><strong>First Two Components:</strong> PC1 and PC2 together explain {global_pc12:.1f}% of variance, useful for 2D visualization.</li>
                    <li><strong>Priority-Level Patterns:</strong> Each EU Priority exhibits different variance patterns, indicating distinct underlying structures.</li>
                </ul>
            </div>
            
            <hr>
            
            <div class="section">
                <h2>📁 Output Files</h2>
                <ul>
                    <li><strong>Global_PCA/</strong> - Global PCA analysis with scree plots, loadings, and biplots</li>
                    <li><strong>Priority_PCA_*/</strong> - Separate analyses for each EU Priority</li>
                    <li><strong>Each folder contains:</strong>
                        <ul>
                            <li>01_scree_plot.html - Variance explained visualization</li>
                            <li>02_loadings_heatmap.html - Component loadings</li>
                            <li>03_biplot.html - PC1 vs PC2 visualization</li>
                            <li>04_pc_distributions.html - Score distributions</li>
                            <li>05_summary_statistics.html - Statistical summary table</li>
                            <li>pca_results.json - Raw results data</li>
                        </ul>
                    </li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Generate priority rows
    priority_rows = []
    for priority, data in priority_results.items():
        indicators = data['indicators']
        results = data['results']
        priority_rows.append(
            f"<tr>"
            f"<td>{priority}</td>"
            f"<td>{len(indicators)} indicators</td>"
            f"<td>{results['n_components']}</td>"
            f"<td>{results['total_variance_explained']*100:.2f}%</td>"
            f"<td>{results['explained_variance'][0]*100:.2f}%</td>"
            f"</tr>"
        )
    
    # Fill template
    global_n_indicators = len(global_results['indicators'])
    global_n_obs = global_results['metadata']['n_observations']
    global_n_pc = global_results['n_components']
    global_variance = global_results['total_variance_explained'] * 100
    global_pc1 = global_results['explained_variance'][0] * 100
    global_pc12 = (global_results['explained_variance'][0] + global_results['explained_variance'][1]) * 100 if len(global_results['explained_variance']) > 1 else global_pc1
    
    summary_html = summary_html.format(
        n_global_indicators=global_n_indicators,
        global_n_obs=global_n_obs,
        global_n_pc=global_n_pc,
        global_variance=global_variance,
        global_pc1=global_pc1,
        global_pc12=global_pc12,
        priority_rows='\n'.join(priority_rows)
    )
    
    with open(MULTIVARIATE_OUTPUT / "00_summary_report.html", 'w', encoding='utf-8') as f:
        f.write(summary_html)
    
    print(f"[OK] Summary report saved: 00_summary_report.html")
    
    return summary_html


# ===============================
# MAIN PROCESSING
# ===============================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("STAGE 2: MULTIVARIATE ANALYSIS")
    print("="*70)
    print("\nAnalyzing underlying structure of well-being indicators using PCA\n")
    
    # Load data
    df = load_imputed_data(MISSING_DATA_OUTPUT)
    if df is None:
        return
    
    # Perform global PCA
    global_results = perform_global_pca(df)
    if global_results is None:
        print("[ERROR] Global PCA failed")
        return
    
    # Perform priority-specific PCA
    priority_results = perform_priority_pca(df)
    
    # Create comparison summary
    create_comparison_summary(global_results, priority_results)
    
    print("\n" + "="*70)
    print("✅ STAGE 2 COMPLETE: Multivariate Analysis")
    print("="*70)
    print(f"📁 Output directory: {MULTIVARIATE_OUTPUT}")
    print(f"\nGenerated analyses:")
    print(f"  • Global PCA on all {len(global_results['indicators'])} indicators")
    print(f"  • Priority-specific PCA for {len(priority_results)} EU Priorities")
    print(f"  • Comprehensive visualizations (scree plots, loadings, biplots)")
    print(f"  • Summary report and statistical tables")
    print(f"\nNext step: Run normalization (3_normalisation_data.py)")


if __name__ == '__main__':
    main()

