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
from scipy.stats import chi2
import json
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from pipeline_env import get_output_dir

warnings.filterwarnings('ignore')

# ===============================
# CONFIGURATION
# ===============================
OUTPUT_DIR = get_output_dir()
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
        'ES-SILC-1', 'ES-SILC-2', 
        'EC-SILC-2', 'EC-SILC-3', 'EC-SILC-4'
    ],
    'Health and Animal Welfare': [
        'AH-SILC-2', 'AH-SILC-3', 'AH-SILC-4', 
        'AC-SILC-3', 'AC-SILC-4'
    ],
    'Intergenerational Fairness, Youth, Culture and Sport': [
        'IS-SILC-3', 
        'IS-SILC-4', 'IS-SILC-5'
    ],
    'Social Rights and Skills, Quality Jobs and Preparedness': [
        'RT-SILC-1', 'RT-SILC-2', 'RT-LFS-1', 'RT-LFS-2', 
        'RT-LFS-3', 'RT-LFS-4', 'RT-LFS-5', 'RT-LFS-6', 
        'RT-LFS-7', 'RT-LFS-8', 'RU-SILC-1'
    ]
}

# ===============================
# HELPER FUNCTIONS
# ===============================

def load_imputed_data(missing_data_dir):
    """Load the break-adjusted raw data from Stage 1."""
    print("📂 Loading break-adjusted raw data from Stage 1...")
    
    input_path = missing_data_dir / 'raw_data_break_adjusted.csv'
    
    if not input_path.exists():
        print(f"[ERROR] Break-adjusted data not found at {input_path}")
        print("Please run 1_missing_data.py first")
        return None
    
    df = pd.read_csv(input_path)
    print(f"[OK] Loaded {len(df):,} records from Stage 1")
    
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


def varimax_rotation(loadings, gamma=1.0, max_iter=1000, tol=1e-10):
    """
    Perform Varimax rotation on factor loadings.
    
    Args:
        loadings: Loading matrix (n_variables x n_factors)
        gamma: Rotation parameter (1.0 = varimax, 0.0 = quartimax)
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        Tuple of (rotated_loadings, rotation_matrix)
    """
    p, k = loadings.shape
    R = np.eye(k)
    d = 0
    
    for i in range(max_iter):
        d_old = d
        Lambda = np.dot(loadings, R)
        u, s, vh = np.linalg.svd(np.dot(loadings.T, np.asarray(Lambda)**3 - gamma/p * Lambda * np.dot(np.ones((p, 1)), np.dot(np.ones((1, p)), Lambda**2))))
        R = np.dot(u, vh)
        d = np.sum(s)
        
        if d_old != 0 and d/d_old < 1 + tol:
            break
    
    rotated_loadings = np.dot(loadings, R)
    return rotated_loadings, R


def apply_jrc_factor_selection(eigenvalues, explained_variance_ratio):
    """
    Apply JRC factor selection criteria:
    1. Eigenvalues > 1.0
    2. Individual variance > 10%
    3. Cumulative variance >= 75% (keep selecting until at least 75% is explained)
    
    Args:
        eigenvalues: Array of eigenvalues
        explained_variance_ratio: Array of explained variance ratios
    
    Returns:
        List of selected factor indices
    """
    selected_factors = []
    cumulative_variance = 0
    
    for i, (eigenvalue, variance_ratio) in enumerate(zip(eigenvalues, explained_variance_ratio)):
        individual_variance_pct = variance_ratio * 100
        
        # JRC criteria: eigenvalue > 1 and individual variance > 10%
        if eigenvalue > 1.0 and individual_variance_pct > 10.0:
            selected_factors.append(i)
            cumulative_variance += individual_variance_pct
            
            # Stop when we reach at least 75% cumulative variance
            if cumulative_variance >= 75.0:
                break
    
    # Ensure at least one factor is selected if criteria are too strict
    if not selected_factors and len(eigenvalues) > 0:
        # Fall back to first factor that meets eigenvalue > 1 criterion 
        for i, eigenvalue in enumerate(eigenvalues):
            if eigenvalue > 1.0:
                selected_factors.append(i)
                break
        
        # Final fallback to first component
        if not selected_factors:
            selected_factors = [0]
    
    return selected_factors


def perform_pca_analysis(data_scaled, n_components=None, apply_jrc_criteria=True):
    """
    Perform JRC-compliant PCA analysis on scaled data.
    
    Args:
        data_scaled: Normalized data
        n_components: Number of components to retain (None = auto-select using JRC criteria)
        apply_jrc_criteria: Whether to apply JRC factor selection criteria
    
    Returns:
        Dictionary with PCA results including JRC-compliant factors and varimax rotation
    """
    if data_scaled is None or len(data_scaled) < 2:
        return None
    
    # Fit full PCA first to get all eigenvalues and loadings
    pca_full = PCA()
    pca_full.fit(data_scaled)
    
    # Calculate cumulative explained variance
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Apply JRC factor selection criteria if requested
    if apply_jrc_criteria:
        selected_factors = apply_jrc_factor_selection(
            pca_full.explained_variance_, 
            pca_full.explained_variance_ratio_
        )
        n_components_optimal = len(selected_factors)
        
        print(f"       JRC factor selection: {len(selected_factors)} factors selected")
        for i, factor_idx in enumerate(selected_factors):
            eigenval = pca_full.explained_variance_[factor_idx]
            var_pct = pca_full.explained_variance_ratio_[factor_idx] * 100
            print(f"         Factor {factor_idx + 1}: eigenvalue={eigenval:.3f}, variance={var_pct:.2f}%")
    else:
        # Original logic: retain components explaining 70% variance
        if n_components is None:
            threshold = 0.70
            n_components_optimal = np.argmax(cumsum_var >= threshold) + 1
            n_components_optimal = min(n_components_optimal, len(pca_full.explained_variance_ratio_))
        else:
            n_components_optimal = n_components
        
        selected_factors = list(range(n_components_optimal))
    
    # Refit with selected number of components
    pca = PCA(n_components=n_components_optimal)
    scores = pca.fit_transform(data_scaled)
    
    # Get loadings (components transposed)
    loadings = pca.components_.T  # Shape: (n_variables, n_components)
    
    # Apply Varimax rotation if we have multiple components
    if n_components_optimal > 1:
        try:
            rotated_loadings, rotation_matrix = varimax_rotation(loadings)
            print(f"       Applied Varimax rotation to {n_components_optimal} factors")
        except Exception as e:
            print(f"       Warning: Varimax rotation failed ({e}), using unrotated loadings")
            rotated_loadings = loadings
            rotation_matrix = np.eye(n_components_optimal)
    else:
        rotated_loadings = loadings
        rotation_matrix = np.eye(n_components_optimal)
    
    # Compute rotated scores
    try:
        rotated_scores = np.dot(scores, rotation_matrix)
    except:
        rotated_scores = scores
    
    results = {
        'pca': pca,
        'scores': scores,
        'rotated_scores': rotated_scores,
        'n_components': n_components_optimal,
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': cumsum_var[:n_components_optimal],
        'total_variance_explained': np.sum(pca.explained_variance_ratio_),
        'loadings': pca.components_,  # Original loadings (components x variables)
        'rotated_loadings': rotated_loadings.T,  # Rotated loadings (components x variables)
        'rotation_matrix': rotation_matrix,
        'eigenvalues': pca.explained_variance_,
        'selected_factors': selected_factors,
        'jrc_criteria_applied': apply_jrc_criteria,
        'full_variance': pca_full.explained_variance_ratio_,
        'full_cumulative': cumsum_var,
        'full_eigenvalues': pca_full.explained_variance_
    }
    
    return results


def create_scree_plot(pca_results, indicators, title, output_path):
    """Create interactive scree plot using Plotly."""
    if pca_results is None:
        return

    if os.getenv("EWBI_SKIP_PLOTS", "").strip().lower() in {"1", "true", "yes", "y"}:
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
    
    try:
        fig.write_image(str(output_path.with_suffix('.png')), width=1200, height=600)
        print(f"[OK] Saved scree plot: {output_path.with_suffix('.png').name}")
    except Exception as e:
        print(f"[WARN] Could not export scree plot image ({e}). Continuing without images.")


def create_loadings_heatmap(pca_results, indicators, title, output_path):
    """Create interactive loadings heatmap using Plotly."""
    if pca_results is None or len(indicators) == 0:
        return

    if os.getenv("EWBI_SKIP_PLOTS", "").strip().lower() in {"1", "true", "yes", "y"}:
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
    
    try:
        fig.write_image(str(output_path.with_suffix('.png')), width=1200, height=max(400, len(indicators) * 20))
        print(f"[OK] Saved loadings heatmap: {output_path.with_suffix('.png').name}")
    except Exception as e:
        print(f"[WARN] Could not export loadings heatmap image ({e}). Continuing without images.")


def create_biplot(pca_results, indicators, title, output_path):
    """Create interactive biplot (PC1 vs PC2) with loadings."""
    if pca_results is None or pca_results['n_components'] < 2:
        return

    if os.getenv("EWBI_SKIP_PLOTS", "").strip().lower() in {"1", "true", "yes", "y"}:
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
    
    try:
        fig.write_image(str(output_path.with_suffix('.png')), width=1200, height=600)
        print(f"[OK] Saved biplot: {output_path.with_suffix('.png').name}")
    except Exception as e:
        print(f"[WARN] Could not export biplot image ({e}). Continuing without images.")


def create_correlation_heatmap(df_pivot, title, output_path):
    """Create correlation matrix heatmap."""
    if df_pivot is None or df_pivot.shape[1] < 2:
        return

    if os.getenv("EWBI_SKIP_PLOTS", "").strip().lower() in {"1", "true", "yes", "y"}:
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
    
    try:
        fig.write_image(str(output_path.with_suffix('.png')), width=1200, height=max(600, len(corr_matrix) * 25))
        print(f"[OK] Saved correlation heatmap: {output_path.with_suffix('.png').name}")
    except Exception as e:
        print(f"[WARN] Could not export correlation heatmap image ({e}). Continuing without images.")


def create_pc_distributions(pca_results, title, output_path):
    """Create PC score distributions."""
    if pca_results is None:
        return

    if os.getenv("EWBI_SKIP_PLOTS", "").strip().lower() in {"1", "true", "yes", "y"}:
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
    
    try:
        fig.write_image(str(output_path.with_suffix('.png')), width=1200, height=600)
        print(f"[OK] Saved PC distributions: {output_path.with_suffix('.png').name}")
    except Exception as e:
        print(f"[WARN] Could not export PC distributions image ({e}). Continuing without images.")


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
    """Perform JRC-compliant PCA across all indicators."""
    print("\n" + "="*70)
    print("GLOBAL PCA: All Indicators (JRC-Compliant)")
    print("="*70)
    
    # Prepare data
    data_scaled, scaler, indicators, metadata = prepare_pca_data(df)
    
    if data_scaled is None:
        print("[ERROR] Could not prepare data for global PCA")
        return None
    
    print(f"\n[INFO] Data prepared: {metadata['n_observations']} observations x {metadata['n_indicators']} indicators")
    print(f"       Indicators: {', '.join(indicators)}")
    
    # Perform JRC-compliant PCA
    print("\n[INFO] Applying JRC factor selection criteria...")
    pca_results = perform_pca_analysis(data_scaled, apply_jrc_criteria=True)
    
    if pca_results is None:
        print("[ERROR] PCA analysis failed")
        return None
    
    print(f"\n[OK] JRC-Compliant PCA Analysis Complete:")
    print(f"     Components retained: {pca_results['n_components']}")
    print(f"     Total variance explained: {pca_results['total_variance_explained']*100:.2f}%")
    print(f"     JRC criteria: {pca_results.get('jrc_criteria_applied', False)}")
    print(f"     Varimax rotation: {'Applied' if pca_results['n_components'] > 1 else 'N/A (single component)'}")
    
    # Create visualizations
    print("\n[GENERATING VISUALIZATIONS]")
    
    subdir = MULTIVARIATE_OUTPUT / "Global_PCA"
    subdir.mkdir(parents=True, exist_ok=True)
    
    # Scree plot
    create_scree_plot(
        pca_results, indicators,
        "Global PCA: Scree Plot (JRC-Compliant) - All Indicators",
        subdir / "01_scree_plot"
    )
    
    # Loadings heatmap (use rotated loadings if available)
    loadings_for_viz = pca_results.get('rotated_loadings', pca_results['loadings'])
    pca_viz = pca_results.copy()
    pca_viz['loadings'] = loadings_for_viz
    
    create_loadings_heatmap(
        pca_viz, indicators,
        "Global PCA: Rotated Loading Matrix (Varimax)",
        subdir / "02_loadings_heatmap_rotated"
    )
    
    # Original loadings heatmap for comparison
    create_loadings_heatmap(
        pca_results, indicators,
        "Global PCA: Original Loading Matrix",
        subdir / "02_loadings_heatmap_original"
    )
    
    # Biplot
    create_biplot(
        pca_results, indicators,
        "Global PCA: Biplot (PC1 vs PC2) - JRC-Compliant",
        subdir / "03_biplot"
    )
    
    # PC distributions
    create_pc_distributions(
        pca_results, 
        "Global PCA: Principal Component Distributions (JRC-Compliant)",
        subdir / "04_pc_distributions"
    )
    
    # Summary table with JRC information
    generate_summary_table(
        pca_results, indicators,
        subdir / "05_summary_statistics"
    )
    
    # Save full results to JSON (enhanced for JRC)
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
        'eigenvalues': [float(e) for e in pca_results['eigenvalues']],
        'loadings_shape': [int(s) for s in pca_results['loadings'].shape],
        'rotated_loadings_shape': [int(s) for s in pca_results['rotated_loadings'].shape],
        'jrc_criteria_applied': pca_results.get('jrc_criteria_applied', True),
        'selected_factors': pca_results.get('selected_factors', []),
        'varimax_rotation_applied': pca_results['n_components'] > 1,
        'indicators': indicators
    }
    
    with open(subdir / "pca_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2)
    
    # Save PCA results for Stage 4 consumption
    stage4_results_path = MULTIVARIATE_OUTPUT / "pca_results_full.json"
    save_pca_results_for_stage4(pca_results, indicators, metadata, stage4_results_path)
    
    print(f"\n✅ Global JRC-Compliant PCA complete. Output: {subdir}")
    
    # Add indicators and metadata to pca_results for use by main()
    pca_results['indicators'] = indicators
    pca_results['metadata'] = metadata
    
    return pca_results


def perform_priority_pca(df):
    """Perform separate JRC-compliant PCA for each EU Priority."""
    print("\n" + "="*70)
    print("PRIORITY-SPECIFIC PCA: JRC-Compliant Analysis per EU Priority")
    print("="*70)
    
    priority_results = {}
    
    # Keep directory names short and Windows-safe.
    # Long base paths + verbose priority names can exceed Win32 MAX_PATH.
    priority_dir_alias = {
        "Energy and Housing": "energy_housing",
        "Equality": "equality",
        "Health and Animal Welfare": "health_animal_welfare",
        "Intergenerational Fairness, Youth, Culture and Sport": "intergenerational_youth_culture_sport",
        "Social Rights and Skills, Quality Jobs and Preparedness": "social_rights_skills_quality_jobs_preparedness",
    }

    def safe_dir_slug(text: str, max_len: int = 48) -> str:
        slug = priority_dir_alias.get(text)
        if not slug:
            slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        if len(slug) > max_len:
            slug = slug[:max_len].rstrip("_")
        return slug or "priority"

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
        
        # Perform JRC-compliant PCA
        print(f"\n[INFO] Applying JRC criteria for {priority}...")
        pca_results = perform_pca_analysis(data_scaled, apply_jrc_criteria=True)
        
        if pca_results is None:
            print(f"[WARN] PCA failed for {priority}, skipping...")
            continue
        
        print(f"\n[OK] JRC-Compliant PCA Analysis Complete:")
        print(f"     Components retained: {pca_results['n_components']}")
        print(f"     Total variance explained: {pca_results['total_variance_explained']*100:.2f}%")
        print(f"     JRC criteria: {pca_results.get('jrc_criteria_applied', False)}")
        print(f"     Varimax rotation: {'Applied' if pca_results['n_components'] > 1 else 'N/A (single component)'}")
        
        # Create visualizations
        print("\n[GENERATING VISUALIZATIONS]")
        
        safe_priority_name = safe_dir_slug(priority)
        subdir = MULTIVARIATE_OUTPUT / f"Priority_PCA_{safe_priority_name}"
        subdir.mkdir(parents=True, exist_ok=True)
        
        # Scree plot
        create_scree_plot(
            pca_results, indicators_available,
            f"PCA: {priority} - Scree Plot (JRC-Compliant)",
            subdir / "01_scree_plot"
        )
        
        # Rotated loadings heatmap
        if 'rotated_loadings' in pca_results:
            loadings_for_viz = pca_results.get('rotated_loadings', pca_results['loadings'])
            pca_viz = pca_results.copy()
            pca_viz['loadings'] = loadings_for_viz
            
            create_loadings_heatmap(
                pca_viz, indicators_available,
                f"PCA: {priority} - Rotated Loading Matrix (Varimax)",
                subdir / "02_loadings_heatmap_rotated"
            )
        
        # Original loadings heatmap
        create_loadings_heatmap(
            pca_results, indicators_available,
            f"PCA: {priority} - Original Loading Matrix",
            subdir / "02_loadings_heatmap_original"
        )
        
        # Biplot
        if pca_results['n_components'] >= 2:
            create_biplot(
                pca_results, indicators_available,
                f"PCA: {priority} - Biplot (PC1 vs PC2) - JRC-Compliant",
                subdir / "03_biplot"
            )
        
        # PC distributions
        create_pc_distributions(
            pca_results,
            f"PCA: {priority} - Principal Component Distributions (JRC-Compliant)",
            subdir / "04_pc_distributions"
        )
        
        # Summary table with JRC information
        generate_summary_table(
            pca_results, indicators_available,
            subdir / "05_summary_statistics"
        )
        
        # Save enhanced results
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
            'eigenvalues': [float(e) for e in pca_results['eigenvalues']],
            'jrc_criteria_applied': pca_results.get('jrc_criteria_applied', True),
            'selected_factors': pca_results.get('selected_factors', []),
            'varimax_rotation_applied': pca_results['n_components'] > 1,
            'indicators': indicators_available
        }
        
        with open(subdir / "pca_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2)
        
        priority_results[priority] = {
            'results': pca_results,
            'indicators': indicators_available,
            'output_dir': str(subdir)
        }
        
        print(f"\n✅ {priority} JRC-compliant analysis complete. Output: {subdir}")
    
    return priority_results


def create_comparison_summary(global_results, priority_results):
    """Create a summary comparing global vs priority-specific JRC-compliant PCA."""
    print("\n" + "="*70)
    print("CREATING JRC-COMPLIANT COMPARISON SUMMARY")
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
            .jrc-highlight {{ background-color: #e8f5e8; padding: 15px; border-left: 4px solid #4CAF50; margin: 15px 0; }}
            hr {{ border: none; border-top: 2px solid #4CAF50; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 JRC-Compliant Multivariate Analysis Summary - PCA Results</h1>
            <p><em>Analysis of underlying indicator structure using JRC-compliant Principal Component Analysis methodology</em></p>
            
            <div class="jrc-highlight">
                <h3>📋 JRC Methodology Compliance</h3>
                <ul>
                    <li><strong>Factor Selection:</strong> Applied JRC criteria (eigenvalue > 1, variance > 10%, cumulative ≥ 75%)</li>
                    <li><strong>Varimax Rotation:</strong> Applied to achieve simpler factor structure with cleaner indicator-factor associations</li>
                    <li><strong>Squared Loadings:</strong> Used for indicator weight computation within factors</li>
                    <li><strong>Component Weights:</strong> Based on proportion of explained variance per factor</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Global JRC-Compliant PCA Results</h2>
                <p>PCA performed on all <span class="stat">{n_global_indicators}</span> indicators across all observations using JRC methodology.</p>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Number of Observations</td>
                        <td class="stat">{global_n_obs:,}</td>
                    </tr>
                    <tr>
                        <td>Number of Indicators</td>
                        <td class="stat">{n_global_indicators}</td>
                    </tr>
                    <tr>
                        <td>Components Retained (JRC Criteria)</td>
                        <td class="stat">{global_n_pc}</td>
                    </tr>
                    <tr>
                        <td>Total Variance Explained</td>
                        <td class="stat">{global_variance:.2f}%</td>
                    </tr>
                    <tr>
                        <td>First Component Variance</td>
                        <td class="stat">{global_pc1:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Varimax Rotation Applied</td>
                        <td class="stat">{varimax_applied}</td>
                    </tr>
                </table>
            </div>
            
            <hr>
            
            <div class="section">
                <h2>Priority-Specific JRC-Compliant PCA Results</h2>
                <p>Separate JRC-compliant PCA analyses for each EU Priority to capture domain-specific structures.</p>
                <table>
                    <tr>
                        <th>EU Priority</th>
                        <th>Indicators</th>
                        <th>Components (JRC)</th>
                        <th>Variance Explained</th>
                        <th>First Component</th>
                        <th>Varimax Applied</th>
                    </tr>
                    {priority_rows}
                </table>
            </div>
            
            <hr>
            
            <div class="section">
                <h2>📊 Key JRC-Compliant Insights</h2>
                <ul>
                    <li><strong>Factor Selection:</strong> JRC criteria ensured retention of {global_n_pc} meaningful factors from {n_global_indicators} indicators, explaining {global_variance:.1f}% of total variance.</li>
                    <li><strong>First Component Dominance:</strong> PC1 explains {global_pc1:.1f}% of variance, indicating {dominance_level} underlying commonality.</li>
                    <li><strong>Rotated Structure:</strong> Varimax rotation {rotation_status} to achieve simpler indicator-factor associations.</li>
                    <li><strong>Priority-Level Patterns:</strong> Each EU Priority exhibits different variance patterns, indicating distinct underlying structures suitable for JRC weighting methodology.</li>
                    <li><strong>Stage 4 Integration:</strong> Results saved in pca_results_full.json format for JRC-compliant weighting in aggregation stage.</li>
                </ul>
            </div>
            
            <hr>
            
            <div class="section">
                <h2>📁 Output Files</h2>
                <ul>
                    <li><strong>Global_PCA/</strong> - JRC-compliant global PCA analysis with scree plots, rotated loadings, and biplots</li>
                    <li><strong>Priority_PCA_*/</strong> - Separate JRC-compliant analyses for each EU Priority</li>
                    <li><strong>pca_results_full.json</strong> - Detailed results for Stage 4 JRC-compliant weighting</li>
                    <li><strong>Each folder contains:</strong>
                        <ul>
                            <li>01_scree_plot.html - Variance explained visualization with JRC factor selection indicators</li>
                            <li>02_loadings_heatmap_rotated.html - Varimax-rotated component loadings</li>
                            <li>02_loadings_heatmap_original.html - Original (unrotated) component loadings</li>
                            <li>03_biplot.html - PC1 vs PC2 visualization</li>
                            <li>04_pc_distributions.html - Score distributions</li>
                            <li>05_summary_statistics.html - Statistical summary with JRC criteria compliance</li>
                            <li>pca_results.json - Raw results data with JRC methodology details</li>
                        </ul>
                    </li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Generate priority rows with JRC information
    priority_rows = []
    for priority, data in priority_results.items():
        indicators = data['indicators']
        results = data['results']
        varimax_applied = 'Yes' if results.get('n_components', 0) > 1 else 'N/A'
        
        priority_rows.append(
            f"<tr>"
            f"<td>{priority}</td>"
            f"<td>{len(indicators)} indicators</td>"
            f"<td>{results['n_components']}</td>"
            f"<td>{results['total_variance_explained']*100:.2f}%</td>"
            f"<td>{results['explained_variance'][0]*100:.2f}%</td>"
            f"<td>{varimax_applied}</td>"
            f"</tr>"
        )
    
    # Fill template with JRC-enhanced information
    global_n_indicators = len(global_results['indicators'])
    global_n_obs = global_results['metadata']['n_observations']
    global_n_pc = global_results['n_components']
    global_variance = global_results['total_variance_explained'] * 100
    global_pc1 = global_results['explained_variance'][0] * 100
    
    # Compute conditional values to avoid formatting issues
    varimax_applied = 'Yes' if global_results.get('n_components', 0) > 1 else 'N/A (Single Component)'
    dominance_level = 'strong' if global_pc1 > 30 else 'moderate'
    rotation_status = 'applied' if global_results.get('n_components', 0) > 1 else 'not needed'
    
    summary_html = summary_html.format(
        n_global_indicators=global_n_indicators,
        global_n_obs=global_n_obs,
        global_n_pc=global_n_pc,
        global_variance=global_variance,
        global_pc1=global_pc1,
        varimax_applied=varimax_applied,
        dominance_level=dominance_level,
        rotation_status=rotation_status,
        priority_rows='\n'.join(priority_rows)
    )
    
    with open(MULTIVARIATE_OUTPUT / "00_summary_report.html", 'w', encoding='utf-8') as f:
        f.write(summary_html)
    
    print(f"[OK] JRC-compliant summary report saved: 00_summary_report.html")
    print(f"[INFO] Added JRC methodology details and compliance indicators")
    
    return summary_html


def save_pca_results_for_stage4(pca_results, indicators, metadata, output_path):
    """
    Save PCA results in format expected by Stage 4 (JRC-compliant weighting).
    
    Creates detailed results including:
    - Rotated loadings for JRC methodology
    - Component weights based on explained variance
    - Eigenvalues for factor selection
    - Complete indicator mapping
    
    Args:
        pca_results: PCA results dictionary
        indicators: List of indicator names
        metadata: Analysis metadata
        output_path: Path to save results
    """
    try:
        # Extract country-year combinations from metadata index
        country_year_combos = {}
        
        if 'index' in metadata:
            for idx_tuple in metadata['index']:
                if len(idx_tuple) >= 2:  # (Country, Year, Decile)
                    country = idx_tuple[0]
                    year = int(idx_tuple[1])
                    key = (country, year)
                    
                    if key not in country_year_combos:
                        country_year_combos[key] = {
                            'country': country,
                            'year': year,
                            'eigenvalues': [float(ev) for ev in pca_results['eigenvalues']],
                            'explained_variance_ratio': [float(evr) for evr in pca_results['explained_variance']],
                            'component_weights': [],
                            'component_loadings': [],
                            'rotated_loadings': [],
                            'indicator_names': indicators,
                            'n_components': pca_results['n_components'],
                            'total_variance_explained': float(pca_results['total_variance_explained']),
                            'jrc_criteria_applied': pca_results.get('jrc_criteria_applied', True),
                            'selected_factors': pca_results.get('selected_factors', list(range(pca_results['n_components'])))
                        }
        
        # If no country-year combinations found, create generic entry
        if not country_year_combos:
            country_year_combos[('ALL', 2020)] = {
                'country': 'ALL',
                'year': 2020,
                'eigenvalues': [float(ev) for ev in pca_results['eigenvalues']],
                'explained_variance_ratio': [float(evr) for evr in pca_results['explained_variance']],
                'component_weights': [],
                'component_loadings': [],
                'rotated_loadings': [],
                'indicator_names': indicators,
                'n_components': pca_results['n_components'],
                'total_variance_explained': float(pca_results['total_variance_explained']),
                'jrc_criteria_applied': pca_results.get('jrc_criteria_applied', True),
                'selected_factors': pca_results.get('selected_factors', list(range(pca_results['n_components'])))
            }
        
        # Add component weights and loadings
        for key in country_year_combos:
            result_entry = country_year_combos[key]
            
            # Component weights based on explained variance (JRC methodology)
            if pca_results['n_components'] > 0:
                total_variance = sum(pca_results['explained_variance'])
                if total_variance > 0:
                    component_weights = [float(var_ratio / total_variance) for var_ratio in pca_results['explained_variance']]
                else:
                    component_weights = [1.0 / pca_results['n_components']] * pca_results['n_components']
            else:
                component_weights = []
            
            result_entry['component_weights'] = component_weights
            
            # Original loadings (transposed for compatibility)
            if 'loadings' in pca_results:
                result_entry['component_loadings'] = pca_results['loadings'].tolist()
            
            # Rotated loadings from Varimax (JRC methodology)
            if 'rotated_loadings' in pca_results:
                result_entry['rotated_loadings'] = pca_results['rotated_loadings'].tolist()
            else:
                result_entry['rotated_loadings'] = result_entry.get('component_loadings', [])
        
        # Convert keys to strings for JSON serialization
        country_year_combos_str = {}
        for key, value in country_year_combos.items():
            key_str = f"('{key[0]}', {key[1]})"
            country_year_combos_str[key_str] = value
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(country_year_combos_str, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved JRC-compliant PCA results for Stage 4: {output_path.name}")
        print(f"     Country-year combinations: {len(country_year_combos)}")
        print(f"     Components per combination: {pca_results['n_components']}")
        print(f"     Indicators: {len(indicators)}")
        
    except Exception as e:
        print(f"[WARN] Failed to save PCA results for Stage 4: {e}")
        print(f"       Stage 4 will use unweighted aggregation")


# ===============================
# MAIN PROCESSING
# ===============================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("STAGE 2: MULTIVARIATE ANALYSIS (JRC-COMPLIANT)")
    print("="*70)
    print("\nJRC-compliant PCA analysis with:")
    print("  • Factor selection criteria (eigenvalue > 1, variance > 10%, cumulative ≥ 75%)")
    print("  • Varimax rotation for simpler factor structure")
    print("  • Squared loadings methodology for indicator weighting")
    print("  • Component weights based on explained variance proportion\n")
    
    # Load data
    df = load_imputed_data(MISSING_DATA_OUTPUT)
    if df is None:
        return
    
    # Perform global JRC-compliant PCA
    global_results = perform_global_pca(df)
    if global_results is None:
        print("[ERROR] Global PCA failed")
        return
    
    # Perform priority-specific JRC-compliant PCA
    priority_results = perform_priority_pca(df)
    
    # Create comparison summary
    create_comparison_summary(global_results, priority_results)
    
    print("\n" + "="*70)
    print("✅ STAGE 2 COMPLETE: JRC-Compliant Multivariate Analysis")
    print("="*70)
    print(f"📁 Output directory: {MULTIVARIATE_OUTPUT}")
    print(f"\nGenerated JRC-compliant analyses:")
    print(f"  • Global PCA on all {len(global_results['indicators'])} indicators")
    print(f"  • Priority-specific PCA for {len(priority_results)} EU Priorities")
    print(f"  • Varimax rotation applied where multiple components exist")
    print(f"  • Factor selection using JRC criteria")
    print(f"  • PCA results saved for Stage 4 weighting (pca_results_full.json)")
    print(f"  • Comprehensive visualizations (scree plots, loadings, biplots)")
    print(f"  • Summary report and statistical tables")
    print(f"\nNext step: Run normalization (3_normalisation_data.py)")
    print(f"Then run: Stage 4 weighting with JRC-compliant PCA weights (4_weighting_aggregation.py)")


if __name__ == '__main__':
    main()

