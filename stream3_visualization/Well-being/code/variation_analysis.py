#!/usr/bin/env python3
"""
Variation Analysis Script for EWBI Data
This script analyzes variations across deciles and years to identify unusual patterns.
"""

import pandas as pd
import numpy as np
import json

def analyze_decile_variations(df):
    """Analyze variations across deciles for each indicator-country combination"""
    print("=== ANALYZING DECILE VARIATIONS ===\n")
    
    # Filter for primary indicators only and exclude 'All' deciles
    primary_df = df[(df['Level'] == '4 (Primary_indicator)') & (df['decile'] != 'All')].copy()
    
    # Calculate statistics for each indicator-country combination
    decile_stats = []
    
    for (indicator, country) in primary_df.groupby(['primary_index', 'country']):
        if len(country) == 10:  # Should have all 10 deciles
            scores = country['Score'].values
            stats = {
                'primary_index': indicator[0],
                'country': indicator[1],
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'range': np.max(scores) - np.min(scores),
                'std': np.std(scores),
                'cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
                'decile_1_score': scores[0],
                'decile_10_score': scores[9],
                'decile_1_to_10_ratio': scores[0] / scores[9] if scores[9] > 0 else 0
            }
            decile_stats.append(stats)
    
    decile_stats_df = pd.DataFrame(decile_stats)
    
    # Sort by coefficient of variation (highest first)
    decile_stats_df = decile_stats_df.sort_values('cv', ascending=False)
    
    print("Top 20 indicators with highest decile variation (CV):")
    print(decile_stats_df.head(20)[['primary_index', 'country', 'cv', 'range', 'decile_1_score', 'decile_10_score']].to_string(index=False))
    
    print(f"\nTop 20 indicators with highest absolute range:")
    range_sorted = decile_stats_df.sort_values('range', ascending=False)
    print(range_sorted.head(20)[['primary_index', 'country', 'range', 'cv', 'decile_1_score', 'decile_10_score']].to_string(index=False))
    
    # Check for extreme decile 1 to decile 10 ratios
    print(f"\nTop 20 indicators with highest decile 1 to decile 10 ratio:")
    ratio_sorted = decile_stats_df.sort_values('decile_1_to_10_ratio', ascending=False)
    print(ratio_sorted.head(20)[['primary_index', 'country', 'decile_1_to_10_ratio', 'cv', 'range']].to_string(index=False))
    
    return decile_stats_df

def analyze_year_variations(df):
    """Analyze variations across years for each indicator-country-decile combination"""
    print("\n=== ANALYZING YEAR VARIATIONS ===\n")
    
    # Load time series data
    try:
        time_series_df = pd.read_csv('../output/ewbi_time_series.csv')
        print(f"Time series data shape: {time_series_df.shape}")
        
        # Filter for primary indicators and 'All' deciles
        ts_primary = time_series_df[
            (time_series_df['Level'] == '4 (Primary_indicator)') & 
            (time_series_df['decile'] == 'All')
        ].copy()
        
        # Calculate year-to-year variations
        year_stats = []
        
        for (indicator, country) in ts_primary.groupby(['primary_index', 'country']):
            if len(country) > 1:  # Need at least 2 years
                scores = country['Score'].values
                years = country['year'].values
                
                # Calculate year-to-year changes
                year_changes = np.diff(scores)
                year_change_pct = np.diff(scores) / scores[:-1] * 100
                
                stats = {
                    'primary_index': indicator[0],
                    'country': indicator[1],
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'range': np.max(scores) - np.min(scores),
                    'std': np.std(scores),
                    'cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
                    'max_year_change': np.max(np.abs(year_changes)),
                    'max_year_change_pct': np.max(np.abs(year_change_pct)),
                    'num_years': len(scores),
                    'first_year': np.min(years),
                    'last_year': np.max(years)
                }
                year_stats.append(stats)
        
        year_stats_df = pd.DataFrame(year_stats)
        
        print("Top 20 indicators with highest year-to-year variation (CV):")
        print(year_stats_df.sort_values('cv', ascending=False).head(20)[['primary_index', 'country', 'cv', 'max_year_change_pct', 'range']].to_string(index=False))
        
        print(f"\nTop 20 indicators with highest year-to-year percentage change:")
        print(year_stats_df.sort_values('max_year_change_pct', ascending=False).head(20)[['primary_index', 'country', 'max_year_change_pct', 'cv', 'range']].to_string(index=False))
        
        return year_stats_df
        
    except FileNotFoundError:
        print("Time series data not found. Skipping year variation analysis.")
        return None

def identify_suspicious_patterns(decile_stats_df, year_stats_df):
    """Identify potentially suspicious or problematic patterns"""
    print("\n=== IDENTIFYING SUSPICIOUS PATTERNS ===\n")
    
    # Decile patterns that might be suspicious
    print("SUSPICIOUS DECILE PATTERNS:")
    
    # 1. Very high CV (> 1.0)
    high_cv = decile_stats_df[decile_stats_df['cv'] > 1.0]
    if len(high_cv) > 0:
        print(f"\n1. Indicators with very high decile variation (CV > 1.0): {len(high_cv)} found")
        print(high_cv[['primary_index', 'country', 'cv', 'range']].head(10).to_string(index=False))
    
    # 2. Extreme decile ratios (> 10 or < 0.1)
    extreme_ratios = decile_stats_df[
        (decile_stats_df['decile_1_to_10_ratio'] > 10) | 
        (decile_stats_df['decile_1_to_10_ratio'] < 0.1)
    ]
    if len(extreme_ratios) > 0:
        print(f"\n2. Indicators with extreme decile 1 to decile 10 ratios: {len(extreme_ratios)} found")
        print(extreme_ratios[['primary_index', 'country', 'decile_1_to_10_ratio', 'cv']].head(10).to_string(index=False))
    
    # 3. Very small ranges (< 0.01) - might indicate data quality issues
    small_ranges = decile_stats_df[decile_stats_df['range'] < 0.01]
    if len(small_ranges) > 0:
        print(f"\n3. Indicators with very small decile ranges (< 0.01): {len(small_ranges)} found")
        print(small_ranges[['primary_index', 'country', 'range', 'cv']].head(10).to_string(index=False))
    
    # Year patterns that might be suspicious
    if year_stats_df is not None:
        print(f"\nSUSPICIOUS YEAR PATTERNS:")
        
        # 1. Very high year-to-year changes (> 50%)
        high_year_changes = year_stats_df[year_stats_df['max_year_change_pct'] > 50]
        if len(high_year_changes) > 0:
            print(f"\n1. Indicators with very high year-to-year changes (> 50%): {len(high_year_changes)} found")
            print(high_year_changes[['primary_index', 'country', 'max_year_change_pct', 'cv']].head(10).to_string(index=False))

def main():
    """Main analysis function"""
    print("EWBI Data Variation Analysis")
    print("=" * 50)
    
    # Load master data
    print("Loading master data...")
    df = pd.read_csv('../output/ewbi_master.csv')
    print(f"Loaded {df.shape[0]} rows of data")
    
    # Analyze decile variations
    decile_stats = analyze_decile_variations(df)
    
    # Analyze year variations
    year_stats = analyze_year_variations(df)
    
    # Identify suspicious patterns
    identify_suspicious_patterns(decile_stats, year_stats)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Check the output above for indicators with unusual variations.")
    print("High CV values, extreme decile ratios, or large year-to-year changes")
    print("may indicate data quality issues or interesting trends worth investigating.")

if __name__ == "__main__":
    main() 