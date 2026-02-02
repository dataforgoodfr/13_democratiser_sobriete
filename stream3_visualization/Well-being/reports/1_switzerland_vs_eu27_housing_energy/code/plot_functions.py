import pandas as pd 
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import mapclassify
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# Set font
plt.rcParams['font.family'] = 'Arial'

# Define study countries EU Countries + EFTA + UK
study_countries = ['AT', 'BE', 'BG', 'HR','CY', 'CZ', 'DK', 'EE', 'FI', 
'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 
'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 
'UK', 'NO','CH','IS', 'LI']


def process_eurostat_tsv(file_path, return_format='long', verbose=True):
    """
    Process Eurostat TSV files with multi-dimensional headers and time series data.
    
    Parameters:
    -----------
    file_path : str
        Path to the Eurostat TSV file
    return_format : str, default 'long'
        Output format: 'wide' for original structure, 'long' for melted format
    verbose : bool, default True
        Whether to print processing information
    
    Returns:
    --------
    pandas.DataFrame
        Processed dataframe in specified format
    """
    
    # 1. Import data with special header handling
    df = pd.read_csv(file_path, sep='\t', header=0)
    
    # 2. Extract dimensions and temporal columns
    first_col_name = df.columns[0]
    dimensions = first_col_name.split(',')
    
    if verbose:
        print(f"Dimensions found: {dimensions}")
        print(f"Temporal columns: {df.columns[1:].tolist()}")
    
    # 3. Split first column into distinct dimensions
    df_expanded = df[first_col_name].str.split(',', expand=True)
    df_expanded.columns = dimensions
    
    # 4. Add temporal columns
    for col in df.columns[1:]:
        df_expanded[col] = df[col]
    
    # 5. Data cleaning - replace typical Eurostat missing values
    df_expanded = df_expanded.replace([':', '- ', ': ', ' :'], np.nan)
    
    # 6. Convert numeric columns (years) to numeric type
    year_columns = [col for col in df_expanded.columns if col.strip().isdigit()]
    for col in year_columns:
        df_expanded[col] = pd.to_numeric(df_expanded[col], errors='coerce')
    
    if verbose:
        print(f"\nProcessed data overview:")
        print(df_expanded.head())
        print(f"\nDataset shape: {df_expanded.shape}")
    
    # 7. Return in requested format
    if return_format.lower() == 'long':
        return _transform_to_long_format(df_expanded)
    else:
        return df_expanded.rename(columns={'geo\\TIME_PERIOD': 'geo'})


def _transform_to_long_format(df):
    """
    Transform data to long format for easier analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Wide format dataframe with year columns
    
    Returns:
    --------
    pandas.DataFrame
        Long format dataframe with year and value columns
    """
    
    # Identify dimension and value columns
    dimension_cols = [col for col in df.columns if not col.strip().isdigit()]
    value_cols = [col for col in df.columns if col.strip().isdigit()]
    
    # Melt transformation
    df_long = df.melt(
        id_vars=dimension_cols,
        value_vars=value_cols,
        var_name='year',
        value_name='value'
    )
    
    # Convert year to integer
    df_long['year'] = df_long['year'].astype(int)
    
    # Clean up column names (handle common Eurostat patterns)
    if 'geo\\TIME_PERIOD' in df_long.columns:
        df_long = df_long.rename(columns={'geo\\TIME_PERIOD': 'geo'})
    
    return df_long


import pandas as pd
import numpy as np
import re

def process_eurostat_tsv_monthly(file_path, verbose=True):
    """
    Process Eurostat TSV with monthly time columns into tidy long format.
    
    Parameters:
    -----------
    file_path : str
        Path to the Eurostat TSV file
    verbose : bool
        Whether to print information during processing
    
    Returns:
    --------
    pandas.DataFrame
        Long format with dimension columns, 'geo', 'year', 'month', and 'value'
    """
    
    # 1. Load the raw data
    df_raw = pd.read_csv(file_path, sep='\t', header=0, dtype=str, low_memory=False)
    
    # 2. Extract dimension names from the first column
    first_col = df_raw.columns[0]
    dimension_names = first_col.split(',')
    
    if verbose:
        print(f"Dimensions detected: {dimension_names}")
    
    # 3. Split the first column into separate dimensions
    df_dims = df_raw[first_col].str.split(',', expand=True)
    df_dims.columns = dimension_names
    
    # 4. Combine with the monthly time columns
    time_columns = df_raw.columns[1:]
    df_data = pd.concat([df_dims, df_raw[time_columns]], axis=1)
    
    # 5. Replace Eurostat missing values with np.nan
    df_data.replace([':', ' :', ': ', '-', ' -'], np.nan, inplace=True)
    
    # 6. Melt to long format
    df_long = df_data.melt(
        id_vars=dimension_names,
        value_vars=time_columns,
        var_name='time',
        value_name='value'
    )
    
    # 7. Extract year and month from time column (e.g. 2000M01)
    df_long = df_long[df_long['time'].notna()]
    df_long['year'] = df_long['time'].str.extract(r'^(\d{4})').astype(float).astype('Int64')

    # 8. Convert value to numeric
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    
    # 9. Clean common column names (like 'geo\\TIME_PERIOD' to 'geo')
    df_long.rename(columns=lambda x: x.replace('geo\\TIME_PERIOD', 'geo'), inplace=True)
    
    # 10. Summarize values on months 
    columns_list = df_long.columns.tolist()

    # Extract values that are not 'time' or 'value'
    extracted_values = [value for value in columns_list if value not in ['time', 'value']]

    df_long = df_long.groupby(extracted_values)['value'].sum().reset_index()

    # 10. Final overview
    if verbose:
        print(f"\nResulting data shape: {df_long.shape}")
        print(df_long.head())

    return df_long




def plot_europe_map(df, year, colormap='YlOrRd', value_title='Value',
                    figsize=(10, 10), shapefile_path=None, k=6):
    """
    Create a choropleth map of Europe with study countries highlighted and non-study countries with stripes.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to visualize. Must have columns 'year', 'geo' (ISO country codes), and 'value'
    year : int
        The year to display data for
    study_countries : list of str
        List of ISO 2-letter country codes representing the study countries
    colormap : str, default 'YlOrRd'
        The matplotlib colormap to use for the visualization
    figsize : tuple, default (10, 10)
        Figure size as (width, height)
    shapefile_path : str, optional
        Path to the shapefile. If None, uses default path
    k : int, default 6
        Number of quantile bins to classify values

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """


    # Default shapefile path if not provided
    if shapefile_path is None:
        shapefile_path = "0_shapefile/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp"

    # Load shapefile
    world = gpd.read_file(shapefile_path)

    # Extended list of European countries including those that might not be classified as 'Europe' in CONTINENT
    european_countries = [
        'AD', 'AL', 'AT', 'BA', 'BE', 'BG', 'BY', 'CH', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR',
        'GB', 'GR', 'HR', 'HU', 'IE', 'IS', 'IT', 'LI', 'LT', 'LU', 'LV', 'MC', 'MD', 'ME', 'MK', 'MT',
        'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'RU', 'SE', 'SI', 'SK', 'SM', 'TR', 'UA', 'VA', 'XK'
    ]

    # Filter to Europe using both CONTINENT field and explicit country list
    europe = world[
        (world['CONTINENT'] == 'Europe') | 
        (world['ISO_A2_EH'].isin(european_countries))
    ].copy()

    # Also include countries that might be coded differently but are geographically in Europe
    # This handles cases where Turkey or other countries might have different continent classifications
    additional_europe = world[
        world['NAME'].isin(['Turkey', 'Cyprus', 'Russia', 'Kazakhstan']) |
        world['ISO_A2_EH'].isin(['TR', 'CY', 'RU', 'KZ'])
    ].copy()
    
    # Combine and remove duplicates
    europe = pd.concat([europe, additional_europe]).drop_duplicates(subset=['ISO_A2_EH'])

    # Prepare data for the specified year
    year_df = df[df['year'] == year].copy()

    # Check if the year exists in the data
    if year_df.empty:
        raise ValueError(f"No data found for year {year}")

    # Handle UK/GB country code mismatch
    year_df['geo'] = year_df['geo'].replace('UK', 'GB')
    year_df['ISO_A2_EH'] = year_df['geo']

    # Merge data
    europe = europe.merge(year_df[['ISO_A2_EH', 'value']], on='ISO_A2_EH', how='left')

    # Mark study countries
    study_countries_mapped = [code.replace('UK', 'GB') for code in study_countries]
    europe['is_study_country'] = europe['ISO_A2_EH'].isin(study_countries_mapped)

    # Reproject to Lambert Conformal Conic for Europe
    europe = europe.to_crs(epsg=3035)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot study countries with data
    study_europe = europe[europe['is_study_country']].copy()
    study_europe.plot(
        column='value',
        cmap=colormap,
        linewidth=0.6,
        ax=ax,
        edgecolor='black',
        scheme='quantiles',
        k=k,
        legend=False,
        missing_kwds={"color": "lightgrey", "label": "Missing values"},
    )

    # Plot non-study countries with stripes
    non_study_europe = europe[~europe['is_study_country']].copy()
    non_study_europe.plot(
        ax=ax,
        color='white',
        linewidth=0.6,
        edgecolor='black',
        hatch='///',
        alpha=0.4
    )

    # Create custom legend with correct quantile bins and colors
    values = study_europe.loc[study_europe['value'].notna(), 'value']

    if not values.empty:
        classifier = mapclassify.Quantiles(values, k=k)
        bins = classifier.bins

        cmap = cm.get_cmap(colormap, k)
        colors = [cmap(i) for i in range(k)]

        legend_elements = []
        for i in range(k):
            if i == 0:
                label = f'{values.min():.1f} – {bins[i]:.1f}'
            else:
                label = f'{bins[i-1]:.1f} – {bins[i]:.1f}'

            patch = mpatches.Rectangle((0, 0), 1, 1,
                                       facecolor=colors[i],
                                       edgecolor='black',
                                       linewidth=0.6)
            legend_elements.append((patch, label))

        if study_europe['value'].isna().any():
            missing_patch = mpatches.Rectangle((0, 0), 1, 1,
                                               facecolor='lightgrey',
                                               edgecolor='black',
                                               linewidth=0.6)
            legend_elements.append((missing_patch, 'Missing values'))

        legend_elements.append((
            mpatches.Rectangle((0, 0), 1, 1,
                                facecolor='white',
                                edgecolor='black',
                                hatch='///',
                                linewidth=0.3,
                                alpha=0.4),
            'Non-study regions'
        ))

        ax.legend([elem[0] for elem in legend_elements],
                  [elem[1] for elem in legend_elements],
                  loc='upper right',
                  title=f'{value_title} ({year})',
                  fontsize=10,
                  title_fontsize=10,
                  fancybox=False,
                  framealpha=1.0,
                  edgecolor='black',
                  facecolor='white')

    # Set the bounds for Central Europe
    ax.set_xlim(2200000, 6600000)
    ax.set_ylim(1200000, 5800000)
    
    ax.set_axis_off()
    plt.tight_layout()

    return fig, ax




def plot_europe_nuts2_map(df, year, colormap='YlOrRd', value_title='Value',
                          figsize=(12, 10), shapefile_path=None, k=6,
                          world_shapefile_path="0_shapefile/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp"):
    """
    Create a choropleth map of Europe at NUTS 2 level with study regions highlighted 
    and non-study regions (including non-EU countries) with stripes and thick country borders.
    """
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import mapclassify
    import matplotlib.patches as mpatches
    from matplotlib import cm


    # Define study countries
    country_code_map = {'UK': 'GB'}
    study_countries_mapped = [country_code_map.get(c, c) for c in study_countries]

    # Load NUTS 2 shapefile
    if shapefile_path is None:
        shapefile_path = r"0_shapefile/NUTS_RG_10M_2024_3035.gpkg"
        #shapefile_path = r"0_shapefile\NUTS_RG_10M_2021_3035.gpkg"

    try:
        try:
            nuts2 = gpd.read_file(shapefile_path, layer='NUTS_RG_10M_2024_3035')
        except:
            nuts2 = gpd.read_file(shapefile_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"NUTS 2 geopackage not found at {shapefile_path}.")

    if 'LEVL_CODE' in nuts2.columns:
        nuts2 = nuts2[nuts2['LEVL_CODE'] == 2].copy()
    elif 'STAT_LEVL_' in nuts2.columns:
        nuts2 = nuts2[nuts2['STAT_LEVL_'] == 2].copy()

    # Identify NUTS code column
    nuts_code_col = None
    for col in ['NUTS_ID', 'geo', 'CNTR_CODE', 'id', 'CODE']:
        if col in nuts2.columns:
            nuts_code_col = col
            break
    if nuts_code_col is None:
        raise ValueError("Could not find a NUTS code column in the shapefile.")

    # Prepare data
    year_df = df[df['year'] == year].copy()
    if year_df.empty:
        raise ValueError(f"No data found for year {year}")
    if nuts_code_col != 'geo':
        nuts2 = nuts2.rename(columns={nuts_code_col: 'geo'})

    # Merge value
    nuts2 = nuts2.merge(year_df[['geo', 'value']], on='geo', how='left')
    nuts2['country_code'] = nuts2['geo'].str[:2]
    nuts2['is_study_region'] = nuts2['country_code'].isin(study_countries_mapped)

    # Convert CRS
    if nuts2.crs != 'EPSG:3035':
        nuts2 = nuts2.to_crs(epsg=3035)

    # Load world shapefile
    world = gpd.read_file(world_shapefile_path)

    # Fix: force inclusion of Turkey
    europe_world = world[(world['CONTINENT'] == 'Europe') | (world['ISO_A2'] == 'TR')].copy()
    europe_world = europe_world.to_crs(epsg=3035)

    # Determine non-study countries
    non_study_countries = europe_world[~europe_world['ISO_A2'].isin(study_countries_mapped)]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Non-study countries with hatch
    if not non_study_countries.empty:
        non_study_countries.plot(
            ax=ax,
            color='white',
            edgecolor='black',
            linewidth=0.3,
            hatch='///',
            alpha=0.4
        )

    # Study regions with data
    study_nuts2 = nuts2[nuts2['is_study_region'] & nuts2['value'].notna()].copy()
    if not study_nuts2.empty:
        study_nuts2.plot(
            column='value',
            cmap=colormap,
            linewidth=0.3,
            ax=ax,
            edgecolor='black',
            scheme='quantiles',
            k=k,
            legend=False,
            missing_kwds={"color": "lightgrey", "label": "Missing values"},
        )

    # Study regions with missing data
    study_missing = nuts2[nuts2['is_study_region'] & nuts2['value'].isna()].copy()
    if not study_missing.empty:
        study_missing.plot(
            ax=ax,
            color='lightgrey',
            linewidth=0.3,
            edgecolor='black'
        )

    # Non-study NUTS2
    non_study_nuts2 = nuts2[~nuts2['is_study_region']].copy()
    if not non_study_nuts2.empty:
        non_study_nuts2.plot(
            ax=ax,
            color='white',
            linewidth=0.3,
            edgecolor='black',
            hatch='///',
            alpha=0.4
        )

    # Countries in study group with no NUTS2 at all
    nuts2_country_codes = nuts2['country_code'].unique()
    study_country_missing_nuts2 = [
        c for c in study_countries_mapped if c not in nuts2_country_codes
    ]
    study_countries_no_nuts2 = europe_world[europe_world['ISO_A2'].isin(study_country_missing_nuts2)]
    if not study_countries_no_nuts2.empty:
        study_countries_no_nuts2.plot(
            ax=ax,
            color='lightgrey',
            edgecolor='black',
            linewidth=0.3
        )

    # Legend
    legend_elements = []
    if not study_nuts2.empty:
        values = study_nuts2['value']
        classifier = mapclassify.Quantiles(values, k=k)
        bins = classifier.bins
        cmap = cm.get_cmap(colormap, k)
        colors = [cmap(i) for i in range(k)]

        for i in range(k):
            label = f'{values.min():.1f} – {bins[i]:.1f}' if i == 0 else f'{bins[i-1]:.1f} – {bins[i]:.1f}'
            patch = mpatches.Rectangle((0, 0), 1, 1,
                                       facecolor=colors[i],
                                       edgecolor='black',
                                       linewidth=0.3)
            legend_elements.append((patch, label))

    if not study_missing.empty or not study_countries_no_nuts2.empty:
        legend_elements.append((
            mpatches.Rectangle((0, 0), 1, 1, facecolor='lightgrey', edgecolor='black', linewidth=0.3),
            'Missing values'
        ))

    if not non_study_nuts2.empty or not non_study_countries.empty:
        legend_elements.append((
            mpatches.Rectangle((0, 0), 1, 1,
                               facecolor='white',
                               edgecolor='black',
                               hatch='///',
                               linewidth=0.3,
                               alpha=0.4),
            'Non-study regions'
        ))

    if legend_elements:
        ax.legend([e[0] for e in legend_elements],
                  [e[1] for e in legend_elements],
                  loc='upper right',
                  title=f'{value_title} ({year})',
                  fontsize=9,
                  title_fontsize=10,
                  fancybox=False,
                  framealpha=1.0,
                  edgecolor='black',
                  facecolor='white')

    # Final: draw borders for all European countries (including Turkey)
    europe_world.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

    ax.set_xlim(2200000, 6600000)
    ax.set_ylim(1200000, 5800000)
    ax.set_axis_off()
    plt.tight_layout()

    return fig, ax
