"""
IEA Energy Sankey Diagram — EU-27
=================================
Reads IEA data files and produces a Sankey diagram of EU-27 energy flows
for the latest common year.

Structure:
  LEFT:   Primary energy supply by source (domestic production + imports)
  MIDDLE: Electricity (transformation hub)
  RIGHT:  Final consumption sectors + Exports + Losses & transformation

Data units: all converted to PJ (petajoules)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
DATA_DIR = os.path.join(BASE_DIR, 'external_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'graphs', 'IEA')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Region configurations
REGIONS = {
    'EU-27': {
        'suffix': 'European Union (EU27)',
        'file_prefix': 'iea_eu27',
    },
    'France': {
        'suffix': 'France',
        'file_prefix': 'iea_france',
    },
    'Switzerland': {
        'suffix': 'Switzerland',
        'file_prefix': 'iea_switzerland',
    },
}

def get_files(region):
    """Return the dict of IEA filenames for a given region."""
    s = REGIONS[region]['suffix']
    return {
        'production': f'iea_Domestic energy production by source - {s}.csv',
        'coal_trade': f'iea_Coal imports vs. exports - {s}.csv',
        'gas_trade': f'iea_Natural gas imports vs. exports - {s}.csv',
        'oil_trade': f'iea_Crude oil imports vs. exports - {s}.csv',
        'oil_prod_trade': f'iea_Oil products imports vs. exports - {s}.csv',
        'elec_trade': f'iea_Electricity imports vs. exports - {s}.csv',
        'elec_gen': f'iea_Electricity generation by source - {s}.csv',
        'elec_cons': f'iea_Electricity consumption by sector - {s}.csv',
        'coal_cons': f'iea_Coal final consumption by sector - {s}.csv',
        'gas_cons': f'iea_Natural gas final consumption by sector - {s}.csv',
        'oil_cons': f'iea_Oil products final consumption by sector - {s}.csv',
        'bio_cons': f'iea_Biofuels and waste final consumption by sector - {s}.csv',
    }

# Colors
DOMESTIC_COLOR = '#2E86C1'
IMPORTS_COLOR = '#E74C3C'
SOURCE_COLORS = {
    'Coal': '#555555',
    'Oil': '#8B4513',
    'Oil Products': '#D2691E',
    'Natural Gas': '#FF8C00',
    'Nuclear': '#9370DB',
    'Renewables': '#32CD32',
    'Biofuels & Waste': '#228B22',
    'Elec. Imports': '#DAA520',
}
ELEC_COLOR = '#FFD700'
SECTOR_COLORS = {
    'Industry': '#4682B4',
    'Transport': '#DC143C',
    'Residential': '#3CB371',
    'Non-residential buildings': '#20B2AA',
    'Agriculture & Other': '#CD853F',
    'Non-energy Use': '#A9A9A9',
    'Exports': '#191970',
    'Losses & Transformation': '#D3D3D3',
}


# ============================================================================
# DATA LOADING
# ============================================================================

def read_iea(filename):
    """Read an IEA CSV file (3 header rows to skip, first col = Year, last col = Units)."""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, skiprows=3)
    df = df.rename(columns={df.columns[0]: 'Year'})
    if 'Units' in df.columns:
        df = df.drop(columns=['Units'])
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def get_row(df, year):
    """Return the row for a specific year as a dict (excluding 'Year')."""
    row = df[df['Year'] == year]
    if row.empty:
        return {}
    return row.iloc[0].drop('Year').to_dict()


# ============================================================================
# COMPUTE FLOWS
# ============================================================================

def compute_sankey_data(year, region='EU-27'):
    """Compute all Sankey nodes & links for the given year. Values in PJ."""

    files = get_files(region)

    # Load all data
    prod = read_iea(files['production'])
    coal_trade = read_iea(files['coal_trade'])
    gas_trade = read_iea(files['gas_trade'])
    oil_trade = read_iea(files['oil_trade'])
    oil_prod_trade = read_iea(files['oil_prod_trade'])
    elec_trade = read_iea(files['elec_trade'])
    elec_gen = read_iea(files['elec_gen'])
    elec_cons = read_iea(files['elec_cons'])
    coal_cons = read_iea(files['coal_cons'])
    gas_cons = read_iea(files['gas_cons'])
    oil_cons = read_iea(files['oil_cons'])
    bio_cons = read_iea(files['bio_cons'])

    # Get data for the target year
    p = get_row(prod, year)       # TJ
    ct = get_row(coal_trade, year)
    gt = get_row(gas_trade, year)
    ot = get_row(oil_trade, year)
    opt = get_row(oil_prod_trade, year)
    et = get_row(elec_trade, year)
    eg = get_row(elec_gen, year)  # GWh
    ec = get_row(elec_cons, year) # TJ
    cc = get_row(coal_cons, year)
    gc = get_row(gas_cons, year)
    oc = get_row(oil_cons, year)
    bc = get_row(bio_cons, year)

    # ---------- SUPPLY (TJ) ----------
    # Domestic production
    coal_prod = p.get('Coal and coal products', 0)
    oil_prod = p.get('Primary oil', 0)
    gas_prod = p.get('Natural gas', 0)
    nuclear_prod = p.get('Nuclear', 0)
    hydro_prod = p.get('Hydropower', 0)
    solar_wind_prod = p.get('Solar, wind and other renewables', 0)
    bio_prod = p.get('Biofuels and waste', 0)
    heat_prod = p.get('Heat', 0)

    # Imports (positive values from the import/export files)
    coal_imp = ct.get('Imports', 0)
    gas_imp = gt.get('Imports', 0)
    oil_crude_imp = ot.get('Imports', 0)
    oil_prod_imp = opt.get('Imports', 0)
    elec_imp = et.get('Imports', 0)

    # Exports (stored as negative, take absolute value)
    coal_exp = abs(ct.get('Exports', 0))
    gas_exp = abs(gt.get('Exports', 0))
    oil_crude_exp = abs(ot.get('Exports', 0))
    oil_prod_exp = abs(opt.get('Exports', 0))
    elec_exp = abs(et.get('Exports', 0))

    # Total supply per source
    coal_supply = coal_prod + coal_imp
    oil_crude_supply = oil_prod + oil_crude_imp
    gas_supply = gas_prod + gas_imp
    renewables_supply = hydro_prod + solar_wind_prod

    # ---------- ELECTRICITY GENERATION (GWh → TJ, ×3.6) ----------
    GWH_TO_TJ = 3.6
    coal_elec = eg.get('Coal', 0) * GWH_TO_TJ
    oil_elec = eg.get('Oil', 0) * GWH_TO_TJ
    gas_elec = eg.get('Natural gas', 0) * GWH_TO_TJ
    nuclear_elec = eg.get('Nuclear', 0) * GWH_TO_TJ
    hydro_elec = eg.get('Hydropower', 0) * GWH_TO_TJ
    solar_elec = eg.get('Solar PV', 0) * GWH_TO_TJ
    wind_elec = eg.get('Wind', 0) * GWH_TO_TJ
    geo_elec = eg.get('Geothermal', 0) * GWH_TO_TJ
    tide_elec = eg.get('Tide', 0) * GWH_TO_TJ
    solar_th_elec = eg.get('Solar thermal', 0) * GWH_TO_TJ
    other_elec = eg.get('Other sources', 0) * GWH_TO_TJ
    bio_gen_elec = eg.get('Biofuels', 0) * GWH_TO_TJ
    waste_gen_elec = eg.get('Waste', 0) * GWH_TO_TJ

    renewables_elec = hydro_elec + solar_elec + wind_elec + geo_elec + tide_elec + solar_th_elec + other_elec
    bio_elec = bio_gen_elec + waste_gen_elec

    # ---------- FINAL CONSUMPTION BY SECTOR (TJ) ----------
    # Sum "Agriculture and forestry" + "Fishing" + "Other non-specified" into Agriculture & Other
    def sector_sums(d):
        """Return dict of aggregated sector values."""
        industry = d.get('Industry', 0)
        transport = d.get('Transport', 0)
        residential = d.get('Residential', 0)
        commercial = d.get('Commercial and public services', 0)
        agri = d.get('Agriculture and forestry', 0) + d.get('Fishing', 0) + d.get('Other non-specified', 0)
        non_energy = d.get('Non-energy use', 0)
        return {
            'Industry': industry,
            'Transport': transport,
            'Residential': residential,
            'Non-residential buildings': commercial,
            'Agriculture & Other': agri,
            'Non-energy Use': non_energy,
        }

    coal_sectors = sector_sums(cc)
    oil_sectors = sector_sums(oc)
    gas_sectors = sector_sums(gc)
    bio_sectors = sector_sums(bc)
    elec_sectors = sector_sums(ec)  # electricity consumption has no Non-energy use column
    elec_sectors['Non-energy Use'] = 0

    # ---------- OIL: CRUDE vs PRODUCTS SPLIT ----------
    # Oil products total outflows = sector consumption + products exports + electricity from oil
    oil_total_cons = sum(oil_sectors.values())
    oil_products_total_out = oil_total_cons + oil_prod_exp + oil_elec
    # Crude oil refined into products = products outflows - products imports
    refinery_output = max(0, oil_products_total_out - oil_prod_imp)
    # Oil products supply (for label)
    oil_prod_supply = refinery_output + oil_prod_imp
    # Crude oil losses (refinery inefficiency etc.)
    oil_crude_loss = max(0, oil_crude_supply - oil_crude_exp - refinery_output)

    # ---------- LOSSES per source (TJ) ----------
    coal_total_cons = sum(coal_sectors.values())
    gas_total_cons = sum(gas_sectors.values())
    bio_total_cons = sum(bio_sectors.values())

    coal_loss = max(0, coal_supply - coal_elec - coal_total_cons - coal_exp)
    gas_loss = max(0, gas_supply - gas_elec - gas_total_cons - gas_exp)
    nuclear_loss = max(0, nuclear_prod - nuclear_elec)
    renewables_loss = max(0, renewables_supply - renewables_elec)
    bio_loss = max(0, bio_prod - bio_elec - bio_total_cons)

    # Electricity T&D losses
    total_elec_input = (coal_elec + oil_elec + gas_elec + nuclear_elec +
                        renewables_elec + bio_elec + elec_imp)
    total_elec_output_sectors = sum(elec_sectors.values())
    elec_td_loss = max(0, total_elec_input - total_elec_output_sectors - elec_exp)

    # ========================================================================
    # BUILD NODES & LINKS
    # ========================================================================

    # 5-column layout:
    #   FAR LEFT (origin):  Domestic Production (0), Imports (1)
    #   SOURCES (top→bot):  Renewables (2), Biofuels (3), Coal (4),
    #                       Nuclear (5), Elec. Imports (6), Nat Gas (7),
    #                       Crude Oil (8)
    #   REFINERY:           Oil Products (9)
    #   MIDDLE:             Electricity (10)
    #   SINKS (top→bot):    Industry (11), Non-res buildings (12),
    #                       Residential (13), Transport (14),
    #                       Agriculture (15), Non-energy Use (16),
    #                       Losses (17), Exports (18)

    TJ_TO_PJ = 1 / 1000  # convert TJ → PJ

    # Totals for origin nodes
    domestic_total = (coal_prod + oil_prod + gas_prod +
                      renewables_supply + bio_prod)
    imports_total = (coal_imp + oil_crude_imp + oil_prod_imp + gas_imp +
                     nuclear_prod + elec_imp)

    node_labels = [
        # FAR LEFT — origin (0-1)
        f"Domestic\nProduction\n({domestic_total * TJ_TO_PJ:,.0f} PJ)",
        f"Imports\n({imports_total * TJ_TO_PJ:,.0f} PJ)",
        # SOURCES top→bottom (2-8)
        f"Renewables\n({renewables_supply * TJ_TO_PJ:,.0f} PJ)",
        f"Biofuels & Waste\n({bio_prod * TJ_TO_PJ:,.0f} PJ)",
        f"Coal\n({coal_supply * TJ_TO_PJ:,.0f} PJ)",
        f"Nuclear\n({nuclear_prod * TJ_TO_PJ:,.0f} PJ)",
        f"Elec. Imports\n({elec_imp * TJ_TO_PJ:,.0f} PJ)",
        f"Natural Gas\n({gas_supply * TJ_TO_PJ:,.0f} PJ)",
        f"Crude Oil\n({oil_crude_supply * TJ_TO_PJ:,.0f} PJ)",
        # REFINERY (9)
        f"Oil Products\n({oil_prod_supply * TJ_TO_PJ:,.0f} PJ)",
        # MIDDLE (10)
        "Electricity",
        # SINKS top→bottom (11-18)
        "Industry",
        "Non-residential buildings",
        "Residential",
        "Transport",
        "Agriculture & Other",
        "Non-energy Use",
        "Losses &\nTransformation",
        "Exports",
    ]

    node_colors = [
        DOMESTIC_COLOR,
        IMPORTS_COLOR,
        SOURCE_COLORS['Renewables'],
        SOURCE_COLORS['Biofuels & Waste'],
        SOURCE_COLORS['Coal'],
        SOURCE_COLORS['Nuclear'],
        SOURCE_COLORS['Elec. Imports'],
        SOURCE_COLORS['Natural Gas'],
        SOURCE_COLORS['Oil'],
        SOURCE_COLORS['Oil Products'],
        ELEC_COLOR,
        SECTOR_COLORS['Industry'],
        SECTOR_COLORS['Non-residential buildings'],
        SECTOR_COLORS['Residential'],
        SECTOR_COLORS['Transport'],
        SECTOR_COLORS['Agriculture & Other'],
        SECTOR_COLORS['Non-energy Use'],
        SECTOR_COLORS['Losses & Transformation'],
        SECTOR_COLORS['Exports'],
    ]

    # Node positions — 5-column layout
    X_ORIGIN  = 0.01
    X_SOURCE  = 0.18
    X_REFINE  = 0.38
    X_ELEC    = 0.56
    X_SINK    = 0.99

    n_sources = 7   # Renewables, Bio, Coal, Nuclear, Elec.Imp, Gas, Crude Oil
    n_sinks = 8     # Industry..Exports

    #                    Renew  Bio    Coal   Nucl   ElImp  Gas    CrudeOil
    source_y =          [0.01,  0.09,  0.17,  0.28,  0.38,  0.50,  0.68]
    #                    Ind    NonRes Resid  Trans  Agri   NonEn  Losses Exports
    sink_y   =          [0.01,  0.12,  0.23,  0.38,  0.52,  0.62,  0.74,  0.88]

    node_x = (
        [X_ORIGIN, X_ORIGIN] +          # Domestic (top), Imports (bottom)
        [X_SOURCE] * n_sources +         # Sources
        [X_REFINE] +                     # Oil Products
        [X_ELEC] +                       # Electricity
        [X_SINK] * n_sinks               # Sinks
    )
    node_y = (
        [0.12, 0.58] +                   # Domestic (top), Imports (bottom)
        source_y +                        # Sources
        [0.86] +                          # Oil Products (below Crude Oil)
        [0.05] +                          # Electricity below title
        sink_y                            # Sinks
    )

    # ---- LINKS ----
    MIN_FLOW_TJ = 20_000  # minimum flow to display (20 PJ)

    raw_links = []

    def add_link(src, tgt, val_tj, color_hex):
        if val_tj > MIN_FLOW_TJ:
            raw_links.append((src, tgt, val_tj * TJ_TO_PJ, color_hex))

    # --- Domestic (0) → Sources ---
    add_link(0, 2, renewables_supply, DOMESTIC_COLOR)  # → Renewables
    add_link(0, 3, bio_prod,          DOMESTIC_COLOR)  # → Biofuels
    add_link(0, 4, coal_prod,         DOMESTIC_COLOR)  # → Coal
    add_link(0, 7, gas_prod,          DOMESTIC_COLOR)  # → Gas
    add_link(0, 8, oil_prod,          DOMESTIC_COLOR)  # → Crude Oil

    # --- Imports (1) → Sources ---
    add_link(1, 4, coal_imp,          IMPORTS_COLOR)   # → Coal
    add_link(1, 5, nuclear_prod,      IMPORTS_COLOR)   # → Nuclear
    add_link(1, 6, elec_imp,          IMPORTS_COLOR)   # → Elec. Imports
    add_link(1, 7, gas_imp,           IMPORTS_COLOR)   # → Gas
    add_link(1, 8, oil_crude_imp,     IMPORTS_COLOR)   # → Crude Oil

    # --- Imports (1) → Oil Products (added first so it stacks below crude oil) ---
    add_link(1, 9, oil_prod_imp,      IMPORTS_COLOR)   # → Oil Products

    # --- Crude Oil (8) → Oil Products (9) (refinery, added last so it stacks at top) ---
    add_link(8, 9, refinery_output, SOURCE_COLORS['Oil'])

    # --- Crude Oil (8) → Exports (18) ---
    add_link(8, 18, oil_crude_exp, SOURCE_COLORS['Oil'])

    # --- Crude Oil (8) → Losses (17) ---
    add_link(8, 17, oil_crude_loss, SOURCE_COLORS['Oil'])

    # --- Oil Products (9) → Electricity (10) ---
    add_link(9, 10, oil_elec, SOURCE_COLORS['Oil Products'])

    # --- Oil Products (9) → Exports (18) ---
    add_link(9, 18, oil_prod_exp, SOURCE_COLORS['Oil Products'])

    # --- Sources → Electricity (10) ---
    add_link(2, 10, renewables_elec,  SOURCE_COLORS['Renewables'])
    add_link(3, 10, bio_elec,         SOURCE_COLORS['Biofuels & Waste'])
    add_link(4, 10, coal_elec,        SOURCE_COLORS['Coal'])
    add_link(5, 10, nuclear_elec,     SOURCE_COLORS['Nuclear'])
    add_link(6, 10, elec_imp,         SOURCE_COLORS['Elec. Imports'])
    add_link(7, 10, gas_elec,         SOURCE_COLORS['Natural Gas'])

    # --- Sources & Oil Products → Direct consumption by sector ---
    sector_idx_map = {
        'Industry': 11, 'Non-residential buildings': 12, 'Residential': 13,
        'Transport': 14, 'Agriculture & Other': 15, 'Non-energy Use': 16,
    }
    for src_idx, sectors, color in [
        (4, coal_sectors, SOURCE_COLORS['Coal']),
        (9, oil_sectors, SOURCE_COLORS['Oil Products']),
        (7, gas_sectors, SOURCE_COLORS['Natural Gas']),
        (3, bio_sectors, SOURCE_COLORS['Biofuels & Waste']),
    ]:
        for sector_name, sector_id in sector_idx_map.items():
            add_link(src_idx, sector_id, sectors.get(sector_name, 0), color)

    # --- Sources → Exports (18) ---
    add_link(4, 18, coal_exp, SOURCE_COLORS['Coal'])
    add_link(7, 18, gas_exp,  SOURCE_COLORS['Natural Gas'])

    # --- Sources → Losses (17) ---
    add_link(2, 17, renewables_loss,  SOURCE_COLORS['Renewables'])
    add_link(3, 17, bio_loss,         SOURCE_COLORS['Biofuels & Waste'])
    add_link(4, 17, coal_loss,        SOURCE_COLORS['Coal'])
    add_link(5, 17, nuclear_loss,     SOURCE_COLORS['Nuclear'])
    add_link(7, 17, gas_loss,         SOURCE_COLORS['Natural Gas'])

    # --- Electricity (10) → Sectors ---
    for sector_name, sector_id in sector_idx_map.items():
        if sector_name != 'Non-energy Use':
            add_link(10, sector_id, elec_sectors.get(sector_name, 0), ELEC_COLOR)

    # --- Electricity → Exports ---
    add_link(10, 18, elec_exp, ELEC_COLOR)

    # --- Electricity → Losses (T&D) ---
    add_link(10, 17, elec_td_loss, ELEC_COLOR)

    # Build arrays
    sources = [r[0] for r in raw_links]
    targets = [r[1] for r in raw_links]
    values  = [r[2] for r in raw_links]  # PJ

    # Link colors: use source color with alpha
    def hex_to_rgba(hex_color, alpha=0.4):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    link_colors = [hex_to_rgba(r[3], 0.35) for r in raw_links]

    # Print summary
    total_supply = (coal_supply + oil_crude_supply + gas_supply + nuclear_prod +
                    renewables_supply + bio_prod + oil_prod_imp + elec_imp)
    total_exp = coal_exp + oil_crude_exp + oil_prod_exp + gas_exp + elec_exp
    total_losses = (coal_loss + oil_crude_loss + gas_loss + nuclear_loss +
                    renewables_loss + bio_loss + elec_td_loss)

    print(f"\n{'='*60}")
    print(f"  {region} Energy Balance — {year}")
    print(f"{'='*60}")
    print(f"  Total primary supply:      {total_supply * TJ_TO_PJ:>10,.0f} PJ")
    print(f"  Direct final consumption:  {(coal_total_cons + oil_total_cons + gas_total_cons + bio_total_cons) * TJ_TO_PJ:>10,.0f} PJ")
    print(f"  Electricity consumption:   {total_elec_output_sectors * TJ_TO_PJ:>10,.0f} PJ")
    print(f"  Total exports:             {total_exp * TJ_TO_PJ:>10,.0f} PJ")
    print(f"  Losses & transformation:   {total_losses * TJ_TO_PJ:>10,.0f} PJ")
    print(f"{'='*60}\n")

    return {
        'node_labels': node_labels,
        'node_colors': node_colors,
        'node_x': node_x,
        'node_y': node_y,
        'sources': sources,
        'targets': targets,
        'values': values,
        'link_colors': link_colors,
        'year': year,
        'region': region,
    }


# ============================================================================
# SIMPLIFIED SANKEY — Domestic / Imports → Electricity → Sectors
# ============================================================================

def compute_simple_sankey_data(year, region='EU-27'):
    """
    Simplified Sankey: Domestic & Imports → (Electricity) → final sectors.

    Nodes (in order):
      0  Domestic Production     (left)
      1  Imports                 (left)
      2  Electricity             (middle-top)
      3  Industry                (right)
      4  Non-residential buildings (right)
      5  Residential             (right)
      6  Transport               (right)
      7  Agriculture & Other     (right)
      8  Losses, Transformation  (right, includes non-energy use)
         & Non-energy Use
      9  Exports                 (right)
    """
    files = get_files(region)

    # Load all data
    prod = read_iea(files['production'])
    coal_trade = read_iea(files['coal_trade'])
    gas_trade = read_iea(files['gas_trade'])
    oil_trade = read_iea(files['oil_trade'])
    oil_prod_trade = read_iea(files['oil_prod_trade'])
    elec_trade = read_iea(files['elec_trade'])
    elec_gen = read_iea(files['elec_gen'])
    elec_cons = read_iea(files['elec_cons'])
    coal_cons = read_iea(files['coal_cons'])
    gas_cons = read_iea(files['gas_cons'])
    oil_cons = read_iea(files['oil_cons'])
    bio_cons = read_iea(files['bio_cons'])

    p = get_row(prod, year)
    ct = get_row(coal_trade, year)
    gt = get_row(gas_trade, year)
    ot = get_row(oil_trade, year)
    opt = get_row(oil_prod_trade, year)
    et = get_row(elec_trade, year)
    eg = get_row(elec_gen, year)
    ec = get_row(elec_cons, year)
    cc = get_row(coal_cons, year)
    gc = get_row(gas_cons, year)
    oc = get_row(oil_cons, year)
    bc = get_row(bio_cons, year)

    # ---- Supply ----
    coal_prod_v = p.get('Coal and coal products', 0)
    oil_prod_v = p.get('Primary oil', 0)
    gas_prod_v = p.get('Natural gas', 0)
    nuclear_prod_v = p.get('Nuclear', 0)
    hydro_prod = p.get('Hydropower', 0)
    solar_wind_prod = p.get('Solar, wind and other renewables', 0)
    bio_prod_v = p.get('Biofuels and waste', 0)

    coal_imp = ct.get('Imports', 0)
    gas_imp = gt.get('Imports', 0)
    oil_crude_imp = ot.get('Imports', 0)
    oil_prod_imp = opt.get('Imports', 0)
    elec_imp = et.get('Imports', 0)

    coal_exp = abs(ct.get('Exports', 0))
    gas_exp = abs(gt.get('Exports', 0))
    oil_crude_exp = abs(ot.get('Exports', 0))
    oil_prod_exp = abs(opt.get('Exports', 0))
    elec_exp = abs(et.get('Exports', 0))

    coal_supply = coal_prod_v + coal_imp
    oil_crude_supply = oil_prod_v + oil_crude_imp
    gas_supply = gas_prod_v + gas_imp
    renewables_supply = hydro_prod + solar_wind_prod

    # ---- Electricity generation (GWh → TJ) ----
    GWH_TO_TJ = 3.6
    coal_elec = eg.get('Coal', 0) * GWH_TO_TJ
    oil_elec = eg.get('Oil', 0) * GWH_TO_TJ
    gas_elec = eg.get('Natural gas', 0) * GWH_TO_TJ
    nuclear_elec = eg.get('Nuclear', 0) * GWH_TO_TJ
    hydro_elec = eg.get('Hydropower', 0) * GWH_TO_TJ
    solar_elec = eg.get('Solar PV', 0) * GWH_TO_TJ
    wind_elec = eg.get('Wind', 0) * GWH_TO_TJ
    geo_elec = eg.get('Geothermal', 0) * GWH_TO_TJ
    tide_elec = eg.get('Tide', 0) * GWH_TO_TJ
    solar_th_elec = eg.get('Solar thermal', 0) * GWH_TO_TJ
    other_elec = eg.get('Other sources', 0) * GWH_TO_TJ
    bio_gen_elec = eg.get('Biofuels', 0) * GWH_TO_TJ
    waste_gen_elec = eg.get('Waste', 0) * GWH_TO_TJ

    renewables_elec = hydro_elec + solar_elec + wind_elec + geo_elec + tide_elec + solar_th_elec + other_elec
    bio_elec = bio_gen_elec + waste_gen_elec

    # ---- Sector consumption (TJ) ----
    def sector_sums(d):
        return {
            'Industry': d.get('Industry', 0),
            'Transport': d.get('Transport', 0),
            'Residential': d.get('Residential', 0),
            'Buildings': d.get('Commercial and public services', 0),
            'Agriculture & Other': (d.get('Agriculture and forestry', 0) +
                                    d.get('Fishing', 0) +
                                    d.get('Other non-specified', 0)),
            'Non-energy Use': d.get('Non-energy use', 0),
        }

    coal_sectors = sector_sums(cc)
    oil_sectors = sector_sums(oc)
    gas_sectors = sector_sums(gc)
    bio_sectors = sector_sums(bc)
    elec_sectors = sector_sums(ec)
    elec_sectors['Non-energy Use'] = 0

    # ---- Oil refinery ----
    oil_total_cons = sum(oil_sectors.values())
    oil_products_total_out = oil_total_cons + oil_prod_exp + oil_elec
    refinery_output = max(0, oil_products_total_out - oil_prod_imp)
    oil_prod_supply = refinery_output + oil_prod_imp

    # ---- Losses ----
    coal_total_cons = sum(coal_sectors.values())
    gas_total_cons = sum(gas_sectors.values())
    bio_total_cons = sum(bio_sectors.values())

    coal_loss = max(0, coal_supply - coal_elec - coal_total_cons - coal_exp)
    gas_loss = max(0, gas_supply - gas_elec - gas_total_cons - gas_exp)
    oil_crude_loss = max(0, oil_crude_supply - oil_crude_exp - refinery_output)
    nuclear_loss = max(0, nuclear_prod_v - nuclear_elec)
    renewables_loss = max(0, renewables_supply - renewables_elec)
    bio_loss = max(0, bio_prod_v - bio_elec - bio_total_cons)

    total_elec_input = (coal_elec + oil_elec + gas_elec + nuclear_elec +
                        renewables_elec + bio_elec + elec_imp)
    total_elec_output_sectors = sum(elec_sectors.values())
    elec_td_loss = max(0, total_elec_input - total_elec_output_sectors - elec_exp)

    # ================================================================
    # Domestic / Import shares per fuel
    # ================================================================
    def safe_share(part, total):
        return part / total if total > 0 else 0

    coal_dom = safe_share(coal_prod_v, coal_supply)
    coal_imp_s = safe_share(coal_imp, coal_supply)
    gas_dom = safe_share(gas_prod_v, gas_supply)
    gas_imp_s = safe_share(gas_imp, gas_supply)
    oil_dom = safe_share(oil_prod_v, oil_crude_supply)
    oil_imp_s = safe_share(oil_crude_imp, oil_crude_supply)
    # Oil products: from domestic crude, imported crude, and product imports
    op_dom = safe_share(oil_dom * refinery_output, oil_prod_supply)
    op_imp = safe_share(oil_imp_s * refinery_output + oil_prod_imp, oil_prod_supply)
    # Nuclear = imported uranium, Renewables = domestic, Biofuels = domestic

    # ---- Domestic → Electricity ----
    dom_to_elec = (coal_dom * coal_elec + gas_dom * gas_elec +
                   oil_dom * oil_elec + renewables_elec + bio_elec)
    # ---- Imports → Electricity ----
    imp_to_elec = (coal_imp_s * coal_elec + gas_imp_s * gas_elec +
                   oil_imp_s * oil_elec + nuclear_elec + elec_imp)

    # ---- Direct consumption per sector (non-electric fuels) ----
    SECTORS = ['Industry', 'Non-residential buildings', 'Residential', 'Transport',
               'Agriculture & Other']
    dom_to_sector = {}
    imp_to_sector = {}
    for s in SECTORS:
        key = 'Buildings' if s == 'Non-residential buildings' else s
        dom_to_sector[s] = (coal_dom * coal_sectors[key] + gas_dom * gas_sectors[key] +
                            op_dom * oil_sectors[key] + bio_sectors[key])
        imp_to_sector[s] = (coal_imp_s * coal_sectors[key] + gas_imp_s * gas_sectors[key] +
                            op_imp * oil_sectors[key])

    # ---- Exports (non-electric) ----
    dom_to_exports = (coal_dom * coal_exp + gas_dom * gas_exp +
                      oil_dom * oil_crude_exp + op_dom * oil_prod_exp)
    imp_to_exports = (coal_imp_s * coal_exp + gas_imp_s * gas_exp +
                      oil_imp_s * oil_crude_exp + op_imp * oil_prod_exp)

    # ---- Losses + Non-energy use (non-electric) ----
    dom_non_energy = (coal_dom * coal_sectors['Non-energy Use'] + gas_dom * gas_sectors['Non-energy Use'] +
                      op_dom * oil_sectors['Non-energy Use'] + bio_sectors['Non-energy Use'])
    imp_non_energy = (coal_imp_s * coal_sectors['Non-energy Use'] + gas_imp_s * gas_sectors['Non-energy Use'] +
                      op_imp * oil_sectors['Non-energy Use'])
    dom_to_losses = (coal_dom * coal_loss + gas_dom * gas_loss +
                     oil_dom * oil_crude_loss + renewables_loss + bio_loss + dom_non_energy)
    imp_to_losses = (coal_imp_s * coal_loss + gas_imp_s * gas_loss +
                     oil_imp_s * oil_crude_loss + nuclear_loss + imp_non_energy)

    # ================================================================
    # BUILD NODES & LINKS
    # ================================================================
    TJ_TO_PJ = 1 / 1000

    domestic_total = (coal_prod_v + oil_prod_v + gas_prod_v +
                      renewables_supply + bio_prod_v)
    imports_total = (coal_imp + oil_crude_imp + oil_prod_imp + gas_imp +
                     nuclear_prod_v + elec_imp)
    total_supply = domestic_total + imports_total

    # Right-side node totals (direct fuel + electricity per sector)
    right_totals = {}
    for s in SECTORS:
        key = 'Buildings' if s == 'Non-residential buildings' else s
        right_totals[s] = dom_to_sector[s] + imp_to_sector[s] + elec_sectors.get(key, 0)
    right_totals['Losses'] = dom_to_losses + imp_to_losses + elec_td_loss
    right_totals['Exports'] = dom_to_exports + imp_to_exports + elec_exp

    def _pct(val):
        return val / total_supply * 100 if total_supply > 0 else 0

    node_labels = [
        f"Domestic Production\n({domestic_total * TJ_TO_PJ:,.0f} PJ — {_pct(domestic_total):.0f}%)",  # 0
        f"Imports\n({imports_total * TJ_TO_PJ:,.0f} PJ — {_pct(imports_total):.0f}%)",                # 1
        f"Electricity\n({total_elec_input * TJ_TO_PJ:,.0f} PJ — {_pct(total_elec_input):.0f}%)",      # 2
        f"Industry\n({right_totals['Industry'] * TJ_TO_PJ:,.0f} PJ — {_pct(right_totals['Industry']):.0f}%)",                                  # 3
        f"Non-residential\nbuildings\n({right_totals['Non-residential buildings'] * TJ_TO_PJ:,.0f} PJ — {_pct(right_totals['Non-residential buildings']):.0f}%)",  # 4
        f"Residential\n({right_totals['Residential'] * TJ_TO_PJ:,.0f} PJ — {_pct(right_totals['Residential']):.0f}%)",                          # 5
        f"Transport\n({right_totals['Transport'] * TJ_TO_PJ:,.0f} PJ — {_pct(right_totals['Transport']):.0f}%)",                                # 6
        f"Agriculture\n& Other\n({right_totals['Agriculture & Other'] * TJ_TO_PJ:,.0f} PJ — {_pct(right_totals['Agriculture & Other']):.0f}%)",  # 7
        f"Losses, Transformation\n& Non-energy Use\n({right_totals['Losses'] * TJ_TO_PJ:,.0f} PJ — {_pct(right_totals['Losses']):.0f}%)",        # 8
        f"Exports\n({right_totals['Exports'] * TJ_TO_PJ:,.0f} PJ — {_pct(right_totals['Exports']):.0f}%)",                                      # 9
    ]

    node_colors = [
        DOMESTIC_COLOR,                              # 0
        IMPORTS_COLOR,                               # 1
        ELEC_COLOR,                                  # 2
        SECTOR_COLORS['Industry'],                   # 3
        SECTOR_COLORS['Non-residential buildings'],  # 4
        SECTOR_COLORS['Residential'],                # 5
        SECTOR_COLORS['Transport'],                  # 6
        SECTOR_COLORS['Agriculture & Other'],        # 7
        SECTOR_COLORS['Losses & Transformation'],    # 8
        SECTOR_COLORS['Exports'],                    # 9
    ]

    X_LEFT  = 0.01
    X_MID   = 0.45
    X_RIGHT = 0.99

    node_x = [X_LEFT, X_LEFT,
              X_MID,
              X_RIGHT, X_RIGHT, X_RIGHT, X_RIGHT,
              X_RIGHT, X_RIGHT, X_RIGHT]
    node_y = [0.015, 0.219,
              0.015,
              0.015, 0.138, 0.206, 0.326,
              0.477, 0.506, 0.790]

    # ---- Links ----
    MIN_FLOW_TJ = 10_000
    raw_links = []

    def add_link(src, tgt, val_tj, color_hex):
        if val_tj > MIN_FLOW_TJ:
            raw_links.append((src, tgt, val_tj * TJ_TO_PJ, color_hex))

    sector_idx = {'Industry': 3, 'Non-residential buildings': 4, 'Residential': 5,
                  'Transport': 6, 'Agriculture & Other': 7}

    # Domestic (0) → Electricity (2)
    add_link(0, 2, dom_to_elec, DOMESTIC_COLOR)
    # Imports (1) → Electricity (2)
    add_link(1, 2, imp_to_elec, IMPORTS_COLOR)
    # Domestic (0) → Sectors direct
    for s, idx in sector_idx.items():
        add_link(0, idx, dom_to_sector[s], DOMESTIC_COLOR)
    # Imports (1) → Sectors direct
    for s, idx in sector_idx.items():
        add_link(1, idx, imp_to_sector[s], IMPORTS_COLOR)
    # Domestic (0) → Losses (8)
    add_link(0, 8, dom_to_losses, DOMESTIC_COLOR)
    # Imports (1) → Losses (8)
    add_link(1, 8, imp_to_losses, IMPORTS_COLOR)
    # Domestic (0) → Exports (9)
    add_link(0, 9, dom_to_exports, DOMESTIC_COLOR)
    # Imports (1) → Exports (9)
    add_link(1, 9, imp_to_exports, IMPORTS_COLOR)
    # Electricity (2) → Sectors
    for s, idx in sector_idx.items():
        add_link(2, idx, elec_sectors.get(
            'Buildings' if s == 'Non-residential buildings' else s, 0), ELEC_COLOR)
    # Electricity (2) → Exports (9)
    add_link(2, 9, elec_exp, ELEC_COLOR)
    # Electricity (2) → Losses (8)
    add_link(2, 8, elec_td_loss, ELEC_COLOR)

    # Build arrays
    sources = [r[0] for r in raw_links]
    targets = [r[1] for r in raw_links]
    values  = [r[2] for r in raw_links]

    def hex_to_rgba(hex_color, alpha=0.4):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    link_colors = [hex_to_rgba(r[3], 0.35) for r in raw_links]

    return {
        'node_labels': node_labels,
        'node_colors': node_colors,
        'node_x': node_x,
        'node_y': node_y,
        'sources': sources,
        'targets': targets,
        'values': values,
        'link_colors': link_colors,
        'year': year,
        'region': region,
        'simple': True,
    }


def create_simple_sankey(data):
    """Render the simplified Sankey diagram using matplotlib for pixel-perfect control."""
    year = data['year']
    region = data['region']
    file_prefix = REGIONS[region]['file_prefix']

    # ---- Geometry constants (in figure coordinates) ----
    FIG_W, FIG_H = 16, 11  # inches
    X_LEFT = 0.06       # left column center
    X_MID = 0.45        # electricity column center
    X_RIGHT = 0.88      # right column center
    NODE_W = 0.04       # node rectangle half-width (full width = 2*NODE_W)
    GAP = 0.008         # vertical gap between right-side nodes
    TOP_MARGIN = 0.06   # top margin for title
    BOT_MARGIN = 0.02   # bottom margin

    # ---- Collect total flows per node ----
    n_nodes = len(data['node_labels'])
    inflow = [0.0] * n_nodes
    outflow = [0.0] * n_nodes
    for s, t, v in zip(data['sources'], data['targets'], data['values']):
        outflow[s] += v
        inflow[t] += v
    node_size = [max(inflow[i], outflow[i]) for i in range(n_nodes)]

    # Scale: the right side has 7 nodes (indices 3-9) that must fit in the available space
    right_indices = [3, 4, 5, 6, 7, 8, 9]
    right_total = sum(node_size[i] for i in right_indices)
    usable_height = 1.0 - TOP_MARGIN - BOT_MARGIN - (len(right_indices) - 1) * GAP
    scale = usable_height / right_total  # PJ -> figure-y units

    # ---- Compute node rectangles: {node_idx: (x_left, y_bottom, width, height)} ----
    node_rects = {}

    # Right column: stack from top
    y_cursor = 1.0 - TOP_MARGIN
    for idx in right_indices:
        h = node_size[idx] * scale
        y_bottom = y_cursor - h
        node_rects[idx] = (X_RIGHT - NODE_W, y_bottom, 2 * NODE_W, h)
        y_cursor = y_bottom - GAP

    # Left column: Domestic (0) + Imports (1) — same total as right side
    left_indices = [0, 1]
    left_total = sum(node_size[i] for i in left_indices)
    left_scale = usable_height / left_total if left_total > 0 else scale
    y_cursor = 1.0 - TOP_MARGIN
    for idx in left_indices:
        h = node_size[idx] * left_scale
        y_bottom = y_cursor - h
        node_rects[idx] = (X_LEFT - NODE_W, y_bottom, 2 * NODE_W, h)
        y_cursor = y_bottom - GAP

    # Middle column: Electricity (2) — place at the top
    elec_h = node_size[2] * scale
    elec_y_top = 1.0 - TOP_MARGIN
    node_rects[2] = (X_MID - NODE_W, elec_y_top - elec_h, 2 * NODE_W, elec_h)

    # ---- Helper: draw a flow band (filled Bezier) between two vertical spans ----
    def draw_flow(ax, x0, y0_bot, y0_top, x1, y1_bot, y1_top, color, alpha=0.35):
        """Draw a curved flow band from (x0, y0_bot..y0_top) to (x1, y1_bot..y1_top)."""
        dx = (x1 - x0) * 0.45
        verts = [
            (x0, y0_bot),
            (x0 + dx, y0_bot),
            (x1 - dx, y1_bot),
            (x1, y1_bot),
            (x1, y1_top),
            (x1 - dx, y1_top),
            (x0 + dx, y0_top),
            (x0, y0_top),
            (x0, y0_bot),
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.LINETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha)
        ax.add_patch(patch)

    # ---- Create figure ----
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ---- Draw nodes ----
    for idx in range(n_nodes):
        rx, ry, rw, rh = node_rects[idx]
        rect = mpatches.FancyBboxPatch(
            (rx, ry), rw, rh,
            boxstyle="round,pad=0.003",
            facecolor=data['node_colors'][idx],
            edgecolor='#333333',
            linewidth=0.8,
        )
        ax.add_patch(rect)

    # ---- Draw node labels ----
    label_fontsize = 9
    for idx in range(n_nodes):
        rx, ry, rw, rh = node_rects[idx]
        label = data['node_labels'][idx]
        cy = ry + rh / 2

        if idx in [0, 1]:  # Left column: label to the right of the node
            ax.text(rx + rw + 0.008, cy, label, va='center', ha='left',
                    fontsize=label_fontsize, fontfamily='Arial', color='#2C3E50')
        elif idx == 2:  # Electricity: label to right
            ax.text(rx + rw + 0.008, cy, label, va='center', ha='left',
                    fontsize=label_fontsize, fontfamily='Arial', color='#2C3E50')
        else:  # Right column: label to the left
            ax.text(rx - 0.008, cy, label, va='center', ha='right',
                    fontsize=label_fontsize, fontfamily='Arial', color='#2C3E50')

    # ---- Draw flows ----
    # Track the current "cursor" on each side of each node for stacking flows
    node_out_cursor = {}   # outflow cursor (right edge of source node)
    node_in_cursor = {}    # inflow cursor (left edge of target node)
    for idx in range(n_nodes):
        rx, ry, rw, rh = node_rects[idx]
        node_out_cursor[idx] = ry + rh  # start from top
        node_in_cursor[idx] = ry + rh   # start from top

    # Sort links: for each source, order targets by their vertical position (top-first)
    # This minimizes flow crossings
    def target_center(li):
        tgt = data['targets'][li]
        ry = node_rects[tgt][1]
        rh = node_rects[tgt][3]
        return -(ry + rh / 2)  # negative because we stack top-down

    def source_center(li):
        src = data['sources'][li]
        ry = node_rects[src][1]
        rh = node_rects[src][3]
        return -(ry + rh / 2)

    # First pass: allocate outflow cursors (group by source, sort targets top-to-bottom)
    from collections import defaultdict
    links_by_source = defaultdict(list)
    links_by_target = defaultdict(list)
    for li in range(len(data['sources'])):
        links_by_source[data['sources'][li]].append(li)
        links_by_target[data['targets'][li]].append(li)

    # For each source, sort its outgoing links by target vertical center (top first)
    out_alloc = {}  # li -> (src_top, src_bot)
    for src_idx in sorted(links_by_source.keys()):
        lis = sorted(links_by_source[src_idx], key=target_center)
        cursor = node_rects[src_idx][1] + node_rects[src_idx][3]  # top of node
        for li in lis:
            val = data['values'][li]
            h = (val / node_size[src_idx]) * node_rects[src_idx][3] if node_size[src_idx] > 0 else 0
            out_alloc[li] = (cursor, cursor - h)
            cursor -= h

    # For each target, sort its incoming links by source vertical center (top first)
    in_alloc = {}  # li -> (tgt_top, tgt_bot)
    for tgt_idx in sorted(links_by_target.keys()):
        lis = sorted(links_by_target[tgt_idx], key=source_center)
        cursor = node_rects[tgt_idx][1] + node_rects[tgt_idx][3]  # top of node
        for li in lis:
            val = data['values'][li]
            h = (val / node_size[tgt_idx]) * node_rects[tgt_idx][3] if node_size[tgt_idx] > 0 else 0
            in_alloc[li] = (cursor, cursor - h)
            cursor -= h

    # Draw all flows
    for li in range(len(data['sources'])):
        src = data['sources'][li]
        tgt = data['targets'][li]

        src_rect = node_rects[src]
        tgt_rect = node_rects[tgt]

        src_x = src_rect[0] + src_rect[2]  # right edge of source
        tgt_x = tgt_rect[0]                # left edge of target

        src_top, src_bot = out_alloc[li]
        tgt_top, tgt_bot = in_alloc[li]

        # Color
        raw_color = data['link_colors'][li]
        if raw_color.startswith('rgba('):
            parts = raw_color.replace('rgba(', '').replace(')', '').split(',')
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
        else:
            hex_color = raw_color

        draw_flow(ax, src_x, src_bot, src_top, tgt_x, tgt_bot, tgt_top, hex_color, alpha=0.35)

    # ---- Title ----
    fig.text(0.5, 0.97,
             f"{region} Energy Flows — Simplified ({year})",
             ha='center', va='top', fontsize=18, fontfamily='Arial',
             fontweight='bold', color='#2C3E50')
    fig.text(0.5, 0.945,
             "Source: IEA World Energy Balances — Values in PJ (petajoules)",
             ha='center', va='top', fontsize=11, fontfamily='Arial', color='#666666')

    # ---- Save ----
    png_path = os.path.join(OUTPUT_DIR, f'{file_prefix}_energy_sankey_simple_{year}.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {png_path}")

    svg_path = os.path.join(OUTPUT_DIR, f'{file_prefix}_energy_sankey_simple_{year}.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {svg_path}")

    plt.close(fig)

    export_sankey_data({**data, 'simple': True})


# ============================================================================
# PNG EXPORT VIA EDGE HEADLESS
# ============================================================================

def _export_png_via_edge(html_path, png_path, width=1600, height=900):
    """Use Edge in headless mode to screenshot the HTML Sankey as PNG."""
    import subprocess
    import shutil

    # Find Edge
    edge_candidates = [
        shutil.which('msedge'),
        r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    ]
    edge_path = None
    for p in edge_candidates:
        if p and os.path.exists(p):
            edge_path = p
            break

    if not edge_path:
        print("  PNG export skipped: Microsoft Edge not found.")
        return

    file_url = 'file:///' + html_path.replace('\\', '/')
    try:
        result = subprocess.run(
            [
                edge_path,
                '--headless',
                '--disable-gpu',
                '--no-sandbox',
                '--hide-scrollbars',
                f'--screenshot={png_path}',
                f'--window-size={width},{height}',
                file_url,
            ],
            capture_output=True,
            timeout=30,
        )
        if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
            print(f"[SAVED] {png_path}")
        else:
            stderr = result.stderr.decode(errors='replace').strip()
            print(f"  PNG export failed via Edge. stderr: {stderr[:200]}")
    except Exception as e:
        print(f"  PNG export failed via Edge: {e}")


# ============================================================================
# PLOT
# ============================================================================

def create_sankey(data):
    """Render the detailed Sankey diagram using matplotlib."""
    from collections import defaultdict

    year = data['year']
    region = data['region']
    file_prefix = REGIONS[region]['file_prefix']

    # ---- Geometry ----
    FIG_W, FIG_H = 22, 14
    TOP_MARGIN = 0.055
    BOT_MARGIN = 0.015
    GAP = 0.006
    NODE_W = 0.018

    # 5 columns: Origin, Sources, Refinery, Electricity, Sinks
    col_x = {
        'origin': 0.045,
        'source': 0.20,
        'refinery': 0.42,
        'elec': 0.58,
        'sink': 0.92,
    }

    # ---- Node sizes ----
    n_nodes = len(data['node_labels'])
    inflow = [0.0] * n_nodes
    outflow = [0.0] * n_nodes
    for s, t, v in zip(data['sources'], data['targets'], data['values']):
        outflow[s] += v
        inflow[t] += v
    node_size = [max(inflow[i], outflow[i]) for i in range(n_nodes)]

    # Column assignments (from compute_sankey_data):
    #   0,1 = origin; 2-8 = sources; 9 = refinery; 10 = electricity; 11-18 = sinks
    origin_ids = [0, 1]
    source_ids = [2, 3, 4, 5, 6, 7, 8]
    refinery_ids = [9]
    elec_ids = [10]
    sink_ids = [11, 12, 13, 14, 15, 16, 17, 18]

    # Use the sinks column (most nodes) to set the scale
    sink_total = sum(node_size[i] for i in sink_ids)
    usable = 1.0 - TOP_MARGIN - BOT_MARGIN
    sink_gaps = (len(sink_ids) - 1) * GAP
    scale = (usable - sink_gaps) / sink_total if sink_total > 0 else 1.0

    node_rects = {}

    def stack_column(ids, x_center):
        n_gaps = max(len(ids) - 1, 0)
        total_h = sum(node_size[i] * scale for i in ids)
        total_needed = total_h + n_gaps * GAP
        col_scale = scale
        if total_needed > usable:
            col_scale = (usable - n_gaps * GAP) / sum(node_size[i] for i in ids) if sum(node_size[i] for i in ids) > 0 else scale
        y_cursor = 1.0 - TOP_MARGIN
        for idx in ids:
            h = node_size[idx] * col_scale
            node_rects[idx] = (x_center - NODE_W, y_cursor - h, 2 * NODE_W, h)
            y_cursor -= h + GAP

    stack_column(sink_ids, col_x['sink'])
    stack_column(source_ids, col_x['source'])
    stack_column(origin_ids, col_x['origin'])

    # Refinery: position near crude oil (8)
    ref_h = node_size[9] * scale
    if 8 in node_rects:
        ref_center = node_rects[8][1] + node_rects[8][3] / 2
    else:
        ref_center = 0.5
    ref_y = max(BOT_MARGIN, min(1.0 - TOP_MARGIN - ref_h, ref_center - ref_h / 2))
    node_rects[9] = (col_x['refinery'] - NODE_W, ref_y, 2 * NODE_W, ref_h)

    # Electricity: position at the top
    elec_h = node_size[10] * scale
    elec_y_top = 1.0 - TOP_MARGIN
    node_rects[10] = (col_x['elec'] - NODE_W, elec_y_top - elec_h, 2 * NODE_W, elec_h)

    # ---- Bezier flow helper ----
    def draw_flow(ax, x0, y0_bot, y0_top, x1, y1_bot, y1_top, color, alpha=0.30):
        dx = (x1 - x0) * 0.45
        verts = [
            (x0, y0_bot), (x0 + dx, y0_bot), (x1 - dx, y1_bot), (x1, y1_bot),
            (x1, y1_top), (x1 - dx, y1_top), (x0 + dx, y0_top), (x0, y0_top),
            (x0, y0_bot),
        ]
        codes = [
            Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.CLOSEPOLY,
        ]
        patch = mpatches.PathPatch(Path(verts, codes), facecolor=color, edgecolor='none', alpha=alpha)
        ax.add_patch(patch)

    # ---- Create figure ----
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ---- Draw nodes ----
    for idx in range(n_nodes):
        if idx not in node_rects:
            continue
        rx, ry, rw, rh = node_rects[idx]
        rect = mpatches.FancyBboxPatch(
            (rx, ry), rw, rh,
            boxstyle="round,pad=0.002",
            facecolor=data['node_colors'][idx],
            edgecolor='#333333', linewidth=0.6,
        )
        ax.add_patch(rect)

    # ---- Labels ----
    fs = 7.5
    for idx in range(n_nodes):
        if idx not in node_rects:
            continue
        rx, ry, rw, rh = node_rects[idx]
        cy = ry + rh / 2
        label = data['node_labels'][idx]
        if idx in origin_ids:
            ax.text(rx + rw + 0.006, cy, label, va='center', ha='left',
                    fontsize=fs, fontfamily='Arial', color='#2C3E50')
        elif idx in sink_ids:
            ax.text(rx - 0.006, cy, label, va='center', ha='right',
                    fontsize=fs, fontfamily='Arial', color='#2C3E50')
        else:
            ax.text(rx + rw + 0.006, cy, label, va='center', ha='left',
                    fontsize=fs, fontfamily='Arial', color='#2C3E50')

    # ---- Flow allocation ----
    links_by_source = defaultdict(list)
    links_by_target = defaultdict(list)
    for li in range(len(data['sources'])):
        links_by_source[data['sources'][li]].append(li)
        links_by_target[data['targets'][li]].append(li)

    def tgt_y(li):
        t = data['targets'][li]
        return -(node_rects[t][1] + node_rects[t][3] / 2) if t in node_rects else 0

    def src_y(li):
        s = data['sources'][li]
        return -(node_rects[s][1] + node_rects[s][3] / 2) if s in node_rects else 0

    out_alloc = {}
    for src_idx in sorted(links_by_source.keys()):
        lis = sorted(links_by_source[src_idx], key=tgt_y)
        r = node_rects[src_idx]
        cursor = r[1] + r[3]
        for li in lis:
            h = (data['values'][li] / node_size[src_idx]) * r[3] if node_size[src_idx] > 0 else 0
            out_alloc[li] = (cursor, cursor - h)
            cursor -= h

    in_alloc = {}
    for tgt_idx in sorted(links_by_target.keys()):
        lis = sorted(links_by_target[tgt_idx], key=src_y)
        r = node_rects[tgt_idx]
        cursor = r[1] + r[3]
        for li in lis:
            h = (data['values'][li] / node_size[tgt_idx]) * r[3] if node_size[tgt_idx] > 0 else 0
            in_alloc[li] = (cursor, cursor - h)
            cursor -= h

    # ---- Draw flows ----
    for li in range(len(data['sources'])):
        src = data['sources'][li]
        tgt = data['targets'][li]
        if src not in node_rects or tgt not in node_rects:
            continue
        src_x = node_rects[src][0] + node_rects[src][2]
        tgt_x = node_rects[tgt][0]
        src_top, src_bot = out_alloc[li]
        tgt_top, tgt_bot = in_alloc[li]

        raw_color = data['link_colors'][li]
        if raw_color.startswith('rgba('):
            parts = raw_color.replace('rgba(', '').replace(')', '').split(',')
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
        else:
            hex_color = raw_color
        draw_flow(ax, src_x, src_bot, src_top, tgt_x, tgt_bot, tgt_top, hex_color, alpha=0.30)

    # ---- Title ----
    fig.text(0.5, 0.975,
             f"{region} Energy Flows ({year})",
             ha='center', va='top', fontsize=18, fontfamily='Arial',
             fontweight='bold', color='#2C3E50')
    fig.text(0.5, 0.955,
             "Source: IEA World Energy Balances — Values in PJ (petajoules)",
             ha='center', va='top', fontsize=11, fontfamily='Arial', color='#666666')

    # ---- Save ----
    png_path = os.path.join(OUTPUT_DIR, f'{file_prefix}_energy_sankey_{year}.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {png_path}")

    svg_path = os.path.join(OUTPUT_DIR, f'{file_prefix}_energy_sankey_{year}.svg')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {svg_path}")

    plt.close(fig)

    export_sankey_data(data)


def export_sankey_data(data):
    """Export the Sankey link data to Excel."""
    import openpyxl

    file_prefix = REGIONS[data['region']]['file_prefix']

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"Energy Flows {data['year']}"
    ws.append(['Source', 'Target', 'Flow (PJ)'])
    for src, tgt, val in zip(data['sources'], data['targets'], data['values']):
        # Use clean label (first line only)
        src_label = data['node_labels'][src].split('\n')[0]
        tgt_label = data['node_labels'][tgt].split('\n')[0]
        ws.append([src_label, tgt_label, round(val, 1)])

    suffix = '_simple' if data.get('simple') else ''
    xlsx_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_energy_sankey{suffix}_{data['year']}.xlsx")
    wb.save(xlsx_path)
    print(f"[SAVED] {xlsx_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    for region in REGIONS:
        print("=" * 60)
        print(f"IEA Energy Sankey — {region}")
        print("=" * 60)

        files = get_files(region)
        # Find latest common year across all files
        all_dfs = {name: read_iea(files[name]) for name in files}
        max_years = {name: df['Year'].max() for name, df in all_dfs.items()}
        latest_common = min(max_years.values())
        print(f"\nLatest year per file:")
        for name, yr in max_years.items():
            print(f"  {name}: {yr}")
        print(f"\nUsing latest common year: {latest_common}")

        # Generate for latest year and 2023 (if data available)
        years_to_run = sorted(set([latest_common, 2023]))
        for yr in years_to_run:
            print(f"\n--- Year {yr} ---")
            data = compute_sankey_data(yr, region)
            create_sankey(data)

            simple_data = compute_simple_sankey_data(yr, region)
            create_simple_sankey(simple_data)

        print("\nDone!\n")


if __name__ == "__main__":
    main()
