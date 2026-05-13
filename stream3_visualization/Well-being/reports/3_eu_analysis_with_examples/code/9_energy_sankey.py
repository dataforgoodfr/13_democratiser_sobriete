"""
9_energy_sankey.py — Energy Sankey diagrams for 4 reports.

Produces 2 visuals per report (detailed + simplified Sankey) using IEA data.

Reports:
  rep_eu   : EU-27 data
  rep_ewbi : EU-27 data (identical)
  rep_fr   : France data
  rep_ch   : Switzerland data

Reuses computation & rendering from iea_sankey.py.
All outputs: PNG + SVG + Excel.
"""

import os
import sys

# Ensure this script's directory is on the path so we can import iea_sankey
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import iea_sankey

# ---------------------------------------------------------------------------
# Report configurations: report key → IEA region name
# ---------------------------------------------------------------------------
REPORT_CONFIGS = {
    'rep_eu': {
        'prefix': 'rep_eu',
        'title_suffix': '(EU-27)',
        'region': 'EU-27',
    },
    'rep_ewbi': {
        'prefix': 'rep_ewbi',
        'title_suffix': '(EU-27 + EFTA)',
        'region': 'EU-27',
    },
    'rep_fr': {
        'prefix': 'rep_fr',
        'title_suffix': '(France)',
        'region': 'France',
    },
    'rep_ch': {
        'prefix': 'rep_ch',
        'title_suffix': '(Switzerland)',
        'region': 'Switzerland',
    },
}

BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
OUTPUT_BASE = os.path.join(BASE_DIR, 'outputs', 'graphs', 'Energy_Sankey')


def generate_report(report_key):
    cfg = REPORT_CONFIGS[report_key]
    region = cfg['region']

    print(f"\n{'='*60}")
    print(f"Generating energy sankey: {cfg['prefix']} {cfg['title_suffix']}")
    print(f"  Region: {region}")
    print(f"{'='*60}")

    out_dir = os.path.join(OUTPUT_BASE, cfg['prefix'])
    os.makedirs(out_dir, exist_ok=True)

    # Point iea_sankey's output to our report-specific folder
    iea_sankey.OUTPUT_DIR = out_dir

    # Find latest common year across all IEA files for this region
    files = iea_sankey.get_files(region)
    all_dfs = {name: iea_sankey.read_iea(files[name]) for name in files}
    max_years = {name: df['Year'].max() for name, df in all_dfs.items()}
    latest_common = min(max_years.values())
    print(f"  Latest common year: {latest_common}")

    years_to_run = sorted(set([latest_common, 2023]))
    for yr in years_to_run:
        print(f"\n  --- Year {yr} ---")

        # Detailed Sankey
        print(f"  [1] Detailed Sankey...")
        data = iea_sankey.compute_sankey_data(yr, region)
        iea_sankey.create_sankey(data)

        # Simplified Sankey
        print(f"  [2] Simplified Sankey...")
        simple_data = iea_sankey.compute_simple_sankey_data(yr, region)
        iea_sankey.create_simple_sankey(simple_data)

    print(f"\n  All outputs saved to: {out_dir}")


def main():
    for report_key in ['rep_eu', 'rep_ewbi', 'rep_fr', 'rep_ch']:
        generate_report(report_key)
    print("\nDone.")


if __name__ == '__main__':
    main()
