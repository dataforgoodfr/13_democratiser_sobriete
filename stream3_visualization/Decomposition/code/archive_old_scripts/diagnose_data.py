import pandas as pd

df = pd.read_excel('data/2025-12-15_FR scenarios data_before computation.xlsx')

print('=== DEMOGRAPHY DATA ===')
demo = df[df['Sector'] == 'Demography']
print(f'Total rows: {len(demo)}')
print()

print('By Scenario:')
for scenario in sorted(demo['Scenario'].unique()):
    subset = demo[demo['Scenario'] == scenario]
    print(f'  {scenario}: {len(subset)} rows')
    years_list = sorted(subset['Year'].unique())
    print(f'    Years: {years_list}')

print()
print('=== ALL SECTORS ===')
for sector in sorted(df['Sector'].unique()):
    count = len(df[df['Sector'] == sector])
    print(f'  {sector}: {count} rows')
    scenarios = df[df['Sector'] == sector]['Scenario'].unique()
    print(f'    Scenarios: {sorted(scenarios)}')

print()
print('=== DEMOGRAPHY DETAILS ===')
demo_grouped = demo.groupby(['Sector', 'Type', 'Variable', 'Scenario']).size()
print(demo_grouped)
