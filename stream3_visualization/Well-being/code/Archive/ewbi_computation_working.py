import json
import numpy as np
import pandas as pd

# Load the preprocessed data
print("Loading primary indicator data...")
df = pd.read_csv('../output/primary_data_preprocessed.csv')
df = df.set_index(['country', 'primary_index', 'decile'])

print(f"Initial data shape: {df.shape}")

# Filter out economic good indicators (only keep satisfiers)
economic_indicators_to_remove = [
    'AN-SILC-1',
    'AE-HBS-1', 'AE-HBS-2',
    'HQ-SILC-2',
    'HH-SILC-1', 'HH-HBS-1', 'HH-HBS-2', 'HH-HBS-3', 'HH-HBS-4',
    'EC-HBS-1', 'EC-HBS-2',
    'ED-ICT-1', 'ED-EHIS-1',
    'AC-SILC-1', 'AC-SILC-2', 'AC-HBS-1', 'AC-HBS-2', 'AC-EHIS-1',
    'IE-HBS-1', 'IE-HBS-2',
    'IC-SILC-1', 'IC-SILC-2', 'IC-HBS-1', 'IC-HBS-2',
    'TT-SILC-1', 'TT-SILC-2', 'TT-HBS-1', 'TT-HBS-2',
    'TS-SILC-1', 'TS-HBS-1', 'TS-HBS-2'
]

print(f"Filtering out {len(economic_indicators_to_remove)} economic indicators")

# Remove economic indicators
df_filtered = df[~df.index.get_level_values('primary_index').isin(economic_indicators_to_remove)]

print(f"After filtering: {df_filtered.shape}")
print(f"Removed {df.shape[0] - df_filtered.shape[0]} rows")

# Use filtered data for the rest of the computation
df = df_filtered

def simple_average(data: list[tuple[pd.Series | pd.DataFrame, float]]):
    # Simple straight average of all indicators (ignoring weights)
    all_values = []
    for values, weight in data:
        all_values.append(values)
    
    # Concatenate all series and calculate mean
    if all_values:
        combined = pd.concat(all_values, axis=1)
        return combined.mean(axis=1)
    else:
        return pd.Series(dtype=float)

def weighted_geometric_mean(data: list[tuple[pd.Series | pd.DataFrame, float]]):
    # Weighted geometric mean
    all_values = []
    all_weights = []
    
    for values, weight in data:
        all_values.append(values)
        all_weights.append(weight)
    
    if all_values:
        # Calculate weighted geometric mean: exp(sum(weight * log(value)) / sum(weights))
        log_values = [np.log(v) for v in all_values]
        weighted_log_sum = sum(w * log_val for w, log_val in zip(all_weights, log_values))
        total_weight = sum(all_weights)
        
        if total_weight > 0:
            return np.exp(weighted_log_sum / total_weight)
        else:
            return pd.Series(dtype=float)
    else:
        return pd.Series(dtype=float)

# Load EWBI structure
with open('../data/ewbi_indicators.json') as f:
    config = json.load(f)['EWBI']

print(f"EWBI structure has {len(config)} EU priorities")

# Calculate secondary indicators using simple averages
print("Calculating secondary indicators...")
secondary = {}
missing = {}

# separate countries as indicators aren't all available for all countries
for country, cdf in df.groupby('country'):
    cdf = cdf.loc[country]
    for prio in config:
        for component in prio['components']:
            factors = []
            for ind in component['indicators']:
                code = ind['code']
                if code in cdf.index:
                    factors.append((cdf.loc[code], 1))  # weight set to 1 since we ignore it
                elif code not in {'IS-SILC-2', 'IS-SILC-1', 'RU-LFS-1'}:
                   print(f"{country},{code}")
            if factors:
                secondary[country, prio['name'], component['name']] = simple_average(factors)
            else:
                #print('Missing', country, component['name'])
                pass

secondary = pd.concat(secondary, names=('country', 'eu_priority', 'secondary_indicator'))
print(f"Created {len(secondary)} secondary indicator scores")

# Save secondary indicators
secondary.to_csv('../output/secondary_indicators.csv')
print("Saved secondary indicators")

# Calculate EU priorities using weighted geometric mean
print("Calculating EU priorities...")
priorities = {}
for country, cdf in secondary.groupby('country'):
    cdf = cdf.loc[country]
    for prio in config:
        pname = prio['name']
        if pname in cdf.index:
            cpdf = cdf.loc[pname]
            factors = []
            for c in prio['components']:
                name = c['name']
                weight = c['weight']
                try:
                    weight = float(weight)
                except ValueError:
                    numerator, denominator = map(int, weight.split('/'))
                    weight = float(numerator) / denominator

                if name in cpdf.index and weight != 0:
                    factors.append((cpdf.loc[name], weight))

            if factors:
                priorities[country, pname] = weighted_geometric_mean(factors)
            else:
                print('Missing', country, pname)                

priorities = pd.concat(priorities, names=['country', 'eu_priority'])
print(f"Created {len(priorities)} EU priority scores")

# Save EU priorities
priorities.to_csv('../output/eu_priorities.csv')
print("Saved EU priorities")

# Calculate EWBI scores
print("Calculating EWBI scores...")
ewbi = {}
for country, cdf in priorities.groupby('country'):
    cdf = cdf.loc[country]
    factors = [(cdf.loc[prio], 1)  for prio in cdf.index.get_level_values('eu_priority')]
    ewbi[country] = simple_average(factors)
    
ewbi = pd.concat(ewbi, names=['country'])
print(f"Created {len(ewbi)} EWBI scores")

# Save EWBI results
ewbi.to_csv('../output/ewbi_results.csv')
print("Saved EWBI results")

print("=== Computation Complete ===")
print(f"Secondary indicators: {len(secondary)} scores")
print(f"EU priorities: {len(priorities)} scores")
print(f"EWBI: {len(ewbi)} scores") 