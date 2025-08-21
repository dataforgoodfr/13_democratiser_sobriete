#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import pandas as pd


# In[16]:


df = pd.read_csv('../output/primary_data_preprocessed.csv')
df = df.set_index(['country', 'primary_index', 'decile'])


# In[17]:


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
print(f"Initial data shape: {df.shape}")

# Remove economic indicators
df_filtered = df[~df.index.get_level_values('primary_index').isin(economic_indicators_to_remove)]

print(f"After filtering: {df_filtered.shape}")
print(f"Removed {df.shape[0] - df_filtered.shape[0]} rows")

# Use filtered data for the rest of the computation
df = df_filtered


# In[18]:


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


# In[23]:


with open('../data/ewbi_indicators.json') as f:
    config = json.load(f)['EWBI']


# In[24]:


print("Present in json file but not in index:", all_codes.difference(df.index.get_level_values('primary_index')))
print("Present in index but not in json file:", df.index.get_level_values('primary_index').difference(all_codes))


# In[27]:


# Calculate secondary indicators using simple averages
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


# In[30]:


secondary.to_csv('../output/secondary_indicators.csv')


# In[31]:


print(all_secondaries.difference(secondary.index.get_level_values('secondary_indicator')))
print(secondary.index.get_level_values('secondary_indicator').difference(all_secondaries))


# In[33]:


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


# In[34]:


priorities.to_csv('../output/eu_priorities.csv')


# In[35]:


ewbi = {}
for country, cdf in priorities.groupby('country'):
    cdf = cdf.loc[country]
    factors = [(cdf.loc[prio], 1)  for prio in cdf.index.get_level_values('eu_priority')]
    ewbi[country] = simple_average(factors)
    
ewbi = pd.concat(ewbi, names=['country'])


# In[36]:


ewbi.to_csv('../output/ewbi_results.csv')

