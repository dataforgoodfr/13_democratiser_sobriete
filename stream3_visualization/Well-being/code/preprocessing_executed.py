#!/usr/bin/env python
# coding: utf-8

# ## Computation of EWBI and wellbeing sub-indicators

# In[1]:


import pandas as pd
import os

# In[2]:


# Build path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', '2025-06-05_df_final_EWBI.csv')

df = pd.read_csv(data_path)
df

import pandas as pd
import os

# Completeness
output_dir = os.path.join(script_dir, '..', 'output')
os.makedirs(output_dir, exist_ok=True)
excel_path = os.path.join(output_dir, 'completeness.xlsx')

with pd.ExcelWriter(excel_path) as writer:
    for primary_index in df['primary_index'].unique():
        sub = df[df['primary_index'] == primary_index]
        # Create a pivot table: index=country, columns=year, values=number of unique deciles
        completeness = sub.groupby(['country', 'year'])['decile'].nunique().unstack(fill_value=0)
        # Ensure all countries and years are present
        all_countries = df['country'].unique()
        all_years = df['year'].unique()
        completeness = completeness.reindex(index=all_countries, columns=sorted(all_years), fill_value=0)
        completeness.to_excel(writer, sheet_name=str(primary_index)[:31])  # Excel sheet names max 31 chars


# ## Preprocessing
# ### Data cleaning

# In[3]:


df = df.drop(columns=['database'])


# In[4]:


df['value'] = df['value'].str.replace(',', '.') # some commas appear as decile separators
df['value'] = df['value'].astype(float)


# In[5]:


df['year'] = df['year'].astype(int)
df.year.unique()


# ### Splitting quintiles into 2 deciles

# In[6]:


def process_quantiles(df):
    """
    Port data in quintiles to deciles by assigning duplicating each row with quintile 
    and assigning it to the two corresponding deciles.
    """
    print("Initial length:", len(df))
    quintile_rows = df[df.quintile.notna()].copy()
    print("Number of rows with quintile:", len(quintile_rows))
    quintile_rows['decile'] = quintile_rows['quintile'] * 2
    quintile_rows_duplicated = quintile_rows.copy()
    quintile_rows_duplicated['decile'] = quintile_rows_duplicated['quintile'] * 2 - 1
    df = pd.concat([df[df.quintile.isna()], quintile_rows, quintile_rows_duplicated], ignore_index=True)
    print("Final length:", len(df))
    df['decile'] = df['decile'].astype(int)
    df = df.drop(columns=['quintile'])
    return df

df = process_quantiles(df)


# ### Fill missing values
# The EU JRC methodology tells us to fill missing values (NaNs) for each indicator using the next last available one, and if absent the next available one. This is preferred to ignoring indicators for the years they're not available.
## Change = only in the future

# In[7]:


wide = df.pivot_table(values='value', index=['primary_index', 'decile', 'country'], columns='year')
wide


# In[8]:


#filled = wide.ffill(axis=1).bfill(axis=1)
filled = wide.ffill(axis=1)
filled


# ### Normalising

# In[9]:


# The normalisation is intra-decile and intra-indicator so we separate using groupby
# We are using a Standardisation (or z-scores) method as described by the JRC 

scaled_min = 0.1

res = []
for (ind, decile), grouped in filled.groupby(['primary_index', 'decile']):
    data = grouped.copy()

    # Z-score normalization: (value - mean) / std, reversed
    norm = -1 * (data - data.mean(axis=0)) / data.std(axis=0)

    # Scale between 0.1 and 1
    norm_min = norm.min().min()
    norm_max = norm.max().max()
    norm = scaled_min + (norm - norm_min) * (1 - scaled_min) / (norm_max - norm_min)

    res.append(norm)


preprocessed = pd.concat(res)
preprocessed



# In[10]:
# Build path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'output', 'primary_data_preprocessed.csv')

preprocessed.swaplevel(1, 2).sort_index().to_csv(data_path)


# In[ ]:




