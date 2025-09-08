#!/usr/bin/env python
# coding: utf-8

# ## Computation of EWBI and wellbeing sub-indicators

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('../data/2025-06-05_df_final_EWBI.csv')
df


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


preprocessed.swaplevel(1, 2).sort_index().to_csv('../output/primary_data_preprocessed.csv')


# In[ ]:




