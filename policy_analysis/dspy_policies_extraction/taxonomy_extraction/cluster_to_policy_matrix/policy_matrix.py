# 1. Imports
from datasets import load_dataset
import json
import pandas as pd

# 2. Load Hugging Face dataset
ds = load_dataset("madoss/wsl_library_filtered_clustered")
hf_df = pd.DataFrame(ds['train'])

# Rename 'chunk_idx' to 'chunk_id' to match the JSONL file
hf_df.rename(columns={'chunk_idx': 'chunk_id'}, inplace=True)

# 3. Load local JSONL file
jsonl_file_path = r"10koutputs.jsonl"
jsonl_data = []

with open(jsonl_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        jsonl_data.append(json.loads(line))

local_df = pd.DataFrame(jsonl_data)

# 4. Merge on 'openalex_id' and 'chunk_id'
merged_df = pd.merge(hf_df, local_df, on=['openalex_id', 'chunk_id'], how='inner')

# 5. Print head of merged dataset
print("Merged dataset head:")
print(merged_df.info())
