import pandas as pd

# Read the parquet file
df = pd.read_parquet("cleaned_results_scaleway.parquet", engine="pyarrow")

# Slice first 100 rows
# We use .copy() to ensure we are working on a new DataFrame, not a view
df_small = df.head(100).copy()

# CRITICAL STEP: Reset index to turn it into a column
df_small.reset_index(inplace=True)

# If you want to rename the index column to something specific (like "id"), uncomment below:
# df_small.rename(columns={"index": "row_id"}, inplace=True)

# Export to JSON (line-delimited)
df_small.to_json("output.jsonl", orient="records", lines=True)
