import os
import joblib
import pandas as pd
import numpy as np
from datasets import load_dataset

# ---------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------
HF_DATASET_ID = None #"sufficiencylab/sufficiency-library"
PARQUET_FILE = "../../results_557k/sample_2000_cleaned_chunks_embedding_Qwen3_06B.parquet"

MODEL_DIR = "../saved_impact_models/impact_model_only_20260122_174131"
OUTPUT_CSV = "sufficiency_library_predictions.csv"
SLICE_ROWS = 10000  # set to None to process all rows (be careful with memory if you do this without streaming or batching)

EMBEDDING_FIELD = "embeddings"

# ---------------------------------------------------------------------
# 1. Load classifiers (NO SBERT)
# ---------------------------------------------------------------------
classifier_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_clf.pkl")]
impact_fields = [f.replace("_clf.pkl", "") for f in classifier_files]

classifiers = {}
mlb_dict = {}

for field in impact_fields:
    classifiers[field] = joblib.load(os.path.join(MODEL_DIR, f"{field}_clf.pkl"))
    mlb_dict[field] = joblib.load(os.path.join(MODEL_DIR, f"{field}_mlb.pkl"))

print(f"Loaded classifiers for {impact_fields}")

# ---------------------------------------------------------------------
# 2. Load embeddings parquet from HF (NO streaming, NO recompute)
# ---------------------------------------------------------------------
print(f"Loading embeddings from {PARQUET_FILE} (first {SLICE_ROWS} rows)...")

if HF_DATASET_ID:
    dataset = load_dataset(
        HF_DATASET_ID,
        data_files=PARQUET_FILE,
        split="train"
    )
    df = dataset.to_pandas().head(SLICE_ROWS)
else:
    df = pd.read_parquet(PARQUET_FILE).head(SLICE_ROWS)

# Extract required fields
chunk_ids = df["chunk_idx"].tolist()
openalex_ids = df["openalex_id"].tolist()

# Convert embeddings to numpy array
X_new = np.vstack(df[EMBEDDING_FIELD].values)

print(f"Loaded {X_new.shape[0]} embeddings with dimension {X_new.shape[1]}")

# ---------------------------------------------------------------------
# 3. Predict
# ---------------------------------------------------------------------
results = pd.DataFrame({
    "chunk_idx": chunk_ids,
    "openalex_id": openalex_ids,
})

for field in impact_fields:
    print(f"Predicting {field} …")
    clf = classifiers[field]
    mlb = mlb_dict[field]

    y_pred = clf.predict(X_new)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    results[f"{field}_pred"] = list(mlb.inverse_transform(y_pred))

# ---------------------------------------------------------------------
# 4. Save as Parquet (NO TEXT)
# ---------------------------------------------------------------------

results.to_csv(OUTPUT_CSV, index=False)

""" output_parquet = OUTPUT_CSV.replace(".csv", ".parquet")

table = pa.Table.from_pandas(results, preserve_index=False)
pq.write_table(table, output_parquet)

print(f"Predictions saved to {output_parquet}") """
