import os
import joblib
import sys

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
from datasets import load_dataset

# ---------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------
HF_DATASET_ID = None #"sufficiencylab/sufficiency-library"
PARQUET_FILE = "../../data/embeddings_chunked_conclusions_557k_Qwen3-06B.parquet"
MODEL_DIR = "../saved_impact_models/impact_model_only_20260311_125622"
OUTPUT_CSV = "../../results_557k/impact_taxonomy_557k_2026-03-11.csv"
SLICE_ROWS = None  # set to None to process all rows in batches
BATCH_SIZE = 1000  # rows per batch when SLICE_ROWS is None
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
# 2. Helper: predict on a DataFrame batch
# ---------------------------------------------------------------------
def predict_batch(df_batch):
    chunk_ids = df_batch["chunk_idx"].tolist()
    openalex_ids = df_batch["openalex_id"].tolist()
    X = np.vstack(df_batch[EMBEDDING_FIELD].values)

    batch_result = pd.DataFrame({
        "chunk_idx": chunk_ids,
        "openalex_id": openalex_ids,
    })
    for field in impact_fields:
        y_pred = classifiers[field].predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        batch_result[f"{field}_pred"] = list(mlb_dict[field].inverse_transform(y_pred))

    return batch_result

# ---------------------------------------------------------------------
# 3. Load embeddings and predict
# ---------------------------------------------------------------------
if HF_DATASET_ID:
    dataset = load_dataset(HF_DATASET_ID, data_files=PARQUET_FILE, split="train")
    df = dataset.to_pandas()
    if SLICE_ROWS is not None:
        df = df.head(SLICE_ROWS)
    print(f"Loaded {len(df)} rows from HF dataset")
    results = predict_batch(df)

elif SLICE_ROWS is not None:
    print(f"Loading first {SLICE_ROWS} rows from {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE).head(SLICE_ROWS)
    print(f"Loaded {len(df)} embeddings with dimension {np.vstack(df[EMBEDDING_FIELD].values).shape[1]}")
    results = predict_batch(df)

else:
    print(f"Processing {PARQUET_FILE} in batches of {BATCH_SIZE}...")
    pf = pq.ParquetFile(PARQUET_FILE)
    total_rows = pf.metadata.num_rows
    total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total rows: {total_rows}, batches: {total_batches}")

    output_parquet = OUTPUT_CSV.replace(".csv", ".parquet")
    parquet_writer = None
    first_batch = True
    try:
        for batch in tqdm(pf.iter_batches(batch_size=BATCH_SIZE), total=total_batches, desc="Predicting batches"):
            df_batch = batch.to_pandas()
            result_batch = predict_batch(df_batch)

            # Write CSV progressively
            result_batch.to_csv(OUTPUT_CSV, mode="a", header=first_batch, index=False)

            # Write Parquet progressively
            table = pa.Table.from_pandas(result_batch, preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(output_parquet, table.schema)
            parquet_writer.write_table(table)

            first_batch = False
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    print(f"Predictions saved to {OUTPUT_CSV}")
    print(f"Predictions saved to {output_parquet}")
    sys.exit(0)

# ---------------------------------------------------------------------
# 4. Save (for HF dataset and SLICE_ROWS paths)
# ---------------------------------------------------------------------

output_parquet = OUTPUT_CSV.replace(".csv", ".parquet")
results.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")

results.to_parquet(output_parquet, index=False)
print(f"Predictions saved to {output_parquet}")
