import os
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = "chunked_conclusions_557k_2026-01-26.parquet"
OUTPUT_FILE = "embeddings_chunked_conclusions_557k_Qwen3-06B.parquet"
MODEL = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 1024

model_path = os.environ["DSDIR"] + "/HuggingFace_Models/" + MODEL  # for Jean Zay

model = SentenceTransformer(model_path)
model.half()
model.to("cuda")
print('Model loaded')

parquet_file = pq.ParquetFile(INPUT_FILE)
total_rows = parquet_file.metadata.num_rows
n_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
print(f"Total rows: {total_rows}, Batches: {n_batches}")

writer = None

try:
    for batch in tqdm(
        parquet_file.iter_batches(batch_size=BATCH_SIZE),
        total=n_batches,
        desc="Computing embeddings",
    ):
        df = batch.to_pandas()

        # Compute embeddings
        embeddings = model.encode(
            df["text"].tolist(),
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Add embeddings to dataframe
        df["embedding"] = list(embeddings)

        # Progressively save to parquet
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_FILE, table.schema)
        writer.write_table(table)

finally:
    if writer:
        writer.close()

print(f"Embeddings saved to {OUTPUT_FILE}")
