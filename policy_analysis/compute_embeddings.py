"""Embed a parquet text column with a SentenceTransformer, streaming output.

Jean Zay-ready: --model may be a plain HF id (resolved under
$DSDIR/HuggingFace_Models when set, so compute nodes stay offline) or a
filesystem path. Used for policy embeddings before clustering:

    python compute_embeddings.py \\
        --input policies.parquet --output policy_embeddings.parquet \\
        --text-column policy_text --model Qwen/Qwen3-Embedding-4B
"""
import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def resolve_model(model: str) -> str:
    if os.path.isdir(model):
        return model
    dsdir = os.environ.get("DSDIR")
    if dsdir:  # Jean Zay pre-staged model store
        return f"{dsdir}/HuggingFace_Models/{model}"
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B",
                    help="HF id (resolved under $DSDIR on Jean Zay) or local path")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="rows per parquet batch (encode batch is 64)")
    args = ap.parse_args()

    model = SentenceTransformer(resolve_model(args.model))
    model.half()
    model.to("cuda")
    print("Model loaded")

    parquet_file = pq.ParquetFile(args.input)
    total_rows = parquet_file.metadata.num_rows
    n_batches = (total_rows + args.batch_size - 1) // args.batch_size
    print(f"Total rows: {total_rows}, Batches: {n_batches}")

    writer = None

    try:
        for batch in tqdm(
            parquet_file.iter_batches(batch_size=args.batch_size),
            total=n_batches,
            desc="Computing embeddings",
        ):
            df = batch.to_pandas()

            embeddings = model.encode(
                df[args.text_column].tolist(),
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            df["embedding"] = list(embeddings)

            # Progressively save to parquet
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema)
            writer.write_table(table)

    finally:
        if writer:
            writer.close()

    print(f"Embeddings saved to {args.output}")


if __name__ == "__main__":
    main()
