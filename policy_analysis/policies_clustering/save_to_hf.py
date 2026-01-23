from huggingface_hub import login, upload_file
from pathlib import Path
import argparse
import pandas as pd
import tempfile
from dotenv import load_dotenv  

load_dotenv()  # Load environment variables from .env file


def main():
    parser = argparse.ArgumentParser(
        description="Convert a local CSV to Parquet and upload to a Hugging Face dataset repo"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the local CSV file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="EdouardCallet/wsl-policy-10k",
        help="Hugging Face dataset repo (username/repo)"
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        default="data/data.parquet",
        help="Path inside the HF repo (e.g. data/train.parquet)"
    )

    args = parser.parse_args()
    print("ARGS:", args)

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load CSV
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path,sep=";")
    print(f"Loaded dataframe with shape {df.shape}")

    # Write Parquet to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / (csv_path.stem + ".parquet")

        print(f"Converting to Parquet: {parquet_path}")
        df.to_parquet(parquet_path, index=False)

        # Authenticate (uses HF_TOKEN env var or cached token)
        login()

        print(f"Uploading {parquet_path} → {args.repo_id}/{args.repo_path}")

        upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=args.repo_path,
            repo_id=args.repo_id,
            repo_type="dataset"
        )

    print("✅ Upload complete!")


if __name__ == "__main__":
    main()
