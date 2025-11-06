from argparse import ArgumentParser
from pathlib import Path

from fast_ingestion_pipeline import IndexingPipeline


def main():
    parser = ArgumentParser(description="Run model smashing job")
    parser.add_argument("--file-path", required=True, help="Path to the file")

    args = parser.parse_args()
    file_path = args.file_path
    folder_path = Path(file_path).parent

    indexing_pipeline = IndexingPipeline(pdf_path=folder_path)
    print(f"Parsing document: {file_path}")

    indexing_pipeline.run(file_path)


if __name__ == "__main__":
    main()
