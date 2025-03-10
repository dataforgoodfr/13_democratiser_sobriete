import argparse

from src.container import Container

container = Container()

def get_args_parser():
    parser = argparse.ArgumentParser(description="Extract text from a PDF file.")

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output file.",
    )
    return parser


def main():
    ingestion_usecase = container.ingestion_usecase()
    ingestion_usecase.ingest_mobility_papers()


if __name__ == "__main__":
    main()