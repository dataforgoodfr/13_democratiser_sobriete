import yaml
import argparse


# Load configuration from a YAML file
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def read_args():
    parser = argparse.ArgumentParser(description="Benchmarking")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )
    return parser.parse_args()


def create_collection_name(embedding_model: str, chunk_size: int, chunk_overlap: int):
    """
    Generate a structured and unique collection name based on key hyperparameters.

    Parameters:
        embedding_model (str): Name of the embedding model.
        chunk_size (int): Size of text chunks.
        chunk_overlap (int): Overlapping tokens between chunks.
    Returns:
        str: A concise, readable, and unique collection name.
    """
    # Format collection name
    collection_name = (
        f"{embedding_model.replace('/', '-')}_cs{chunk_size}_co{chunk_overlap}"
    )
    return collection_name
