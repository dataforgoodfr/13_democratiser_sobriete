from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="EdouardCallet/wsl-policy-10k")
    parser.add_argument("--text-column", default="single_policy_item")
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()

    login()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = load_dataset(args.repo_id)

    model = SentenceTransformer(
        args.model_name,
        device=device,
        trust_remote_code=True
    )

    def embed_batch(batch):
        return {
            "embedding": model.encode(
                batch[args.text_column],
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        }

    def process_split(ds, split_name):
        print(f"Embedding {split_name}: {len(ds)} rows")
        return ds.map(
            embed_batch,
            batched=True,
            batch_size=args.batch_size,
            load_from_cache_file=False,
            desc=f"Embedding {split_name}"
        )

    if isinstance(dataset, DatasetDict):
        if args.split:
            dataset[args.split] = process_split(dataset[args.split], args.split)
        else:
            for split in dataset:
                dataset[split] = process_split(dataset[split], split)
    else:
        dataset = process_split(dataset, "data")

    dataset.push_to_hub(
        args.repo_id,
        commit_message="Add Qwen3 embeddings (fast)"
    )

    print("✅ Done")


if __name__ == "__main__":
    main()
