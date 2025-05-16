from datetime import datetime
import hydra
import pandas as pd
import rootutils
import torch
import transformers
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.bert_module import BertLitModule


class CSVDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.doi = self.df["doi"].values
        self.title = self.df["title"].values
        self.X = self.df["abstract"].values

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        out = self.tokenizer(
            self.X[idx],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        out["title"] = self.title[idx]
        out["doi"] = self.doi[idx]
        out["input_ids"] = out["input_ids"].squeeze(0)
        out["attention_mask"] = out["attention_mask"].squeeze(0)
        return out


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    save_path = f"predict_results_{datetime.now().strftime('%m-%d_%H-%M-%S')}.csv"
    print("Loading data...")
    # df = pd.DataFrame(
    #     {
    #         "doi": [
    #             "10.1016/j.buildenv.2023.110000",
    #             "10.1016/j.buildenv.2023.110000",
    #             "10.1016/j.buildenv.2023.110000",
    #         ],
    #         "title": [
    #             "Title 1",
    #             "Title 2",
    #             "Title 3",
    #         ],
    #         "abstract": [
    #             "Abstract 1",
    #             "Abstract 2",
    #             "Abstract 3",
    #         ],
    #     }
    # )

    df = (
        pd.read_csv(
            "data/all_documents_general_query.csv",
            usecols=["doi", "title", "abstract"],
            dtype={
                "doi": str,
                "title": str,
                "abstract": str,
            },
        )
        .drop_duplicates()
        .dropna()
        .reset_index(drop=True)
    )

    # df = pd.read_csv(
    #     "data/sector_positive.csv",
    #     usecols=["title", "abstract"],
    # )
    # df["doi"] = ["oui"] * len(df)
    print(f"Loaded {len(df)} rows")
    print("Data loaded")

    # Create pytorch dataset and dataloader from the dataframe
    tokenizer = transformers.BertTokenizer.from_pretrained(cfg.model.model_name)
    dataset = CSVDataset(df, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=256, num_workers=4, pin_memory=True)

    ckpt_path = [
        {
            "model_name": "buildings",
            "path": "logs/train/runs/2025-05-12_15-43-16/checkpoints/epoch_007.ckpt",
        },
        {
            "model_name": "digitalisation",
            "path": "logs/train/runs/2025-05-09_00-28-50/checkpoints/epoch_012.ckpt",
        },
        {
            "model_name": "freight",
            "path": "logs/train/runs/2025-05-12_17-38-01/checkpoints/epoch_003.ckpt",
        },
        {
            "model_name": "mobility",
            "path": "logs/train/runs/2025-05-12_16-13-47/checkpoints/epoch_004.ckpt",
        },
        {
            "model_name": "nutrition",
            "path": "logs/train/runs/2025-05-09_00-35-45/checkpoints/epoch_012.ckpt",
        },
        {
            "model_name": "trade",
            "path": "logs/train/runs/2025-05-12_21-47-51/checkpoints/epoch_022.ckpt",
        },
        {
            "model_name": "urban_ecology",
            "path": "logs/train/runs/2025-05-12_16-54-29/checkpoints/epoch_010.ckpt",
        },
        {
            "model_name": "urban_governance",
            "path": "logs/train/runs/2025-05-12_17-22-32/checkpoints/epoch_003.ckpt",
        },
        {
            "model_name": "urban_infra",
            "path": "logs/train/runs/2025-05-12_17-09-01/checkpoints/epoch_012.ckpt",
        },
    ]
    print("Loading models...")
    models = {
        d["model_name"]: BertLitModule.load_from_checkpoint(d["path"])
        for d in ckpt_path
    }

    # Set models to evaluation mode
    for model in models.values():
        model.eval()
        model.to(device)

    all_outputs = {model_name: [] for model_name in models.keys()}
    all_outputs["doi"] = []
    all_outputs["title"] = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch = batch.to(device)
            all_outputs["doi"] += batch["doi"]
            all_outputs["title"] += batch["title"]
            # Store predictions for each model
            for model_name, model in models.items():
                output = model.net(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                preds = torch.argmax(output.logits, dim=1)  # for classification
                all_outputs[f"{model_name}"] += preds.tolist()
            if i % 50 == 0:
                pd.DataFrame(all_outputs).set_index("doi").to_csv(save_path)

    pd.DataFrame(all_outputs).set_index("doi").to_csv(save_path)

    print(f"Predictions saved to {save_path}")


if __name__ == "__main__":
    main()
