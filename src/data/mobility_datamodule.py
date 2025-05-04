import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
from lightning import LightningDataModule


class MobilityDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer(
            row["abstract"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs["labels"] = torch.tensor(row["true_label"])
        return inputs


class MobilityDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer,
        batch_size=32,
        max_length=512,
        seed=42,
        val_split=0.1,
        test_split=0.2,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self.val_split = val_split
        self.test_split = test_split

    def prepare_data(self):
        self.data = pd.read_csv(self.data_path).dropna()
        self.data["true_label"] = self.data["true_label"].replace(
            {"Not About Sufficiency": 0, "About Sufficiency": 1}
        )

    def setup(self, stage=None):
        train_data, test_data = train_test_split(
            self.data,
            test_size=self.test_split,
            stratify=self.data["true_label"],
            random_state=self.seed,
        )
        test_data, val_data = train_test_split(
            test_data,
            test_size=self.test_split / (self.val_split + self.test_split),
            stratify=test_data["true_label"],
            random_state=self.seed,
        )
        self.train_dataset = MobilityDataset(
            train_data, self.tokenizer, self.max_length
        )
        self.val_dataset = MobilityDataset(val_data, self.tokenizer, self.max_length)
        self.test_dataset = MobilityDataset(test_data, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dm = MobilityDataModule(
        data_path="data/full_concat_dataset.csv", tokenizer=tokenizer
    )
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(batch)
        break
