import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import lightning as L


class LitTextClassification(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased").train()

    def training_step(self, batch):
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        self.log("train_loss", output.loss)
        return output.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-5)


class TextClassificationData(L.LightningDataModule):
    def prepare_data(self):
        load_dataset("imdb")

    def train_dataloader(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = load_dataset("imdb")["train"]
        dataset = dataset.map(lambda sample: tokenizer(sample["text"], padding="max_length", truncation=True))
        dataset.set_format(type="torch")
        return torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)


if __name__ == "__main__":
    model = LitTextClassification()
    data = TextClassificationData()
    trainer = L.Trainer(max_epochs=3)
    trainer.fit(model, data)
