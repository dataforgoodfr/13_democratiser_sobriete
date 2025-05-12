from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import ConfusionMatrix
from transformers import BertForSequenceClassification
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score


class BertLitModule(LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        criterion: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        tensor_cores: bool,
        probability_loss: bool,
        cache_dir: str = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.net = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes, cache_dir=cache_dir
        )
        self.num_classes = num_classes
        self.criterion = criterion
        self.probability_loss = probability_loss

        self.val_preds = []
        self.val_targets = []

        if num_classes == 2:
            metric_kwargs = {"task": "binary"}
        else:
            metric_kwargs = {"task": "multiclass", "num_classes": num_classes}

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)

        # Precision, Recall, and F1-score metrics
        self.train_precision = Precision(**metric_kwargs)
        self.train_recall = Recall(**metric_kwargs)
        self.train_f1 = F1Score(**metric_kwargs)

        self.val_precision = Precision(**metric_kwargs)
        self.val_recall = Recall(**metric_kwargs)
        self.val_f1 = F1Score(**metric_kwargs)

        self.test_precision = Precision(**metric_kwargs)
        self.test_recall = Recall(**metric_kwargs)
        self.test_f1 = F1Score(**metric_kwargs)

        # Confusion matrix metrics
        self.train_conf_matrix = ConfusionMatrix(**metric_kwargs)
        self.val_conf_matrix = ConfusionMatrix(**metric_kwargs)
        self.test_conf_matrix = ConfusionMatrix(**metric_kwargs)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.net.train()
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        target = batch["labels"]
        output = self.net(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = output.logits
        if self.probability_loss:
            # Convert labels to probabilities
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).float()
        loss = self.criterion(logits, target)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, batch["labels"]

    def training_step(self, batch):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_f1(preds, targets)
        self.train_conf_matrix.update(preds, targets)

        # self.log("lr", self.trainer.lr_schedulers[0]["scheduler"].get_lr()[0])
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/precision",
            self.train_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/recall",
            self.train_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)
        self.test_conf_matrix.update(preds, targets)

        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/precision",
            self.test_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)


    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Log confusion matrix
        test_conf_matrix = self.test_conf_matrix.compute()
        print("test/conf_matrix\n", test_conf_matrix.numpy(force=True))

        # Reset confusion matrix for the next epoch
        self.test_conf_matrix.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        with torch.no_grad():
            loss, preds, targets = self.model_step(batch)
        self.val_preds += preds.tolist()
        self.val_targets += targets.tolist()

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)
        self.val_conf_matrix.update(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc

        # Log confusion matrix
        val_conf_matrix = self.val_conf_matrix.compute()
        print("val/conf_matrix\n", val_conf_matrix.numpy(force=True))

        # Reset confusion matrix for the next epoch
        self.val_conf_matrix.reset()

        # Log `val_acc_best`
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

        # report_table 
        # self.log("val/classification_report", wandb.Table(data=report_table, columns=report_columns))

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.tensor_cores:
            torch.set_float32_matmul_precision("medium")
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/recall",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = BertLitModule("bert-base-uncased")
