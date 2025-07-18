"""
Description:
    - Base class for all models

Raises:
    NotImplementedError:
        - If the forward method is not implemented
        - If the BaseModel class is instantiated directly

Returns:
    - BaseModel: Base class for all models
"""

from abc import abstractmethod
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.optim as optim

from src.config import Config


class BaseModel(pl.LightningModule):
    """
    Description:
        - Base class for all models
        - Other models should inherit from this class and take the _config parameter

    Args:
        - pl.LightningModule: PyTorch Lightning module
        - _config (Config): Configuration settings
    """

    def __init__(self, _config: Config):
        super().__init__()
        self.batch_size = _config.model.batch_size
        self.num_epochs = _config.trainer.max_epochs
        self.image_size = _config.data.image_size

        self.optimizer = _config.optimizer.optimizer
        self.lr = _config.optimizer.lr
        self.weight_decay = _config.optimizer.weight_decay
        self.momentum = _config.optimizer.momentum
        self.betas = _config.optimizer.betas

        self.scheduler = _config.scheduler.scheduler
        self.step_size = _config.scheduler.step_size
        self.step_size_up = _config.scheduler.step_size_up
        self.patience = _config.scheduler.patience
        self.factor = _config.scheduler.factor
        self.min_lr = _config.scheduler.min_lr
        self.max_lr = _config.scheduler.max_lr
        self.T_max = _config.scheduler.T_max
        self.gamma = _config.scheduler.gamma

        self.task = _config.task

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    # Needed due to custom loss functions for some classes
    def loss_fn(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy_metric(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        acc = self.accuracy_metric(y_hat, y)
        self.log("val_loss", val_loss)
        self.log("val_acc", acc)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy_metric(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    # TODO: Fix this and put it properly with the _configure_accuracy method
    def accuracy_metric(self, predictions, labels):
        """
        Need a top 1 accuracy function for the model - no need for configurability
        """
        _, preds = torch.max(
            predictions, 1
        )  # Get the predicted class index (highest probability)
        correct = torch.sum(preds == labels)  # Compare predicted vs actual labels
        accuracy = correct.float() / labels.size(
            0
        )  # Compute the percentage of correct predictions
        return accuracy

    def configure_optimizers(self):
        optimizer: optim.Optimizer
        if self.optimizer == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
        elif self.optimizer == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        else:
            raise ValueError(f"Optimizer '{self.optimizer}' is not recognized")

        scheduler = self._configure_scheduler(optimizer)

        lr_scheduler = {
            "scheduler": scheduler,
            "name": "learning rate",
            "monitor": "val_loss",
        }

        return ([optimizer], [lr_scheduler])

    def _configure_scheduler(self, optimizer):
        if self.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs
            )
        elif self.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.gamma
            )
        elif self.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.factor, patience=self.patience
            )
        elif self.scheduler == "cyclic":
            return optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.lr,
                max_lr=self.max_lr,
                step_size_up=self.step_size_up,
                mode="triangular",
            )
        elif self.scheduler == "none":
            return None
        else:
            raise ValueError(f"Scheduler '{self.scheduler}' is not recognized")

    def get_num_params(self) -> None:
        # print number of parameters in comma number format
        params = "{:,}".format(
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        )

        print(f"Number of trainable parameters: {params}")
