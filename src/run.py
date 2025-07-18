import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import wandb
from fire import Fire
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.config import Config
from src.datasets import RawPngLoader
from src.models import ResidualCNN, ThreeLayerCnn

ABS_PATH: str = os.path.abspath(os.path.dirname(__file__))

# with open(os.path.join(ABS_PATH, "../", "wandb_key.txt"), "r") as f:
#     key = f.read().strip()

# wandb.login(key=key)

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


models: Dict = {
    "ThreeLayerCnn": ThreeLayerCnn,
    "ResidualCNN": ResidualCNN,
}

datamodules: Dict = {"RawPngLoader": RawPngLoader}


def main(config_file: str):
    """
    Description:
    - Main function to train the model.

    Args:
        config_file (str): Path to the configuration file

    Raises:
        ValueError: if model name is not recognized
        ValueError: if dataloader name is not recognized
    """
    # get abspath
    _config = Config.from_yaml(os.path.join(ABS_PATH, "../", "configs", config_file))

    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("medium")

    seed_everything(_config.seed)
    log_dir = Path(_config.trainer.log_dir)

    model_class = _config.model.model_name
    if model_class:
        model = models[model_class](_config)
    else:
        raise ValueError(f"Model '{_config.model.model_name}' is not recognized.")

    # Access the data module based on the config
    data_module_class = _config.data.data_loader
    if data_module_class:
        data_module = datamodules[data_module_class](_config)
    else:
        raise ValueError(f"Data source '{_config.data.data_loader}' is not recognized.")

    # for using in file names for artifacts
    dataset_name = os.path.basename(_config.data.data_dir)

    checkpoint_dir = os.path.join(
        ABS_PATH, "../checkpoints", _config.trainer.checkpoint_dir
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.6f}",
        save_top_k=3,  # Save top k models
        mode="min",  # Save model with minimum validation loss
    )

    wandb_logging = _config.trainer.wandb_logging

    if wandb_logging:
        wandb_logger = WandbLogger(
            project=_config.wandb.project_name,
            name=f"{_config.wandb.experiment_id}-{_config.model.model_name}-{dataset_name}-{_config.wandb.ex_description}",
            save_dir=log_dir,
            version=(
                f"{_config.wandb.run_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                if not _config.test_only
                else None
            ),
            config=_config,
        )

        wandb_logger.experiment.config.batch_size = _config.model.batch_size

        trainer = pl.Trainer(
            default_root_dir=os.getcwd(),
            max_epochs=_config.trainer.max_epochs,
            num_nodes=_config.trainer.num_nodes if _config.accelerator == "cuda" else 1,
            accelerator=_config.accelerator,
            devices=(
                _config.trainer.devices if _config.accelerator == "cuda" else "auto"
            ),
            callbacks=[checkpoint_callback],
            log_every_n_steps=_config.trainer.log_every_n_steps,
            logger=wandb_logger,  # type: ignore
        )

    else:
        trainer = pl.Trainer(
            default_root_dir=os.getcwd(),
            max_epochs=_config.trainer.max_epochs,
            num_nodes=_config.trainer.num_nodes if _config.accelerator == "cuda" else 1,
            accelerator=_config.accelerator,
            devices=(
                _config.trainer.devices if _config.accelerator == "cuda" else "auto"
            ),
            callbacks=[checkpoint_callback],
            log_every_n_steps=_config.trainer.log_every_n_steps,
        )

    if not _config.test_only:
        data_module.setup(stage="fit")
        if not _config.trainer.resume_from:
            trainer.fit(model, datamodule=data_module)
        else:
            trainer.fit(
                model, datamodule=data_module, ckpt_path=_config.trainer.resume_from
            )
            trainer.test(model, datamodule=data_module, ckpt_path="best")
    else:
        data_module.setup(stage="test")
        trainer.test(model, datamodule=data_module, ckpt_path=_config.trainer.load_path)


if __name__ == "__main__":
    Fire(main)
