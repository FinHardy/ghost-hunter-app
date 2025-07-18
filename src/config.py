"""
Description:
- This file contains the configuration settings for the project,
    in the form of a Config() dataclass object.

Returns:
- Config: A dataclass containing the configuration settings
"""

import os
from dataclasses import dataclass
from typing import Tuple, Union

import yaml


@dataclass
class DataConfig:
    data_dir: str
    train_dir: str
    data_loader: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    num_workers: int
    augment: bool
    image_size: int
    labels_file: str


@dataclass
class WandbConfig:
    project_name: str
    run_name: str
    experiment_id: str
    ex_description: str  # needs to be written in underscores. Keep short


@dataclass
class ModelConfig:
    in_channels: int
    batch_size: int
    model_name: str
    accuracy_metric: str


@dataclass
class OptimizerConfig:
    optimizer: str
    lr: float
    weight_decay: float
    momentum: float
    betas: Tuple[float, float]


@dataclass
class SchedulerConfig:
    scheduler: str
    step_size: int
    step_size_up: int
    patience: int
    factor: float
    min_lr: float
    max_lr: float
    T_max: int
    gamma: float


@dataclass
class TrainingConfig:
    num_nodes: int
    devices: int
    max_epochs: int
    log_dir: str
    log_every_n_steps: int
    load_path: str
    resume_from: Union[str, bool]
    # e.g. ./checkpoints/basicCNN/
    checkpoint_dir: str
    wandb_logging: bool


@dataclass
class TestConfig:
    load_path: str


@dataclass
class Config:
    """
    Description:
    - Dataclass containing configuration settings for the project
    """

    seed: int
    test_only: bool
    accelerator: str
    task: str

    data: DataConfig
    wandb: WandbConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainingConfig
    test: TestConfig

    abs_path: str = os.path.abspath(
        os.path.dirname(__file__)
    )  # keep this here for ease of use

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "Config":
        """Loads configuration from a YAML file."""
        try:
            with open(yaml_file, "r", encoding="utf-8") as file:
                config_dict = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{yaml_file}' not found.")

        config_dict["data"] = DataConfig(**config_dict["data"])
        config_dict["wandb"] = WandbConfig(**config_dict["wandb"])
        config_dict["model"] = ModelConfig(**config_dict["model"])
        config_dict["optimizer"] = OptimizerConfig(**config_dict["optimizer"])
        config_dict["scheduler"] = SchedulerConfig(**config_dict["scheduler"])
        config_dict["trainer"] = TrainingConfig(**config_dict["trainer"])
        config_dict["test"] = TestConfig(**config_dict["test"])

        return cls(**config_dict)
