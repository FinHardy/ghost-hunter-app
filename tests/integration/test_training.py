"""
Integration tests for training pipeline
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.config import Config
from src.run import main as train_main


class TestTrainingPipeline:
    """Test complete training pipeline"""

    @pytest.fixture
    def setup_training_data(self, temp_dir):
        """Setup complete training environment"""
        # Create directories
        data_dir = os.path.join(temp_dir, "data")
        config_dir = os.path.join(temp_dir, "configs")
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")

        os.makedirs(data_dir)
        os.makedirs(config_dir)
        os.makedirs(checkpoint_dir)

        # Create test images
        for i in range(10):  # Small dataset for testing
            image_array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            image = Image.fromarray(image_array).convert("L")  # Convert to grayscale
            image.save(os.path.join(data_dir, f"test_{i}.png"))

        key = {0: "na", 1: "horizontal", 2: "vertical"}
        labels_data = {
            "labels": [
                {"file": f"test_{i}.png", "label": key[i % 3]}  # Binary labels
                for i in range(10)
            ]
        }

        labels_file = os.path.join(data_dir, "test_labels.yaml")
        with open(labels_file, "w") as f:
            yaml.dump(labels_data, f)

        # Create config
        config_data = {
            "seed": 42,
            "test_only": False,
            "accelerator": "cpu",
            "task": "binary",
            "data": {
                "data_dir": str(data_dir),
                "train_dir": str(data_dir),
                "data_loader": "RawPngLoader",
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "num_workers": 1,
                "augment": False,
                "image_size": 64,
                "labels_file": os.path.join(data_dir, "test_labels.yaml"),
            },
            "model": {
                "in_channels": 1,
                "batch_size": 2,
                "model_name": "ThreeLayerCnn",
                "accuracy_metric": "cross_entropy",
            },
            "optimizer": {
                "optimizer": "adamw",
                "lr": 0.001,
                "weight_decay": 0.01,
                "momentum": 0.9,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "scheduler": "plateau",
                "step_size": 30,
                "step_size_up": 1000,
                "patience": 2,
                "factor": 0.5,
                "min_lr": 3e-5,
                "max_lr": 5e-5,
                "T_max": 10,
                "gamma": 0.1,
            },
            "trainer": {
                "num_nodes": 1,
                "devices": 1,
                "max_epochs": 2,  # Very short for testing
                "log_dir": os.path.join(temp_dir, "logs"),
                "log_every_n_steps": 1,
                "checkpoint_dir": "test_checkpoints/",
                "wandb_logging": False,
                "load_path": "",
                "resume_from": False,
            },
            "wandb": {
                "project_name": "test",
                "run_name": "",
                "experiment_id": "test",
                "ex_description": "test",
            },
            "test": {
                "load_path": "",
            },
        }

        config_file = os.path.join(config_dir, "test_config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        return {
            "config_file": str(config_file),
            "data_dir": str(data_dir),
            "temp_dir": temp_dir,
        }

    @patch("src.run.pl.Trainer")
    def test_training_initialization(self, mock_trainer, setup_training_data):
        """Test that training initializes without errors"""
        config_file = setup_training_data["config_file"]

        # Mock trainer to avoid actual training
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance


        # This should not raise an exception
        train_main(config_file)

        # Verify trainer was called
        mock_trainer.assert_called_once()
        mock_trainer_instance.fit.assert_called_once()

    def test_config_loading(self, setup_training_data):
        """Test config loading functionality"""
        config_file = setup_training_data["config_file"]
        old_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(config_file))
            # This should not raise an exception
            config = Config.from_yaml(os.path.basename(config_file))
        finally:
            os.chdir(old_cwd)

        assert config.seed == 42
        assert config.accelerator == "cpu"
        assert config.model.model_name == "ThreeLayerCnn"
        assert config.trainer.max_epochs == 2


class TestModelSaving:
    """Test model saving and loading"""

    def test_model_checkpoint_creation(self, temp_dir):
        """Test that model checkpoints can be created"""
        from src.models import ThreeLayerCnn

        config = MagicMock()
        config.model.in_channels = 1
        config.model.batch_size = 2
        config.accelerator = "cpu"
        config.task = "binary"

        model = ThreeLayerCnn(config)

        # Save model
        checkpoint_path = os.path.join(temp_dir, "test_model.ckpt")
        torch.save(model.state_dict(), checkpoint_path)

        # Verify file was created
        assert os.path.exists(checkpoint_path)

        # Test loading
        loaded_state = torch.load(checkpoint_path, map_location="cpu")
        assert isinstance(loaded_state, dict)

        # Create new model and load state
        new_model = ThreeLayerCnn(config)
        new_model.load_state_dict(loaded_state)

        # Verify parameters match
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
