"""
Tests for dataset loaders
"""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.datasets import RawPngLoader


class TestRawPngLoader:
    """Test RawPngLoader dataset"""

    @pytest.fixture
    def setup_test_dataset(self, temp_dir):
        """Create test dataset with images and labels in separate directories"""
        data_dir = os.path.join(temp_dir, "test_data")
        images_dir = os.path.join(data_dir, "images")
        os.makedirs(images_dir)
        # Create test images
        for i in range(6):  # 6 images for train/val/test split
            image_array = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            image = Image.fromarray(image_array).convert("L")  # Convert to grayscale
            image_path = os.path.join(images_dir, f"test_image_{i}.png")
            image.save(image_path)

        # assert that files were created
        assert len(os.listdir(images_dir)) == 6

        # Create labels file (relative paths to images directory)
        key = {0: "na", 1: "horizontal", 2: "vertical"}
        labels_data = {
            "labels": [
                {"file": f"test_image_{i}.png", "label": key[i % 3]} for i in range(6)
            ]
        }

        labels_file_path = os.path.join(data_dir, "test_labels.yaml")
        with open(labels_file_path, "w") as f:
            yaml.dump(labels_data, f)

        return str(data_dir), str(labels_file_path)

    @pytest.fixture
    def mock_config(self, setup_test_dataset):
        """Create mock config for RawPngLoader"""
        data_dir, labels_file = setup_test_dataset

        config = MagicMock()
        config.data.data_dir = os.path.join(data_dir, "images")
        config.data.train_dir = os.path.join(data_dir, "images")
        config.data.labels_file = labels_file
        config.data.image_size = 256
        config.data.train_ratio = 0.6
        config.data.val_ratio = 0.2
        config.data.test_ratio = 0.2
        config.data.num_workers = 1
        config.data.augment = False
        config.data.data_loader = "RawPngLoader"  # Add required field
        config.model.batch_size = 2
        config.seed = 42
        config.accelerator = "cpu"
        config.task = "binary"
        return config

    def test_dataloader_initialization(self, mock_config):
        """Test RawPngLoader initialization"""
        dataloader = RawPngLoader(mock_config)
        assert dataloader is not None

    def test_setup_splits_data(self, mock_config):
        """Test that setup correctly splits data"""
        dataloader = RawPngLoader(mock_config)
        dataloader.setup(stage="fit")

        # Check that datasets were created
        assert hasattr(dataloader, "train_dataset")
        assert hasattr(dataloader, "val_dataset")

        # Check split ratios (approximately)
        total_samples = len(dataloader.train_dataset) + len(dataloader.val_dataset)
        assert total_samples > 0

        train_ratio = len(dataloader.train_dataset) / total_samples
        assert 0.5 <= train_ratio <= 0.8  # Roughly 60% ± some tolerance

    def test_train_dataloader(self, mock_config):
        """Test train dataloader creation"""
        dataloader = RawPngLoader(mock_config)
        dataloader.setup(stage="fit")

        train_loader = dataloader.train_dataloader()

        assert train_loader is not None
        assert hasattr(train_loader, "__iter__")

        # Test one batch
        batch = next(iter(train_loader))
        images, labels = batch

        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images.shape[1] == 1  # Single channel
        assert images.shape[2] == 256  # Height
        assert images.shape[3] == 256  # Width

    def test_val_dataloader(self, mock_config):
        """Test validation dataloader creation"""
        dataloader = RawPngLoader(mock_config)
        dataloader.setup(stage="fit")

        val_loader = dataloader.val_dataloader()

        assert val_loader is not None

        # Test one batch
        batch = next(iter(val_loader))
        images, labels = batch

        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_test_setup(self, mock_config):
        """Test test dataset setup"""
        dataloader = RawPngLoader(mock_config)
        dataloader.setup(stage="test")

        assert hasattr(dataloader, "test_dataset")

        test_loader = dataloader.test_dataloader()
        assert test_loader is not None

    def test_image_preprocessing(self, mock_config):
        """Test that images are properly preprocessed"""
        dataloader = RawPngLoader(mock_config)
        dataloader.setup(stage="fit")

        train_loader = dataloader.train_dataloader()
        batch = next(iter(train_loader))
        images, labels = batch

        # Check image tensor properties
        assert images.dtype == torch.float32
        assert images.min() >= -1.0  # Assuming normalization
        assert images.max() <= 1.0

        # Check label properties
        assert labels.dtype == torch.long

    def test_empty_dataset_handling(self, temp_dir):
        """Test handling of empty dataset"""
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir)

        config = MagicMock()
        config.data.data_dir = str(empty_dir)
        config.data.train_dir = str(empty_dir)
        # Use full path for nonexistent labels file
        config.data.labels_file = os.path.join(empty_dir, "nonexistent.yaml")
        config.data.image_size = 64
        config.data.train_ratio = 0.8
        config.data.val_ratio = 0.1
        config.data.test_ratio = 0.1
        config.data.num_workers = 1
        config.data.augment = False
        config.data.data_loader = "default"
        config.model.batch_size = 2
        config.seed = 42
        config.accelerator = "cpu"
        config.task = "classification"

        # Should handle gracefully or raise appropriate error
        with pytest.raises((FileNotFoundError, ValueError)):
            RawPngLoader(config)

    def test_incorrect_labels_field(self, temp_dir):
        """Test error raised if labels file contains incorrect field name"""
        data_dir = os.path.join(temp_dir, "bad_labels")
        images_dir = os.path.join(data_dir, "images")
        os.makedirs(images_dir)

        # Create one image
        image_array = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        image = Image.fromarray(image_array).convert("L")
        image_path = os.path.join(images_dir, "test_image_0.png")
        image.save(image_path)

        # Write labels file with incorrect field name ('filename' instead of 'file')
        labels_data = {"labels": [{"filename": "test_image_0.png", "label": "na"}]}
        labels_file_path = os.path.join(data_dir, "bad_labels.yaml")
        with open(labels_file_path, "w") as f:
            yaml.dump(labels_data, f)

        config = MagicMock()
        config.data.data_dir = images_dir
        config.data.train_dir = images_dir
        config.data.labels_file = labels_file_path
        config.data.image_size = 256
        config.data.train_ratio = 1.0
        config.data.val_ratio = 0.0
        config.data.test_ratio = 0.0
        config.data.num_workers = 1
        config.data.augment = False
        config.data.data_loader = "RawPngLoader"
        config.model.batch_size = 1
        config.seed = 42
        config.accelerator = "cpu"
        config.task = "binary"

        # Should raise KeyError or ValueError due to missing 'file' field
        with pytest.raises((KeyError, ValueError)):
            RawPngLoader(config)
