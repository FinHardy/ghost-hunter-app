"""
Tests for model architectures
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from src.models import ResidualCNN, ThreeLayerCnn


class TestThreeLayerCnn:
    """Test ThreeLayerCnn model"""

    @pytest.fixture
    def model_config(self):
        """Create mock config for testing"""
        config = MagicMock()
        config.model.in_channels = 1
        config.model.batch_size = 2
        config.accelerator = "cpu"
        config.task = "binary"
        return config

    def test_model_initialization(self, model_config):
        """Test model initialization"""
        model = ThreeLayerCnn(model_config)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "conv3")
        assert hasattr(model, "fc1")
        assert hasattr(model, "fc2")

    def test_forward_pass_shape(self, model_config):
        """Test forward pass output shape"""
        model = ThreeLayerCnn(model_config)
        model.eval()

        # Test with typical input size
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 256, 256)

        with torch.no_grad():
            output = model(input_tensor)

        # For binary classification, expect shape [batch_size, num_classes]
        assert output.shape[0] == batch_size
        assert len(output.shape) == 2

    def test_model_parameters_require_grad(self, model_config):
        """Test that model parameters require gradients"""
        model = ThreeLayerCnn(model_config)

        for param in model.parameters():
            assert param.requires_grad


class TestResidualCNN:
    """Test ResidualCNN model"""

    @pytest.fixture
    def model_config(self):
        """Create mock config for testing"""
        config = MagicMock()
        config.model.in_channels = 1
        config.model.batch_size = 2
        config.accelerator = "cpu"
        config.task = "binary"
        return config

    def test_model_initialization(self, model_config):
        """Test ResidualCNN initialization"""
        model = ResidualCNN(model_config)

        assert isinstance(model, nn.Module)
        # Check for expected ResidualCNN components
        assert hasattr(model, "block1")

    def test_forward_pass_shape(self, model_config):
        """Test ResidualCNN forward pass"""
        model = ResidualCNN(model_config)
        model.eval()

        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 256, 256)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape[0] == batch_size
        assert len(output.shape) == 2


class TestModelUtils:
    """Test model utility functions"""

    def test_model_device_consistency(self):
        """Test that models can be moved to different devices"""
        config = MagicMock()
        config.model.in_channels = 1
        config.model.batch_size = 2
        config.accelerator = "cpu"
        config.task = "binary"

        model = ThreeLayerCnn(config)

        # Test moving to CPU (should always work)
        model = model.to("cpu")

        # Check that all parameters are on CPU
        for param in model.parameters():
            assert param.device.type == "cpu"
