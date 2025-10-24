"""
Integration tests for inference pipeline
"""

import os
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from src.models import ThreeLayerCnn


class TestInferenceCore:
    """Test core inference functionality"""

    def test_model_loading_for_inference(self, temp_dir):
        """Test loading model from checkpoint for inference"""
        # Create test model and save checkpoint
        config = MagicMock()
        config.model.in_channels = 1
        config.model.batch_size = 2
        config.accelerator = "cpu"
        config.task = "binary"

        model = ThreeLayerCnn(config)
        checkpoint_path = os.path.join(temp_dir, "test_model.ckpt")
        torch.save(model.state_dict(), checkpoint_path)

        # Load model
        new_model = ThreeLayerCnn(config)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        new_model.load_state_dict(state_dict)

        # Set to eval mode
        new_model.eval()

        # Test inference on dummy data
        with torch.no_grad():
            test_input = torch.randn(1, 1, 256, 256)
            output = new_model(test_input)

            # Check output shape for binary classification
            assert output.shape == (1, 3)

            # Check that output is valid probabilities after softmax
            probs = torch.softmax(output, dim=1)
            assert torch.allclose(probs.sum(dim=1), torch.tensor(1.0))

    def test_single_image_inference(self, temp_dir):
        """Test inference on a single image"""
        # Create model
        config = MagicMock()
        config.model.in_channels = 1
        config.model.batch_size = 2
        config.accelerator = "cpu"
        config.task = "binary"

        model = ThreeLayerCnn(config)

        # Create test image
        image_array = np.random.randint(0, 1, (256, 256), dtype=np.uint8)
        image = Image.fromarray(image_array).convert("L")
        test_image_path = os.path.join(temp_dir, "test_image.png")
        image.save(test_image_path)

        # Load and preprocess image
        loaded_image = Image.open(test_image_path).convert("L")
        image_array = np.array(loaded_image) / 255.0  # Normalize
        image_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)

        # Run inference
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            predictions = torch.softmax(output, dim=1)

        # Verify output
        assert predictions.shape == (1, 3)
        assert torch.allclose(predictions.sum(dim=1), torch.tensor(1.0))
        assert 0 <= predictions.max().item() <= 1
