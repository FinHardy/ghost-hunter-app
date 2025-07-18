"""
Test configuration and setup for Ghost Hunter project
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

# Add src to path for imports
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
sys.path.insert(0, str(PROJECT_ROOT))

# Test data paths
TEST_DATA_DIR = os.path.join(TEST_DIR, "test_data")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def device():
    """Get appropriate device for testing"""
    return torch.device("cpu")  # Force CPU for consistent testing
