"""
Tests for create_average_boxes.py script
"""

import os
import pytest
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path

from scripts.create_average_boxes import (
    logical_sort,
    infer_array_dimensions,
    convert_all_to_boxes
)


class TestLogicalSort:
    """Test logical sorting functionality"""
    
    def test_logical_sort_basic(self):
        """Test basic logical sorting"""
        file_list = [
            "test_1_2.png",
            "test_0_1.png", 
            "test_2_0.png",
            "test_0_0.png"
        ]
        
        sorted_files, coordinates = logical_sort(file_list)
        
        # Should be sorted by coordinates
        expected_order = [
            "test_0_0.png",
            "test_0_1.png",
            "test_1_2.png", 
            "test_2_0.png"
        ]
        
        assert sorted_files == expected_order
        assert len(coordinates) == len(file_list)


class TestInferArrayDimensions:
    """Test automatic array dimension inference"""
    
    def test_infer_dimensions_3x3(self):
        """Test inferring dimensions for 3x3 grid"""
        file_list = [
            f"test_{i}_{j}.png" 
            for i in range(3) 
            for j in range(3)
        ]
        
        array_length, dim1, dim2 = infer_array_dimensions(file_list)
        
        assert dim1 == 3  # max x + 1
        assert dim2 == 3  # max y + 1
        assert array_length == 3