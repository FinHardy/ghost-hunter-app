"""
Tests for create_average_boxes.py script
"""

from scripts.create_average_boxes import (
    # convert_all_to_boxes,
    logical_sort,
)


class TestLogicalSort:
    """Test logical sorting functionality"""

    def test_logical_sort_basic(self):
        """Test basic logical sorting"""
        file_list = ["test_1_2.png", "test_0_1.png", "test_2_0.png", "test_0_0.png"]

        sorted_files, coordinates = logical_sort(file_list)

        # Should be sorted by coordinates
        expected_order = [
            "test_0_0.png",
            "test_0_1.png",
            "test_1_2.png",
            "test_2_0.png",
        ]

        assert sorted_files == expected_order
        assert len(coordinates) == len(file_list)
