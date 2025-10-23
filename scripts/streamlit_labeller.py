"""
Streamlit-native labelling interface for Ghost Hunter
Refactored from binary_search_labeller.py to work seamlessly within Streamlit
"""

import os
import numpy as np
import yaml
from typing import Optional, Tuple, List, Dict, Any


def logical_sort_coordinates(file_list: List[str]) -> List[str]:
    """
    Sorts a list of file paths logically based on the numeric parts of the filenames.
    
    Args:
        file_list: List of file paths to sort.
    
    Returns:
        Sorted list of file paths.
    """
    def logical_sort_key(filepath: str) -> Tuple[int, ...]:
        filename = os.path.basename(filepath)
        # Extract numeric parts from filename (e.g., _0_15 -> [0, 15])
        coordinates_plus_boxsize = filename.split("_")
        numbers = coordinates_plus_boxsize[0:2]
        return tuple(map(int, numbers))
    
    return sorted(file_list, key=logical_sort_key)


def get_files(file_path: str) -> List[str]:
    """Get all PNG files from directory."""
    files = []
    for root, dirs, filenames in os.walk(file_path):
        for filename in filenames:
            if filename.endswith('.png'):
                files.append(os.path.join(root, filename))
    return files


def do_sparse_sampling(step: int, width: int, height: int) -> List[int]:
    """Generate sparse sampling indices in a grid pattern."""
    sparse_indices = []
    for i in range(0, height, step):
        for j in range(0, width, step):
            sparse_indices.append((i * width) + j)
    return sparse_indices


def save_label(file_name: str, output_file: str, label: str) -> int:
    """
    Save a label to the output YAML file.
    Optimized to reduce I/O overhead.
    
    Returns:
        Total number of labels saved so far
    """
    # Read existing data
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data = yaml.safe_load(f)
            if data is None:
                data = {"labels": []}
            elif "labels" not in data:
                data["labels"] = []
    else:
        data = {"labels": []}
    
    # Append new label
    data["labels"].append({"file": file_name, "label": label})
    
    # Write with faster YAML dumping (no sorting, no explicit start)
    with open(output_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    return len(data["labels"])


def load_existing_labels(output_file: str) -> Dict[str, Any]:
    """Load existing labels from YAML file."""
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data = yaml.safe_load(f) or {"labels": []}
    else:
        data = {"labels": []}
    
    if "labels" not in data or not isinstance(data["labels"], list):
        data["labels"] = []
    
    return data


def get_label_statistics(output_file: str) -> Dict[str, int]:
    """Get statistics on label distribution."""
    data = load_existing_labels(output_file)
    label_counts = {}
    for item in data.get("labels", []):
        label = item.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    return label_counts


class StreamlitLabellerState:
    """
    State management class for the Streamlit labeller.
    This replaces the Tkinter-based LabelingApp class.
    """
    
    def __init__(self, file_path: str, output_file: str, labels_to_assign: int, step: int = 20):
        """Initialize the labeller state."""
        self.image_directory = file_path
        self.output_file = output_file
        self.labels_to_assign = labels_to_assign
        self.step = step
        
        # Validate path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path '{file_path}' does not exist.")
        
        # Get and sort files
        files = [os.path.basename(f) for f in get_files(file_path)]
        self.file_list = logical_sort_coordinates(files)
        
        if len(self.file_list) == 0:
            raise ValueError(f"No PNG files found in {file_path}")
        
        if labels_to_assign > len(self.file_list):
            raise ValueError(
                f"Number of labels to assign ({labels_to_assign}) is greater than "
                f"the number of files available ({len(self.file_list)})."
            )
        
        # Get scan dimensions from last file
        final_file = self.file_list[-1]
        dimensions = final_file.split("_")[0:2]
        self.height = int(dimensions[0]) + 1
        self.width = int(dimensions[1]) + 1
        
        # Reshape file list to 2D array
        self.file_list_array = np.array(self.file_list)
        self.file_list_array_reshaped = self.file_list_array.reshape(
            (self.height, self.width)
        )
        
        # Track labels in sparse array (0=unlabeled, 1=horizontal, 2=vertical, 3=none)
        self.sparse_array = np.zeros((self.height, self.width))
        
        # Generate sparse indices for initial sampling
        self.sparse_indices = do_sparse_sampling(
            step=self.step, width=self.width, height=self.height
        )
        
        # Check if we have enough sparse indices
        self.label_number_after_sparse_labelling = labels_to_assign - len(self.sparse_indices)
        if self.label_number_after_sparse_labelling < 0:
            raise ValueError(
                f"Label number {labels_to_assign} is less than the number of "
                f"sparse indices {len(self.sparse_indices)}."
            )
        
        # Initialize state variables
        self.step_through_sparse_indices = 0
        self.labels_assigned = 0
        self.binary_search_iteration = 0
        self.cache = {}  # Cache for binary search
        self.non_zero_coords = []
        self.num_non_zero_coords = 0
        self.current_index = 1
        self.current_file = None
    
    def get_next_file(self) -> Optional[str]:
        """
        Get the next file to label based on sparse sampling or binary search.
        
        Returns:
            Filename to label next, or None if all labels assigned
        """
        if self.labels_assigned >= self.labels_to_assign:
            return None
        
        # First, do sparse sampling
        if self.step_through_sparse_indices < len(self.sparse_indices):
            selected_index = self.sparse_indices[self.step_through_sparse_indices]
            selected_file = str(self.file_list[selected_index])
            self.step_through_sparse_indices += 1
            self.current_file = selected_file
            return selected_file
        
        # Then, use binary search for boundary refinement
        coord = None
        attempts = 0
        max_attempts = 100  # Prevent infinite loops
        
        while coord is None and attempts < max_attempts:
            coord = self._binary_search_for_boundary()
            attempts += 1
        
        if coord is None:
            # If binary search fails, we're done
            return None
        
        i, j = coord
        selected_file = str(self.file_list_array_reshaped[i, j])
        self.current_file = selected_file
        return selected_file
    
    def _binary_search_for_boundary(self) -> Optional[Tuple[int, int]]:
        """
        Search for the boundary between differently labeled regions.
        Returns the coordinate of the next point to label.
        """
        # Refresh non-zero coordinates if needed
        if self.current_index >= self.num_non_zero_coords:
            self.binary_search_iteration += 1
            self.non_zero_coords = np.argwhere(self.sparse_array != 0)
            
            if len(self.non_zero_coords) == 0:
                return None
            
            self.non_zero_coords = [tuple(coord) for coord in self.non_zero_coords]
            self.num_non_zero_coords = len(self.non_zero_coords)
            self.current_index = 0
        
        # Get current coordinate
        current_coord = self.non_zero_coords[self.current_index]
        current_i, current_j = current_coord
        current_coord_value = self.sparse_array[current_i, current_j]
        
        # Find closest coordinate with different label
        ret_closest_coord = None
        min_distance = float("inf")
        max_distance_between_sparse_coords = np.sqrt(2 * self.step**2)
        
        for coord in self.non_zero_coords:
            if coord != current_coord:
                dist = np.linalg.norm(np.array(coord) - np.array(current_coord))
                if (dist < min_distance and 
                    self.sparse_array[coord] != current_coord_value and 
                    dist <= max_distance_between_sparse_coords):
                    
                    mid_coord = (
                        int((current_i + coord[0]) / 2),
                        int((current_j + coord[1]) / 2),
                    )
                    
                    if mid_coord not in self.cache:
                        ret_closest_coord = coord
                        min_distance = dist
        
        self.current_index += 1
        
        if ret_closest_coord is not None:
            mid_coord = (
                int((current_i + ret_closest_coord[0]) / 2),
                int((current_j + ret_closest_coord[1]) / 2),
            )
            self.cache[mid_coord] = True
            return mid_coord
        
        return None
    
    def update_sparse_array(self, label_value: int):
        """
        Update the sparse array with the label for the current file.
        
        Args:
            label_value: 1=horizontal, 2=vertical, 3=none
        """
        if self.current_file is None:
            return
        
        pos = np.where(self.file_list_array_reshaped == self.current_file)
        if pos[0].size > 0 and pos[1].size > 0:
            i, j = pos[0][0], pos[1][0]
            self.sparse_array[i, j] = label_value
            self.labels_assigned += 1
        else:
            raise ValueError(f"File '{self.current_file}' not found in file array.")
    
    def save_final_heatmap(self) -> np.ndarray:
        """
        Save and return the final label heatmap.
        Applies transformations to match original image orientation.
        """
        out_image = self.sparse_array.copy()
        return out_image
    
    def get_current_position(self) -> Optional[Tuple[int, int]]:
        """Get the (i, j) position of the current file in the array."""
        if self.current_file is None:
            return None
        
        pos = np.where(self.file_list_array_reshaped == self.current_file)
        if pos[0].size > 0 and pos[1].size > 0:
            return (pos[0][0], pos[1][0])
        return None
    
    def is_complete(self) -> bool:
        """Check if all labels have been assigned."""
        return self.labels_assigned >= self.labels_to_assign
    
    def get_progress(self) -> Tuple[int, int]:
        """Get (current, total) labels assigned."""
        return (self.labels_assigned, self.labels_to_assign)
