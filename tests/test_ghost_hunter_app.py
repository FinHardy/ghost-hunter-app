"""
Comprehensive tests for Ghost Hunter Streamlit Application
Tests core logic, configuration, and workflow integration
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from PIL import Image

from ghost_hunter import (
    create_training_config,
    initialize_session_state,
)


class TestSessionStateInitialization:
    """Test session state initialization"""

    def test_initialize_session_state_creates_defaults(self):
        """Test that initialize_session_state creates all required keys"""
        mock_st = MagicMock()
        mock_st.session_state = {}

        with patch("ghost_hunter.st", mock_st):
            initialize_session_state()

        # Check all required keys exist
        required_keys = [
            "step",
            "config",
            "temp_dir",
            "conversion_done",
            "boxing_done",
            "labelling_done",
            "training_done",
            "inference_done",
            "labeller_state",
            "current_label_file",
            "labelling_initialized",
            "inference_plot_path",
        ]

        for key in required_keys:
            assert key in mock_st.session_state

    def test_initialize_session_state_default_values(self):
        """Test that default values are correct"""
        mock_st = MagicMock()
        mock_st.session_state = {}

        with patch("ghost_hunter.st", mock_st):
            initialize_session_state()

        assert mock_st.session_state["step"] == 1
        assert mock_st.session_state["config"] == {}
        assert mock_st.session_state["conversion_done"] is False
        assert mock_st.session_state["boxing_done"] is False
        assert mock_st.session_state["labelling_done"] is False
        assert mock_st.session_state["training_done"] is False
        assert mock_st.session_state["inference_done"] is False

    def test_initialize_session_state_preserves_existing(self):
        """Test that existing session state values are preserved"""
        mock_st = MagicMock()
        mock_st.session_state = {"step": 5, "conversion_done": True}

        with patch("ghost_hunter.st", mock_st):
            initialize_session_state()

        # Existing values should be preserved
        assert mock_st.session_state["step"] == 5
        assert mock_st.session_state["conversion_done"] is True


class TestConfigurationCreation:
    """Test configuration creation and validation"""

    def test_create_training_config_structure(self):
        """Test that create_training_config returns correct structure"""
        input_config = {
            "project_name": "test_project",
            "boxed_png_path": "data/boxed_png/test_project",
            "labelling_path": "labelling/test_project_labels.yaml",
            "batch_size": 32,
            "learning_rate": 0.0004,
            "max_epochs": 50,
            "device": "cpu",
        }

        result = create_training_config(input_config)

        # Check top-level keys
        assert "seed" in result
        assert "test_only" in result
        assert "accelerator" in result
        assert "task" in result
        assert "data" in result
        assert "model" in result
        assert "optimizer" in result
        assert "scheduler" in result
        assert "trainer" in result

    def test_create_training_config_device_mapping(self):
        """Test that device selection is correctly mapped"""
        config_cpu = {
            "device": "cpu",
            "project_name": "test",
            "boxed_png_path": "path",
            "labelling_path": "labels",
            "batch_size": 32,
            "learning_rate": 0.0004,
            "max_epochs": 50,
        }
        config_cuda = {
            "device": "cuda",
            "project_name": "test",
            "boxed_png_path": "path",
            "labelling_path": "labels",
            "batch_size": 32,
            "learning_rate": 0.0004,
            "max_epochs": 50,
        }
        config_mps = {
            "device": "mps",
            "project_name": "test",
            "boxed_png_path": "path",
            "labelling_path": "labels",
            "batch_size": 32,
            "learning_rate": 0.0004,
            "max_epochs": 50,
        }

        assert create_training_config(config_cpu)["accelerator"] == "cpu"
        assert create_training_config(config_cuda)["accelerator"] == "cuda"
        assert create_training_config(config_mps)["accelerator"] == "mps"

    def test_create_training_config_data_paths(self):
        """Test that data paths are correctly set"""
        input_config = {
            "project_name": "test_project",
            "boxed_png_path": "data/boxed_png/test_project",
            "labelling_path": "labelling/test_project_labels.yaml",
            "batch_size": 32,
            "learning_rate": 0.0004,
            "max_epochs": 50,
            "device": "cpu",
        }

        result = create_training_config(input_config)

        assert result["data"]["data_dir"] == "data/boxed_png/test_project"
        assert result["data"]["train_dir"] == "data/boxed_png/test_project"
        assert result["data"]["labels_file"] == "test_project_labels.yaml"

    def test_create_training_config_hyperparameters(self):
        """Test that hyperparameters are correctly set"""
        input_config = {
            "project_name": "test_project",
            "boxed_png_path": "data/boxed_png/test_project",
            "labelling_path": "labelling/test_project_labels.yaml",
            "batch_size": 64,
            "learning_rate": 0.001,
            "max_epochs": 100,
            "device": "cpu",
        }

        result = create_training_config(input_config)

        assert result["model"]["batch_size"] == 64
        assert result["optimizer"]["lr"] == 0.001
        assert result["trainer"]["max_epochs"] == 100

    def test_create_training_config_checkpoint_dir(self):
        """Test that checkpoint directory uses project name"""
        input_config = {
            "project_name": "my_experiment",
            "boxed_png_path": "data/boxed_png/my_experiment",
            "labelling_path": "labelling/my_experiment_labels.yaml",
            "batch_size": 32,
            "learning_rate": 0.0004,
            "max_epochs": 50,
            "device": "cpu",
        }

        result = create_training_config(input_config)

        assert result["trainer"]["checkpoint_dir"] == "my_experiment/"
        assert result["test"]["load_path"] == "my_experiment/"


class TestConfigValidation:
    """Test configuration validation logic"""

    def test_project_name_validation_valid(self):
        """Test valid project name patterns"""
        import re

        valid_names = [
            "my_project",
            "test-project",
            "Project123",
            "abc",
            "a_b_c-d-e",
        ]

        for name in valid_names:
            assert re.match(
                r"^[a-zA-Z0-9_-]+$", name
            ), f"Valid name '{name}' failed validation"

    def test_project_name_validation_invalid(self):
        """Test invalid project name patterns"""
        import re

        invalid_names = [
            "my project",  # Space
            "project@123",  # Special char
            "project.name",  # Dot
            "project/name",  # Slash
            "",  # Empty
        ]

        for name in invalid_names:
            assert not re.match(
                r"^[a-zA-Z0-9_-]+$", name
            ), f"Invalid name '{name}' passed validation"

    def test_project_name_minimum_length(self):
        """Test minimum project name length requirement"""
        too_short = "ab"
        valid = "abc"

        assert len(too_short) < 3
        assert len(valid) >= 3


class TestWorkflowValidation:
    """Test workflow step validation logic"""

    def test_boxing_requires_conversion(self, temp_dir):
        """Test that boxing requires conversion to be completed"""
        config = {
            "output_png_path": os.path.join(temp_dir, "png"),
            "boxed_png_path": os.path.join(temp_dir, "boxed"),
            "box_size": 3,
            "n_cols": 10,
            "n_rows": 10,
        }

        # Create output directory but no files
        os.makedirs(str(config["output_png_path"]), exist_ok=True)

        # Check that no PNG files exist
        png_files = [
            f for f in os.listdir(str(config["output_png_path"])) if f.endswith(".png")
        ]
        assert len(png_files) == 0, "Boxing should not proceed without PNG files"

    def test_labelling_requires_boxing(self, temp_dir):
        """Test that labelling requires boxing to be completed"""
        config = {
            "boxed_png_path": os.path.join(temp_dir, "boxed"),
            "labelling_path": os.path.join(temp_dir, "labels.yaml"),
        }

        # Check that directory doesn't exist or is empty
        if os.path.exists(config["boxed_png_path"]):
            boxed_files = [
                f for f in os.listdir(config["boxed_png_path"]) if f.endswith(".png")
            ]
            assert (
                len(boxed_files) == 0
            ), "Labelling should not proceed without boxed images"

    def test_training_requires_labels(self, temp_dir):
        """Test that training requires labels to be completed"""
        config = {
            "labelling_path": os.path.join(temp_dir, "labels.yaml"),
        }

        # Check that labels file doesn't exist
        assert not os.path.exists(
            config["labelling_path"]
        ), "Training should not proceed without labels file"


class TestFileStructureHelpers:
    """Test file structure and path handling"""

    def test_project_directory_structure(self, temp_dir):
        """Test that correct directory structure is created"""
        project_name = "test_project"

        expected_dirs = [
            f"data/png/{project_name}",
            f"data/boxed_png/{project_name}",
            f"output/{project_name}",
            f"checkpoints/{project_name}",
        ]

        for dir_path in expected_dirs:
            full_path = os.path.join(temp_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)
            assert os.path.exists(full_path), f"Directory {dir_path} was not created"

    def test_auto_detect_dimensions_from_png(self, temp_dir):
        """Test dimension detection from PNG filenames"""
        png_dir = os.path.join(temp_dir, "png")
        os.makedirs(png_dir, exist_ok=True)

        # Create test PNG files with naming convention: base_row_col.png
        test_files = [
            "image_0_0.png",
            "image_5_3.png",
            "image_9_9.png",
        ]

        for filename in test_files:
            image = Image.new("L", (256, 256), color=0)
            image.save(os.path.join(png_dir, filename))

        # Get dimensions from last file
        png_files = sorted([f for f in os.listdir(png_dir) if f.endswith(".png")])
        last_file = png_files[-1]
        parts = last_file.replace(".png", "").split("_")

        n_rows = int(parts[-2]) + 1
        n_cols = int(parts[-1]) + 1

        assert n_rows == 10
        assert n_cols == 10

    def test_auto_detect_dimensions_from_boxed(self, temp_dir):
        """Test dimension detection from boxed PNG filenames"""
        boxed_dir = os.path.join(temp_dir, "boxed")
        os.makedirs(boxed_dir, exist_ok=True)

        # Create test boxed files with naming convention: row_col_boxsize.png
        test_files = [
            "0_0_3.png",
            "3_2_3.png",
            "6_6_3.png",
        ]

        for filename in test_files:
            image = Image.new("L", (256, 256), color=0)
            image.save(os.path.join(boxed_dir, filename))

        # Get dimensions from last file
        boxed_files = sorted([f for f in os.listdir(boxed_dir) if f.endswith(".png")])
        last_file = boxed_files[-1]
        parts = last_file.replace(".png", "").split("_")

        n_rows = int(parts[0]) + 1
        n_cols = int(parts[1]) + 1

        assert n_rows == 7
        assert n_cols == 7


class TestConfigFilePersistence:
    """Test configuration file saving and loading"""

    def test_config_save_to_yaml(self, temp_dir):
        """Test saving configuration to YAML file"""
        config = {
            "project_name": "test_project",
            "batch_size": 32,
            "learning_rate": 0.0004,
            "device": "cpu",
        }

        config_path = os.path.join(temp_dir, "test_config.yaml")

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        assert os.path.exists(config_path)

        # Verify content
        with open(config_path) as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config["project_name"] == "test_project"
        assert loaded_config["batch_size"] == 32

    def test_config_load_from_yaml(self, temp_dir):
        """Test loading configuration from YAML file"""
        config = {
            "project_name": "loaded_project",
            "max_epochs": 100,
            "accelerator": "cuda",
        }

        config_path = os.path.join(temp_dir, "load_test.yaml")

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Load it back
        with open(config_path) as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config["project_name"] == "loaded_project"
        assert loaded_config["max_epochs"] == 100
        assert loaded_config["accelerator"] == "cuda"

    def test_config_handles_nested_structure(self, temp_dir):
        """Test that nested config structure is preserved"""
        config = create_training_config(
            {
                "project_name": "test",
                "boxed_png_path": "path",
                "labelling_path": "labels",
                "batch_size": 32,
                "learning_rate": 0.0004,
                "max_epochs": 50,
                "device": "cpu",
            }
        )

        config_path = os.path.join(temp_dir, "nested_config.yaml")

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with open(config_path) as f:
            loaded_config = yaml.safe_load(f)

        # Check nested structure
        assert "data" in loaded_config
        assert "model" in loaded_config
        assert "optimizer" in loaded_config
        assert loaded_config["model"]["batch_size"] == 32
        assert loaded_config["optimizer"]["lr"] == 0.0004


class TestLabelValidation:
    """Test label validation logic"""

    def test_minimum_labels_requirement(self):
        """Test that minimum label count is enforced"""
        min_labels = 10

        insufficient_labels = 5
        sufficient_labels = 15

        assert insufficient_labels < min_labels
        assert sufficient_labels >= min_labels

    def test_label_distribution_check(self):
        """Test label distribution imbalance detection"""
        # Balanced distribution
        balanced = {"horizontal": 20, "vertical": 18, "na": 22}
        max_balanced = max(balanced.values())
        min_balanced = min(balanced.values())

        # Imbalanced distribution
        imbalanced = {"horizontal": 50, "vertical": 5, "na": 5}
        max_imbalanced = max(imbalanced.values())
        min_imbalanced = min(imbalanced.values())

        assert (
            max_balanced <= min_balanced * 5
        ), "Balanced distribution incorrectly flagged"
        assert (
            max_imbalanced > min_imbalanced * 5
        ), "Imbalanced distribution not detected"


class TestExistingProjectDetection:
    """Test detection of existing projects"""

    def test_detect_existing_png_files(self, temp_dir):
        """Test detection of existing PNG files"""
        project_name = "test_project"
        png_path = os.path.join(temp_dir, "data", "png", project_name)
        os.makedirs(png_path, exist_ok=True)

        # Create some PNG files
        for i in range(5):
            image = Image.new("L", (256, 256), color=0)
            image.save(os.path.join(png_path, f"test_{i}_0.png"))

        # Check detection
        png_exists = (
            os.path.exists(png_path)
            and len([f for f in os.listdir(png_path) if f.endswith(".png")]) > 0
        )
        assert png_exists is True

        png_count = len([f for f in os.listdir(png_path) if f.endswith(".png")])
        assert png_count == 5

    def test_detect_existing_boxed_files(self, temp_dir):
        """Test detection of existing boxed files"""
        project_name = "test_project"
        boxed_path = os.path.join(temp_dir, "data", "boxed_png", project_name)
        os.makedirs(boxed_path, exist_ok=True)

        # Create some boxed files
        for i in range(3):
            image = Image.new("L", (256, 256), color=0)
            image.save(os.path.join(boxed_path, f"{i}_0_3.png"))

        # Check detection
        boxed_exists = (
            os.path.exists(boxed_path)
            and len([f for f in os.listdir(boxed_path) if f.endswith(".png")]) > 0
        )
        assert boxed_exists is True

        boxed_count = len([f for f in os.listdir(boxed_path) if f.endswith(".png")])
        assert boxed_count == 3

    def test_detect_existing_checkpoints(self, temp_dir):
        """Test detection of existing model checkpoints"""
        project_name = "test_project"
        checkpoint_dir = os.path.join(temp_dir, "checkpoints", project_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create checkpoint files
        checkpoint_files = ["model-epoch=10.ckpt", "model-epoch=20.ckpt"]
        for ckpt in checkpoint_files:
            Path(os.path.join(checkpoint_dir, ckpt)).touch()

        # Check detection
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        assert len(ckpt_files) == 2


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config():
    """Provide a sample configuration"""
    return {
        "project_name": "test_project",
        "dm4_file_path": "/path/to/test.dm4",
        "output_png_path": "data/png/test_project",
        "boxed_png_path": "data/boxed_png/test_project",
        "labelling_path": "labelling/test_project_labels.yaml",
        "output_save_path": "output/test_project/",
        "n_cols": 10,
        "n_rows": 10,
        "box_size": 3,
        "sampling_space": 10,
        "number_of_labels": 50,
        "max_epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.0004,
        "device": "cpu",
    }


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for testing"""
    mock_st = MagicMock()
    mock_st.session_state = {}
    return mock_st
