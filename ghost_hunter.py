"""
Advanced Ghost Hunter Web App
Complete Streamlit implementation with full workflow integration
"""

import os
import random
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent))

try:
    from scripts.create_average_boxes import convert_all_to_boxes
    from scripts.dm4_to_png import export_dm4_bf_images_to_png
    from scripts.hdf5_to_png import save_h5_diffraction_to_png
    from scripts.streamlit_labeller import (
        StreamlitLabellerState,
        get_label_statistics,
        load_existing_labels,
        save_label,
    )
    from src.inference import plot_embeddings
    from src.run import main as train_model
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Ghost Hunter ML Pipeline",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "step": 1,
        "config": {},
        "temp_dir": tempfile.mkdtemp(),
        "conversion_done": False,
        "boxing_done": False,
        "labelling_done": False,
        "training_done": False,
        "inference_done": False,
        # Labelling-specific state
        "labeller_state": None,
        "current_label_file": None,
        "labelling_initialized": False,
        # Inference result
        "inference_plot_path": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Main application function"""
    initialize_session_state()

    # Header
    st.title("👻 Ghost Hunter: Interactive ML Pipeline")
    st.markdown(
        "**Transform DM4/HDF5 files → Label data → Train models → Generate insights**"
    )

    # Sidebar with status indicators
    with st.sidebar:
        st.title("🚀 Pipeline Status")

        # Status indicators
        st.write("### Status")
        status_icons = {
            "conversion_done": "✅ Conversion Complete"
            if st.session_state.conversion_done
            else "⏳ Conversion Pending",
            "boxing_done": "✅ Boxing Complete"
            if st.session_state.boxing_done
            else "⏳ Boxing Pending",
            "labelling_done": "✅ Labelling Complete"
            if st.session_state.labelling_done
            else "⏳ Labelling Pending",
            "training_done": "✅ Training Complete"
            if st.session_state.training_done
            else "⏳ Training Pending",
            "inference_done": "✅ Inference Complete"
            if st.session_state.inference_done
            else "⏳ Inference Pending",
        }

        for status in status_icons.values():
            st.write(status)

    # Display all sections on one page
    setup_configuration()
    st.divider()

    dm4_conversion()
    st.divider()

    average_boxing()
    st.divider()

    data_labelling()
    st.divider()

    model_training()
    st.divider()

    inference_results()


def setup_configuration():
    """Step 1: Setup and Configuration"""
    st.header("⚙️ Step 1: Setup & Configuration")

    # Create two columns for new project vs existing project
    st.subheader("Choose Your Path")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🆕 Create New Project")
        st.caption("Start a fresh project from scratch")

        project_name = st.text_input(
            "Enter a unique project name",
            value="my_ghost_hunter_project",
            key="project_name_new",
            help="Use only letters, numbers, underscores, and hyphens.",
        )

        # Validate project name
        project_name_valid = False
        if project_name:
            import re

            if not re.match(r"^[a-zA-Z0-9_-]+$", project_name):
                st.error("❌ Invalid characters in name")
            elif len(project_name) < 3:
                st.warning("⚠️ Name too short (min 3 chars)")
            else:
                st.success("✅ Valid name")
                project_name_valid = True

        use_new_project = st.checkbox("Use this new project", key="use_new", value=True)

    with col_right:
        st.markdown("### 📂 Load Existing Project")
        st.caption("Continue working on a previous project")

        # Get existing config files
        configs_dir = "configs"
        existing_configs = []
        if os.path.exists(configs_dir):
            existing_configs = [
                f.replace(".yaml", "")
                for f in os.listdir(configs_dir)
                if f.endswith(".yaml") and not f.startswith(".")
            ]

        if existing_configs:
            selected_config = st.selectbox(
                "Select an existing project",
                options=["", *sorted(existing_configs)],
                key="existing_project_select",
                help="Choose from previously created projects",
            )

            if selected_config:
                st.success(f"✅ Selected: `{selected_config}`")
                # Load the config to show summary
                try:
                    config_path = os.path.join(configs_dir, f"{selected_config}.yaml")
                    with open(config_path) as f:
                        loaded_config = yaml.safe_load(f)

                    with st.expander("📋 Project Info"):
                        # Display data configuration
                        if "data" in loaded_config:
                            st.markdown("**📁 Data Configuration:**")
                            data_config = loaded_config["data"]
                            st.write(
                                f"- Data Directory: `{data_config.get('data_dir', 'N/A')}`"
                            )
                            st.write(
                                f"- Image Size: {data_config.get('image_size', 'N/A')}px"
                            )
                            st.write(
                                f"- Labels File: `{data_config.get('labels_file', 'N/A')}`"
                            )
                            st.write(
                                f"- Train/Val/Test: {data_config.get('train_ratio', 'N/A')}/{data_config.get('val_ratio', 'N/A')}/{data_config.get('test_ratio', 'N/A')}"
                            )

                        # Display model configuration
                        if "model" in loaded_config:
                            st.markdown("**🧠 Model Configuration:**")
                            model_config = loaded_config["model"]
                            st.write(
                                f"- Model: {model_config.get('model_name', 'N/A')}"
                            )
                            st.write(
                                f"- Batch Size: {model_config.get('batch_size', 'N/A')}"
                            )
                            st.write(
                                f"- Channels: {model_config.get('in_channels', 'N/A')}"
                            )

                        # Display training configuration
                        st.markdown("**⚙️ Training Configuration:**")
                        st.write(
                            f"- Device: {loaded_config.get('accelerator', 'N/A').upper()}"
                        )
                        st.write(f"- Task: {loaded_config.get('task', 'N/A')}")

                        if "optimizer" in loaded_config:
                            opt_config = loaded_config["optimizer"]
                            st.write(
                                f"- Optimizer: {opt_config.get('optimizer', 'N/A')}"
                            )
                            st.write(f"- Learning Rate: {opt_config.get('lr', 'N/A')}")
                            st.write(
                                f"- Weight Decay: {opt_config.get('weight_decay', 'N/A')}"
                            )

                except Exception as e:
                    st.warning(f"⚠️ Could not load config details: {str(e)}")

            use_existing_project = st.checkbox(
                "Use this existing project", key="use_existing", value=False
            )
        else:
            st.info("No existing projects found")
            st.caption("Create a new project to get started")
            selected_config = None
            use_existing_project = False

    st.divider()

    # Determine which project to use
    if use_existing_project and selected_config:
        project_name = selected_config
        st.info(f"🔄 Loading existing project: **{project_name}**")
        load_from_config = True
    elif use_new_project and project_name_valid:
        st.info(f"🆕 Creating new project: **{project_name}**")
        load_from_config = False
    else:
        st.warning(
            "⚠️ Please select either a new project name or an existing project to continue."
        )
        return

    # Check if project already exists
    png_path = f"data/png/{project_name}"
    boxed_path = f"data/boxed_png/{project_name}"
    project_exists = os.path.exists(png_path) or os.path.exists(boxed_path)

    if project_exists:
        st.info(
            f"ℹ*Existing project detected!** If PNG files and/or boxed images already exist for '{project_name}', you can skip those steps and proceed directly to labelling or training."
        )

        # Show what exists
        png_exists = (
            os.path.exists(png_path)
            and len([f for f in os.listdir(png_path) if f.endswith(".png")]) > 0
        )
        boxed_exists = (
            os.path.exists(boxed_path)
            and len([f for f in os.listdir(boxed_path) if f.endswith(".png")]) > 0
        )

        if png_exists:
            png_count = len([f for f in os.listdir(png_path) if f.endswith(".png")])
            st.success(f"✅ Found {png_count} PNG files in `{png_path}`")

        if boxed_exists:
            boxed_count = len([f for f in os.listdir(boxed_path) if f.endswith(".png")])
            st.success(f"✅ Found {boxed_count} boxed images in `{boxed_path}`")

    # File path input
    st.subheader("📁 Specify Data File Path")

    # Make file path optional if project exists
    help_text = (
        "Example: /home/user/data/experiment.dm4 or /home/user/data/experiment.h5"
    )
    if project_exists:
        help_text += (
            " (Optional for existing projects - leave blank to skip conversion steps)"
        )

    # Add tabs for file selection method
    tab_browse, tab_manual = st.tabs(["📂 Browse Files", "📝 Manual Path Entry"])

    dm4_file_path = ""

    with tab_browse:
        st.markdown("**Select your data file from the filesystem**")
        st.caption("💡 Navigate to your file using your system's file browser")

        # Provide a directory browser hint
        browse_dir = st.text_input(
            "Start browsing from directory (optional)",
            value=os.path.expanduser("~"),
            key="browse_start_dir",
            help="Enter a directory path to start browsing from, or leave as home directory",
        )

        if os.path.isdir(browse_dir):
            # List files in the directory
            try:
                all_files = []
                for root, dirs, files in os.walk(browse_dir):
                    # Only go one level deep to avoid performance issues
                    if root == browse_dir:
                        for file in files:
                            if file.endswith((".dm4", ".h5", ".hdf5")):
                                all_files.append(os.path.join(root, file))
                        for dir_name in dirs[:20]:  # Limit subdirectories shown
                            dir_path = os.path.join(root, dir_name)
                            for file in os.listdir(dir_path):
                                if file.endswith((".dm4", ".h5", ".hdf5")):
                                    all_files.append(os.path.join(dir_path, file))

                if all_files:
                    all_files.sort()
                    # Create display names (show relative path from browse_dir)
                    display_files = [os.path.relpath(f, browse_dir) for f in all_files]

                    selected_file = st.selectbox(
                        f"Found {len(all_files)} data file(s) - Select one:",
                        options=range(len(all_files)),
                        format_func=lambda i: f"{display_files[i]} ({os.path.getsize(all_files[i]) / (1024*1024):.2f} MB)",
                        key="file_selector",
                    )

                    if selected_file is not None:
                        dm4_file_path = all_files[selected_file]
                        st.success(
                            f"✅ Selected: **{os.path.basename(dm4_file_path)}**"
                        )
                        st.code(dm4_file_path, language="")
                else:
                    st.info(
                        f"No .dm4, .h5, or .hdf5 files found in {browse_dir} or immediate subdirectories"
                    )
                    st.caption(
                        "Try entering the full file path in the 'Manual Path Entry' tab, or change the browse directory above"
                    )
            except PermissionError:
                st.error(f"❌ Permission denied accessing {browse_dir}")
            except Exception as e:
                st.error(f"❌ Error browsing directory: {e}")
        else:
            st.warning(f"⚠️ Directory not found: {browse_dir}")

    with tab_manual:
        dm4_file_path_manual = st.text_input(
            "Enter the full path to your DM4 or HDF5 (.h5) file",
            value="",
            key="dm4_file_path_manual",
            help=help_text,
        )
        if dm4_file_path_manual:
            dm4_file_path = dm4_file_path_manual

    st.subheader("Training Parameters")

    # Load defaults from existing config if available
    default_device = "cpu"
    default_epochs = 50
    default_batch_size = 32
    default_lr = 0.0004

    if load_from_config and selected_config:
        try:
            config_path = os.path.join("configs", f"{selected_config}.yaml")
            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)

            default_device = loaded_config.get("accelerator", "cpu")
            if "model" in loaded_config:
                default_batch_size = loaded_config["model"].get("batch_size", 32)
            if "optimizer" in loaded_config:
                default_lr = loaded_config["optimizer"].get("lr", 0.0004)

            st.caption(
                "ℹ️ Using parameters from existing config (you can modify them below)"
            )
        except Exception as e:
            st.warning(f"⚠️ Could not load all parameters from config: {str(e)}")

    # Device selection
    col1, col2 = st.columns([2, 1])
    with col1:
        device_options = ["cpu", "cuda", "mps"]
        device_descriptions = {
            "cpu": "CPU - Compatible with all systems (slower)",
            "cuda": "CUDA - NVIDIA GPU acceleration (fastest, requires CUDA)",
            "mps": "MPS - Apple Silicon GPU acceleration (M1/M2 Macs)",
        }

        # Set default index based on loaded config
        default_device_index = (
            device_options.index(default_device)
            if default_device in device_options
            else 0
        )

        device = st.selectbox(
            "Training Device",
            options=device_options,
            index=default_device_index,
            format_func=lambda x: device_descriptions[x],
            key="device_select",
            help="Select the device for model training. CUDA requires NVIDIA GPU, MPS requires Apple Silicon.",
        )

    with col2:
        # Device availability check
        try:
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                st.warning("⚠️ CUDA not available")
            elif device == "mps" and not torch.backends.mps.is_available():
                st.warning("⚠️ MPS not available")
            else:
                st.success(f"✅ {device.upper()} ready")
        except ImportError:
            st.info("ℹ️ Install PyTorch to check device availability")

    max_epochs = st.number_input(
        "Max Epochs",
        value=default_epochs,
        min_value=1,
        key="max_epochs",
        help="Number of complete passes through the training data",
    )

    # Set default batch size index
    batch_size_options = [16, 32, 64, 128]
    default_batch_index = (
        batch_size_options.index(default_batch_size)
        if default_batch_size in batch_size_options
        else 1
    )

    batch_size = st.selectbox(
        "Batch Size",
        batch_size_options,
        index=default_batch_index,
        key="batch_size",
        help="Number of samples processed together. Larger = faster but needs more memory",
    )
    learning_rate = st.number_input(
        "Learning Rate",
        value=default_lr,
        format="%.6f",
        key="learning_rate",
        help="Step size for model weight updates. Typical range: 0.0001 - 0.001",
    )

    # Validate file path (optional for existing projects)
    dm4_file_valid = False
    file_type = None  # Track file type: 'dm4' or 'h5'
    n_cols = None  # Initialize dimension variables
    n_rows = None
    if dm4_file_path:
        if os.path.exists(dm4_file_path):
            if dm4_file_path.endswith(".dm4"):
                dm4_file_valid = True
                file_type = "dm4"
                st.success(f"✅ Valid DM4 file: `{os.path.basename(dm4_file_path)}`")
            elif dm4_file_path.endswith(".h5") or dm4_file_path.endswith(".hdf5"):
                dm4_file_valid = True
                file_type = "h5"
                st.success(f"✅ Valid HDF5 file: `{os.path.basename(dm4_file_path)}`")
            else:
                st.error(
                    "❌ Invalid file type! Please provide a file with `.dm4`, `.h5`, or `.hdf5` extension."
                )
                st.info(
                    "Tip: Make sure your file path ends with `.dm4` or `.h5` (e.g., `/path/to/data.dm4` or `/path/to/data.h5`)"
                )
        else:
            st.error(f"❌ File not found at the specified path: `{dm4_file_path}`")
            st.info(
                "Tips:\n- Check for typos in the file path\n- Use absolute paths (e.g., `/home/user/data.dm4`)\n- Ensure you have read permissions for the file"
            )
    elif not project_exists:
        st.warning(
            "⚠️ Data file path is required for new projects. Please provide a valid path above."
        )
        st.info("If you're loading an existing project, the data file is optional.")

    # Show file info and images if valid path provided
    if dm4_file_valid:
        if file_type == "dm4":
            st.info("Preview: Random diffraction image and virtual image from DM4 file")
            temp_dm4_path = dm4_file_path

            # Get dataset info and show dimensions
            import py4DSTEM

            try:
                dataset = py4DSTEM.import_file(temp_dm4_path)  # type: ignore
                shape = dataset.data.shape  # type: ignore
                n_rows = shape[0]
                n_cols = shape[1]

                diffraction_image_size = dataset.data[0][0].shape  # type: ignore

                st.success(f"✅ Detected 4D-STEM data shape: `{shape}`")
                st.write(
                    f"**Real space scan grid:** {n_cols} x {n_rows} = {n_cols * n_rows} positions"
                )
                st.write(
                    f"**Diffraction pattern size:** {diffraction_image_size[1]} x {diffraction_image_size[0]} pixels"
                )

                # Create a figure with two subplots side by side
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

                # Load dataset and show random diffraction pattern
                random_i = random.randint(0, shape[0] - 1)
                random_j = random.randint(0, shape[1] - 1)
                diffraction_pattern = dataset[random_i, random_j].data  # type: ignore

                im1 = ax1.imshow(diffraction_pattern, cmap="gray")
                ax1.set_title(
                    f"Random Diffraction Pattern\nat ({random_i}, {random_j})"
                )
                ax1.set_xlabel("Detector X (pixels)")
                ax1.set_ylabel("Detector Y (pixels)")
                plt.colorbar(im1, ax=ax1, label="Intensity", fraction=0.046)

                # Create virtual image
                virtual_image = dataset.data.sum(axis=(2, 3))  # type: ignore

                im2 = ax2.imshow(virtual_image, cmap="gray")
                ax2.set_title("Virtual Image\n(summed intensity)")
                ax2.set_xlabel("Scan X")
                ax2.set_ylabel("Scan Y")
                plt.colorbar(im2, ax=ax2, label="Intensity", fraction=0.046)

                plt.tight_layout()
                st.pyplot(fig, width="content")
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not read DM4 file or display images: {e}")

        elif file_type == "h5":
            st.info("Preview: Random diffraction image from HDF5 file")
            import h5py

            try:
                with h5py.File(dm4_file_path, "r") as f:
                    # Display HDF5 metadata tree
                    with st.expander("📊 HDF5 File Metadata Tree", expanded=True):
                        st.markdown("### Complete HDF5 File Structure")
                        st.markdown(
                            "Use this tree to locate your data and copy the exact path."
                        )

                        def build_h5_tree(group, prefix="", path=""):
                            """Recursively build HDF5 tree structure with full paths"""
                            tree_lines = []
                            items = list(group.items())

                            for idx, (key, item) in enumerate(items):
                                is_last = idx == len(items) - 1
                                connector = "└── " if is_last else "├── "
                                next_prefix = prefix + ("    " if is_last else "│   ")
                                current_path = f"{path}/{key}" if path else key

                                if isinstance(item, h5py.Dataset):
                                    # Dataset information
                                    shape_str = f"{item.shape}"
                                    dtype_str = f"{item.dtype}"
                                    size_mb = (
                                        item.size * item.dtype.itemsize / (1024 * 1024)
                                    )

                                    tree_lines.append(
                                        f"{prefix}{connector}📄 **{key}** `[Dataset]`"
                                    )
                                    tree_lines.append(
                                        f"{next_prefix}📐 Shape: `{shape_str}` | Type: `{dtype_str}` | Size: `{size_mb:.2f} MB`"
                                    )
                                    tree_lines.append(
                                        f"{next_prefix}📎 Path: `{current_path}`"
                                    )

                                    # Show attributes if any
                                    if len(item.attrs) > 0:
                                        tree_lines.append(
                                            f"{next_prefix}⚙️ Attributes ({len(item.attrs)}):"
                                        )
                                        for attr_idx, (attr_key, attr_val) in enumerate(
                                            list(item.attrs.items())[:5]
                                        ):
                                            val_str = str(attr_val)
                                            if len(val_str) > 60:
                                                val_str = val_str[:57] + "..."
                                            attr_connector = (
                                                "└─"
                                                if attr_idx
                                                == min(len(item.attrs) - 1, 4)
                                                else "├─"
                                            )
                                            tree_lines.append(
                                                f"{next_prefix}   {attr_connector} `{attr_key}`: {val_str}"
                                            )
                                        if len(item.attrs) > 5:
                                            tree_lines.append(
                                                f"{next_prefix}   └─ ... and {len(item.attrs) - 5} more attributes"
                                            )

                                elif isinstance(item, h5py.Group):
                                    # Group information
                                    num_children = len(item.keys())
                                    tree_lines.append(
                                        f"{prefix}{connector}📁 **{key}** `[Group - {num_children} items]`"
                                    )
                                    tree_lines.append(
                                        f"{next_prefix}📎 Path: `{current_path}`"
                                    )

                                    # Show group attributes if any
                                    if len(item.attrs) > 0:
                                        tree_lines.append(
                                            f"{next_prefix}⚙️ Attributes ({len(item.attrs)}):"
                                        )
                                        for attr_idx, (attr_key, attr_val) in enumerate(
                                            list(item.attrs.items())[:3]
                                        ):
                                            val_str = str(attr_val)
                                            if len(val_str) > 60:
                                                val_str = val_str[:57] + "..."
                                            attr_connector = (
                                                "└─"
                                                if attr_idx
                                                == min(len(item.attrs) - 1, 2)
                                                else "├─"
                                            )
                                            tree_lines.append(
                                                f"{next_prefix}   {attr_connector} `{attr_key}`: {val_str}"
                                            )
                                        if len(item.attrs) > 3:
                                            tree_lines.append(
                                                f"{next_prefix}   └─ ... and {len(item.attrs) - 3} more attributes"
                                            )

                                    # Recurse into subgroups
                                    tree_lines.extend(
                                        build_h5_tree(item, next_prefix, current_path)
                                    )

                            return tree_lines

                        # Build the complete tree
                        st.markdown(f"📦 **Root:** `{os.path.basename(dm4_file_path)}`")

                        # Show file-level attributes first
                        if len(f.attrs) > 0:
                            st.markdown(
                                f"⚙️ **File-level Attributes ({len(f.attrs)}):**"
                            )
                            for attr_key, attr_val in list(f.attrs.items())[:5]:
                                val_str = str(attr_val)
                                if len(val_str) > 80:
                                    val_str = val_str[:77] + "..."
                                st.markdown(f"  └─ `{attr_key}`: {val_str}")
                            if len(f.attrs) > 5:
                                st.markdown(
                                    f"  └─ ... and {len(f.attrs) - 5} more attributes"
                                )
                            st.markdown("")

                        # Display the tree
                        tree_lines = build_h5_tree(f)
                        for line in tree_lines:
                            st.markdown(line)

                        st.info(
                            "💡 **Tip:** Copy the path shown (e.g., `group/subgroup/dataset`) to use in the dataset path field below."
                        )

                    # Try common dataset keys
                    dataset_key = None
                    for key in ["frame", "data", "frames", "images"]:
                        if key in f:  # type: ignore
                            item = f[key]  # type: ignore
                            # Check if it's actually a dataset, not a group
                            if isinstance(item, h5py.Dataset):
                                dataset_key = key
                                break

                    if dataset_key is None:
                        st.warning(
                            f"Available root-level keys in file: {list(f.keys())}"
                        )  # type: ignore
                        st.info(
                            "Please specify the dataset path. Use the tree view above to find the exact path to your 4D data array."
                        )
                        dataset_key = st.text_input(
                            "Dataset path (e.g., 'frame', 'data', or 'group/dataset')",
                            value="frame",
                            help="Enter the full path to the dataset as shown in the tree above",
                        )

                    # Try to access the dataset
                    try:
                        data = f[dataset_key]  # type: ignore

                        # Check if it's actually a dataset
                        if not isinstance(data, h5py.Dataset):
                            st.error(
                                f"❌ '{dataset_key}' is a {type(data).__name__}, not a Dataset. Please use the tree view above to find the actual dataset path."
                            )
                        else:
                            shape = data.shape  # (scan_y, scan_x, height, width)

                            st.success(f"✅ Found dataset at '{dataset_key}'")
                            st.write(
                                f"**Full dataset shape:** `{shape}` (should be 4D: scan_y, scan_x, detector_height, detector_width)"
                            )

                            if len(shape) == 4:
                                n_rows = shape[0]
                                n_cols = shape[1]
                                diffraction_image_size = (shape[2], shape[3])

                                st.success(
                                    "✅ Detected 4D-STEM data - Shape interpretation:"
                                )
                                st.write(
                                    f"**Real space scan grid:** {n_cols} x {n_rows} = {n_cols * n_rows} positions (from shape[1] x shape[0])"
                                )
                                st.write(
                                    f"**Diffraction pattern size:** {diffraction_image_size[1]} x {diffraction_image_size[0]} pixels (from shape[3] x shape[2])"
                                )

                                # Show random diffraction pattern
                                random_i = random.randint(0, n_rows - 1)
                                random_j = random.randint(0, n_cols - 1)
                                diffraction_pattern = data[random_i, random_j, :, :]

                                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                                im = ax.imshow(diffraction_pattern, cmap="gray")
                                ax.set_title(
                                    f"Random Diffraction Pattern\nat ({random_i}, {random_j})"
                                )
                                ax.set_xlabel("Detector X (pixels)")
                                ax.set_ylabel("Detector Y (pixels)")
                                plt.colorbar(
                                    im, ax=ax, label="Intensity", fraction=0.046
                                )

                                plt.tight_layout()
                                st.pyplot(fig, width="content")
                                plt.close(fig)
                            else:
                                st.warning(
                                    f"Unexpected data shape: {shape}. Expected 4D array (scan_y, scan_x, height, width)"
                                )
                    except KeyError:
                        st.error(
                            f"❌ Dataset '{dataset_key}' not found in HDF5 file. Please check the tree view above for the correct path."
                        )
                    except Exception as e:
                        st.error(f"❌ Error accessing dataset: {e}")

            except Exception as e:
                st.warning(f"Could not read HDF5 file or display images: {e}")

    # Save configuration
    if st.button("Save Configuration & Proceed", type="primary"):
        # Validate project name before proceeding
        import re

        if not project_name or len(project_name) < 3:
            st.error("❌ Please enter a valid project name (at least 3 characters).")
            return
        if not re.match(r"^[a-zA-Z0-9_-]+$", project_name):
            st.error(
                "❌ Project name contains invalid characters. Use only letters, numbers, underscores, and hyphens."
            )
            return

        # For existing projects, try to auto-detect dimensions
        n_cols_detected = None
        n_rows_detected = None

        if project_exists:
            # Try to get dimensions from existing files
            if os.path.exists(png_path):
                png_files = [f for f in os.listdir(png_path) if f.endswith(".png")]
                if png_files:
                    # Get dimensions from last file
                    last_file = sorted(png_files)[-1]
                    parts = last_file.replace(".png", "").split("_")
                    if len(parts) >= 3:  # base_row_col format
                        try:
                            n_rows_detected = int(parts[-2]) + 1
                            n_cols_detected = int(parts[-1]) + 1
                        except ValueError:
                            st.warning(
                                "⚠️ Could not auto-detect dimensions from PNG filenames. Please check file naming."
                            )

            elif os.path.exists(boxed_path):
                boxed_files = [f for f in os.listdir(boxed_path) if f.endswith(".png")]
                if boxed_files:
                    # Get dimensions from last boxed file
                    last_file = sorted(boxed_files)[-1]
                    parts = last_file.replace(".png", "").split("_")
                    if len(parts) >= 2:  # row_col_boxsize format
                        try:
                            n_rows_detected = int(parts[0]) + 1
                            n_cols_detected = int(parts[1]) + 1
                        except ValueError:
                            st.warning(
                                "⚠️ Could not auto-detect dimensions from boxed filenames. Please check file naming."
                            )

        # Use detected dimensions or require DM4 file for new projects
        if n_cols_detected and n_rows_detected:
            n_cols = n_cols_detected
            n_rows = n_rows_detected
            st.info(
                f"✅ Auto-detected dimensions from existing files: {n_cols} x {n_rows}"
            )
        elif not dm4_file_path or not dm4_file_valid:
            st.error("❌ Cannot proceed without valid DM4 file for new projects!")
            st.info(
                "For new projects: Provide a valid DM4 file path above.\n💡 For existing projects: Make sure PNG or boxed files exist in the project directories."
            )
            return
        else:
            # n_cols and n_rows should have been set from DM4 file in the dm4_file_valid section
            # If not set, we cannot proceed
            if "n_cols" not in locals() or "n_rows" not in locals():
                st.error(
                    "❌ Could not determine array dimensions. Please check your DM4 file."
                )
                return

        config = {
            "project_name": project_name,
            "dm4_file_path": dm4_file_path if dm4_file_valid else "",
            "output_png_path": f"data/png/{project_name}",
            "boxed_png_path": f"data/boxed_png/{project_name}",
            "labelling_path": f"labelling/{project_name}_labels.yaml",
            "output_save_path": f"output/{project_name}/",
            "n_cols": n_cols,
            "n_rows": n_rows,
            "box_size": 3,  # Default, will be set in boxing section
            "sampling_space": 10,  # Default, will be set in labelling section
            "number_of_labels": 50,  # Default, will be set in labelling section
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,  # Training device (cpu, cuda, mps)
        }
        st.session_state.config = config

        # Auto-mark steps as done if files exist
        if project_exists:
            png_exists = (
                os.path.exists(png_path)
                and len([f for f in os.listdir(png_path) if f.endswith(".png")]) > 0
            )
            boxed_exists = (
                os.path.exists(boxed_path)
                and len([f for f in os.listdir(boxed_path) if f.endswith(".png")]) > 0
            )

            if png_exists:
                st.session_state.conversion_done = True
            if boxed_exists:
                st.session_state.boxing_done = True

            # Check for completed labelling file
            labelling_path = str(config["labelling_path"])
            if os.path.exists(labelling_path):
                try:
                    labels_data = load_existing_labels(labelling_path)
                    num_labels = len(labels_data.get("labels", []))
                    if num_labels >= int(config["number_of_labels"]):  # type: ignore
                        st.session_state.labelling_done = True
                        st.success(
                            f"✅ Found completed labelling file with {num_labels} labels"
                        )
                except Exception:
                    pass

            # Check for model checkpoint
            checkpoint_dir = f"checkpoints/{project_name}"
            if os.path.exists(checkpoint_dir):
                checkpoint_files = [
                    f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")
                ]
                if checkpoint_files:
                    st.session_state.training_done = True
                    st.success(f"✅ Found {len(checkpoint_files)} model checkpoint(s)")

        st.success("✅ Configuration saved successfully!")

        # Provide navigation guidance based on what exists
        if st.session_state.training_done:
            st.info(
                "Existing model checkpoints found! You can proceed directly to Step 6: Inference"
            )
        elif st.session_state.labelling_done:
            st.info("Labelling completed! You can proceed to Step 5: Model Training")
        elif st.session_state.conversion_done and st.session_state.boxing_done:
            st.info(
                "Existing project loaded! You can proceed to Step 4: Data Labelling"
            )
        elif st.session_state.conversion_done:
            st.info("PNG files found! You can proceed to Step 3: Average Boxing")
        else:
            st.info("You can now proceed to Step 2: File Conversion")


def dm4_conversion():
    """Step 2: Data File to PNG Conversion"""
    st.header("🔄 Step 2: Data File to PNG Conversion")

    if not st.session_state.config:
        st.error("❌ Please complete Step 1 first!")
        return

    config = st.session_state.config

    # Display current settings
    with st.expander("Current Configuration", expanded=False):
        st.json(config)

    st.subheader("🛠️ Conversion Settings")

    # Image enhancement options
    st.markdown("**Image Enhancement Options**")
    col_enhance1, col_enhance2, col_enhance3 = st.columns(3)

    with col_enhance1:
        use_log_scale = st.checkbox(
            "📈 Log Scaling",
            key="use_log_scale",
            value=False,
            help="Apply logarithmic scaling to enhance visibility of ghost disks and low-intensity features",
        )

    with col_enhance2:
        use_histogram_equalization = st.checkbox(
            "📊 Histogram Equalization",
            key="use_histogram_eq",
            value=False,
            help="Improve contrast by redistributing intensity values",
        )

    with col_enhance3:
        use_gamma_correction = st.checkbox(
            "🔆 Gamma Correction",
            key="use_gamma",
            value=False,
            help="Adjust brightness using gamma correction",
        )

    gamma_value = 1.0
    if use_gamma_correction:
        gamma_value = st.slider(
            "Gamma Value",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key="gamma_value",
            help="Values < 1 brighten the image, values > 1 darken it",
        )

    # Show preview if any enhancement is selected
    if use_log_scale or use_histogram_equalization or use_gamma_correction:
        with st.expander("🔍 Preview Enhancement Effects", expanded=True):
            st.markdown("**Live Preview: Original vs Enhanced**")

            # Try to load a sample diffraction pattern
            try:
                dm4_file = config.get("dm4_file_path", "")
                mid_i, mid_j = 0, 0  # Initialize for caption
                sample_pattern = None

                if dm4_file and os.path.exists(dm4_file):
                    if dm4_file.endswith(".dm4"):
                        import py4DSTEM

                        dataset = py4DSTEM.import_file(dm4_file)  # type: ignore
                        # Get a pattern from middle of scan
                        mid_i = dataset.data.shape[0] // 2  # type: ignore
                        mid_j = dataset.data.shape[1] // 2  # type: ignore
                        sample_pattern = dataset.data[mid_i, mid_j].copy()  # type: ignore
                    elif dm4_file.endswith((".h5", ".hdf5")):
                        import h5py

                        dataset_key_preview = st.session_state.get(
                            "h5_dataset_key", "frame"
                        )
                        with h5py.File(dm4_file, "r") as f:
                            try:
                                data = f[dataset_key_preview]  # type: ignore
                                if (
                                    isinstance(data, h5py.Dataset)
                                    and len(data.shape) == 4
                                ):
                                    mid_i = data.shape[0] // 2
                                    mid_j = data.shape[1] // 2
                                    sample_pattern = data[mid_i, mid_j, :, :].copy()  # type: ignore
                                else:
                                    raise ValueError("Invalid dataset shape")
                            except:
                                st.warning(
                                    "Could not load preview from HDF5. Using synthetic pattern."
                                )
                                sample_pattern = np.random.rand(256, 256) * 1000

                if sample_pattern is not None:
                    # Apply enhancements
                    from scripts.hdf5_to_png import apply_image_enhancements

                    enhanced_pattern = apply_image_enhancements(
                        sample_pattern,
                        use_log_scale=use_log_scale,
                        use_histogram_eq=use_histogram_equalization,
                        gamma=gamma_value if use_gamma_correction else None,
                    )

                    # Display side by side
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                    im1 = ax1.imshow(sample_pattern, cmap="gray")
                    ax1.set_title("Original", fontsize=14, fontweight="bold")
                    ax1.set_xlabel("X (pixels)")
                    ax1.set_ylabel("Y (pixels)")
                    plt.colorbar(im1, ax=ax1, label="Intensity", fraction=0.046)

                    im2 = ax2.imshow(enhanced_pattern, cmap="gray")
                    enhancements = []
                    if use_log_scale:
                        enhancements.append("Log")
                    if use_histogram_equalization:
                        enhancements.append("HistEq")
                    if use_gamma_correction:
                        enhancements.append(f"γ={gamma_value:.1f}")
                    title = "Enhanced: " + " + ".join(enhancements)
                    ax2.set_title(title, fontsize=14, fontweight="bold")
                    ax2.set_xlabel("X (pixels)")
                    ax2.set_ylabel("Y (pixels)")
                    plt.colorbar(im2, ax=ax2, label="Intensity", fraction=0.046)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.caption(f"📍 Sample from scan position ({mid_i}, {mid_j})")
                else:
                    st.info("No file loaded for preview")
            except Exception as e:
                st.warning(f"Could not generate preview: {e}")

    st.divider()

    crop_images = st.checkbox(
        "Crop Images",
        key="crop_images",
        help="Enable to crop diffraction patterns to a specific region",
    )

    crop_values = None
    x_min = 0
    x_max = 512
    y_min = 0
    y_max = 512

    if crop_images:
        st.write("**Crop Parameters (x_min, x_max, y_min, y_max):**")
        st.caption("⚠️ Make sure crop values are within your image dimensions!")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_min = st.number_input("X Min", value=0, min_value=0, key="x_min")
        with col2:
            x_max = st.number_input("X Max", value=512, min_value=1, key="x_max")
        with col3:
            y_min = st.number_input("Y Min", value=0, min_value=0, key="y_min")
        with col4:
            y_max = st.number_input("Y Max", value=512, min_value=1, key="y_max")

        # Validate crop values
        if x_min >= x_max:
            st.error("❌ X Min must be less than X Max!")
        if y_min >= y_max:
            st.error("❌ Y Min must be less than Y Max!")
        if (x_max - x_min) < 10 or (y_max - y_min) < 10:
            st.warning("⚠️ Crop region is very small. Make sure this is intentional.")

        crop_values = (x_min, x_max, y_min, y_max)

        # Show crop preview
        if x_min < x_max and y_min < y_max:
            with st.expander("✂️ Preview Crop Region", expanded=True):
                st.markdown("**Live Preview: Full Image vs Cropped Region**")

                try:
                    dm4_file = config.get("dm4_file_path", "")
                    mid_i, mid_j = 0, 0
                    sample_pattern = None

                    if dm4_file and os.path.exists(dm4_file):
                        if dm4_file.endswith(".dm4"):
                            import py4DSTEM

                            dataset = py4DSTEM.import_file(dm4_file)  # type: ignore
                            mid_i = dataset.data.shape[0] // 2  # type: ignore
                            mid_j = dataset.data.shape[1] // 2  # type: ignore
                            sample_pattern = dataset.data[mid_i, mid_j].copy()  # type: ignore
                        elif dm4_file.endswith((".h5", ".hdf5")):
                            import h5py

                            dataset_key_preview = st.session_state.get(
                                "h5_dataset_key", "frame"
                            )
                            with h5py.File(dm4_file, "r") as f:
                                try:
                                    data = f[dataset_key_preview]  # type: ignore
                                    if (
                                        isinstance(data, h5py.Dataset)
                                        and len(data.shape) == 4
                                    ):
                                        mid_i = data.shape[0] // 2
                                        mid_j = data.shape[1] // 2
                                        sample_pattern = data[mid_i, mid_j, :, :].copy()  # type: ignore
                                except:
                                    st.warning("Could not load preview from HDF5.")
                                    sample_pattern = None

                    if sample_pattern is not None:
                        # Check if crop values are valid for this image
                        img_height, img_width = sample_pattern.shape

                        if x_max > img_width or y_max > img_height:
                            st.error(
                                f"❌ Crop values exceed image dimensions! Image size: {img_width} x {img_height}"
                            )
                        else:
                            # Apply crop
                            cropped_pattern = sample_pattern[y_min:y_max, x_min:x_max]

                            # Display side by side
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                            # Full image with crop rectangle overlay
                            im1 = ax1.imshow(sample_pattern, cmap="gray")
                            ax1.set_title(
                                "Full Image with Crop Region",
                                fontsize=14,
                                fontweight="bold",
                            )
                            ax1.set_xlabel("X (pixels)")
                            ax1.set_ylabel("Y (pixels)")

                            # Draw rectangle showing crop region
                            from matplotlib.patches import Rectangle

                            rect = Rectangle(
                                (x_min, y_min),
                                x_max - x_min,
                                y_max - y_min,
                                linewidth=2,
                                edgecolor="red",
                                facecolor="none",
                                linestyle="--",
                            )
                            ax1.add_patch(rect)
                            ax1.text(
                                x_min,
                                y_min - 5,
                                "Crop Region",
                                color="red",
                                fontsize=10,
                                fontweight="bold",
                                va="bottom",
                            )

                            plt.colorbar(im1, ax=ax1, label="Intensity", fraction=0.046)

                            # Cropped image
                            im2 = ax2.imshow(cropped_pattern, cmap="gray")
                            crop_size_str = f"{x_max - x_min} × {y_max - y_min}"
                            ax2.set_title(
                                f"Cropped Image ({crop_size_str} px)",
                                fontsize=14,
                                fontweight="bold",
                            )
                            ax2.set_xlabel("X (pixels)")
                            ax2.set_ylabel("Y (pixels)")
                            plt.colorbar(im2, ax=ax2, label="Intensity", fraction=0.046)

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                            st.caption(
                                f"📍 Sample from scan position ({mid_i}, {mid_j}) | Original: {img_width} × {img_height} px → Cropped: {crop_size_str} px"
                            )
                    else:
                        st.info("No file loaded for crop preview")

                except Exception as e:
                    st.warning(f"Could not generate crop preview: {e}")

    # HDF5-specific settings
    dataset_key = "frame"
    if config.get("dm4_file_path", "").endswith((".h5", ".hdf5")):
        st.subheader("📊 HDF5 Dataset Settings")
        dataset_key = st.text_input(
            "Dataset Key",
            value="frame",
            key="h5_dataset_key",
            help="Name of the dataset in the HDF5 file (common values: 'frame', 'data', 'frames')",
        )

    if st.button("🚀 Start File Conversion", type="primary"):
        if not config.get("dm4_file_path"):
            st.error("❌ No data file path found in configuration!")
            st.info(
                "Please go back to Step 1 and provide a valid DM4 or HDF5 file path, then save the configuration."
            )
            return

        # Validate crop parameters if cropping is enabled
        if crop_images and (x_min >= x_max or y_min >= y_max):
            st.error(
                "❌ Invalid crop parameters! Please fix the values above before proceeding."
            )
            return

        try:
            # Use the file path directly - no need to copy
            dm4_file_path = config["dm4_file_path"]

            if not os.path.exists(dm4_file_path):
                st.error(f"❌ File not found at: `{dm4_file_path}`")
                st.info(
                    "The file may have been moved or deleted. Please update the path in Step 1."
                )
                return

            # Determine file type
            file_type = None
            if dm4_file_path.endswith(".dm4"):
                file_type = "dm4"
            elif dm4_file_path.endswith((".h5", ".hdf5")):
                file_type = "h5"
            else:
                st.error("❌ Unsupported file type. Please use .dm4 or .h5 files.")
                return

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text(
                f"🔄 Converting {file_type.upper()} to PNG format... This may take a few minutes."
            )
            progress_bar.progress(50)

            # Call appropriate conversion function based on file type
            if file_type == "dm4":
                export_dm4_bf_images_to_png(
                    dm4_file_path,
                    config["output_png_path"],
                    crop=crop_images,
                    crop_values=crop_values,
                    use_log_scale=use_log_scale,
                    use_histogram_eq=use_histogram_equalization,
                    gamma=gamma_value if use_gamma_correction else None,
                )
            elif file_type == "h5":
                save_h5_diffraction_to_png(
                    dm4_file_path,
                    config["output_png_path"],
                    dataset_key=dataset_key,
                    crop=crop_images,
                    crop_values=crop_values,
                    use_log_scale=use_log_scale,
                    use_histogram_eq=use_histogram_equalization,
                    gamma=gamma_value if use_gamma_correction else None,
                )

            progress_bar.progress(100)
            status_text.text("✅ Conversion completed successfully!")

            st.session_state.conversion_done = True
            st.success("File conversion completed! Ready for average boxing.")

        except Exception as e:
            st.error(f"❌ Error during conversion: {str(e)}")


def average_boxing():
    """Step 3: Average Boxing"""
    st.header("Step 3: Average Boxing")

    if not st.session_state.config:
        st.error("❌ Please complete previous steps first!")
        return

    if not st.session_state.conversion_done:
        st.warning("⚠️ Please complete file conversion first!")
        return

    config = st.session_state.config

    # Display current settings
    with st.expander("Current Configuration", expanded=False):
        st.json(config)

    st.subheader("Boxing Configuration")

    # Box size slider
    box_size = st.slider(
        "Box Size",
        min_value=1,
        max_value=5,
        value=config.get("box_size", 3),
        key="box_size_slider",
        help="Size of the averaging box for creating averaged images",
    )

    # Update config with selected box size
    st.session_state.config["box_size"] = box_size

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"""
        **Input Settings:**
        - 📁 PNG Directory: `{config["output_png_path"]}`
        - � Box Size: {config["box_size"]}
        - � Array Length: Auto-detected from filenames
        """
        )

    with col2:
        st.info(
            f"""
        **Output Settings:**
        - 📁 Output Directory: `{config["boxed_png_path"]}`
        - 🖼️ Processing: Max values with {config["box_size"]}x{config["box_size"]} boxes
        """
        )

        st.subheader("ℹ️ About Boxing")
        st.write(
            """
        **Average boxing** creates averaged images from neighboring regions:
        - Takes a box of images (e.g., 3x3 grid)
        - Computes max/average values across the box
        - Reduces noise and enhances features
        - Essential preprocessing step for labelling
        """
        )

    if st.button("🚀 Start Average Boxing", type="primary"):
        # Validate input directory
        if not os.path.exists(config["output_png_path"]):
            st.error(f"❌ Input directory not found: `{config['output_png_path']}`")
            st.info(
                "Please complete Step 2 (DM4 Conversion) first to generate PNG files."
            )
            return

        # Check for PNG files
        png_files = [
            f for f in os.listdir(config["output_png_path"]) if f.endswith(".png")
        ]
        if len(png_files) == 0:
            st.error(f"❌ No PNG files found in `{config['output_png_path']}`")
            st.info(
                "Please complete Step 2 (DM4 Conversion) first to generate PNG files."
            )
            return

        # Validate box size
        if config["box_size"] < 1:
            st.error("❌ Box size must be at least 1!")
            return

        # Check if output would have enough images
        expected_output = (config["n_rows"] // config["box_size"]) * (
            config["n_cols"] // config["box_size"]
        )
        if expected_output < 10:
            st.warning(
                f"⚠️ Box size of {config['box_size']} will result in only {expected_output} output images. Consider using a smaller box size."
            )
            if not st.checkbox(
                "I understand and want to proceed anyway", key="confirm_small_output"
            ):
                return

        try:
            if not os.path.exists(config["output_png_path"]):
                st.error(
                    "❌ PNG directory not found! Please complete DM4 conversion first."
                )
                return

            # Count PNG files
            png_files = [
                f for f in os.listdir(config["output_png_path"]) if f.endswith(".png")
            ]
            if len(png_files) == 0:
                st.error("❌ No PNG files found in the output directory!")
                return

            st.info(f"Found {len(png_files)} PNG files to process")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Initializing boxing process...")
            progress_bar.progress(10)

            # Ensure output directory exists
            os.makedirs(config["boxed_png_path"], exist_ok=True)

            progress_bar.progress(25)
            status_text.text("Creating averaged boxes... (This may take a while)")

            # Call boxing function
            convert_all_to_boxes(
                stem_image_dir=config["output_png_path"],
                output_dir=config["boxed_png_path"],
                box_size=config["box_size"],
                n_cols=config["n_cols"],
                n_rows=config["n_rows"],
            )

            progress_bar.progress(100)
            status_text.text("✅ Boxing completed successfully!")

            st.session_state.boxing_done = True
            st.success("Average boxing completed! Ready for data labelling.")

            # Show results summary
            boxed_files = [
                f for f in os.listdir(config["boxed_png_path"]) if f.endswith(".png")
            ]
            st.info(f"Generated {len(boxed_files)} boxed images")

        except Exception as e:
            st.error(f"❌ Error during boxing: {str(e)}")
            st.write("**Debugging info:**")
            st.write(f"- Input directory: {config['output_png_path']}")
            st.write(f"- Output directory: {config['boxed_png_path']}")
            st.write(f"- Box size: {config['box_size']}")
            st.write("- Array length: Auto-detected from filenames")


def data_labelling():
    """Step 4: Data Labelling - Streamlit Native Interface"""
    st.header("Step 4: Data Labelling")

    if not st.session_state.config:
        st.error("❌ Configuration not found!")
        st.info("Please complete Step 1 (Setup & Configuration) first.")
        return

    if not st.session_state.boxing_done:
        st.warning("⚠️ Boxing step not completed!")
        st.info(
            "Please complete Step 3 (Average Boxing) before labelling. You need boxed images to label."
        )
        return

    config = st.session_state.config

    # Check if boxed images exist
    if not os.path.exists(config["boxed_png_path"]):
        st.error(f"❌ Boxed images directory not found: `{config['boxed_png_path']}`")
        st.info("Please complete Step 3 (Average Boxing) to generate boxed images.")
        return

    boxed_files = [
        f for f in os.listdir(config["boxed_png_path"]) if f.endswith(".png")
    ]
    if len(boxed_files) == 0:
        st.error(f"❌ No boxed images found in `{config['boxed_png_path']}`")
        st.info("Please complete Step 3 (Average Boxing) to generate boxed images.")
        return

    # Check if labelling is already complete
    if os.path.exists(config["labelling_path"]):
        try:
            labels_data = load_existing_labels(config["labelling_path"])
            existing_num_labels = len(labels_data.get("labels", []))
            if existing_num_labels > 0:
                st.info(
                    f"Found existing labels file with {existing_num_labels} labels. You can continue labelling or proceed to training if complete."
                )
        except Exception:
            pass

    # Display configuration
    st.subheader("Labelling Configuration")

    # Labelling parameter sliders
    sampling_interval = st.slider(
        "Sampling Interval",
        min_value=1,
        max_value=50,
        value=config.get("sampling_space", 10),
        key="sampling_interval",
        help="Grid spacing for initial sparse sampling. Smaller = more initial labels, longer setup time.",
    )
    number_of_labels = st.slider(
        "Number of Labels",
        min_value=10,
        max_value=1000,
        value=config.get("number_of_labels", 50),
        key="number_of_labels",
        help="Total number of labels to assign. More labels = better training, but more work.",
    )

    # Validate labelling parameters
    if number_of_labels > len(boxed_files):
        st.error(
            f"❌ Number of labels ({number_of_labels}) exceeds available images ({len(boxed_files)})!"
        )
        st.info(f"Please reduce the number of labels to {len(boxed_files)} or less.")
        return

    # Calculate expected sparse samples
    expected_sparse = (config["n_rows"] // sampling_interval) * (
        config["n_cols"] // sampling_interval
    )
    if expected_sparse > number_of_labels:
        st.warning(
            f"⚠️ Sampling interval ({sampling_interval}) will generate ~{expected_sparse} samples, which exceeds your target of {number_of_labels} labels."
        )
        st.info(
            f"Consider increasing the sampling interval to at least {int(np.sqrt(config['n_rows'] * config['n_cols'] / number_of_labels))} for efficiency."
        )
    elif expected_sparse < number_of_labels * 0.3:
        st.warning(
            f"⚠️ Sampling interval ({sampling_interval}) will only generate ~{expected_sparse} initial samples. Most labelling will use binary search refinement."
        )
        st.info("Consider decreasing the sampling interval for more uniform coverage.")

    # Update config with selected parameters
    st.session_state.config["sampling_space"] = sampling_interval
    st.session_state.config["number_of_labels"] = number_of_labels

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"""
        **Current Settings:**
        - 📁 Data Path: `{config["boxed_png_path"]}`
        - 🎯 Sampling Space: {config["sampling_space"]}
        - 🔢 Number of Labels: {config["number_of_labels"]}
        """
        )

    with col2:
        st.info(
            f"""
        **Output:**
        - 📄 Labels File: `{config["labelling_path"]}`
        """
        )

    # Initialize labeller state if needed
    if (
        st.button("🚀 Initialize Labelling Session", type="primary")
        or st.session_state.labelling_initialized
    ):
        try:
            # Ensure labelling directory exists
            os.makedirs(os.path.dirname(config["labelling_path"]), exist_ok=True)

            # Initialize labeller state only once
            if not st.session_state.labelling_initialized:
                st.session_state.labeller_state = StreamlitLabellerState(
                    file_path=config["boxed_png_path"],
                    output_file=config["labelling_path"],
                    labels_to_assign=config["number_of_labels"],
                    step=config["sampling_space"],
                )
                st.session_state.labelling_initialized = True
                st.rerun()

            labeller = st.session_state.labeller_state

            # Check if labelling is complete
            if labeller.is_complete():
                st.success("All labels have been assigned!")
                st.session_state.labelling_done = True

                # Show final heatmap
                st.subheader("📊 Final Label Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                final_heatmap = labeller.save_final_heatmap()
                cmap = plt.get_cmap("viridis", 4)
                im = ax.imshow(final_heatmap, cmap=cmap, vmin=0, vmax=3)
                ax.set_title("Final Label Map")
                ax.axis("off")
                cbar = plt.colorbar(
                    im,
                    ax=ax,
                    ticks=[0, 1, 2, 3],
                    label="Label",
                    fraction=0.046,
                )
                cbar.ax.set_yticklabels(
                    ["No Label", "Horizontal", "Vertical", "No Polarisation"]
                )
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)
                plt.close(fig)

                # Show statistics
                st.subheader("📈 Label Statistics")
                label_stats = get_label_statistics(config["labelling_path"])
                for label, count in label_stats.items():
                    st.write(f"- **{label}**: {count}")

                return

            # Get next file to label
            if st.session_state.current_label_file is None:
                next_file = labeller.get_next_file()
                if next_file:
                    st.session_state.current_label_file = next_file
                else:
                    st.error("❌ Could not get next file to label")
                    return

            current_file = st.session_state.current_label_file

            # Progress bar
            current, total = labeller.get_progress()
            st.progress(current / total)
            st.write(f"**Progress: {current} / {total} labels assigned**")

            # Display current heatmap and image side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🗺️ Label Heatmap")
                fig, ax = plt.subplots(figsize=(5, 5))
                cmap = plt.get_cmap("viridis", 4)
                im = ax.imshow(labeller.sparse_array, cmap=cmap, vmin=0, vmax=3)

                # Highlight current position
                pos = labeller.get_current_position()
                if pos:
                    i, j = pos
                    ax.plot(
                        j,
                        i,
                        marker="s",
                        color="red",
                        markersize=10,
                        markeredgewidth=2,
                        markeredgecolor="black",
                    )

                ax.set_title("Current Labelling Progress", fontsize=10)
                ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            with col2:
                st.subheader("🖼️ Current Image")
                st.write(f"**File:** `{current_file}`")

                # Load and display image with caching
                img_path = os.path.join(config["boxed_png_path"], current_file)
                if os.path.exists(img_path):
                    # Use st.image directly with file path (faster than loading with PIL)
                    st.image(img_path, width="stretch")
                else:
                    st.error(f"❌ Image not found: {img_path}")

            # Labelling buttons
            st.subheader("🏷️ Assign Label")
            st.write("**Choose the label for this image:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    "🔴 Horizontal Polarisation",
                    key="label_horizontal",
                    width="stretch",
                ):
                    save_label(current_file, config["labelling_path"], "horizontal")
                    labeller.update_sparse_array(1)
                    st.session_state.current_label_file = None
                    st.rerun()

            with col2:
                if st.button(
                    "🟢 Vertical Polarisation", key="label_vertical", width="stretch"
                ):
                    save_label(current_file, config["labelling_path"], "vertical")
                    labeller.update_sparse_array(2)
                    st.session_state.current_label_file = None
                    st.rerun()

            with col3:
                if st.button(
                    "🟡 No Observable Polarisation", key="label_none", width="stretch"
                ):
                    save_label(current_file, config["labelling_path"], "na")
                    labeller.update_sparse_array(3)
                    st.session_state.current_label_file = None
                    st.rerun()

            # Option to reset labelling
            st.divider()
            if st.button("🔄 Reset Labelling Session", type="secondary"):
                st.session_state.labeller_state = None
                st.session_state.current_label_file = None
                st.session_state.labelling_initialized = False
                st.warning(
                    "⚠️ Labelling session reset. Note: Previously saved labels in the YAML file are preserved."
                )
                st.rerun()

        except FileNotFoundError as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure you have completed the average boxing step first.")
        except Exception as e:
            st.error(f"❌ Error during labelling: {str(e)}")
            import traceback

            st.code(traceback.format_exc())

    # Show existing labels if file exists
    if os.path.exists(config["labelling_path"]):
        with st.expander("View Existing Labels"):
            try:
                labels_data = load_existing_labels(config["labelling_path"])
                if labels_data and "labels" in labels_data:
                    num_labels = len(labels_data["labels"])
                    st.write(f"**Total labels saved:** {num_labels}")

                    # Show statistics
                    label_stats = get_label_statistics(config["labelling_path"])
                    st.write("**Label Distribution:**")
                    for label, count in label_stats.items():
                        st.write(f"- {label}: {count}")
            except Exception as e:
                st.warning(f"⚠️ Could not read labels file: {str(e)}")


def model_training():
    """Step 4: Model Training"""
    st.header("🤖 Step 4: Model Training")

    if not st.session_state.config:
        st.error("❌ Configuration not found!")
        st.info("Please complete Step 1 (Setup & Configuration) first.")
        return

    config = st.session_state.config

    # Validate labels file exists (more lenient - check file, not just session state)
    if not os.path.exists(config["labelling_path"]):
        st.error(f"❌ Labels file not found: `{config['labelling_path']}`")
        st.info("Please complete Step 4 (Data Labelling) to generate the labels file.")

        # Still show warning about session state if that's the issue
        if not st.session_state.labelling_done:
            st.warning("⚠️ Labelling step not marked as complete in this session.")
        return

    # Check number of labels
    try:
        labels_data = load_existing_labels(config["labelling_path"])
        num_labels = len(labels_data.get("labels", []))

        if num_labels < 10:
            st.error(f"❌ Insufficient labels! Found only {num_labels} labels.")
            st.info(
                "You need at least 10 labels for training. Please complete the labelling step."
            )
            return
        elif num_labels < 30:
            st.warning(
                f"⚠️ Low number of labels ({num_labels}). Training may not produce good results."
            )
            st.info(
                "Consider labelling more images (recommended: 50+) for better model performance."
            )
        else:
            st.success(f"✅ Found {num_labels} labels - good for training!")

        # Check label distribution
        label_stats = get_label_statistics(config["labelling_path"])
        if len(label_stats) < 2:
            st.error(
                "❌ Only one label type found! You need at least 2 different label types for classification."
            )
            st.info("Please add more diverse labels in the labelling step.")
            return

        # Check for imbalanced labels
        max_count = max(label_stats.values())
        min_count = min(label_stats.values())
        if max_count > min_count * 5:
            st.warning(
                "⚠️ Imbalanced label distribution detected! Some labels are 5x more common than others."
            )
            st.info(
                "Consider labelling more images of underrepresented classes for better model balance."
            )
            with st.expander("View Label Distribution"):
                for label, count in label_stats.items():
                    st.write(f"- {label}: {count}")

    except Exception as e:
        st.error(f"❌ Could not validate labels: {str(e)}")
        return

    st.subheader("⚙️ Training Configuration")
    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"""
        **Model Settings:**
        - 🧠 Model: ThreeLayerCnn
        - �️ Device: {config.get("device", "cpu").upper()}
        - 📊 Max Epochs: {config["max_epochs"]}
        - 📦 Batch Size: {config["batch_size"]}
        - 📈 Learning Rate: {config["learning_rate"]}
        """
        )

    with col2:
        st.info(
            """
        **Data Configuration:**
        - 🎯 Train Ratio: 80%
        - ✅ Validation Ratio: 10%
        - 🧪 Test Ratio: 10%
        - 🖼️ Image Size: 256x256
        """
        )

    if st.button("🚀 Start Model Training", type="primary"):
        # Validate device availability before training
        selected_device = config.get("device", "cpu")
        try:
            import torch

            if selected_device == "cuda" and not torch.cuda.is_available():
                st.error("❌ CUDA device selected but not available!")
                st.info(
                    "Options:\n- Install CUDA and PyTorch with CUDA support\n- Change device to 'cpu' in Step 1 and re-save configuration"
                )
                return
            elif selected_device == "mps" and not torch.backends.mps.is_available():
                st.error("❌ MPS device selected but not available!")
                st.info(
                    "Options:\n- MPS requires Apple Silicon Mac (M1/M2)\n- Change device to 'cpu' in Step 1 and re-save configuration"
                )
                return
        except ImportError:
            if selected_device != "cpu":
                st.warning(
                    f"⚠️ Cannot verify {selected_device.upper()} availability (PyTorch not imported). Training will attempt to use {selected_device}."
                )

        # Initialize variables before try block to avoid unbound errors
        config_file_path = f"configs/{config['project_name']}.yaml"
        training_config = None

        try:
            # Create config file for training
            training_config = create_training_config(config)

            # Ensure configs directory exists
            os.makedirs("configs", exist_ok=True)

            # Save config file
            with open(config_file_path, "w") as f:
                yaml.dump(training_config, f, default_flow_style=False)

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text(
                f"🚀 Initializing training on {selected_device.upper()}..."
            )
            progress_bar.progress(10)

            status_text.text("🧠 Training model... (This may take a while)")
            progress_bar.progress(25)

            # Call training function
            train_model(f"{config['project_name']}.yaml")

            progress_bar.progress(100)
            status_text.text("✅ Training completed successfully!")

            st.session_state.training_done = True
            st.success("🎉 Model training completed! Ready for inference.")

            # Display training statistics
            display_training_stats(config)

        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.write("**Debugging info:**")
            st.write(f"- Config file path: {config_file_path}")
            if training_config is not None:
                st.write(f"- Training config: {training_config}")

    # Display training statistics if training is complete
    if st.session_state.training_done:
        with st.expander("📊 View Training Statistics", expanded=False):
            display_training_stats(config)


def display_training_stats(config):
    """Display training statistics from PyTorch Lightning logs"""
    st.subheader("📊 Training Statistics")

    # Find the most recent lightning logs
    lightning_logs_dir = "lightning_logs"

    if not os.path.exists(lightning_logs_dir):
        st.info("No training logs found yet.")
        return

    # Get all version directories and sort by modification time
    version_dirs = [
        d for d in os.listdir(lightning_logs_dir) if d.startswith("version_")
    ]
    if not version_dirs:
        st.info("No training logs found yet.")
        return

    # Get the most recent version directory
    version_dirs_full = [os.path.join(lightning_logs_dir, d) for d in version_dirs]
    latest_log_dir = max(version_dirs_full, key=os.path.getmtime)
    metrics_file = os.path.join(latest_log_dir, "metrics.csv")

    if not os.path.exists(metrics_file):
        st.info("ℹ️ No metrics file found yet.")
        return

    try:
        import pandas as pd

        # Read metrics CSV
        df = pd.read_csv(metrics_file)

        # Display key metrics
        col1, col2, col3 = st.columns(3)

        # Get final metrics
        if "train_loss" in df.columns:
            final_train_loss = df["train_loss"].dropna().iloc[-1]
            with col1:
                st.metric("Final Train Loss", f"{final_train_loss:.4f}")

        if "val_loss" in df.columns:
            final_val_loss = df["val_loss"].dropna().iloc[-1]
            best_val_loss = df["val_loss"].min()
            with col2:
                st.metric("Final Val Loss", f"{final_val_loss:.4f}")
                st.metric("Best Val Loss", f"{best_val_loss:.4f}")

        if "train_acc" in df.columns:
            final_train_acc = df["train_acc"].dropna().iloc[-1]
            with col3:
                st.metric("Final Train Accuracy", f"{final_train_acc:.2%}")

        if "val_acc" in df.columns:
            final_val_acc = df["val_acc"].dropna().iloc[-1]
            best_val_acc = df["val_acc"].max()
            with col3:
                st.metric("Final Val Accuracy", f"{final_val_acc:.2%}")
                st.metric("Best Val Accuracy", f"{best_val_acc:.2%}")

        # Plot training curves
        st.write("### 📈 Training Curves")

        # Loss plot
        if "train_loss" in df.columns or "val_loss" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))

            if "train_loss" in df.columns:
                train_loss_clean = df[["epoch", "train_loss"]].dropna()
                ax.plot(
                    train_loss_clean["epoch"],
                    train_loss_clean["train_loss"],
                    label="Train Loss",
                    marker="o",
                    markersize=3,
                )

            if "val_loss" in df.columns:
                val_loss_clean = df[["epoch", "val_loss"]].dropna()
                ax.plot(
                    val_loss_clean["epoch"],
                    val_loss_clean["val_loss"],
                    label="Validation Loss",
                    marker="s",
                    markersize=3,
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training and Validation Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        # Accuracy plot
        if "train_acc" in df.columns or "val_acc" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))

            if "train_acc" in df.columns:
                train_acc_clean = df[["epoch", "train_acc"]].dropna()
                ax.plot(
                    train_acc_clean["epoch"],
                    train_acc_clean["train_acc"] * 100,
                    label="Train Accuracy",
                    marker="o",
                    markersize=3,
                )

            if "val_acc" in df.columns:
                val_acc_clean = df[["epoch", "val_acc"]].dropna()
                ax.plot(
                    val_acc_clean["epoch"],
                    val_acc_clean["val_acc"] * 100,
                    label="Validation Accuracy",
                    marker="s",
                    markersize=3,
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Training and Validation Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        # Show full metrics table
        if st.checkbox("Show detailed metrics table"):
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Could not load training statistics: {str(e)}")


def inference_results():
    """Step 5: Inference and Results"""
    st.header("Step 5: Inference & Results")

    if not st.session_state.config:
        st.error("❌ Configuration not found!")
        st.info("Please complete Step 1 (Setup & Configuration) first.")
        return

    config = st.session_state.config

    # Check for model checkpoints (more lenient - check files, not just session state)
    checkpoint_dir = f"checkpoints/{config['project_name']}"
    if not os.path.exists(checkpoint_dir):
        st.error(f"❌ Checkpoint directory not found: `{checkpoint_dir}`")
        st.info(
            "Please complete Step 5 (Model Training) to generate model checkpoints."
        )

        if not st.session_state.training_done:
            st.warning("⚠️ Training step not marked as complete in this session.")
        return

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not checkpoint_files:
        st.error(f"❌ No checkpoint files found in `{checkpoint_dir}`")
        st.info(
            "Please complete Step 5 (Model Training) to generate model checkpoints."
        )
        return

    # Display available checkpoints
    st.success(f"✅ Found {len(checkpoint_files)} model checkpoint(s)")
    with st.expander("View Available Checkpoints"):
        for ckpt in sorted(checkpoint_files):
            st.write(f"- `{ckpt}`")

    st.subheader("Generate Results")

    # Add softmax toggle
    col1, col2 = st.columns([2, 1])
    with col1:
        use_softmax = st.checkbox(
            "Apply Softmax Activation",
            value=True,
            help="Apply softmax to the model output for probability distribution. Uncheck for raw logits.",
        )
    with col2:
        st.write("")  # Spacing

    if st.button("Run Inference", type="primary"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Running inference...")
            progress_bar.progress(25)

            config_file = f"{config['project_name']}.yaml"

            # Ensure output directory exists
            os.makedirs(config["output_save_path"], exist_ok=True)

            progress_bar.progress(50)

            # Call inference function and store the returned plot path
            plot_path = plot_embeddings(
                config_file,
                config["n_cols"],
                config["n_rows"],
                save_path=config["output_save_path"],
                with_softmax=use_softmax,
            )

            progress_bar.progress(100)
            status_text.text("✅ Inference completed successfully!")

            # Store the plot path in session state
            st.session_state.inference_plot_path = plot_path
            st.session_state.inference_done = True
            st.success("🎉 Inference completed!")

        except Exception as e:
            st.error(f"❌ Error during inference: {str(e)}")

    # Display results if inference has been completed
    if st.session_state.inference_done and st.session_state.inference_plot_path:
        display_results(st.session_state.inference_plot_path)


def display_results(plot_path):
    """Display generated inference results

    Args:
        plot_path: Path to the inference result image
    """
    st.divider()
    st.subheader("📈 Inference Results")

    if plot_path and os.path.exists(plot_path):
        st.success("✅ Inference completed successfully!")
        st.write(f"**Output:** `{os.path.basename(plot_path)}`")

        # Display the inference image in full width
        st.image(plot_path, width="stretch")

        # Add download button centered below the image
        _, col2, _ = st.columns([1, 2, 1])
        with col2, open(plot_path, "rb") as file:
            st.download_button(
                label="📥 Download Result",
                data=file,
                file_name=os.path.basename(plot_path),
                mime="image/png",
                width="stretch",
                type="primary",
            )
    else:
        st.info("ℹ️ No result image found. Run inference to generate results.")


def create_training_config(config):
    """Create a training configuration dictionary"""
    return {
        "seed": 42,
        "test_only": False,
        "accelerator": config.get("device", "cpu"),  # Use selected device
        "task": "binary",
        "data": {
            "data_dir": config["boxed_png_path"],  # Use boxed images for training
            "train_dir": config["boxed_png_path"],  # Use boxed images for training
            "data_loader": "RawPngLoader",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "num_workers": 4,
            "augment": False,
            "image_size": 256,
            "labels_file": os.path.basename(config["labelling_path"]),
        },
        "wandb": {
            "project_name": "ghost-hunter-web",
            "run_name": "",
            "experiment_id": "1",
            "ex_description": config["project_name"],
        },
        "model": {
            "in_channels": 1,
            "batch_size": config["batch_size"],
            "model_name": "ThreeLayerCnn",
            "accuracy_metric": "cross_entropy",
        },
        "optimizer": {
            "optimizer": "adamw",
            "lr": config["learning_rate"],
            "weight_decay": 0.002,
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
            "max_epochs": config["max_epochs"],
            "log_dir": "./logs",
            "log_every_n_steps": 100,
            "load_path": "",
            "resume_from": False,
            "checkpoint_dir": f"{config['project_name']}/",
            "wandb_logging": False,
        },
        "test": {
            "load_path": f"{config['project_name']}/",
        },
    }


if __name__ == "__main__":
    main()
