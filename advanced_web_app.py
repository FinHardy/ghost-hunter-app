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
import py4DSTEM
import streamlit as st
import yaml
from PIL import Image

from scripts.binary_search_labeller import label

# Import the DM4 image display function
from scripts.show_image_from_dm4 import (
    show_random_diffraction_pattern,
    show_virtual_image,
)

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent))

try:
    from scripts.create_average_boxes import convert_all_to_boxes
    from scripts.dm4_to_png import save_dm4_BF_to_png
    from scripts.streamlit_labeller import (
        StreamlitLabellerState,
        save_label,
        get_label_statistics,
        load_existing_labels
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
        "**Transform DM4 files → Label data → Train models → Generate insights**"
    )

    # Sidebar navigation
    with st.sidebar:
        st.title("🚀 Pipeline Steps")

        # Step selection
        steps = [
            "1️⃣ Setup & Configuration",
            "2️⃣ DM4 to PNG Conversion",
            "3️⃣ Average Boxing",
            "4️⃣ Data Labelling",
            "5️⃣ Model Training",
            "6️⃣ Inference & Results",
        ]

        selected_step = st.selectbox(
            "Choose step:", steps, index=st.session_state.step - 1
        )
        st.session_state.step = steps.index(selected_step) + 1  # type: ignore

        # Progress indicator
        progress = st.session_state.step / 6
        st.progress(progress)
        st.write(f"Progress: {st.session_state.step}/6 steps")

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

    # Route to appropriate page
    if st.session_state.step == 1:
        setup_configuration()
    elif st.session_state.step == 2:
        dm4_conversion()
    elif st.session_state.step == 3:
        average_boxing()
    elif st.session_state.step == 4:
        data_labelling()
    elif st.session_state.step == 5:
        model_training()
    elif st.session_state.step == 6:
        inference_results()


def setup_configuration():
    """Step 1: Setup and Configuration"""
    st.header("⚙️ Step 1: Setup & Configuration")

    st.subheader("Project Name")
    project_name = st.text_input(
        "Enter a unique project name",
        value="my_ghost_hunter_project",
        key="project_name",
        help="This will be used to name output directories and files",
    )

    # File path input
    st.subheader("� Specify DM4 File Path")
    dm4_file_path = st.text_input(
        "Enter the full path to your DM4 file",
        value="",
        key="dm4_file_path",
        help="Example: /home/user/data/experiment.dm4"
    )


    st.subheader("🏷️ Labelling Parameters")
    sampling_interval = st.slider(
        "Sampling Interval", min_value=1, max_value=50, value=10, key="sampling_interval"
    )
    number_of_labels = st.slider(
        "Number of Labels",
        min_value=10,
        max_value=1000,
        value=50,
        key="number_of_labels",
    )

    st.subheader("📦 Boxing Parameters")
    box_size = st.slider(
        "Box Size",
        min_value=1,
        max_value=5,
        value=3,
        key="box_size",
        help="Size of the averaging box for creating averaged images",
    )

    st.info(
        "ℹ️ **Array dimensions will be automatically detected** from your image filenames!"
    )

    st.subheader("🤖 Training Parameters")
    max_epochs = st.number_input(
        "Max Epochs", value=50, min_value=1, key="max_epochs"
    )
    batch_size = st.selectbox(
        "Batch Size", [16, 32, 64, 128], index=1, key="batch_size"
    )
    learning_rate = st.number_input(
        "Learning Rate", value=0.0004, format="%.6f", key="learning_rate"
    )


    
    # Validate file path
    dm4_file_valid = False
    if dm4_file_path:
        if os.path.exists(dm4_file_path):
            if dm4_file_path.endswith('.dm4'):
                dm4_file_valid = True
                st.success(f"✅ Valid DM4 file: {os.path.basename(dm4_file_path)}")
            else:
                st.error("❌ File must have .dm4 extension")
        else:
            st.error("❌ File path does not exist")

    # Show DM4 info and images if valid path provided
    if dm4_file_valid:
        st.info(
            "Preview: Random diffraction image and virtual image from DM4 file"
        )
        temp_dm4_path = dm4_file_path

        # Get dataset info and show dimensions
        import py4DSTEM

        try:
            dataset = py4DSTEM.import_file(temp_dm4_path)
            shape = dataset.data.shape  # type: ignore
            dim2 = shape[0]
            dim1 = shape[1]
            st.write(
                f"**Real space image dimensions:** {dim1} x {dim2} (scan positions)"
            )
            st.write(
                f"**Diffraction pattern size:** {dim1} x {dim2} (detector pixels)"
            )
        except Exception as e:
            st.warning(f"Could not read DM4 file info: {e}")

        # Show random diffraction image
        try:
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            
            # Load dataset and show random diffraction pattern
            shape = dataset.data.shape
            random_i = random.randint(0, shape[0] - 1)
            random_j = random.randint(0, shape[1] - 1)
            diffraction_pattern = dataset[random_i, random_j].data
            
            ax1.imshow(diffraction_pattern, cmap="gray")
            ax1.set_title(f"Random Diffraction Pattern at ({random_i}, {random_j})")
            ax1.set_xlabel("Detector X (pixels)")
            ax1.set_ylabel("Detector Y (pixels)")
            plt.colorbar(ax1.images[0], ax=ax1, label="Intensity")
            
            st.pyplot(fig1)
            plt.close(fig1)
        except Exception as e:
            st.warning(f"Could not display random diffraction image: {e}")

        # Show virtual image
        try:
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            
            # Load dataset and create virtual image
            virtual_image = dataset.data.sum(axis=(2, 3))
            
            ax2.imshow(virtual_image, cmap="gray")
            ax2.set_title("Virtual Image (summed intensity)")
            ax2.set_xlabel("Scan X")
            ax2.set_ylabel("Scan Y")
            plt.colorbar(ax2.images[0], ax=ax2, label="Intensity")
            
            st.pyplot(fig2)
            plt.close(fig2)
        except Exception as e:
            st.warning(f"Could not display virtual image: {e}")

    # Save configuration
    if st.button("💾 Save Configuration & Proceed", type="primary"):
        if not dm4_file_path or not dm4_file_valid:
            st.error("❌ Please provide a valid DM4 file path!")
            return

        config = {
            "project_name": project_name,
            "dm4_file_path": dm4_file_path,
            "output_png_path": f"data/png/{project_name}",
            "boxed_png_path": f"data/boxed_png/{project_name}",
            "labelling_path": f"labelling/{project_name}_labels.yaml",
            "output_save_path": f"output/{project_name}/",
            "dim1": dim1,
            "dim2": dim2,
            "box_size": box_size,
            "sampling_space": sampling_interval,
            "number_of_labels": number_of_labels,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        st.session_state.config = config
        st.success("✅ Configuration saved successfully!")
        st.info("👉 You can now proceed to Step 2: DM4 Conversion")


def dm4_conversion():
    """Step 2: DM4 to PNG Conversion"""
    st.header("🔄 Step 2: DM4 to PNG Conversion")

    if not st.session_state.config:
        st.error("❌ Please complete Step 1 first!")
        return

    config = st.session_state.config

    # Display current settings
    with st.expander("Current Configuration", expanded=False):
        st.json(config)

    st.subheader("🛠️ Conversion Settings")
    crop_images = st.checkbox("Crop Images", key="crop_images")

    crop_values = None
    if crop_images:
        st.write("**Crop Parameters (x_min, x_max, y_min, y_max):**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_min = st.number_input("X Min", value=0, key="x_min")
        with col2:
            x_max = st.number_input("X Max", value=512, key="x_max")
        with col3:
            y_min = st.number_input("Y Min", value=0, key="y_min")
        with col4:
            y_max = st.number_input("Y Max", value=512, key="y_max")
        crop_values = (x_min, x_max, y_min, y_max)

    if st.button("🚀 Start DM4 Conversion", type="primary"):
        if not config.get("dm4_file_path"):
            st.error("❌ Please provide a valid DM4 file path in Step 1!")
            return

        try:
            # Use the file path directly - no need to copy
            dm4_file_path = config["dm4_file_path"]
            
            if not os.path.exists(dm4_file_path):
                st.error(f"❌ File not found: {dm4_file_path}")
                return

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔄 Converting DM4 to PNG format...")
            progress_bar.progress(50)

            # Call conversion function directly with the file path
            save_dm4_BF_to_png(
                dm4_file_path,
                config["output_png_path"],
                crop=crop_images,
                crop_values=crop_values,
            )

            progress_bar.progress(100)
            status_text.text("✅ Conversion completed successfully!")

            st.session_state.conversion_done = True
            st.success("🎉 DM4 conversion completed! Ready for average boxing.")

        except Exception as e:
            st.error(f"❌ Error during conversion: {str(e)}")


def average_boxing():
    """Step 3: Average Boxing"""
    st.header("📦 Step 3: Average Boxing")

    if not st.session_state.config:
        st.error("❌ Please complete previous steps first!")
        return

    if not st.session_state.conversion_done:
        st.warning("⚠️ Please complete DM4 conversion first!")
        return

    config = st.session_state.config

    # Display current settings
    with st.expander("Current Configuration", expanded=False):
        st.json(config)

    st.subheader("📦 Boxing Configuration")
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Input Settings:**
        - 📁 PNG Directory: `{config["output_png_path"]}`
        - � Box Size: {config["box_size"]}
        - � Array Length: Auto-detected from filenames
        """)

    with col2:
        st.info(f"""
        **Output Settings:**
        - 📁 Output Directory: `{config["boxed_png_path"]}`
        - 🖼️ Processing: Max values with {config["box_size"]}x{config["box_size"]} boxes
        """)

        st.subheader("ℹ️ About Boxing")
        st.write("""
        **Average boxing** creates averaged images from neighboring regions:
        - Takes a box of images (e.g., 3x3 grid)
        - Computes max/average values across the box
        - Reduces noise and enhances features
        - Essential preprocessing step for labelling
        """)

    if st.button("🚀 Start Average Boxing", type="primary"):
        try:
            # Ensure input directory exists and has files
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

            st.info(f"📊 Found {len(png_files)} PNG files to process")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("📦 Initializing boxing process...")
            progress_bar.progress(10)

            # Ensure output directory exists
            os.makedirs(config["boxed_png_path"], exist_ok=True)

            progress_bar.progress(25)
            status_text.text("📦 Creating averaged boxes... (This may take a while)")

            # Call boxing function
            convert_all_to_boxes(
                stem_image_dir=config["output_png_path"],
                output_dir=config["boxed_png_path"],
                box_size=config["box_size"],
            )

            progress_bar.progress(100)
            status_text.text("✅ Boxing completed successfully!")

            st.session_state.boxing_done = True
            st.success("🎉 Average boxing completed! Ready for data labelling.")

            # Show results summary
            boxed_files = [
                f for f in os.listdir(config["boxed_png_path"]) if f.endswith(".png")
            ]
            st.info(f"📊 Generated {len(boxed_files)} boxed images")

        except Exception as e:
            st.error(f"❌ Error during boxing: {str(e)}")
            st.write("**Debugging info:**")
            st.write(f"- Input directory: {config['output_png_path']}")
            st.write(f"- Output directory: {config['boxed_png_path']}")
            st.write(f"- Box size: {config['box_size']}")
            st.write("- Array length: Auto-detected from filenames")


def data_labelling():
    """Step 4: Data Labelling - Streamlit Native Interface"""
    st.header("🏷️ Step 4: Data Labelling")

    if not st.session_state.config:
        st.error("❌ Please complete previous steps first!")
        return

    if not st.session_state.boxing_done:
        st.warning("⚠️ Please complete average boxing first!")
        return

    config = st.session_state.config

    # Display configuration
    st.subheader("📋 Labelling Configuration")
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Current Settings:**
        - 📁 Data Path: `{config["boxed_png_path"]}`
        - 🎯 Sampling Space: {config["sampling_space"]}
        - 🔢 Number of Labels: {config["number_of_labels"]}
        """)

    with col2:
        st.info(f"""
        **Output:**
        - 📄 Labels File: `{config["labelling_path"]}`
        """)

    # Initialize labeller state if needed
    if st.button("🚀 Initialize Labelling Session", type="primary") or st.session_state.labelling_initialized:
        try:
            # Ensure labelling directory exists
            os.makedirs(os.path.dirname(config["labelling_path"]), exist_ok=True)
            
            # Initialize labeller state only once
            if not st.session_state.labelling_initialized:
                st.session_state.labeller_state = StreamlitLabellerState(
                    file_path=config["boxed_png_path"],
                    output_file=config["labelling_path"],
                    labels_to_assign=config["number_of_labels"],
                    step=config["sampling_space"]
                )
                st.session_state.labelling_initialized = True
                st.rerun()
            
            labeller = st.session_state.labeller_state
            
            # Check if labelling is complete
            if labeller.is_complete():
                st.success("🎉 All labels have been assigned!")
                st.session_state.labelling_done = True
                
                # Show final heatmap
                st.subheader("📊 Final Label Distribution")
                fig, ax = plt.subplots(figsize=(10, 10))
                final_heatmap = labeller.save_final_heatmap()
                cmap = plt.get_cmap("viridis", 4)
                im = ax.imshow(final_heatmap, cmap=cmap, vmin=0, vmax=3)
                ax.set_title("Final Label Map")
                ax.axis("off")
                plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], label="Label")
                st.pyplot(fig)
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
                fig, ax = plt.subplots(figsize=(8, 8))
                cmap = plt.get_cmap("viridis", 4)
                im = ax.imshow(labeller.sparse_array, cmap=cmap, vmin=0, vmax=3)
                
                # Highlight current position
                pos = labeller.get_current_position()
                if pos:
                    i, j = pos
                    ax.plot(j, i, marker="s", color="red", markersize=12, 
                           markeredgewidth=2, markeredgecolor="black")
                
                ax.set_title("Current Labelling Progress")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                st.subheader("🖼️ Current Image")
                st.write(f"**File:** `{current_file}`")
                
                # Load and display image
                img_path = os.path.join(config["boxed_png_path"], current_file)
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("L")
                    st.image(img, width="stretch")
                else:
                    st.error(f"❌ Image not found: {img_path}")
            
            # Labelling buttons
            st.subheader("🏷️ Assign Label")
            st.write("**Choose the label for this image:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔴 Horizontal Polarisation", key="label_horizontal", width="stretch"):
                    save_label(current_file, config["labelling_path"], "horizontal")
                    labeller.update_sparse_array(1)
                    st.session_state.current_label_file = None
                    st.rerun()
            
            with col2:
                if st.button("🟢 Vertical Polarisation", key="label_vertical", width="stretch"):
                    save_label(current_file, config["labelling_path"], "vertical")
                    labeller.update_sparse_array(2)
                    st.session_state.current_label_file = None
                    st.rerun()
            
            with col3:
                if st.button("🟡 No Observable Polarisation", key="label_none", width="stretch"):
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
                st.warning("⚠️ Labelling session reset. Note: Previously saved labels in the YAML file are preserved.")
                st.rerun()
                
        except FileNotFoundError as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Make sure you have completed the average boxing step first.")
        except Exception as e:
            st.error(f"❌ Error during labelling: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Show existing labels if file exists
    if os.path.exists(config["labelling_path"]):
        with st.expander("📊 View Existing Labels"):
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
        st.error("❌ Please complete previous steps first!")
        return

    if not st.session_state.labelling_done:
        st.warning("⚠️ Please complete data labelling first!")
        return

    config = st.session_state.config

    st.subheader("⚙️ Training Configuration")
    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Model Settings:**
        - 🧠 Model: ThreeLayerCnn
        - 📊 Max Epochs: {config["max_epochs"]}
        - 📦 Batch Size: {config["batch_size"]}
        - 📈 Learning Rate: {config["learning_rate"]}
        """)

    with col2:
        st.info(f"""
        **Data Configuration:**
        - 🎯 Train Ratio: 80%
        - ✅ Validation Ratio: 10%
        - 🧪 Test Ratio: 10%
        - 🖼️ Image Size: 256x256
        """)

    if st.button("🚀 Start Model Training", type="primary"):
        try:
            # Create config file for training
            training_config = create_training_config(config)
            config_file_path = f"configs/{config['project_name']}.yaml"

            # Ensure configs directory exists
            os.makedirs("configs", exist_ok=True)

            # Save config file
            with open(config_file_path, "w") as f:
                yaml.dump(training_config, f, default_flow_style=False)

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🚀 Initializing training...")
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
        st.info("ℹ️ No training logs found yet.")
        return
    
    # Get all version directories and sort by modification time
    version_dirs = [d for d in os.listdir(lightning_logs_dir) if d.startswith("version_")]
    if not version_dirs:
        st.info("ℹ️ No training logs found yet.")
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
        if 'train_loss' in df.columns:
            final_train_loss = df['train_loss'].dropna().iloc[-1]
            with col1:
                st.metric("Final Train Loss", f"{final_train_loss:.4f}")
        
        if 'val_loss' in df.columns:
            final_val_loss = df['val_loss'].dropna().iloc[-1]
            best_val_loss = df['val_loss'].min()
            with col2:
                st.metric("Final Val Loss", f"{final_val_loss:.4f}")
                st.metric("Best Val Loss", f"{best_val_loss:.4f}")
        
        if 'train_acc' in df.columns:
            final_train_acc = df['train_acc'].dropna().iloc[-1]
            with col3:
                st.metric("Final Train Accuracy", f"{final_train_acc:.2%}")
        
        if 'val_acc' in df.columns:
            final_val_acc = df['val_acc'].dropna().iloc[-1]
            best_val_acc = df['val_acc'].max()
            with col3:
                st.metric("Final Val Accuracy", f"{final_val_acc:.2%}")
                st.metric("Best Val Accuracy", f"{best_val_acc:.2%}")
        
        # Plot training curves
        st.write("### 📈 Training Curves")
        
        # Loss plot
        if 'train_loss' in df.columns or 'val_loss' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            
            if 'train_loss' in df.columns:
                train_loss_clean = df[['epoch', 'train_loss']].dropna()
                ax.plot(train_loss_clean['epoch'], train_loss_clean['train_loss'], 
                       label='Train Loss', marker='o', markersize=3)
            
            if 'val_loss' in df.columns:
                val_loss_clean = df[['epoch', 'val_loss']].dropna()
                ax.plot(val_loss_clean['epoch'], val_loss_clean['val_loss'], 
                       label='Validation Loss', marker='s', markersize=3)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        # Accuracy plot
        if 'train_acc' in df.columns or 'val_acc' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            
            if 'train_acc' in df.columns:
                train_acc_clean = df[['epoch', 'train_acc']].dropna()
                ax.plot(train_acc_clean['epoch'], train_acc_clean['train_acc'] * 100, 
                       label='Train Accuracy', marker='o', markersize=3)
            
            if 'val_acc' in df.columns:
                val_acc_clean = df[['epoch', 'val_acc']].dropna()
                ax.plot(val_acc_clean['epoch'], val_acc_clean['val_acc'] * 100, 
                       label='Validation Accuracy', marker='s', markersize=3)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Training and Validation Accuracy')
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
    st.header("📊 Step 5: Inference & Results")

    if not st.session_state.config:
        st.error("❌ Please complete previous steps first!")
        return

    if not st.session_state.training_done:
        st.warning("⚠️ Please complete model training first!")
        return

    config = st.session_state.config

    st.subheader("🔮 Generate Results")
    
    # Add softmax toggle
    col1, col2 = st.columns([2, 1])
    with col1:
        use_softmax = st.checkbox(
            "Apply Softmax Activation",
            value=True,
            help="Apply softmax to the model output for probability distribution. Uncheck for raw logits."
        )
    with col2:
        st.write("")  # Spacing

    if st.button("🚀 Run Inference", type="primary"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔮 Running inference...")
            progress_bar.progress(25)

            config_file = f"{config['project_name']}.yaml"

            # Ensure output directory exists
            os.makedirs(config["output_save_path"], exist_ok=True)

            progress_bar.progress(50)

            # Call inference function and store the returned plot path
            plot_path = plot_embeddings(
                config_file,
                config["dim1"],
                config["dim2"],
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
        st.success(f"✅ Inference completed successfully!")
        st.write(f"**Output:** `{os.path.basename(plot_path)}`")
        
        # Display the inference image in full width
        st.image(plot_path, width="stretch")
        
        # Add download button centered below the image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with open(plot_path, "rb") as file:
                st.download_button(
                    label=f"📥 Download Result",
                    data=file,
                    file_name=os.path.basename(plot_path),
                    mime="image/png",
                    width="stretch",
                    type="primary"
                )
    else:
        st.info("ℹ️ No result image found. Run inference to generate results.")


def create_training_config(config):
    """Create a training configuration dictionary"""
    return {
        "seed": 42,
        "test_only": False,
        "accelerator": "cuda",
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
        }
    }


if __name__ == "__main__":
    main()
