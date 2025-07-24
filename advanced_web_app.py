"""
Advanced Ghost Hunter Web App
Complete Streamlit implementation with full workflow integration
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import yaml

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent))

try:
    from scripts.create_average_boxes import convert_all_to_boxes
    from scripts.dm4_to_png import save_dm4_BF_to_png
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
        st.session_state.step = steps.index(selected_step) + 1

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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Project Settings")
        project_name = st.text_input(
            "Project Name", value="my_experiment", key="project_name"
        )

        st.subheader("📊 Image Dimensions")
        dim1 = st.number_input("Height (dim1)", value=512, min_value=1, key="dim1")
        dim2 = st.number_input("Width (dim2)", value=512, min_value=1, key="dim2")

    with col2:
        st.subheader("🏷️ Labelling Parameters")
        sampling_space = st.slider(
            "Sampling Space", min_value=1, max_value=50, value=10, key="sampling_space"
        )
        number_of_labels = st.slider(
            "Number of Labels",
            min_value=10,
            max_value=1000,
            value=100,
            key="number_of_labels",
        )

        st.subheader("📦 Boxing Parameters")
        box_size = st.slider(
            "Box Size",
            min_value=1,
            max_value=10,
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

    # File upload
    st.subheader("📤 Upload DM4 File")
    dm4_file = st.file_uploader("Choose a DM4 file", type=["dm4"], key="dm4_file")

    # Save configuration
    if st.button("💾 Save Configuration & Proceed", type="primary"):
        config = {
            "project_name": project_name,
            "dm4_file": dm4_file,
            "output_png_path": f"data/png/{project_name}",
            "boxed_png_path": f"data/boxed_png/{project_name}",
            "labelling_path": f"labelling/{project_name}_labels.yaml",
            "output_save_path": f"output/{project_name}/",
            "dim1": dim1,
            "dim2": dim2,
            "box_size": box_size,
            "sampling_space": sampling_space,
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
        if config["dm4_file"] is None:
            st.error("❌ Please upload a DM4 file in Step 1!")
            return

        try:
            # Create temporary file for DM4
            temp_dm4_path = os.path.join(
                st.session_state.temp_dir, config["dm4_file"].name
            )
            with open(temp_dm4_path, "wb") as f:
                f.write(config["dm4_file"].getbuffer())

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔄 Converting DM4 to PNG format...")
            progress_bar.progress(25)

            # Ensure output directory exists
            os.makedirs(config["output_png_path"], exist_ok=True)

            progress_bar.progress(50)

            # Call conversion function
            save_dm4_BF_to_png(
                temp_dm4_path,
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

        # Optional parameters
        st.subheader("🎛️ Advanced Options")
        smoothing_type = st.selectbox(
            "Smoothing Type", ["gamma", "sigmoid"], index=0, key="smoothing_type"
        )
        gamma_value = st.slider(
            "Gamma Correction",
            min_value=0.1,
            max_value=3.0,
            value=1.2,
            step=0.1,
            key="gamma_value",
        )

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
                smoothing=smoothing_type,
                gamma=gamma_value,
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
    """Step 4: Data Labelling"""
    st.header("🏷️ Step 4: Data Labelling")

    if not st.session_state.config:
        st.error("❌ Please complete previous steps first!")
        return

    if not st.session_state.boxing_done:
        st.warning("⚠️ Please complete average boxing first!")
        return

    config = st.session_state.config

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

    st.warning(
        "⚠️ **Important:** The labelling process will open a separate GUI window. Please complete the labelling there and return here when finished."
    )

    if st.button("🎯 Start Interactive Labelling", type="primary"):
        try:
            status_text = st.empty()
            status_text.text("🚀 Opening labelling interface...")

            # Ensure labelling directory exists
            os.makedirs(os.path.dirname(config["labelling_path"]), exist_ok=True)

            # Import and call labelling function - use boxed images now
            from labelling.binary_search_labeler import label

            label(
                config[
                    "boxed_png_path"
                ],  # Changed from output_png_path to boxed_png_path
                config["labelling_path"],
                config["sampling_space"],
                config["number_of_labels"],
            )

            st.success("🎉 Labelling interface launched!")

        except Exception as e:
            st.error(f"❌ Error starting labelling: {str(e)}")

    # Check if labelling is complete
    if os.path.exists(config["labelling_path"]):
        try:
            with open(config["labelling_path"], "r") as f:
                labels_data = yaml.safe_load(f)
                if labels_data and "labels" in labels_data:
                    num_labels = len(labels_data["labels"])
                    st.success(f"✅ Found {num_labels} labels in the file!")
                    st.session_state.labelling_done = True

                    # Show some statistics
                    if st.checkbox("Show label statistics"):
                        label_counts = {}
                        for item in labels_data["labels"]:
                            label = item.get("label", "unknown")
                            label_counts[label] = label_counts.get(label, 0) + 1

                        st.write("**Label Distribution:**")
                        for label, count in label_counts.items():
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

        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.write("**Debugging info:**")
            st.write(f"- Config file path: {config_file_path}")
            st.write(f"- Training config: {training_config}")


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

    if st.button("🚀 Run Inference", type="primary"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔮 Running inference...")
            progress_bar.progress(25)

            config_file = f"configs/{config['project_name']}.yaml"

            # Ensure output directory exists
            os.makedirs(config["output_save_path"], exist_ok=True)

            progress_bar.progress(50)

            # Call inference function
            plot_embeddings(
                config_file,
                config["dim1"],
                config["dim2"],
                save_path=config["output_save_path"],
            )

            progress_bar.progress(100)
            status_text.text("✅ Inference completed successfully!")

            st.session_state.inference_done = True
            st.success("🎉 Inference completed!")

            # Display results
            display_results(config)

        except Exception as e:
            st.error(f"❌ Error during inference: {str(e)}")


def display_results(config):
    """Display generated results"""
    st.subheader("📈 Results")

    output_dir = config["output_save_path"]
    if os.path.exists(output_dir):
        # List all image files
        image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp")
        image_files = [
            f for f in os.listdir(output_dir) if f.lower().endswith(image_extensions)
        ]

        if image_files:
            st.write(f"✅ Found {len(image_files)} result images:")

            # Display images in a grid
            cols = st.columns(min(len(image_files), 3))
            for i, image_file in enumerate(image_files[:6]):  # Show max 6 images
                with cols[i % 3]:
                    image_path = os.path.join(output_dir, image_file)
                    st.image(image_path, caption=image_file, use_column_width=True)

                    # Download button for each image
                    with open(image_path, "rb") as file:
                        st.download_button(
                            label=f"📥 Download {image_file}",
                            data=file.read(),
                            file_name=image_file,
                            mime="image/png",
                        )
        else:
            st.info("ℹ️ No result images found yet.")
    else:
        st.info("ℹ️ Results directory not found.")


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
    }


if __name__ == "__main__":
    main()
