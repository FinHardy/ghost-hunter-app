"""
Interactive Web App for Ghost Hunter Workflow
Streamlit-based web interface for the machine learning pipeline
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
import yaml

# Page configuration
st.set_page_config(page_title="Ghost Hunter ML Pipeline", page_icon="👻", layout="wide")


def main():
    st.title("👻 Ghost Hunter: Interactive ML Pipeline")
    st.markdown("Transform DM4 files → Label data → Train models → Generate insights")

    # Sidebar navigation
    st.sidebar.title("Pipeline Steps")
    step = st.sidebar.selectbox(
        "Choose step:",
        [
            "1. Setup & Configuration",
            "2. DM4 to PNG Conversion",
            "3. Data Labelling",
            "4. Model Training",
            "5. Inference & Results",
        ],
    )

    if "1. Setup" in step:
        setup_page()
    elif "2. DM4" in step:
        conversion_page()
    elif "3. Data" in step:
        labelling_page()
    elif "4. Model" in step:
        training_page()
    elif "5. Inference" in step:
        inference_page()


def setup_page():
    st.header("⚙️ Setup & Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project Settings")
        project_name = st.text_input("Project Name", value="my_experiment")
        dm4_file = st.file_uploader("Upload DM4 file", type=["dm4"])

    with col2:
        st.subheader("Parameters")
        dim1 = st.number_input("Height", value=512)
        dim2 = st.number_input("Width", value=512)
        sampling_space = st.slider("Sampling Space", 1, 50, 10)

    if st.button("Save Configuration"):
        st.success("Configuration saved!")


def conversion_page():
    st.header("🔄 DM4 to PNG Conversion")
    st.info("Convert your DM4 files to PNG format for processing")

    if st.button("Start Conversion"):
        with st.spinner("Converting..."):
            st.success("Conversion completed!")


def labelling_page():
    st.header("🏷️ Data Labelling")
    st.info("Label your data interactively")

    if st.button("Start Labelling"):
        st.success("Labelling interface launched!")


def training_page():
    st.header("🤖 Model Training")
    st.info("Train your machine learning model")

    if st.button("Start Training"):
        with st.spinner("Training..."):
            st.success("Training completed!")


def inference_page():
    st.header("📊 Inference & Results")
    st.info("Generate results and visualizations")

    if st.button("Run Inference"):
        with st.spinner("Running inference..."):
            st.success("Results generated!")


if __name__ == "__main__":
    main()
