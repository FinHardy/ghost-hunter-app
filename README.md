# Advanced Ghost Hunter Web App

This project provides an interactive Streamlit-based web application for the Ghost Hunter ML pipeline.

## Prerequisites

- Python 3.8+
- Install dependencies:

```sh
pip install -r requirements_web.txt
```

## Running the Web App

To start the advanced web app, run:

```sh
streamlit run advanced_web_app.py
```

This will launch the Ghost Hunter ML Pipeline interface in your browser.

## Usage

1. **Setup & Configuration**: Enter project details and parameters.
2. **DM4 to PNG Conversion**: Convert DM4 files to PNG format.
3. **Average Boxing**: Preprocess images using average boxing.
4. **Data Labelling**: Label your data interactively.
5. **Model Training**: Train your ML model.
6. **Inference & Results**: Generate and view results.

## Troubleshooting

- If you encounter import errors, ensure you have installed all dependencies from `requirements_web.txt`.
- For GPU acceleration, ensure CUDA is available and compatible with your PyTorch installation.

## License