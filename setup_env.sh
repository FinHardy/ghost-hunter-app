#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="ghost-hunting-env"
PYTHON_VERSION="3.9"

echo ">>> Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

echo ">>> Activating '${ENV_NAME}'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo ">>> Installing packages from requirements.txt using pip..."
pip install -r requirements.txt

echo ">>> Setup complete! To activate your environment, run:"
echo "    conda activate ${ENV_NAME}"