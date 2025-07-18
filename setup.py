from pathlib import Path

from setuptools import find_packages, setup

directory = Path(__file__).resolve().parent

setup(
    name="src",
    packages=find_packages(),
    description="polarisation mapping",
    author="Fintan Hardy",
    license="MIT",
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "lightning",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "numpy",
        "scipy",
        "scikit-image",
        "wandb",
        "pyyaml",
        "pillow",
        "fire",
        "py4DSTEM",
    ],
    extras_require={
        "testing": [
            "pytest",
            "pytest-flake8",
            "pytest-pylint",
            "pytorch-lightning",
            "lightning",
            "torch",
            "pyyaml",
            "torchvision",
            "pillow",
            "numpy",
            "flake8",
            "pylint",
            "mypy",
            "black",
            "isort",
            "pre-commit",
        ]
    },
    python_requires=">=3.6",
)
