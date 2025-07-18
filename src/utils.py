"""
Place for redundant functions that are used across multiple files
"""

import math
import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import py4DSTEM
import py4DSTEM.preprocess.preprocess as pre
import torch
from fire import Fire
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def load_state_dict(model, load_path: str, accelerator: str) -> None:
    """
    Description: Load the state dict into the model
    Args:
    - model: Model to load the state dict into
    - load_path (str): Path to the checkpoint (has to be absolute)
    - accelerator (str): Device to load the model on
    """

    assert os.path.exists(str(load_path)), f"Checkpoint {str(load_path)} does not exist"
    checkpoint = torch.load(str(load_path), weights_only=True, map_location=accelerator)
    model.load_state_dict(checkpoint["state_dict"])


def load_random_png_as_tensor(data_dir: str, image_size: int) -> torch.Tensor:
    """
    Loads a random PNG file from the specified directory and transforms it into a tensor.

    Args:
    - data_dir (str): Directory containing PNG files.
    - image_size (Tuple[int, int]): Desired image size (height, width).

    Returns:
    - torch.Tensor: Transformed image tensor.
    """
    # List all PNG files in the directory
    png_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]
    if not png_files:
        raise FileNotFoundError("No PNG files found in the specified directory.")

    # Randomly select one file
    random_file = random.choice(png_files)
    image_path = os.path.join(data_dir, random_file)

    # Load the image using PIL
    image = Image.open(image_path).convert("L")  # Convert to grayscale if needed

    # Transform the image into a tensor
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    image_tensor = transform(image)

    return image_tensor


def generate_image_with_missing_pixels(
    image: torch.Tensor, noise_ratio: float = 0.25
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description: Generates an image with missing pixels.

    Args:
    - image (torch.Tensor): Image tensor
    - noise_ratio (float): Ratio of missing pixels

    Returns:
    - modified_image (torch.Tensor): Image with missing pixels
    - ground_truth (torch.Tensor): Original image
    """
    ground_truth = image.clone()
    modified_image = image.clone()
    num_pixels = modified_image.shape[1] * modified_image.shape[2]
    num_missing_pixels = int(math.ceil(num_pixels * noise_ratio))

    image_flat = modified_image.view(-1)
    missing_indices = torch.randperm(num_pixels)[:num_missing_pixels]
    image_flat[missing_indices] = 0

    missing_pixels = image_flat.view(image.shape)
    return missing_pixels, ground_truth


def save_dm4_BF_to_png(*files, binning_param: int = 2):
    """
    Description:
    - Save the Bright Field images from the dm4 files to png format
    Args:
    - files: List of dm4 files (do not require absolute path)

    IMPORTANT: Files do not need absolute path but do need to be stored inside
    /data/dm4 directory

    e.g. save_dm4_BF_to_png(["file1.dm4", "file2.dm4"])

    """
    base_file = os.getcwd()
    dm4_files = os.path.join(base_file, "../data/dm4")

    for file in files:
        assert ".dm4" in file, "File must be a dm4 file"
        output_dir = os.path.join(dm4_files, file.split(".")[0])

        os.makedirs(output_dir, exist_ok=True)

        print(os.path.join(dm4_files, file))

        try:
            image = py4DSTEM.import_file(os.path.join(dm4_files, file))
            pre_shape = image.data.shape
            pre.bin_data_diffraction(image, binning_param)
            shape = image.data.shape
            assert pre_shape != shape, "Data shape should change after binning"
            for i in tqdm(range(shape[0])):
                for j in range(shape[1]):
                    diffraction_pattern = image[i, j].data
                    filename = os.path.join(output_dir, f"{file}_{i}_{j}.png")
                    plt.imsave(filename, diffraction_pattern, cmap="gray")

        except Exception as e:
            # if not 4d STEM image, will get caught here e.g. dm4 files that only show ADF images
            print(e)


if __name__ == "__main__":
    # using this file to run the save file function since its just a utility function
    Fire(save_dm4_BF_to_png)
