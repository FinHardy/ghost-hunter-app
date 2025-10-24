import os
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

ABS_PATH = os.path.abspath(os.path.dirname(__file__))


def logical_sort(file_list):
    """
    Sorts a list of file paths logically based on the numeric parts of the filenames.
    Also returns extracted coordinates for inferring array dimensions.

    Args:
        file_list (list of str): List of file paths to sort.

    Returns:
        tuple: (sorted list of file paths, list of coordinates)
    """

    def logical_sort_key(filepath):
        filename = os.path.basename(filepath)
        # Extract all numeric parts from the filename (e.g., _0_15 -> [0, 15])
        numbers = re.findall(r"\d+", filename)
        # Convert all extracted numbers to integers and return as a tuple
        return tuple(map(int, numbers))

    sorted_files = sorted(file_list, key=logical_sort_key)
    coordinates = [logical_sort_key(f) for f in sorted_files]
    return sorted_files, coordinates


def average_vals_image(
    box_size: int,
    image_list: list[str],
    array_width: int,
    starting_x: int,
    starting_y: int,
    to_save: bool = False,
    output_dir: str = ".",
):
    """
    Compute an average values image by taking the average intensity at each pixel
    within a specified box_size region across multiple images.

    Args:
        box_size (int): Size of the box to extract.
        image_list (list): Flattened list of image paths.
        array_width (int): Width of the 2D grid of images.
        starting_x (int): Starting x-coordinate.
        starting_y (int): Starting y-coordinate.
        to_save (bool): Whether to save the output image.
        output_dir (str): Directory to save the image if `to_save` is True.

    Returns:
        None. Displays or saves the averaged image.
    """

    accum_image = None
    count = 0

    for i in tqdm(range(box_size)):
        image_list_x = image_list[
            starting_x + ((starting_y + i) * array_width) : starting_x
            + ((starting_y + i) * array_width)
            + box_size
        ]

        for image in image_list_x:
            img = Image.open(image).convert("L")
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array.flatten()
            img_array[img_array == 255] = 0

            if accum_image is None:
                accum_image = np.zeros_like(img_array)

            accum_image += img_array
            count += 1

    averaged_image = accum_image / count  # type: ignore

    averaged_image = np.clip(averaged_image, 0, 255).astype(np.uint8)

    averaged_image = averaged_image.reshape(1024, 1024)
    if to_save:
        plt.imsave(
            f"{output_dir}/{starting_x}_{starting_y}_avg.png",
            averaged_image,
            cmap="gray",
        )
    else:
        plt.imshow(averaged_image, cmap="gray")
        plt.show()


def max_vals_image(
    box_size: int,
    image_list: list[str],
    # array_width: int,
    n_cols: int,
    n_rows: int,
    starting_x: int,
    starting_y: int,
    to_save: bool = False,
    output_dir: str = ".",
    smoothing: str = "sigmoid",
    gamma: float = 1.2,
):
    """
    Efficiently processes images and computes a max-value image across a box of images.

    Args:
        box_size (int): Size of the box to extract.
        image_list (list): Flattened list of image paths.
        array_width (int): Width of the 2D grid of images.
        starting_x (int): Starting x-coordinate.
        starting_y (int): Starting y-coordinate.
        to_save (bool): Whether to save the output image.
        output_dir (str): Directory to save the image if `to_save` is True.

    Returns:
        None. Displays or saves the averaged image.
    """
    max_vals_image = None

    # n_rows = len(image_list) // array_width
    # n_cols = array_width

    if starting_x < 0 or starting_y < 0 or starting_x > n_cols or starting_y > n_rows:
        raise ValueError("Starting x and y values bust be inside range of the grid")

    for i in range(box_size):
        # Dynamically compute the row indices, ensuring they stay in bounds
        row_idx = starting_y + i - box_size // 2

        if row_idx < 0:
            continue

        if row_idx >= n_rows:
            break  # Stop processing if we exceed the grid height

        col_start = max(starting_x - (box_size // 2), 0)
        col_end = min(starting_x + (box_size // 2), n_cols)

        row_start = row_idx * n_cols + col_start
        row_end = row_idx * n_cols + col_end
        image_list_x = image_list[row_start : row_end + 1]

        images = [np.array(Image.open(image).convert("L")) for image in image_list_x]
        images = np.stack(images, axis=0)  # 3D array (n_images, height, width)

        if max_vals_image is None:
            max_vals_image = images.max(axis=0)  # type: ignore
        else:
            max_vals_image = np.maximum(max_vals_image, images.max(axis=0))  # type: ignore

    # Normalize the resulting image to [0, 1]
    normalized_image = (max_vals_image - max_vals_image.min()) / (  # type: ignore
        max_vals_image.max() - max_vals_image.min()  # type: ignore
    )

    if smoothing == "gamma":
        gamma_corrected_image = normalized_image**gamma
        final_image = gamma_corrected_image
    elif smoothing == "sigmoid":
        final_image = 1 / (1 + np.exp(-1 * (normalized_image - 0.5)))
    else:
        final_image = normalized_image

    final_image = np.array(final_image)
    if to_save:
        # check if directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.imsave(
            f"{output_dir}/{starting_y}_{starting_x}_boxsize{box_size}.png",
            final_image,
            cmap="gray",
        )
    else:
        plt.imshow(final_image, cmap="gray")
        plt.show()


def convert_all_to_boxes(
    stem_image_dir, output_dir, box_size, n_cols, n_rows, **kwargs
):
    """
    Convert all images to boxes of size box_size x box_size and save them to the output_dir.
    Automatically detects array dimensions if not provided.

    Args:
        stem_image_dir (str): Directory containing the original PNG images.
        output_dir (str): Path to the directory to save the images.
        box_size (int): Size of the box to extract.
        array_length (int, optional): Length of the 2D grid of images. If None, will auto-detect.
        **kwargs: Optional keyword arguments:
            - smoothing (str): Type of smoothing to apply. Default is "gamma".
            - gamma (float): Gamma correction value. Default is 1.2.

    Returns:
        None. Saves the images to the output_dir.
    """
    image_list = []

    for root, _, files in os.walk(stem_image_dir):
        for file in files:
            if file.endswith(".png"):
                image_list.append(os.path.join(root, file))

    if not image_list:
        raise ValueError(f"No PNG files found in directory: {stem_image_dir}")

    sorted_image_list, _ = logical_sort(image_list)

    print(f"Processing {len(sorted_image_list)} images with box_size={box_size}")

    for i in tqdm(range(len(sorted_image_list)), desc="Creating boxed images"):
        row = i // n_cols
        col = i % n_cols
        max_vals_image(
            box_size,
            sorted_image_list,
            n_cols,
            n_rows,
            col,
            row,
            to_save=True,
            output_dir=output_dir,
            smoothing=kwargs.get("smoothing", "gamma"),
            gamma=kwargs.get("gamma", 1.2),
        )
