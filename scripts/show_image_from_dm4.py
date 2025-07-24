import os
import random
from typing import Optional

import matplotlib.pyplot as plt
import py4DSTEM
import py4DSTEM.preprocess.preprocess as pre


def show_random_diffraction_pattern(
    dm4_file: str,
    binning_param: int = 1,
    save_image: bool = False,
    output_path: Optional[str] = None,
) -> None:
    """
    Display a random diffraction pattern from a py4DSTEM dm4 file.

    Args:
        dm4_file: Path to the dm4 file
        binning_param: Binning parameter for data reduction
        save_image: Whether to save the displayed image
        output_path: Path to save the image (if save_image is True)
    """
    assert ".dm4" in dm4_file, "File must be a dm4 file"
    assert os.path.isfile(dm4_file), f"File does not exist: {dm4_file}"

    try:
        # Load the 4D-STEM dataset
        print(f"Loading dm4 file: {dm4_file}")
        dataset = py4DSTEM.import_file(dm4_file)

        # Apply binning if specified
        if binning_param > 1:
            print(f"Applying binning with parameter: {binning_param}")
            pre.bin_data_diffraction(dataset, binning_param)

        # Get the shape of the dataset (scan positions)
        shape = dataset.data.shape  # type: ignore
        print(f"Dataset shape (scan positions): {shape[0]} x {shape[1]}")
        print(
            f"Diffraction pattern shape: {dataset.data.shape[2]} x {dataset.data.shape[3]}"
        )  # type: ignore

        # Select random scan position
        random_i = random.randint(0, shape[0] - 1)
        random_j = random.randint(0, shape[1] - 1)
        print(f"Selected random scan position: ({random_i}, {random_j})")

        # Extract the diffraction pattern at the random position
        diffraction_pattern = dataset[random_i, random_j].data  # type: ignore

        # Display the diffraction pattern
        plt.figure(figsize=(8, 8))
        plt.imshow(diffraction_pattern, cmap="gray")
        plt.title(f"Diffraction Pattern at position ({random_i}, {random_j})")
        plt.colorbar(label="Intensity")
        plt.xlabel("Detector X (pixels)")
        plt.ylabel("Detector Y (pixels)")

        # Save the image if requested
        if save_image:
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(dm4_file))[0]
                output_path = (
                    f"{base_name}_random_diffraction_{random_i}_{random_j}.png"
                )

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Image saved to: {output_path}")

        plt.show()

    except Exception as e:
        print(f"Error loading or processing dm4 file: {e}")
        print("This might not be a 4D-STEM dataset or the file might be corrupted.")
        # Try a fallback approach without advanced features
        try:
            print("Trying fallback approach without advanced features...")
            dataset = py4DSTEM.import_file(dm4_file)
            shape = dataset.data.shape  # type: ignore
            random_i = random.randint(0, shape[0] - 1)
            random_j = random.randint(0, shape[1] - 1)
            diffraction_pattern = dataset[random_i, random_j].data  # type: ignore

            plt.figure(figsize=(8, 8))
            plt.imshow(diffraction_pattern, cmap="gray")
            plt.title(
                f"Diffraction Pattern at position ({random_i}, {random_j}) - Fallback"
            )
            plt.colorbar(label="Intensity")
            plt.show()
        except Exception as fallback_e:
            print(f"Fallback also failed: {fallback_e}")


def show_virtual_image(dm4_file: str, binning_param: int = 1) -> None:
    """
    Display the virtual image (summed intensity across all scan positions) from a py4DSTEM dm4 file.

    Args:
        dm4_file: Path to the dm4 file
        binning_param: Binning parameter for data reduction
    """
    assert ".dm4" in dm4_file, "File must be a dm4 file"
    assert os.path.isfile(dm4_file), f"File does not exist: {dm4_file}"

    try:
        print(f"Loading dm4 file: {dm4_file}")
        dataset = py4DSTEM.import_file(dm4_file)

        if binning_param > 1:
            print(f"Applying binning with parameter: {binning_param}")
            pre.bin_data_diffraction(dataset, binning_param)

        # Virtual image: sum intensity at each scan position
        # shape: (scan_x, scan_y, det_x, det_y)
        # Sum over detector axes to get a 2D image of scan positions
        virtual_image = dataset.data.sum(axis=(2, 3))  # type: ignore

        plt.figure(figsize=(8, 8))
        plt.imshow(virtual_image, cmap="gray")
        plt.title("Virtual Image (summed intensity across scan positions)")
        plt.colorbar(label="Intensity")
        plt.xlabel("Scan X")
        plt.ylabel("Scan Y")
        plt.show()

    except Exception as e:
        print(f"Error loading or processing dm4 file: {e}")


def show_dataset_info(dm4_file: str) -> None:
    """
    Display basic information about the 4D-STEM dataset.

    Args:
        dm4_file: Path to the dm4 file
    """
    assert ".dm4" in dm4_file, "File must be a dm4 file"
    assert os.path.isfile(dm4_file), f"File does not exist: {dm4_file}"

    try:
        print(f"Loading dm4 file for info: {dm4_file}")
        dataset = py4DSTEM.import_file(dm4_file)

        shape = dataset.data.shape  # type: ignore
        print("Dataset information:")
        print(f"  - Scan dimensions: {shape[0]} x {shape[1]} positions")
        print(f"  - Diffraction pattern size: {shape[2]} x {shape[3]} pixels")
        print(f"  - Total diffraction patterns: {shape[0] * shape[1]}")
        print(f"  - Data type: {dataset.data.dtype}")  # type: ignore
        print(f"  - Memory size: ~{dataset.data.nbytes / (1024**3):.2f} GB")  # type: ignore

    except Exception as e:
        print(f"Error loading dm4 file: {e}")


if __name__ == "__main__":
    # Example usage
    ABS_PATH = os.path.abspath(os.path.dirname(__file__))

    # Example dm4 file path (adjust this to your actual file)
    dm4_file_path = os.path.join(
        ABS_PATH,
        "../../data/mariana_boracite-2025-06-25/InSitu10/SI_Diffraction.dm4",
    )

    # Check if the example file exists
    if os.path.exists(dm4_file_path):
        print("=== Dataset Information ===")
        show_dataset_info(dm4_file_path)

        print("\n=== Showing random diffraction pattern ===")
        show_random_diffraction_pattern(
            dm4_file=dm4_file_path,
            binning_param=1,
            save_image=False,
        )

        print("\n=== Showing virtual image ===")
        show_virtual_image(
            dm4_file=dm4_file_path,
            binning_param=1,
        )
    else:
        print(f"Example dm4 file not found at: {dm4_file_path}")
        print("Please update the dm4_file_path variable to point to your dm4 file.")
        print("\nExample usage:")
        print("show_random_diffraction_pattern('path/to/your/file.dm4')")
        print("show_virtual_image('path/to/your/file.dm4')")
        print("show_dataset_info('path/to/your/file.dm4')")
