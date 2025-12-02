import os
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def apply_image_enhancements(
    image: np.ndarray,
    use_log_scale: bool = False,
    use_histogram_eq: bool = False,
    gamma: Optional[float] = None,
) -> np.ndarray:
    """
    Apply various image enhancement techniques to improve visibility.

    Args:
        image: Input image array
        use_log_scale: Apply logarithmic scaling
        use_histogram_eq: Apply histogram equalization
        gamma: Gamma correction value (None for no correction)

    Returns:
        Enhanced image array
    """
    # Ensure we have a numpy array (not memoryview)
    enhanced = np.asarray(image).copy()

    # Apply log scaling (helps with ghost disks)
    if use_log_scale:
        # Add small constant to avoid log(0)
        enhanced = np.log1p(enhanced - enhanced.min() + 1)

    # Normalize to 0-1 range
    if enhanced.max() > enhanced.min():
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

    # Apply gamma correction
    if gamma is not None and gamma != 1.0:
        enhanced = np.power(enhanced, gamma)

    # Apply histogram equalization
    if use_histogram_eq:
        # Simple histogram equalization
        hist, bins = np.histogram(enhanced.flatten(), bins=256, range=[0, 1])
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # Normalize
        enhanced = np.interp(enhanced.flatten(), bins[:-1], cdf).reshape(enhanced.shape)

    return enhanced


def save_h5_diffraction_to_png(
    h5_file: str,
    output_file_path: str,
    dataset_key: str = "frame",
    crop_values: Optional[
        tuple[int, int, int, int]
    ] = None,  # x_min, x_max, y_min, y_max
    crop: bool = False,
    binning_param: int = 1,
    use_log_scale: bool = False,
    use_histogram_eq: bool = False,
    gamma: Optional[float] = None,
):
    """
    Save individual 2D diffraction patterns from a 4D STEM HDF5 file to PNG.

    Output directory will be created at the output_file_path:
    Example:
    output_file_path/
        - basename_0_0.png
        - basename_0_1.png
        - basename_1_0.png
        - basename_1_1.png
        - ...

    Args:
    - h5_file: Path to HDF5 file
    - output_file_path: Output directory path
    - dataset_key: Dataset name inside HDF5 (default is "frame")
    - crop_values: (x_min, x_max, y_min, y_max) crop window
    - crop: Enable cropping
    - binning_param: Integer factor for downsampling
    """
    assert h5_file.endswith(".h5") or h5_file.endswith(
        ".hdf5"
    ), "Input file must be an HDF5 file (.h5 or .hdf5)"
    # if the file does not exist fail quietly
    if not os.path.isfile(h5_file):
        print(f"File does not exist: {h5_file}")
        return

    # Use output_file_path directly as the output directory
    output_dir = output_file_path
    os.makedirs(output_dir, exist_ok=True)
    base_file_name = os.path.splitext(os.path.basename(h5_file))[0]

    try:
        with h5py.File(h5_file) as f:
            data = f[dataset_key]  # shape: (scan_y, scan_x, height, width)
            scan_y, scan_x = data.shape[0], data.shape[1]  # type: ignore

            for i in tqdm(range(scan_y), desc="Saving PNGs"):
                for j in range(scan_x):
                    # Convert to numpy array to avoid memoryview issues
                    pattern = np.asarray(data[i, j])  # type: ignore

                    # Optional cropping
                    if crop and crop_values is not None:
                        x_min, x_max, y_min, y_max = crop_values
                        pattern = pattern[y_min:y_max, x_min:x_max]  # type: ignore

                    # Optional binning
                    if binning_param > 1:
                        pattern = pattern.reshape(  # type: ignore
                            pattern.shape[0] // binning_param,  # type: ignore
                            binning_param,
                            pattern.shape[1] // binning_param,  # type: ignore
                            binning_param,
                        ).mean(axis=(1, 3))

                    # Apply image enhancements
                    pattern = apply_image_enhancements(
                        pattern,  # type: ignore
                        use_log_scale=use_log_scale,
                        use_histogram_eq=use_histogram_eq,
                        gamma=gamma,
                    )

                    # Match DM4 filename format: basename_row_col.png
                    filename = os.path.join(output_dir, f"{base_file_name}_{i}_{j}.png")
                    plt.imsave(filename, pattern, cmap="gray", format="png")  # type: ignore

    except Exception as e:
        print(f"Error processing file {h5_file}: {e}")


if __name__ == "__main__":
    ABS_PATH = os.path.abspath(os.path.dirname(__file__))
    crop_values = (125, 275, 200, 350)
    for i in range(1, 14):
        for N in range(10):
            save_h5_diffraction_to_png(
                h5_file=os.path.join(
                    ABS_PATH,
                    f"../data/BTO/InSitu_({i})/STEM SI_Diffraction SI_frames_h5/frame_{N}.h5",
                ),
                output_file_path=os.path.join(
                    ABS_PATH, f"../data/BTO/InSitu_({i})/png/frame_{N}"
                ),
                crop=True,
                binning_param=1,
                crop_values=crop_values,
            )
