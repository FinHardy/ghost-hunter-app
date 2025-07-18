import os
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def save_h5_diffraction_to_png(
    h5_file: str,
    output_file_path: str,
    dataset_key: str = "frame",
    crop_values: Optional[
        tuple[int, int, int, int]
    ] = None,  # x_min, x_max, y_min, y_max
    crop: bool = False,
    binning_param: int = 1,
):
    """
    Save individual 2D diffraction patterns from a 4D STEM HDF5 file to PNG.

    Output directory will be created in the same path as the input file:
    Example:
    h5_file/
    - example.h5
    - example_png_bin1/
        - example_0_0.png
        - example_0_1.png
        - ...

    Args:
    - h5_file: Path to HDF5 file
    - output_file_path: Base output path (same as input filename typically)
    - dataset_key: Dataset name inside HDF5 (default is "slice")
    - crop_values: (x_min, x_max, y_min, y_max) crop window
    - crop: Enable cropping
    - binning_param: Integer factor for downsampling
    """
    assert h5_file.endswith(".h5"), "Input file must be an .h5 file"
    # if the file does not exist fail quietly
    if not os.path.isfile(h5_file):
        print(f"File does not exist: {h5_file}")
        return

    if crop:
        output_dir = os.path.join(
            output_file_path + "_png_cropped_bin" + str(binning_param)
        )
    else:
        output_dir = os.path.join(output_file_path + "_png_bin" + str(binning_param))

    os.makedirs(output_dir, exist_ok=True)
    base_file_name = os.path.splitext(os.path.basename(h5_file))[0]

    try:
        with h5py.File(h5_file, "r") as f:
            data = f[dataset_key]  # shape: (scan_y, scan_x, height, width)
            scan_y, scan_x = data.shape[0], data.shape[1]

            for i in tqdm(range(scan_y), desc="Saving PNGs"):
                for j in range(scan_x):
                    pattern = data[i, j]

                    # Optional cropping
                    if crop and crop_values is not None:
                        x_min, x_max, y_min, y_max = crop_values
                        pattern = pattern[y_min:y_max, x_min:x_max]

                    # Optional binning
                    if binning_param > 1:
                        pattern = pattern.reshape(
                            pattern.shape[0] // binning_param,
                            binning_param,
                            pattern.shape[1] // binning_param,
                            binning_param,
                        ).mean(axis=(1, 3))

                    filename = os.path.join(output_dir, f"{j}_{i}_{base_file_name}.png")
                    plt.imsave(filename, pattern, cmap="gray", format="png")

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
