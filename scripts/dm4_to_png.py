import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import py4DSTEM
import py4DSTEM.preprocess.preprocess as pre
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
    enhanced = image.copy()

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


def export_dm4_bf_images_to_png(
    dm4_file,
    output_file_path,
    crop_values: Optional[tuple[int, int, int, int]],  # x_min, x_max, y_min, y_max
    crop: bool = False,
    binning_param: int = 1,
    use_log_scale: bool = False,
    use_histogram_eq: bool = False,
    gamma: Optional[float] = None,
):
    """
    Description:
    - Save the Bright Field images from the dm4 files to png format


    Saves files in a directory in the same location as the dm4 file(s)
    e.g.
    dm4/
    - dm4_file.dm4
    - dm4_file/
        - dm4_file_0_0.png
        - dm4_file_0_1.png
        - dm4_file_1_0.png
        - dm4_file_1_1.png
        - ...

    Args:
    - files: List of dm4 files (do not require absolute path)

    """
    assert ".dm4" in dm4_file, "File must be a dm4 file"
    # Check if the file exists
    assert os.path.isfile(dm4_file), "File does not exist"

    output_dir = os.path.join(output_file_path)

    os.makedirs(output_dir, exist_ok=True)
    base_file_name = os.path.splitext(os.path.basename(dm4_file))[0]

    try:
        image = py4DSTEM.import_file(dm4_file)
        if binning_param > 1:
            pre.bin_data_diffraction(image, binning_param)
        shape = image.data.shape  # type: ignore
        # TODO: put *args in the function signature
        if crop:
            image = image.crop_Q(crop_values)  # type: ignore
        for row in tqdm(range(shape[0])):
            for col in range(shape[1]):
                diffraction_pattern = image[row, col].data  # type: ignore

                # Apply image enhancements
                diffraction_pattern = apply_image_enhancements(
                    diffraction_pattern,
                    use_log_scale=use_log_scale,
                    use_histogram_eq=use_histogram_eq,
                    gamma=gamma,
                )

                # TODO: i think this is a bug and the i and j are the wrong way around which is what
                # causes the images to be flipped weirdly at inference time
                filename = os.path.join(output_dir, f"{base_file_name}_{row}_{col}.png")
                plt.imsave(filename, diffraction_pattern, cmap="gray")

    except Exception as e:
        # if not 4d STEM image, will get caught here e.g. dm4 files that only show ADF images
        print(e)
