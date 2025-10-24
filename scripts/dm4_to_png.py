import os
from typing import Optional

import matplotlib.pyplot as plt
import py4DSTEM
import py4DSTEM.preprocess.preprocess as pre
from tqdm import tqdm


def export_dm4_bf_images_to_png(
    dm4_file,
    output_file_path,
    crop_values: Optional[tuple[int, int, int, int]],  # x_min, x_max, y_min, y_max
    crop: bool = False,
    binning_param: int = 1,
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
                # TODO: i think this is a bug and the i and j are the wrong way around which is what
                # causes the images to be flipped weirdly at inference time
                filename = os.path.join(output_dir, f"{base_file_name}_{row}_{col}.png")
                plt.imsave(filename, diffraction_pattern, cmap="gray")

    except Exception as e:
        # if not 4d STEM image, will get caught here e.g. dm4 files that only show ADF images
        print(e)
