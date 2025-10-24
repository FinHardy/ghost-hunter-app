import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from fire import Fire
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.config import Config
from src.models import ResidualCNN, ThreeLayerCnn
from src.utils import load_state_dict

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("medium")

models = {
    "ThreeLayerCnn": ThreeLayerCnn,
    "ResidualCNN": ResidualCNN,
}

ABS_PATH = os.path.abspath(os.path.dirname(__file__))


def logical_sort_coordinates(file_list):
    """
    Sorts a list of file paths logically based on the numeric parts of the filenames.
    To account for my sins of my dumb filenaming system I have had to create this ghastly function to sort the files as would be expected

    Args:
        file_list (list of str): List of file paths to sort.

    Returns:
        list of str: Sorted list of file paths.
    """

    def logical_sort_key(filepath):
        filename = os.path.basename(filepath)
        # Extract all numeric parts from the filename (e.g., _0_15 -> [0, 15])
        # EXCEPT boxsize_5 section (for example)
        coordinates_plus_boxsize = filename.split("_")
        # make 0 index position at 1 index and 1 index at zero position
        coordinates_plus_boxsize[0], coordinates_plus_boxsize[1] = (
            coordinates_plus_boxsize[1],
            coordinates_plus_boxsize[0],
        )
        numbers = coordinates_plus_boxsize[0:2]
        # Convert all extracted numbers to integers and return as a tuple
        return tuple(map(int, numbers))

    return sorted(file_list, key=logical_sort_key)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, accelerator="cuda"):
        self.image_list = image_list
        self.device = accelerator

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert("L")
        image_array = np.array(image)
        image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0).float()
        transform = transforms.Compose([transforms.Resize((256, 256))])
        image_tensor = transform(image_tensor)
        return image_tensor.to(self.device)


def output_polarisation_image(
    _config: Config,
    model,
    stem_image_dir: str,
    save_path: str,
    dim1: int,
    dim2: int,
    save_name=None,
    with_softmax: bool = True,
) -> str:
    image_list: list = []
    ex_description = _config.wandb.ex_description

    for root, _, files in os.walk(stem_image_dir):
        for file in files:
            if file.endswith(".png"):
                image_list.append(os.path.join(root, file))

    image_list = logical_sort_coordinates(image_list)

    assert len(image_list) > 0, "No images found in the directory"

    dataloader = ImageDataset(image_list, _config.accelerator)

    output_list: list = []

    model.eval()

    if with_softmax:
        with torch.no_grad():
            for image in tqdm(dataloader):
                image.to(_config.accelerator)
                output = model(image)
                # apply softmax to output
                output = torch.nn.functional.softmax(output, dim=1)
                output_list.append(output.cpu())
    else:
        with torch.no_grad():
            for image in tqdm(dataloader):
                image.to(_config.accelerator)
                output = model(image)
                output_list.append(output.cpu())

    all_outputs = np.array(output_list)

    out_image = all_outputs.reshape((dim1, dim2, 3))

    # out_image = np.clip(out_image, 0, 1)
    out_image = (out_image / out_image.max() * 255).astype(np.uint8)  # For scaling

    # TODO: fix this vicsious hack that gets around weird bug where it ends up flipped and rotated the wrong way
    out_image = np.flip(out_image, axis=0)
    out_image = np.rot90(out_image, k=3)

    # make sure no spaces in ex_description and compatible with file save name
    ex_description = ex_description.replace(" ", "_")
    ex_description = ex_description.replace(":", "_")

    os.makedirs(os.path.join(ABS_PATH, "../", save_path), exist_ok=True)

    if save_name is not None:
        plot_save_path = os.path.join(
            ABS_PATH,
            "../",
            save_path,
            f"{save_name}_{_config.model.model_name}_{ex_description}.png",
        )
    else:  # if no dir is provided, just save to the save_path
        plot_save_path = os.path.join(
            ABS_PATH,
            "../",
            save_path,
            f"{_config.model.model_name}_{ex_description}.png",
        )

    plt.imsave(
        plot_save_path,
        out_image,
    )

    print(f"Plot saved to {plot_save_path}")

    return plot_save_path


def plot_embeddings(
    config_file: str,
    dim1: int,
    dim2: int,
    save_path: str = "./images/",
    with_softmax: bool = True,
):
    """
    Description: Extracts the embeddings from the network and plots them to a figure

    IMPORTANT: Have to make sure that the embedding size of the model is the same
    as the embedding size in the config file

    Args:
    - data (torch.Tensor): Input data
    - batch_size (int): Batch size for processing
    - _config (Config): Configuration settings
    - save_path (str): Path to save the plot

    Returns:
    - torch.Tensor: Embeddings from the encoder
    """
    _config = Config.from_yaml(os.path.join(ABS_PATH, "../", "configs", config_file))

    model_class = _config.model.model_name
    if model_class:
        model = models[model_class](_config)
    else:
        raise ValueError(f"Model '{_config.model.model_name}' is not recognized.")

    model.to(_config.accelerator)

    # WARNING: this is just a hack to get the model to load the best weights within a file
    # to automate the inference process after training
    route_to_checkpoints = os.path.join(
        ABS_PATH, "../checkpoints", _config.test.load_path
    )

    print(route_to_checkpoints)

    best_checkpoint = str(os.listdir(route_to_checkpoints)[-1])
    print(f"Best checkpoint: {best_checkpoint}")

    test_load_path = os.path.join(route_to_checkpoints, best_checkpoint)

    load_state_dict(model, test_load_path, _config.accelerator)

    stem_image_dir = os.path.join(ABS_PATH, "../", _config.data.data_dir)
    assert os.path.exists(stem_image_dir), "Data directory does not exist"

    plot_path = output_polarisation_image(
        _config,
        model,
        stem_image_dir,
        save_path,
        dim1=dim1,
        dim2=dim2,
        with_softmax=with_softmax,
    )

    return plot_path


if __name__ == "__main__":
    Fire(plot_embeddings)
