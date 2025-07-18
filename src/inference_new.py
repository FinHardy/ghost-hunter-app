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
    Also returns extracted coordinates for inferring array dimensions.
    """

    def logical_sort_key(filepath):
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        parts[0], parts[1] = parts[1], parts[0]
        coords = parts[0:2]
        return tuple(map(int, coords))

    sorted_files = sorted(file_list, key=logical_sort_key)
    print(len(sorted_files), "files found")
    coordinates = [logical_sort_key(f) for f in sorted_files]
    return sorted_files, coordinates


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
):
    image_list = []
    ex_description = _config.wandb.ex_description

    for root, _, files in os.walk(stem_image_dir):
        for file in files:
            if file.endswith(".png"):
                image_list.append(os.path.join(root, file))

    image_list, _ = logical_sort_coordinates(image_list)

    assert len(image_list) > 0, "No images found in the directory"

    dataloader = ImageDataset(image_list, _config.accelerator)

    output_list = []

    model.eval()
    with torch.no_grad():
        for image in tqdm(dataloader):
            image.to(_config.accelerator)
            output = model(image)
            output = torch.nn.functional.softmax(output, dim=1)
            output_list.append(output.cpu())

    all_outputs = np.array(output_list)

    if _config.task == "binary":
        out_image = all_outputs.reshape((dim2, dim1, 3))
    elif _config.task == "polar":
        out_image = all_outputs.reshape((dim2, dim1))

    out_image = (out_image / out_image.max() * 255).astype(np.uint8)
    out_image = np.flip(out_image, axis=0)
    out_image = np.rot90(out_image, k=3)

    ex_description = ex_description.replace(" ", "_").replace(":", "_")
    os.makedirs(os.path.join(ABS_PATH, "../", save_path), exist_ok=True)

    if save_name is not None:
        plot_save_path = os.path.join(
            ABS_PATH,
            "../",
            save_path,
            f"{save_name}_{_config.model.model_name}_{ex_description}.png",
        )
    else:
        plot_save_path = os.path.join(
            ABS_PATH,
            "../",
            save_path,
            f"{_config.model.model_name}_{ex_description}.png",
        )

    plt.imsave(plot_save_path, out_image)
    print(f"Plot saved to {plot_save_path}")


def plot_embeddings(config_file: str, save_path: str = "./images/"):
    """
    Extracts the embeddings from the network and plots them to a figure.
    Automatically infers dimensions based on filenames.
    """
    _config = Config.from_yaml(os.path.join(ABS_PATH, "../", "configs", config_file))

    model_class = _config.model.model_name
    if model_class:
        model = models[model_class](_config)
    else:
        raise ValueError(f"Model '{_config.model.model_name}' is not recognized.")

    model.to(_config.accelerator)

    route_to_checkpoints = os.path.join(
        ABS_PATH, "../checkpoints", _config.test.load_path
    )
    best_checkpoint = str(os.listdir(route_to_checkpoints)[-1])
    print(f"Best checkpoint: {best_checkpoint}")

    test_load_path = os.path.join(route_to_checkpoints, best_checkpoint)
    load_state_dict(model, test_load_path, _config.accelerator)

    stem_image_dir = os.path.join(ABS_PATH, "../", _config.data.data_dir)
    assert os.path.exists(stem_image_dir), "Data directory does not exist"

    print(
        "Checking if multiple diffraction files exist (so we can do multiple in a row)"
    )
    for _, dirs, _ in os.walk(stem_image_dir):
        if len(dirs) > 0:
            print("Multiple directories found, so doing each one in turn")
            for dir in dirs:
                dir_path = os.path.join(stem_image_dir, dir)
                image_files = [
                    os.path.join(root, file)
                    for root, _, files in os.walk(dir_path)
                    for file in files
                    if file.endswith(".png")
                ]
                image_files, coords = logical_sort_coordinates(image_files)
                xs, ys = zip(*coords)
                dim1 = max(xs) + 1
                dim2 = max(ys) + 1

                print(f"Image dimensions inferred: {dim1} x {dim2}")

                output_polarisation_image(
                    _config,
                    model,
                    dir_path,
                    os.path.join("images", dir),
                    dim1=dim1,
                    dim2=dim2,
                    save_name=dir,
                )
        else:
            print("No directories found, so just doing one image")
            image_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(stem_image_dir)
                for file in files
                if file.endswith(".png")
            ]
            image_files, coords = logical_sort_coordinates(image_files)
            xs, ys = zip(*coords)
            dim1 = max(xs) + 1
            dim2 = max(ys) + 1

            print(f"Image dimensions inferred: {dim1} x {dim2}")

            output_polarisation_image(
                _config,
                model,
                stem_image_dir,
                "images/",
                dim1=dim1,
                dim2=dim2,
            )


if __name__ == "__main__":
    Fire(plot_embeddings)
