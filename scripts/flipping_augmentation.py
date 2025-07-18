import os

import numpy as np
import tqdm as tqdm
import yaml  # type: ignore
from PIL import Image

ABS_PATH = os.path.abspath(os.path.dirname(__file__))


def flip_all_images(labels, train_dir, data_dir, labels_file):
    for data in tqdm(labels["labels"]):
        file = os.path.join(data_dir, data["file"])
        label = data["label"]

        assert label in ["vertical", "horizontal", "na"], f"Label {label} is not valid."

        if label == "vertical":
            img90_label = "horizontal"
            img180_label = "vertical"
            img270_label = "horizontal"
        elif label == "horizontal":
            img90_label = "vertical"
            img180_label = "horizontal"
            img270_label = "vertical"
        elif label == "na":
            img90_label = "na"
            img180_label = "na"
            img270_label = "na"

        if not os.path.exists(file):
            print(f"File {file} does not exist.")
            continue

        img = Image.open(file)
        img = np.array(img)  # type: ignore
        # file image by 90 degrees

        if not os.path.exists(os.path.join(train_dir, data["file"])):
            Image.fromarray(img).save(os.path.join(train_dir, data["file"]))

        img90 = np.rot90(img)  # type: ignore
        img90_file = os.path.join(train_dir, data["file"].replace(".png", "_90rot.png"))
        if os.path.exists(img90_file):
            Image.fromarray(img90).save(img90_file)
        new_labels.append({"file": os.path.basename(img90_file), "label": img90_label})

        # 180-degree rotation
        img180 = np.rot90(img90)
        img180_file = os.path.join(
            train_dir, data["file"].replace(".png", "_180rot.png")
        )
        if os.path.exists(img180_file):
            Image.fromarray(img180).save(img180_file)
        new_labels.append(
            {"file": os.path.basename(img180_file), "label": img180_label}
        )

        # 270-degree rotation
        img270 = np.rot90(img180)
        img270_file = os.path.join(
            train_dir, data["file"].replace(".png", "_270rot.png")
        )
        if os.path.exists(img270_file):
            Image.fromarray(img270).save(img270_file)
        new_labels.append(
            {"file": os.path.basename(img270_file), "label": img270_label}
        )

        print(f"Rotated images for {data['file']} saved.")

        with open(labels_file, "w") as stream:
            yaml.dump({"labels": new_labels}, stream)


if __name__ == "__main__":
    labels_file = "../labelling/labels.yaml"
    with open(labels_file, "r") as stream:
        try:
            labels = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_dir = os.path.join(ABS_PATH, "../../data/dm4/Diffraction_SI_averaged_images")
    train_dir = os.path.join(
        ABS_PATH, "../../data/dm4/Diffraction_SI_averaged_images_train"
    )
    # check if data dir exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    new_labels = labels.get("labels", [])
    flip_all_images(labels, train_dir, data_dir, labels_file)
