"""
Summary: This module contains the RawPngLoader class,
    which is responsible for loading the raw PNG dataset.

Returns:
    DataLoader: DataLoader for the training/val/test set
"""

import os

import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import Config


class RawPngDataset(Dataset):
    """
    A PyTorch Dataset class for loading

    Args:
        files (list[str]): List of file paths
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(
        self,
        files: list[str],
        device: str,
        labels: list[dict[str, str]],
        train_dir: str,
        task: str,
        transform=None,
    ):
        self.files = files
        self.transform = transform
        self.device = device
        self.train_dir = train_dir
        self.task = task

        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        self.labels = labels

        self.label_map = {
            "na": 0,
            "vertical": 1,
            "horizontal": 2,
        }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        entry: dict[str, str] = self.labels[idx]
        raw_img_path: str = entry["file"]
        label: str = entry["label"]

        label_out: int = self.label_map[label]

        # Use train_dir and raw_img_path directly
        img_path = os.path.join(self.train_dir, raw_img_path)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("L")
        else:
            raise FileNotFoundError(f"File {img_path} does not exist.")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32)

        return image_tensor, label_out


class RawPngLoader(LightningDataModule):
    """
    Description:
    - A PyTorch Lightning DataModule class for loading the raw PNG dataset.

    Args:
    - _config (Config): Configuration object
    """

    def __init__(self, _config: Config):
        super().__init__()

        # abs_path is the path to the root directory of the repo
        self.abs_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "../../"
        )
        self.train_dir: str = _config.data.train_dir
        self.train_ratio: float = _config.data.train_ratio
        self.test_ratio: float = _config.data.test_ratio
        self.val_ratio: float = _config.data.val_ratio
        self.batch_size: int = _config.model.batch_size
        self.data_loader: str = _config.data.data_loader
        self.num_workers: int = _config.data.num_workers
        self.seed: int = _config.seed
        self.device: str = _config.accelerator
        self.labels_file: str = _config.data.labels_file
        self.task: str = _config.task

        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                # Add any data augmentation transforms if necessary
            ]
        )

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((256, 256))]
        )

        self.labels: list[dict[str, str]] = []

        # Use absolute path if provided, else fall back to labelling subdirectory
        if os.path.isabs(self.labels_file):
            labels_file = self.labels_file
        else:
            labels_file = os.path.join(self.abs_path, "labelling", self.labels_file)

        if os.path.exists(labels_file) and labels_file.endswith(".yaml"):
            with open(labels_file) as f:
                data = yaml.safe_load(f)
                self.labels = data["labels"]
        else:
            raise FileNotFoundError(
                f"Labels file {labels_file} does not exist or is not a yaml file."
            )

        self.files = [entry["file"] for entry in self.labels]

        # File placeholders
        self.train_files: list[str] = []
        self.val_files: list[str] = []
        self.test_files: list[str] = []

        self.train_dataset: RawPngDataset
        self.test_dataset: RawPngDataset
        self.val_dataset: RawPngDataset

    def prepare_data(self, print_lens: bool = True) -> None:
        """
        Description:
        - Prepare the dataset by splitting it into train, val, and test sets.

        Args:
        - print_lens (bool): Whether to print the lengths of the datasets

        Returns:
        - None
        """
        # Use train_dir and file directly, not abs_path
        files = [os.path.join(self.train_dir, f) for f in self.files]

        data_train, data_val_test = [], []

        aug_dataset = False

        # check if the dataset has been augmented
        for i in range(len(files)):
            if "rot" in str(files[i]) or "stride" in str(files[i]):
                aug_dataset = True

        if aug_dataset:
            # need to make sure that the augmented images are not included in the val/test sets
            for i in range(len(files)):
                if not os.path.exists(files[i]):
                    raise FileNotFoundError(f"File {files[i]} does not exist.")

                if "rot" not in str(files[i]) and "stride" not in str(files[i]):
                    data_val_test.append(files[i])
                else:
                    data_train.append(files[i])

            # have to change test and val ratio to ratios of each other, since we are only using a subset of the data
            test_ratio = self.test_ratio / (self.val_ratio + self.test_ratio)
            val_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)

            data_val, data_test = train_test_split(
                data_val_test,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=self.seed,
            )
        else:  # if the dataset has not been augmented
            data_train, data_val_test = train_test_split(
                files,
                test_size=self.val_ratio + self.test_ratio,
                random_state=self.seed,
            )

            data_val, data_test = train_test_split(
                data_val_test,
                test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
                random_state=self.seed,
            )

        # Save split paths for later use
        self.train_files = data_train
        self.val_files = data_val
        self.test_files = data_test

        if print_lens:
            print(
                f"Total data: {len(files)},\n"
                f"Train data: {len(data_train)},\n"
                f"Val data: {len(data_val)},\n"
                f"Test data: {len(data_test)}"
            )

    def setup(self, stage: str = "") -> None:
        """
        Description:
        - Setup datasets for the specific stage (fit, test).

        Args:
        - stage (str): Stage of the data pipeline

        Returns:
        - None
        """
        if stage in ("fit", ""):
            if not self.train_files or not self.val_files:
                self.prepare_data()
            self.train_dataset = RawPngDataset(
                self.train_files,
                device=self.device,
                transform=self.train_transform,
                labels=self.labels,
                train_dir=self.train_dir,
                task=self.task,
            )
            self.val_dataset = RawPngDataset(
                self.val_files,
                device=self.device,
                transform=self.test_transform,
                labels=self.labels,
                train_dir=self.train_dir,
                task=self.task,
            )
        if stage in ("test", ""):
            if not self.test_files:
                self.prepare_data()
            self.test_dataset = RawPngDataset(
                self.test_files,
                device=self.device,
                transform=self.test_transform,
                labels=self.labels,
                train_dir=self.train_dir,
                task=self.task,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Description: Return DataLoader for the training set.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Description: Return DataLoader for the validation set.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Description: Return DataLoader for the test set.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers,
        )
