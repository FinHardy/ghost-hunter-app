import matplotlib.pyplot as plt
import torch.nn as nn

from src.config import Config
from src.models.basemodel import BaseModel


class ThreeLayerCnn(BaseModel):
    def __init__(self, _config: Config):
        super(ThreeLayerCnn, self).__init__(_config)

        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)

        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        if self.task == "polar":
            self.fc2 = nn.Linear(128, 1)
        elif self.task == "binary":
            self.fc2 = nn.Linear(128, 3)

        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # B, 256, 256 -> B, 256, 256, 16
        x = self.max_pool2d(x)  # B, 256, 256, 16 -> B, 128, 128, 16
        x = self.relu(self.conv2(x))  # B, 128, 128, 16 -> B, 128, 128, 32
        x = self.max_pool2d(x)  # B, 128, 128, 32 -> B, 64, 64, 32
        x = self.relu(self.conv3(x))  # B, 64, 64, 32 -> B, 64, 64, 64
        x = self.max_pool2d(x)  # B, 64, 64, 64 -> B, 32, 32, 64
        x = x.view(x.size(0), -1)  # B, 32, 32, 64 -> B, 32 * 32 * 64
        x = self.relu(self.fc1(x))  # B, 32 * 32 * 64 -> B, 128
        x = self.fc2(x)  # B, 128 -> B, 3
        return x

    def visualise_feature_maps(self, x, layer_idx):
        if layer_idx == 1:
            x = self.relu(self.conv1(x))
        elif layer_idx == 2:
            x = self.relu(self.conv1(x))
            x = self.max_pool2d(x)
            x = self.relu(self.conv2(x))
        elif layer_idx == 3:
            x = self.relu(self.conv1(x))
            x = self.max_pool2d(x)
            x = self.relu(self.conv2(x))
            x = self.max_pool2d(x)
            x = self.relu(self.conv3(x))
        else:
            raise ValueError("Layer index must be between 1 and 3")

        feature_maps = x.detach().cpu().numpy()
        print(feature_maps.shape)

        num_filters = feature_maps.shape[0]
        print(f"Num filters: {num_filters}")
        fig, axes = plt.subplots(
            1, min(num_filters, 8), figsize=(20, 10)
        )  # Display up to 8 filters
        for i, ax in enumerate(axes):
            if i >= num_filters:
                break
            ax.imshow(
                feature_maps[i], cmap="viridis"
            )  # Display the first example in the batch
            ax.axis("off")
            ax.set_title(f"Filter {i + 1}")
        plt.show()

    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)
