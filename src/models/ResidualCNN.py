import matplotlib.pyplot as plt
import torch.nn as nn

from src.config import Config
from src.models.basemodel import BaseModel


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        if self.projection is not None:
            identity = self.projection(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResidualCNN(BaseModel):
    def __init__(self, _config: Config):
        super(ResidualCNN, self).__init__(_config)

        self.task = _config.task  # assuming _config has an attribute `task`

        self.block1 = ResidualBlock(1, 16)
        self.block2 = ResidualBlock(16, 32)
        self.block3 = ResidualBlock(32, 64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

        # Use AdaptiveAvgPool to make the model more flexible with input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Output size: (64, 4, 4)

        self.fc1 = nn.Linear(64 * 4 * 4, 128)

        if self.task == "polar":
            self.fc2 = nn.Linear(128, 1)
        elif self.task == "binary":
            self.fc2 = nn.Linear(128, 3)
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.block1(x))  # B, 1, H, W -> B, 16, H/2, W/2
        x = self.pool(self.block2(x))  # -> B, 32, H/4, W/4
        x = self.pool(self.block3(x))  # -> B, 64, H/8, W/8

        x = self.adaptive_pool(x)  # -> B, 64, 4, 4
        x = self.flatten(x)  # -> B, 64*4*4
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def visualise_feature_maps(self, x, layer_idx):
        if layer_idx == 1:
            x = self.pool(self.block1(x))
        elif layer_idx == 2:
            x = self.pool(self.block1(x))
            x = self.pool(self.block2(x))
        elif layer_idx == 3:
            x = self.pool(self.block1(x))
            x = self.pool(self.block2(x))
            x = self.pool(self.block3(x))

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
