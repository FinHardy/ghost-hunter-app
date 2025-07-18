# Test Data Generation Utilities
import os

import numpy as np
import yaml
from PIL import Image


def generate_test_dataset(output_dir, num_images=20, image_size=64):
    """Generate synthetic test dataset"""
    os.makedirs(output_dir, exist_ok=True)

    # Generate images
    labels = []
    for i in range(num_images):
        # Create random image
        image_array = np.random.randint(
            0, 256, (image_size, image_size), dtype=np.uint8
        )
        image = Image.fromarray(image_array, mode="L")

        filename = f"test_image_{i:03d}.png"
        image.save(os.path.join(output_dir, filename))

        # Random binary label
        label = np.random.randint(0, 2)
        key = {0: "na", 1: "horizontal", 2: "vertical"}
        labels.append({"file": filename, "label": key[label]})

    # Save labels
    labels_data = {"labels": labels}
    with open(os.path.join(output_dir, "labels.yaml"), "w") as f:
        yaml.dump(labels_data, f)

    return output_dir


if __name__ == "__main__":
    generate_test_dataset("test_data")
