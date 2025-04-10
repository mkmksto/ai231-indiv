"""
Michael Quinto
AI 231 | HW7
Instructor: Dr. Remolona
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from IPython.display import display
from torchvision import models, transforms

# make path invariant of where the script is run from
SCRIPT_DIR = Path(__file__).parent.resolve()
training_file_path = SCRIPT_DIR.parent / "brain_tumor_dataset" / "Training"
testing_file_path = SCRIPT_DIR.parent / "brain_tumor_dataset" / "Testing"


categories = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def count_categories():
    train_images_counts = {}
    test_images_counts = {}

    # Training
    for category in categories:
        # print(category)
        train_category_path = training_file_path / category
        test_category_path = testing_file_path / category

        train_num_images = len(list(train_category_path.glob("*.jpg")))
        train_images_counts[category] = train_num_images

        test_num_images = len(list(test_category_path.glob("*.jpg")))
        test_images_counts[category] = test_num_images

    return train_images_counts, test_images_counts


def show_one_image_per_category(
    categories: list[str], training_file_path: Path
) -> None:
    plt.figure(figsize=(15, 10))

    for i, category in enumerate(categories):
        category_path = training_file_path / category
        images = list(category_path.glob("*.jpg"))
        if images:  # Check if there are any images in the category
            # Take the first image from each category
            # image_path = images[0]
            image_path = random.choice(images)
            plt.subplot(2, 2, i + 1)
            plt.imshow(plt.imread(image_path))
            plt.title(category)
            plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_images_counts, test_images_counts = count_categories()
    print(train_images_counts)
    print(test_images_counts)
    # show_one_image_per_category()
