import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import (load_and_preprocess_images, load_preprocessed_data,
                   save_preprocessed_data)

training_file_path: Path = Path("./brain_tumor_dataset/Training")
testing_file_path: Path = Path("./brain_tumor_dataset/Testing")

train_images_counts: Dict[str, int] = {}
test_images_counts: Dict[str, int] = {}

categories: List[str] = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]

# Training
for category in categories:
    # print(category)
    train_category_path: Path = training_file_path / category
    test_category_path: Path = testing_file_path / category

    train_num_images: int = len(list(train_category_path.glob("*.jpg")))
    train_images_counts[category] = train_num_images

    test_num_images: int = len(list(test_category_path.glob("*.jpg")))
    test_images_counts[category] = test_num_images


print(train_images_counts)
print(test_images_counts)

# Vectorize and resize the images in preparation for training

# X_train = []
# y_train = []
# X_test = []
# y_test = []

# contains vectorized stuff, models, weights, etc.
TRAINING_DATA_PATH = Path("./training_data")


if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())

    # # only run when you need to preprocess the data into numpy arrays
    # X_train, y_train, X_test, y_test = load_and_preprocess_images(
    #     training_file_path, testing_file_path, categories
    # )
    # save_preprocessed_data(X_train, y_train, X_test, y_test, TRAINING_DATA_PATH)

    # load the data from a file
    X_train, y_train, X_test, y_test = load_preprocessed_data(TRAINING_DATA_PATH)

    # Loaded data from files
    print("Training Data, Shape:")
    print(X_train.shape)
    # print(X_train[:5])
    print(y_train.shape)
    print(y_train[:5])
    print("Testing Data, Shape:")
    print(X_test.shape)
    # print(X_test[:5])
    print(y_test.shape)
    print(y_test[:5])
