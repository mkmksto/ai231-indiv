import warnings
from pathlib import Path

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

training_file_path = Path("./brain_tumor_dataset/Training")
testing_file_path = Path("./brain_tumor_dataset/Testing")

train_images_counts = {}
test_images_counts = {}

categories = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Training
for category in categories:
    # print(category)
    train_category_path = training_file_path / category
    test_category_path = testing_file_path / category

    train_num_images = len(list(train_category_path.glob("*.jpg")))
    train_images_counts[category] = train_num_images

    test_num_images = len(list(test_category_path.glob("*.jpg")))
    test_images_counts[category] = test_num_images


print(train_images_counts)
print(test_images_counts)


if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
