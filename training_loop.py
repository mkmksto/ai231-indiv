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
from torchinfo import summary
from torchvision import datasets, transforms

# from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    VGG19_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet50,
    vgg19,
)
from tqdm import tqdm

from utils import (
    TumorClassifier,
    TumorDataset,
    encode_labels,
    load_and_preprocess_images,
    load_model_weights,
    load_preprocessed_data,
    save_preprocessed_data,
    test_model,
    train_model,
)

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
    print("Cuda Available: ", torch.cuda.is_available())

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

    # ---------------------------------------------------------------------
    # Actual training
    #
    #
    # ---------------------------------------------------------------------

    # Encode the labels
    y_train_encoded, y_test_encoded = encode_labels(y_train, y_test)
    train_dataset = TumorDataset(X_train, y_train_encoded)
    test_dataset = TumorDataset(X_test, y_test_encoded)

    # #
    # # EffNet
    # #
    # model_effnet = TumorClassifier(
    #     num_classes=4,
    #     pretrained_model=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
    # )  # 2 classes for your tumor types
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_effnet = model_effnet.to(device)
    # summary(model_effnet, input_size=(1, 3, 224, 224))
    # print("***** Training EffNet *****")
    # # Training and saving weights (effnet)
    # train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    #     model_effnet,
    #     train_dataset,
    #     batch_size=32,
    #     epochs=10,
    #     val_split=0.2,
    #     lr=0.001,
    #     save_path="./training_data/effnet_weights.pth",
    # )

    # #
    # # ResNet
    # #
    # model_resnet = TumorClassifier(
    #     num_classes=4,
    #     pretrained_model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
    # )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_resnet = model_resnet.to(device)
    # print("***** Training ResNet *****")
    # # training resnet
    # train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    #     model_resnet,
    #     train_dataset,
    #     batch_size=32,
    #     epochs=10,
    #     val_split=0.2,
    #     lr=0.001,
    #     save_path="./training_data/resnet_weights.pth",
    # )

    #
    # MobileNet
    #
    model_mobilenet = TumorClassifier(
        num_classes=4,
        pretrained_model=mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_mobilenet = model_mobilenet.to(device)
    print("***** Training MobileNet *****")
    # training mobilenet
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model_mobilenet,
        train_dataset,
        batch_size=32,
        epochs=10,
        val_split=0.2,
        lr=0.001,
        save_path="./training_data/mobilenet_weights.pth",
    )

    # #
    # # VGG19
    # #
    # model_vgg19 = TumorClassifier(
    #     num_classes=4,
    #     pretrained_model=vgg19(weights=VGG19_Weights.IMAGENET1K_V1),
    # )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_vgg19 = model_vgg19.to(device)
    # print("**** Training VGG19 ****")
    # # training vgg19
    # train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    #     model_vgg19,
    #     train_dataset,
    #     batch_size=32,
    #     epochs=10,
    #     val_split=0.2,
    #     lr=0.001,
    #     save_path="./training_data/vgg19_weights.pth",
    # )

    # ---------------------------------------------------------------------
    # Testing the model
    #
    #
    # ---------------------------------------------------------------------

    # # Later, to load the weights into a new model:
    # new_model = TumorClassifier(num_classes=4)
    # new_model = load_model_weights(
    #     new_model, "./training_data/effnet_weights.pth"
    # )

    # # Later, to load the weights into a new model:
    # new_model = TumorClassifier(num_classes=4)
    # new_model = load_model_weights(
    #     new_model, "./training_data/effnet_weights.pth"
    # )

    # # Later, to load the weights into a new model:
    # new_model = TumorClassifier(num_classes=4)
    # new_model = load_model_weights(
    #     new_model, "./training_data/effnet_weights.pth"
    # )

    # # Test the model
    # test_loss, test_accuracy = test_model(new_model, test_dataset)
    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
