from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm


# Neural Network
class TumorClassifier(nn.Module):
    # # sample usage
    # model = TumorClassifier(num_classes=2)  # 2 classes for your tumor types

    def __init__(self, num_classes=2):  # num_classes=2 for glioma vs meningioma
        super(TumorClassifier, self).__init__()

        # Load pretrained EfficientNetB0
        self.effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Remove the last layer
        num_features = self.effnet.classifier[1].in_features
        self.effnet = nn.Sequential(*list(self.effnet.children())[:-1])

        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes),
        )

        # Add softmax for probability output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass through EfficientNet features
        x = self.effnet(x)
        # Forward pass through classifier
        x = self.classifier(x)
        # Apply softmax
        x = self.softmax(x)
        return x


def train_model(
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    epochs: int = 10,
    val_split: float = 0.2,
    lr: float = 0.001,
):
    # # Sample Usage
    # # Usage:
    # train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    #     model,
    #     train_dataset,  # Your dataset
    #     batch_size=32,
    #     epochs=10,
    #     val_split=0.2,
    #     lr=0.001,
    # )

    # Split dataset into train and validation
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update total loss
            total_train_loss += loss.item()

            # Update progress bar
            train_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100 * correct_train / total_train:.2f}%",
                }
            )

        # Calculate average training metrics
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                total_val_loss += loss.item()

                # Update progress bar
                val_pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{100 * correct_val / total_val:.2f}%",
                    }
                )

        # Calculate average validation metrics
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(
            f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%"
        )
        print(
            f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n"
        )

    # Plot training curves
    plt.figure(figsize=(12, 4))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return train_losses, val_losses, train_accuracies, val_accuracies


# ------------------------------------------------------------------------------------------------
# Utils
#
#
# ------------------------------------------------------------------------------------------------


def encode_labels(
    y_train: np.ndarray, y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode the labels in the training and testing sets.

    Args:
        y_train: Training labels
        y_test: Testing labels
    """

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # show the equivalent of the numbers to the categories (only the unique values)
    print("Encoded labels")
    for class_label in label_encoder.classes_:
        encoded_value = label_encoder.transform([class_label])[0]
        print(f"{class_label}: {encoded_value}")

    return y_train_encoded, y_test_encoded


def load_and_preprocess_images(
    training_path: Path,
    testing_path: Path,
    categories: List[str],
    image_size: int = 150,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train: List[np.ndarray] = []
    y_train: List[str] = []
    X_test: List[np.ndarray] = []
    y_test: List[str] = []

    # Training
    for category in categories:
        folderPath = training_path / category
        for img_path in tqdm(folderPath.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (image_size, image_size))
            X_train.append(img)
            y_train.append(category)

    # Testing
    for category in categories:
        folderPath = testing_path / category
        for img_path in tqdm(folderPath.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (image_size, image_size))
            X_test.append(img)
            y_test.append(category)

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def save_preprocessed_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path = Path("./training_data"),
) -> None:
    """
    Save preprocessed data arrays to files.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        save_path: Directory to save the files
    """
    print(f"Saving preprocessed data to {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / "X_train.npy", X_train)
    np.save(save_path / "y_train.npy", y_train)
    np.save(save_path / "X_test.npy", X_test)
    np.save(save_path / "y_test.npy", y_test)


def load_preprocessed_data(
    load_path: Path = Path("./training_data"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed data arrays from files.

    Args:
        load_path: Directory containing the saved files

    Returns:
        Tuple containing (X_train, y_train, X_test, y_test)
    """
    X_train = np.load(load_path / "X_train.npy")
    y_train = np.load(load_path / "y_train.npy")
    X_test = np.load(load_path / "X_test.npy")
    y_test = np.load(load_path / "y_test.npy")

    return X_train, y_train, X_test, y_test
