from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm


# Neural Network
class TumorClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained_model: nn.Module = None,
        weights: str = None,
    ):
        """
        Initialize the TumorClassifier with a custom pretrained model.
        Defaults to EfficientNetB0, but can be used with any other model
        e.g. ResNet, VGG, MobileNet, etc.

        Args:
            num_classes (int): Number of output classes
            pretrained_model (nn.Module, optional): Pretrained model to use. If None, defaults to EfficientNetB0
            weights (str, optional): Weights to load for the pretrained model. If None, uses default weights
        """
        super(TumorClassifier, self).__init__()

        # Use provided model or default to EfficientNetB0
        if pretrained_model is None:
            self.base_model = efficientnet_b0(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            self.base_model = pretrained_model
            if weights:
                self.base_model.load_state_dict(torch.load(weights))

        # # Get number of features based on model architecture
        # if hasattr(self.base_model, "classifier"):
        #     # For EfficientNet, MobileNet
        #     num_features = self.base_model.classifier[-1].in_features
        #     self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        # Get number of features based on model architecture
        if isinstance(self.base_model, models.mobilenet.MobileNetV3):
            # For MobileNetV3
            num_features = self.base_model.classifier[-1].in_features
            # Remove the classifier but keep the avgpool
            self.base_model = nn.Sequential(
                self.base_model.features, self.base_model.avgpool, nn.Flatten()
            )
        elif hasattr(self.base_model, "fc"):
            # For ResNet
            num_features = self.base_model.fc.in_features
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
        elif hasattr(self.base_model, "classifier"):
            # For VGG
            num_features = self.base_model.classifier[-1].in_features
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        else:
            raise ValueError("Unsupported model architecture")

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
        # Forward pass through base model features
        x = self.base_model(x)
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
    save_path: str = "model_weights.pth",  # New parameter for save path
):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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

    # At the end of training, after the plotting code:
    print(f"Saving model weights to {save_path}")
    torch.save(model.state_dict(), save_path)

    return train_losses, val_losses, train_accuracies, val_accuracies


def load_model_weights(model: nn.Module, weights_path: str) -> nn.Module:
    """
    Load saved weights into a model.

    Args:
        model: The model to load weights into
        weights_path: Path to the saved weights file

    Returns:
        The model with loaded weights

    Sample Usage
    # Training and saving weights
    model = TumorClassifier(num_classes=2)
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model,
        train_dataset,
        batch_size=32,
        epochs=10,
        save_path="tumor_classifier_weights.pth"
    )

    # Later, to load the weights into a new model:
    new_model = TumorClassifier(num_classes=2)
    new_model = load_model_weights(new_model, "tumor_classifier_weights.pth")
    """

    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Set the model to evaluation mode
    return model


class TumorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images) / 255.0  # Normalize to [0,1]
        self.images = self.images.permute(0, 3, 1, 2)  # Change to (N, C, H, W)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


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


def test_model(
    model: nn.Module,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
):
    """
    Test the model on the test dataset and return metrics and data for visualization.

    Args:
        model: The trained model to test
        test_dataset: The test dataset
        batch_size: Batch size for testing

    Returns:
        Tuple containing (test_loss, test_accuracy, classification_report, predictions, true_labels, probabilities)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    # Lists to store predictions and true labels
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            probabilities = outputs.softmax(dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions, labels and probabilities
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            total_loss += loss.item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Generate classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=[f"Class {i}" for i in range(len(np.unique(all_labels)))],
        digits=4,
    )

    return avg_loss, accuracy, report, all_predictions, all_labels, all_probabilities


def plot_confusion_matrix(true_labels, predictions, class_names=None):
    """
    Plot confusion matrix for the model predictions.

    Args:
        true_labels: True labels from the test set
        predictions: Model predictions
        class_names: List of class names (optional)
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(true_labels)))]

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(true_labels, probabilities, class_names=None):
    """
    Plot ROC curves for each class.

    Args:
        true_labels: True labels from the test set
        probabilities: Model prediction probabilities
        class_names: List of class names (optional)
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(true_labels)))]

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve((true_labels == i).astype(int), probabilities[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


"""
Sample Usage:
# 1. Create/load your model
model = TumorClassifier(num_classes=4)
model = load_model_weights(model, "path/to/your/saved/weights.pth")

# 2. Create test dataset
test_dataset = TumorDataset(X_test, y_test)

# 3. Run the test
test_loss, test_accuracy, classification_report, predictions, true_labels, probabilities = test_model(
    model=model,
    test_dataset=test_dataset,
    batch_size=32
)

# 4. Print numerical results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report)

# 5. Create visualizations
# You can use your actual class names
class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Plot confusion matrix
plot_confusion_matrix(true_labels, predictions, class_names)

# Plot ROC curves
plot_roc_curves(true_labels, probabilities, class_names)
"""
