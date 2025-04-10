from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


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
