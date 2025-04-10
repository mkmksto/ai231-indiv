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
