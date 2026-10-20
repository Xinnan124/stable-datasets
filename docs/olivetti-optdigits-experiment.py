"""Reproduce KNN classification results for two image datasets."""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from stable_datasets.images import OlivettiFaces, Optdigits


def prepare_data(dataset):
    """Convert dataset images and labels to NumPy arrays."""
    images = np.stack([np.asarray(sample["image"], dtype=np.float32).reshape(-1) / 255.0 for sample in dataset])
    labels = np.asarray([int(sample["label"]) for sample in dataset])
    return images, labels


def evaluate_dataset(name, builder):
    """Train a 1-NN classifier and evaluate its test accuracy."""
    train_dataset = builder(split="train")
    test_dataset = builder(split="test")

    x_train, y_train = prepare_data(train_dataset)
    x_test, y_test = prepare_data(test_dataset)

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    correct = int(np.sum(predictions == y_test))

    print(f"Dataset: {name}")
    print("Model: KNN (k=1)")
    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples: {len(y_test)}")
    print(f"Feature dimensions: {x_train.shape[1]}")
    print(f"Correct predictions: {correct}")
    print(f"Test accuracy: {accuracy:.4f}")
    print()


def main():
    """Run classification experiments for both datasets."""
    evaluate_dataset("Olivetti Faces", OlivettiFaces)
    evaluate_dataset("UCI Optdigits", Optdigits)


if __name__ == "__main__":
    main()
