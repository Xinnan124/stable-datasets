from collections import Counter

import numpy as np
from PIL import Image

from stable_datasets.images import OlivettiFaces


def test_olivetti_faces_dataset():
    """Validate the Olivetti Faces dataset and its splits."""
    train_ds = OlivettiFaces(split="train")
    test_ds = OlivettiFaces(split="test")

    assert len(train_ds) == 320
    assert len(test_ds) == 80

    train_labels = []
    test_labels = []

    for dataset, labels in [
        (train_ds, train_labels),
        (test_ds, test_labels),
    ]:
        for sample in dataset:
            assert set(sample.keys()) == {"image", "label"}

            image = sample["image"]
            label = sample["label"]

            assert isinstance(image, Image.Image)
            assert image.size == (64, 64)
            assert image.mode == "L"

            pixels = np.asarray(image)
            assert pixels.shape == (64, 64)
            assert pixels.dtype == np.uint8

            assert isinstance(label, int)
            assert 0 <= label < 40

            labels.append(label)

    assert set(train_labels) == set(range(40))
    assert set(test_labels) == set(range(40))

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    assert all(train_counts[label] == 8 for label in range(40))
    assert all(test_counts[label] == 2 for label in range(40))
