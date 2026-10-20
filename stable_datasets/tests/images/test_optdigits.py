import numpy as np
from PIL import Image

from stable_datasets.images import Optdigits


def test_optdigits_dataset():
    """Validate the UCI Optdigits dataset and its official splits."""
    train_ds = Optdigits(split="train")
    test_ds = Optdigits(split="test")

    assert len(train_ds) == 3823
    assert len(test_ds) == 1797

    for dataset in [train_ds, test_ds]:
        labels = []

        for sample in dataset:
            assert set(sample.keys()) == {"image", "label"}

            image = sample["image"]
            label = sample["label"]

            assert isinstance(image, Image.Image)
            assert image.size == (8, 8)
            assert image.mode == "L"

            pixels = np.asarray(image)
            assert pixels.shape == (8, 8)
            assert pixels.dtype == np.uint8
            assert pixels.min() >= 0
            assert pixels.max() <= 255

            assert isinstance(label, int)
            assert 0 <= label < 10

            labels.append(label)

        assert set(labels) == set(range(10))
