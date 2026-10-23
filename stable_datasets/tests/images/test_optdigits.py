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


def test_optdigits_source_consistency():
    """Compare all samples with the official UCI source files."""
    from io import BytesIO
    from urllib.request import urlopen

    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"

    for split, filename, expected_count in [
        ("train", "optdigits.tra", 3823),
        ("test", "optdigits.tes", 1797),
    ]:
        with urlopen(base_url + filename, timeout=120) as response:
            source_bytes = response.read()

        source_data = np.loadtxt(BytesIO(source_bytes), delimiter=",", dtype=np.int32)

        dataset = Optdigits(split=split)

        assert source_data.shape == (expected_count, 65)
        assert len(dataset) == expected_count

        for index in range(expected_count):
            raw_pixels = source_data[index, :-1].reshape(8, 8)
            expected_image = np.rint(raw_pixels * (255.0 / 16.0)).astype(np.uint8)

            sample = dataset[index]

            np.testing.assert_array_equal(
                np.asarray(sample["image"]),
                expected_image,
            )
            assert int(sample["label"]) == int(source_data[index, -1])
