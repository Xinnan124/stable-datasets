import numpy as np
from PIL import Image as PILImage

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Image,
    Version,
)
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class Optdigits(BaseDatasetBuilder):
    """UCI Optical Recognition of Handwritten Digits dataset."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        citation="""@misc{uci_optdigits,
            title={Optical Recognition of Handwritten Digits},
            author={Alpaydin, E. and Kaynak, C.},
            year={1998},
            publisher={UCI Machine Learning Repository}
        }""",
        assets={
            "train": DownloadInfo(
                url=("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra")
            ),
            "test": DownloadInfo(
                url=("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes")
            ),
        },
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "The UCI Optical Recognition of Handwritten Digits "
                "dataset contains 5,620 samples across 10 classes. "
                "Each sample consists of 64 intensity features "
                "representing an 8x8 grayscale image. "
                "The original training and test splits contain "
                "3,823 and 1,797 samples, respectively."
            ),
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(num_classes=10),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        assets = self._source()["assets"]

        local_paths = bulk_download(
            list(assets.values()),
            dest_folder=self._raw_download_dir,
        )

        paths = dict(zip(assets.keys(), local_paths))

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"data_path": paths["train"]},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"data_path": paths["test"]},
            ),
        ]

    def _generate_examples(self, data_path):
        """Generate 8x8 grayscale images and digit labels."""

        data = np.loadtxt(
            data_path,
            delimiter=",",
            dtype=np.int32,
        )

        if data.ndim != 2 or data.shape[1] != 65:
            raise ValueError(f"Unexpected Optdigits data shape: {data.shape}")

        features = data[:, :-1]
        labels = data[:, -1]

        if np.any((features < 0) | (features > 16)):
            raise ValueError("Pixel intensities must be in [0, 16].")

        if np.any((labels < 0) | (labels > 9)):
            raise ValueError("Digit labels must be in [0, 9].")

        for idx, (pixels, label) in enumerate(zip(features, labels)):
            # Convert the original intensities to 8-bit grayscale.
            image = np.rint(pixels.reshape(8, 8) * (255.0 / 16.0)).astype(np.uint8)

            pil_image = PILImage.fromarray(image)

            yield (
                idx,
                {
                    "image": pil_image,
                    "label": int(label),
                },
            )
