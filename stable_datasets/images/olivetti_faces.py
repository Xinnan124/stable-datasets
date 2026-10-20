import numpy as np
from PIL import Image as PILImage
from scipy.io import loadmat

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


class OlivettiFaces(BaseDatasetBuilder):
    """Olivetti Faces dataset for 40-class face classification."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset",
        citation="""@misc{olivettifaces,
            title={The Olivetti Faces Dataset},
            howpublished={AT&T Laboratories Cambridge}
        }""",
        assets={
            "data": DownloadInfo(url="https://ndownloader.figshare.com/files/5976027"),
        },
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "The Olivetti Faces dataset contains 400 grayscale images "
                "of 40 individuals, with 10 images per individual. "
                "Each image has a resolution of 64x64 pixels. "
                "A deterministic stratified split assigns 8 images per "
                "individual to training and 2 to testing."
            ),
            features=Features(
                {
                    "image": Image(),
                    "label": ClassLabel(num_classes=40),
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

        data_path = local_paths[0]

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "data_path": data_path,
                    "split": "train",
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "data_path": data_path,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, data_path, split):
        """Generate grayscale images and identity labels."""

        raw_faces = loadmat(data_path)["faces"]

        if raw_faces.shape != (4096, 400):
            raise ValueError(f"Unexpected Olivetti Faces shape: {raw_faces.shape}")

        # Reproduce the official scikit-learn image orientation.
        images = raw_faces.T.copy().reshape((400, 64, 64)).transpose(0, 2, 1)

        labels = np.repeat(np.arange(40), 10)

        for idx, (image, label) in enumerate(zip(images, labels)):
            # Keep a deterministic 8/2 split for each identity.
            position = idx % 10

            if split == "train" and position >= 8:
                continue

            if split == "test" and position < 8:
                continue

            pil_image = PILImage.fromarray(image, mode="L")

            yield (
                idx,
                {
                    "image": pil_image,
                    "label": int(label),
                },
            )
