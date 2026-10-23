# Olivetti Faces and UCI Optdigits: Dataset Validation

## Overview

This contribution adds two small image classification datasets to stable-datasets:

- Olivetti Faces: 400 grayscale face images across 40 identity classes.
- UCI Optdigits: 5,620 handwritten digit samples across 10 classes.

Both datasets are accessible through the `stable_datasets.images` API.

## Data Sources

### Olivetti Faces

- Source: https://ndownloader.figshare.com/files/5976027
- Original format: MATLAB `.mat`
- Image size: 64 x 64
- Number of classes: 40
- Train: 320 samples
- Test: 80 samples

A deterministic per-identity split is used: the first eight images
of each identity form the training split, and the last two form the
test split. This split is defined by this contribution.

### UCI Optdigits

- Source: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- Original files: `optdigits.tra` and `optdigits.tes`
- Image size: 8 x 8
- Number of classes: 10
- Train: 3,823 samples
- Test: 1,797 samples

The original UCI train/test split is preserved. Each sample's 64
intensity features are reshaped into an 8 x 8 grayscale image.
Original intensity values in [0, 16] are mapped to 8-bit pixel
values in [0, 255] using rounding.

## Dataset Tests

The following pytest files validate dataset loading, split sizes,
image properties, label types, and class coverage:

- `stable_datasets/tests/images/test_olivetti_faces.py`
- `stable_datasets/tests/images/test_optdigits.py`

All four tests passed locally.

## Source Consistency Validation

All 6,020 samples were compared against their corresponding
original source data.

| Dataset | Training samples | Test samples | Verified samples |
| --- | ---: | ---: | ---: |
| Olivetti Faces | 320 | 80 | 400 |
| UCI Optdigits | 3,823 | 1,797 | 5,620 |
| Total | 4,143 | 1,877 | 6,020 |

For Olivetti Faces, every image and identity label matched the
expected source entry, and the train/test source indices were
disjoint with complete coverage.

For UCI Optdigits, every image and digit label matched the
corresponding original training or test file entry after the
documented intensity conversion.

## Classification Smoke Tests

A scikit-learn 1-nearest-neighbor classifier was trained separately
on each dataset's training split and evaluated on its test split.

The classifier used flattened grayscale pixels normalized to
[0, 1] and `KNeighborsClassifier(n_neighbors=1)`.

| Dataset | Train | Test | Correct | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Olivetti Faces | 320 | 80 | 71 | 88.75% |
| UCI Optdigits | 3,823 | 1,797 | 1,759 | 97.89% |

These results demonstrate that the dataset builders support a
basic training and evaluation workflow. They are not intended
as state-of-the-art benchmark results.

## Local Verification

```bash
python -m pytest \
    stable_datasets/tests/images/test_olivetti_faces.py \
    stable_datasets/tests/images/test_optdigits.py -v

python -m ruff check \
    stable_datasets/images/olivetti_faces.py \
    stable_datasets/images/optdigits.py \
    stable_datasets/images/__init__.py \
    stable_datasets/tests/images/test_olivetti_faces.py \
    stable_datasets/tests/images/test_optdigits.py
```

## Reproducing Classification Results

From the repository root, install the project and scikit-learn,
then run:

```bash
python -m pip install -e .
python -m pip install scikit-learn
PYTHONPATH=. python docs/olivetti-optdigits-experiment.py
```

The script trains a 1-nearest-neighbor classifier separately
for both datasets and reports the test accuracy.

Expected results from the validated run:

- Olivetti Faces: 71/80 correct, accuracy 0.8875.
- UCI Optdigits: 1759/1797 correct, accuracy 0.9789.
