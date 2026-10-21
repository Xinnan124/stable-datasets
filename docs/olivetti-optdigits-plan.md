# Olivetti Faces and UCI Optdigits Dataset Integration

## Overview

This PR aims to add two image classification datasets to `stable-datasets`:

1. Olivetti Faces — 400 grayscale facial images representing 40 individuals.
2. UCI Optdigits — 5,620 handwritten digit samples across 10 classes.

## Planned Implementation

- Add dataset builders for both datasets.
- Define image and classification-label schemas.
- Implement reproducible training and evaluation splits.
- Support data downloading and caching.
- Add automated tests for metadata, sample loading, and splits.
- Validate both datasets using an existing classification method.
- Document dataset sources, usage, and experimental results.

## Status

Implementation completed. Both dataset builders, automated tests, and reproducible classification experiments have been added. See docs/olivetti-optdigits-validation.md for validation results.
