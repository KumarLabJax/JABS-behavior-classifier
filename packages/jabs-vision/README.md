# JABS Vision (`jabs-vision`)

This package handles raw video processing and deep learning training and inference.

## Overview

`jabs-vision` is responsible for converting raw video frames into pose estimation data.
It houses the heavy machine learning frameworks and GPU-accelerated code.

## Responsibilities

- **Pose Estimation Inference**: Running deep learning models (PyTorch) on video frames
  to detect keypoints.
- **Static Object Detection**
- **Sementation Masking**
- **Identity Matching**: Tracking individual animals across frames using vectorized
  features.
- **Video Processing**: Handling frame extraction and normalization for input to vision
  models.

## Key Components

- `jabs.vision.inference`: Wrappers for pose estimation model execution.
- `jabs.vision.tracking`: Identity matching and re-identification logic.

## Dependencies

- `torch` / `torchvision`
- `opencv-python-headless`
- `jabs-io`, `jabs-core`
