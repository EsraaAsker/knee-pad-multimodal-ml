# Multidimensional Evaluation of Wearable Knee Rehabilitation Assessment

## Overview

This repository contains the source code supporting:

**“Beyond a Single Accuracy: A Multidimensional Evaluation of Wearable Knee Rehabilitation Assessment.”**

The code covers data preparation, multimodal CNN--Transformer training, participant-independent evaluation, sensor-configuration evaluation, and metric aggregation.

## Code and Data Availability

The source code is provided in the `code/` directory.

The study uses the publicly available **KneE-PAD** dataset:

Kasnesis, P., Plavoukou, T., Syropoulou, A. C., Toumanidis, L., and Georgoudis, G.,  
“A Knee Rehabilitation Exercises Dataset for Postural Assessment using Wearable Devices,”  
*Scientific Data*, 2025.  
DOI: **10.1038/s41597-025-04963-4**

The dataset is **not redistributed in this repository**. The recordings should be obtained from the original dataset source and supplied locally to the preprocessing scripts.

## Experimental Protocols

The repository includes code for multimodal development/benchmark experiments and for the strict participant-independent squat-quality and predefined sensor-configuration analyses. These procedures use different validation protocols and should not be treated as interchangeable.

## Data Setup

The preprocessing scripts expect the KneE-PAD files in the following general layout:

```
KneE-PAD/
└── dataset/
    └── Subject_*/
        └── <label>/
            └── Trial_*/
                ├── imu.npy
                └── emg.npy
```

The preprocessing scripts generate the processed arrays and metadata required by the training and evaluation code.

## Strict protocol summary (squat Correct vs Wrong)

- 31 participants, 1,075 trials, and 1,747 segments (840 Correct, 907 Wrong); windows of 593 samples with stride 37.
- Participant-grouped, label-stratified outer folds: 5 outer folds. Epoch selection uses a participant-grouped inner split within the training participants. Normalization statistics are computed from the training partition only.
- Loss: cross-entropy with class weights proportional to the square root of inverse class frequency, with label smoothing 0.04.
- Metrics: accuracy, balanced accuracy, and Macro-F1; trial-level aggregation is provided by the evaluation script.
- IMU configurations supported by the participant-independent script, with all 8 sEMG channels retained when `USE_EMG=1): full `[1,2,3,4,5,6,7,8]`, right-side `[1,2,3,4]`, right thigh--shank `[1,3]`, left thigh--shank `[5,7]`, and single `[1]`.
- The `SEED` environment variable controls the run seed; repeat the strict run with the study's specified seeds when reproducing the multi-seed analysis.

## Models

### Multimodal CNN--Transformer

The reported multimodal model uses separate 1D-CNN branches for IMU and sEMG, feature concatenation and projection to 128 dimensions, a 2-layer Transformer encoder, attention pooling, and a linear classification head.

The participant-independent script supports both multimodal operation and IMU-only operation through `USE_EMG`.

### Other training levels

`code/training/train_levels.py` provides the repository's Level-1 exercise classification and Level-3 multiclass classification training procedures.

## Essential preprocessing

For the fixed squat representation, the repository uses 593-sample IMU windows with stride 37. The multimodal preprocessing aligns each raw EMG interval to the corresponding IMU window and resamples it to 5,037 points. During model input preparation, the training scripts downsample the stored EMG and IMU arrays and compute normalization statistics from the training partition.

## Installation

Install Python dependencies required by the scripts, including NumPy, pandas, scikit-learn, and PyTorch.

## Example Workflows

### Preprocessing

```bash
python code/preprocessing/build_deep_data_from_raw.py \
  --raw /path/to/KneE-PAD \
  --out /path/to/processed/deep_data
```

For the fixed Squat representation:

```bash
python code/preprocessing/build_fixed_squat_segments_multimodal.py \
  --raw /path/to/KneE-PAD \
  --out /path/to/processed/fixed_squat_multimodal
```

### Multimodal training/evaluation

```bash
KNEEPAD_DATA=/path/to/processed/deep_data \
KNEEPAD_OUT=/path/to/output \
python code/training/train_multimodal_dl.py
```

### Strict participant-independent evaluation

```bash
KNEEPAD_DATA=/path/to/processed/deep_data \
KNEEPAD_OUT=/path/to/output \
EXERCISE=0 USE_EMG=1 \
SUBSET='[1,2,3,4,5,6,7,8]' \
CONFIG=ex0_all_sensors SEED=2026 \
python code/training/train_subject_predictions.py
```

The same participant-independent script accepts the predefined sensor configurations through `SUBSET`, including `[5,7]` for the left thigh--shank configuration.

## Citation

Please cite the associated paper and the original KneE-PAD dataset publication when using this repository.
