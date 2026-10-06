# Multidimensional Evaluation of Wearable Knee Rehabilitation Assessment

## Overview

This repository contains the source code supporting:

**“Beyond a Single Accuracy: A Multidimensional Evaluation of Wearable Knee Rehabilitation Assessment.”**

The code covers the data-preparation and evaluation procedures used in the study, including preprocessing, multimodal model training, participant-independent evaluation, sensor-configuration evaluation, and metric aggregation.

## Code and Data Availability

The source code is provided in the `code/` directory.

The study uses the publicly available **KneE-PAD** dataset:

Kasnesis, P., Plavoukou, T., Syropoulou, A. C., Toumanidis, L., and Georgoudis, G.,  
“A Knee Rehabilitation Exercises Dataset for Postural Assessment using Wearable Devices,”  
*Scientific Data*, 2025.  
DOI: **10.1038/s41597-025-04963-4**

The dataset is **not redistributed in this repository**. The recordings should be obtained from the original dataset source and supplied locally to the preprocessing scripts.

## Experimental Protocols

The manuscript distinguishes two experimental families.

**Archived development and benchmark family.** These experiments use trial-level validation and provide the development/benchmark results reported for contextual comparison in the paper.

**Strict participant-independent family.** This evaluation uses participant-grouped outer folds, grouped inner validation for epoch selection, training-only normalization, and evaluation on unseen participants. It supports the participant-independent squat-quality and predefined sensor-configuration analyses.

Because the two families use different task formulations and validation procedures, their results are reported separately and are not interchangeable.

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

The scripts generate the processed arrays and metadata required by the training and evaluation code.

## Installation

Install the Python dependencies required by the scripts, including NumPy, pandas, scikit-learn, and PyTorch.

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
