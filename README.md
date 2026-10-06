# Multidimensional Evaluation of Wearable Knee Rehabilitation Assessment

## Overview

This repository contains the source code, configuration files, and selected result summaries supporting:

**“Beyond a Single Accuracy: A Multidimensional Evaluation of Wearable Knee Rehabilitation Assessment.”**

The repository provides the materials needed to inspect the reported processing and evaluation procedures and to rerun the supported analyses using the publicly available KneE-PAD dataset.

## Code and Data Availability

The source code and experiment configuration are provided in this repository. The repository includes preprocessing, model training, participant-independent evaluation, sensor-configuration evaluation, and result-aggregation scripts, together with selected machine-readable result summaries.

The **KneE-PAD** dataset is publicly available from its original publication:

Kasnesis, P., Plavoukou, T., Syropoulou, A. C., Toumanidis, L., and Georgoudis, G.,  
“A Knee Rehabilitation Exercises Dataset for Postural Assessment using Wearable Devices,”  
*Scientific Data*, 2025.  
DOI: **10.1038/s41597-025-04963-4**

The dataset is **not redistributed in this repository**. Users should obtain the recordings from the original dataset source and provide the local data path required by the scripts.

## Repository Contents

- `code/preprocessing/` — raw-data preparation and fixed-window construction.
- `code/training/` — multimodal CNN--Transformer training and participant-independent evaluation.
- `code/evaluation/` — prediction and trial-level metric aggregation.
- `code/analysis/` — fold and sensor-configuration result aggregation.
- `configs/` — experiment settings.
- `results/summaries/` — selected machine-readable result summaries.
- `CITATION.cff` — software citation metadata.

## Experimental Families

The manuscript reports two distinct experimental families.

**Archived development and benchmark family.** These experiments use trial-level validation and include the historical model-development/benchmark results reported for contextual comparison.

**Strict participant-independent family.** This evaluation uses participant-grouped outer folds, grouped inner validation for epoch selection, training-only normalization, and evaluation on unseen participants. The strict family is used for the participant-independent squat-quality and predefined sensor-configuration analyses.

Because the task formulation and validation procedures differ between the two families, their reported results are presented separately and are not interchangeable.

## Data Setup

After obtaining KneE-PAD, provide the dataset path to the preprocessing scripts. The expected source layout is:

```
KneE-PAD/
└── dataset/
    └── Subject_*/
        └── <label>/
            └── Trial_*/
                ├── imu.npy
                └── emg.npy
```

The preprocessing code retains the IMU and sEMG arrays together with labels, trial identifiers, and subject identifiers required by the evaluation scripts.

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## Example Workflows

### Preprocessing

```bash
python code/preprocessing/build_deep_data_from_raw.py \
  --raw /path/to/KneE-PAD \
  --out data/processed/deep_data
```

For the fixed Squat representation:

```bash
python code/preprocessing/build_fixed_squat_segments_multimodal.py \
  --raw /path/to/KneE-PAD \
  --out data/processed/fixed_squat_multimodal
```

### Multimodal training/evaluation

```bash
KNEEPAD_DATA=data/processed/deep_data \
KNEEPAD_OUT=results/dl_results \
python code/training/train_multimodal_dl.py
```

### Strict participant-independent evaluation

```bash
KNEEPAD_DATA=data/processed/deep_data \
KNEEPAD_OUT=results/strict \
EXERCISE=0 USE_EMG=1 \
SUBSET='[1,2,3,4,5,6,7,8]' \
CONFIG=ex0_all_sensors SEED=2026 \
python code/training/train_subject_predictions.py
```

The same participant-independent script supports the predefined sensor configurations through `SUBSET`, including the two-sensor left thigh--shank configuration:

```bash
KNEEPAD_DATA=data/processed/deep_data \
KNEEPAD_OUT=results/strict \
EXERCISE=0 USE_EMG=1 \
SUBSET='[5,7]' \
CONFIG=p57 SEED=2026 \
python code/training/train_subject_predictions.py
```

## Reproducibility Scope

The repository documents the preprocessing, windowing, validation structures, model configuration, training-time class weighting, and evaluation metrics used by the supported analyses.

The archived benchmark results are retained as reported reference results, while the strict participant-independent results are represented by the dedicated participant-grouped evaluation code and summaries.

## Citation

Please cite the associated paper and the original KneE-PAD dataset publication when using this repository.
