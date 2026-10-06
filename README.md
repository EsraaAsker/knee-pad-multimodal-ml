# Multidimensional Evaluation of Wearable Knee Rehabilitation Assessment

## Overview

This repository provides the preprocessing, training, evaluation, and analysis code associated with the study:

**“Beyond a Single Accuracy: A Multidimensional Evaluation of Wearable Knee Rehabilitation Assessment.”**

The repository contains the code and selected result summaries supporting the experimental framework reported in the paper, including exercise recognition, execution-quality assessment, participant-independent evaluation, and sensor-configuration analysis.

## Dataset

The study uses the publicly available **KneE-PAD** dataset:

Kasnesis, P., Plavoukou, T., Syropoulou, A. C., Toumanidis, L., and Georgoudis, G.,
“A Knee Rehabilitation Exercises Dataset for Postural Assessment using Wearable Devices,”
Scientific Data, 2025.
DOI: **10.1038/s41597-025-04963-4**

The dataset is **not redistributed in this repository**. Users should obtain it from its original publication/source and place the required files in the local data directory before running the provided scripts.

## Repository Contents

- `code/preprocessing/` — data preparation, segmentation, and preprocessing.
- `code/training/` — model training scripts.
- `code/evaluation/` — validation and performance evaluation.
- `code/analysis/` — aggregation, diagnostics, and result analysis.
- `configs/` — experiment settings and reproducibility parameters.
- `results/` — selected result summaries and diagnostic outputs.
- `figures/` — figures associated with the reported analyses.

## Experimental Protocols

The manuscript distinguishes two experimental families with different evaluation protocols.

### Archived Development and Benchmark Family

This family contains the archived development experiments and benchmark results used for contextual comparison in the manuscript, including engineered-feature and raw-sequence model results.

### Strict Participant-Independent Family

This family contains the participant-independent evaluation used to assess generalization to unseen participants. It includes the CNN--Transformer execution-quality experiment and the predefined sensor-configuration analysis.

The two families are reported separately because their validation protocols differ and their results are not interchangeable.

## Reproducibility

The repository provides the experiment code and configuration information needed to reproduce the supported analyses from the publicly available dataset.

The documented settings include preprocessing, windowing, validation structure, random seeds, model configuration, and class-weighting strategy.

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

After obtaining the KneE-PAD dataset from its original source, place the required files according to the directory structure expected by the preprocessing scripts and configuration files.

Do not include dataset files in the GitHub repository.

For the processed multimodal workflows, the local data directory contains the processed IMU/EMG arrays together with the corresponding labels, trial identifiers, and subject identifiers.

## Running the Experiments

### 1. Preprocessing

The raw-data preparation scripts accept explicit paths. For example:

```bash
python code/preprocessing/build_deep_data_from_raw.py \
  --raw /path/to/KneE-PAD \
  --out data/processed/deep_data
```

For the fixed Squat window representation:

```bash
python code/preprocessing/build_fixed_squat_segments_multimodal.py \
  --raw /path/to/KneE-PAD \
  --out data/processed/fixed_squat_multimodal
```

### 2. Model training

The main CNN--Transformer workflows are:

```bash
KNEEPAD_DATA=data/processed/deep_data \
KNEEPAD_OUT=results/levels \
python code/training/train_levels.py
```

and:

```bash
KNEEPAD_DATA=data/processed/deep_data \
KNEEPAD_OUT=results/dl_results \
python code/training/train_multimodal_dl.py
```

### 3. Participant-independent evaluation

The strict participant-independent trainer uses participant-grouped outer folds and grouped inner validation:

```bash
KNEEPAD_DATA=data/processed/deep_data \
KNEEPAD_OUT=results/strict \
EXERCISE=0 USE_EMG=1 \
SUBSET='[1,2,3,4,5,6,7,8]' \
CONFIG=ex0_all_sensors SEED=2026 \
python code/training/train_subject_predictions.py
```

### 4. Sensor-configuration evaluation

The same participant-independent training script supports the predefined sensor configurations through `SUBSET`, for example:

```bash
KNEEPAD_DATA=data/processed/deep_data \
KNEEPAD_OUT=results/strict \
EXERCISE=0 USE_EMG=1 \
SUBSET='[5,7]' \
CONFIG=p57 SEED=2026 \
python code/training/train_subject_predictions.py
```

### 5. Result aggregation and diagnostics

Scripts in `code/evaluation/` and `code/analysis/` aggregate fold-level outputs, predictions, confusion matrices, and diagnostic summaries.

## Reported Metrics

The repository supports the metrics reported in the paper, including:

- Accuracy
- Balanced Accuracy
- Macro-F1
- Precision
- Recall
- Confusion matrices
- Per-class analysis

## Code and Data Availability

The source code, configuration information, selected result summaries, and diagnostics supporting the revision are provided in this repository. The KneE-PAD dataset is publicly available from its original source and is not redistributed here.

## Citation

Please cite the associated paper and the original KneE-PAD dataset publication when using this repository.
