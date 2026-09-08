# Multimodal Knee-Exercise Quality Classification

A reproducible multimodal machine-learning project for classifying rehabilitation exercise activity and movement quality from synchronized EMG and IMU signals.

## What this demonstrates

- Dual-branch EMG–IMU deep learning.
- Temporal convolutional feature extraction.
- Transformer sequence modeling and attention pooling.
- Trial-level cross-validation for overlapping time-series windows.
- Train-fold-only normalization and class weighting.
- Three-class exercise identification and nine-class joint exercise-quality classification.

## Architecture

The model uses separate EMG and IMU convolutional branches, temporal alignment and projection, a two-layer eight-head Transformer encoder, attention pooling, and a task-specific linear head. The implementation is documented in `train_multimodal_dl.py` and `train_levels.py`.

## Verified results

| Task | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Exercise identification (3 classes) | 99.938% ± 0.092% | 99.944% ± 0.083% | 99.940% ± 0.086% |
| Joint exercise + quality (9 classes) | 71.193% ± 6.505% | 76.300% ± 4.515% | 76.433% ± 4.251% |

These values are copied from the stored fold-level artifacts. Raw participant data are not included in this repository.

## Reproducibility

The scripts expect the official KneE-PAD dataset to be downloaded separately and prepared into the documented `deep_data/` representation. Do not commit raw data or credentials.

```bash
python train_levels.py
```

## Evaluation policy

Splits are made at the trial level before overlapping windows are expanded. Normalization statistics, class weights, and model selection are restricted to training/development data in the retained protocol.

## Limitations

This repository is a research artifact, not a medical diagnostic system. The stored results are protocol-specific and should not be compared with numbers produced under different windowing or split definitions.

## License and data

Code is provided for portfolio and educational use. The dataset remains subject to its original license and citation requirements.
