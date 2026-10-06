# Experimental Evidence and Provenance

This file summarizes the manuscript-relevant evidence retained in the repository and clarifies the distinction between the experimental families reported in the paper.

## 1. CNN--Transformer development result

Source: `results/summaries/baseline_B_ensemble_summary.csv`.

- Accuracy: **85.58 ± 3.56%**
- Balanced Accuracy: **86.06 ± 3.02%**
- Macro-F1: **85.85 ± 3.26%**

This result belongs to the archived development family and is reported in the manuscript as a development/benchmark result.

## 2. Strict participant-independent squat result

Source: `results/summaries/strict_subject_heldout_summary.csv`.

- Segment-level Accuracy: **62.03 ± 7.11%**
- Segment-level Balanced Accuracy: **62.55 ± 6.97%**
- Segment-level Macro-F1: **61.37 ± 7.21%**

Protocol: participant-grouped outer evaluation with grouped inner validation, train-only normalization, and no participant overlap between training and test partitions.

These results belong to the strict participant-independent evaluation family. They use a different validation protocol from the archived development result and therefore are not presented as a replacement or direct re-run of it.

## 3. Archived benchmark results

The manuscript's archived benchmark includes:

- Squat execution-quality ExtraTrees: **89.43 ± 2.60% Balanced Accuracy**.
- Binary execution-quality assessment: **68.82 ± 3.71% Balanced Accuracy** for the eight-IMU configuration.
- Binary execution-quality assessment: **64.73 ± 5.46% Balanced Accuracy** for the archived one-IMU configuration.

These results are retained as historical benchmark results and are clearly distinguished from the strict participant-independent evaluation.

## 4. Gait result definitions

The manuscript reports gait results under different task definitions. In particular:

- **92.67% / 94.29%** are recalls for the two gait error categories in the gait-specific binary error analysis.
- **30.95% / 44.44%** are recalls for Gait_NF and Gait_HA in the joint nine-class classification analysis.

These values therefore refer to different classification tasks and should not be interpreted as conflicting measurements of the same task.

## 5. Dataset availability

The KneE-PAD dataset is publicly available from its original publication. The dataset itself is **not redistributed** in this repository.

The repository provides source code, configuration information, selected machine-readable result summaries, and diagnostic outputs supporting the manuscript revision.

## 6. Reproducibility scope

The repository is intended to support the code/data-availability request by providing:

- preprocessing scripts;
- training scripts;
- evaluation and aggregation scripts;
- experiment settings;
- selected result summaries;
- diagnostic outputs;
- a clear separation between archived and strict experimental families.

Raw patient data are not included in the repository.
