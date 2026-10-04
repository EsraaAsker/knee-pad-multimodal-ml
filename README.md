# Multidimensional Evaluation of Wearable Knee Rehabilitation Assessment

Research code and evaluation artifacts associated with the study:

> **Beyond a Single Accuracy: A Multidimensional Evaluation of Wearable Knee Rehabilitation Assessment**

This repository contains the reproducible code and selected result artifacts for multimodal wearable-sensing experiments on the public **KneE-PAD** rehabilitation dataset. The repository is intended as a research artifact accompanying the manuscript and is organized to make the data assumptions, experimental protocol, evaluation metrics, and reported results explicit.

> **Artifact status.** This public release contains the currently validated multimodal training/evaluation implementation and its stored fold-level results. Raw participant recordings are intentionally excluded. Experimental families that are not represented by executable files in this repository are not claimed as reproducible here.

## Research scope

The study investigates rehabilitation exercise assessment using synchronized inertial (IMU) and surface electromyography (sEMG) signals. Rather than relying on a single aggregate accuracy value, the evaluation emphasizes:

- exercise-level recognition;
- execution/quality classification;
- balanced accuracy and macro-F1 in addition to accuracy;
- trial-level separation before overlapping windows are expanded;
- train-fold-only normalization and class weighting;
- sensor/modality considerations;
- explicit reporting of the evaluation protocol.

The repository is a **research artifact**, not a clinical diagnostic system.

## Dataset

Experiments use the publicly available **KneE-PAD** dataset:

Kasnesis, P., Plavoukou, T., Syropoulou, A. C., Toumanidis, L., and Georgoudis, G.,  
*A Knee Rehabilitation Exercises Dataset for Postural Assessment using Wearable Devices*,  
Scientific Data, 2025.

**DOI:** https://doi.org/10.1038/s41597-025-04963-4

The dataset is **not redistributed** in this repository. Users must obtain it from the original publication/source and comply with the dataset's terms of use and citation requirements.

Expected processed inputs include synchronized IMU and EMG arrays together with labels, trial identifiers, and subject identifiers.

## Repository contents

| Path | Purpose |
|---|---|
| `train_multimodal_dl.py` | Multimodal EMG–IMU training/evaluation workflow |
| `train_levels.py` | Level-1 exercise and Level-3 joint exercise/quality workflow |
| `level1_fold_metrics.csv` | Stored five-fold Level-1 metrics |
| `level3_fold_metrics.csv` | Stored five-fold Level-3 metrics |
| `levels_summary.csv` | Mean and standard-deviation summary of stored metrics |
| `.gitignore` | Prevents raw data, generated arrays, checkpoints, and environments from being committed |
| `requirements.txt` | Python dependencies |

## Experimental protocol represented in this release

### Trial-held-out multimodal evaluation

The supplied training workflow constructs folds at the **raw trial level before overlapping windows are expanded**. This prevents windows originating from the same trial from appearing in both training and test partitions.

Normalization statistics and class weights are computed from the training portion of each fold.

The model consists of:

1. an EMG temporal convolutional branch;
2. an IMU temporal convolutional branch;
3. feature projection and temporal alignment;
4. a two-layer, eight-head Transformer encoder;
5. attention pooling;
6. a task-specific classification head.

Training uses AdamW, cosine learning-rate scheduling, gradient clipping, class-weighted cross-entropy, and label smoothing. The implementation also applies small Gaussian signal augmentation during training.

### Level-1 and Level-3 workflow

`train_levels.py` evaluates:

- **Level 1:** three exercise/activity classes;
- **Level 3:** nine joint exercise/quality classes.

The stored results use five stratified trial-level folds and 18 training epochs.

## Reported stored results

The following values are calculated from the committed fold-level CSV artifacts.

| Task | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Level 1 — exercise/activity | 99.938% ± 0.092% | 99.944% ± 0.083% | 99.940% ± 0.086% |
| Level 3 — joint exercise/quality | 71.193% ± 6.505% | 76.300% ± 4.515% | 76.433% ± 4.251% |

The variation across folds is intentionally reported rather than hidden behind a single score.

## Reproducibility

### 1. Create an environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Do not place raw participant data in the Git repository.

Prepare the official dataset into a local `deep_data/` directory containing the processed arrays expected by the scripts, including:

```
deep_data/
├── emg_official.dat
├── imu_official.dat
├── labels.npy
├── trial_ids.npy
└── subjects.npy
```

The array dimensions expected by the current implementation are documented in the source files.

### 3. Run the Level-1 / Level-3 workflow

The current source implementation uses local paths from the original experimental environment. Before running on another machine, set the `ROOT` and `OUT` paths in `train_levels.py` to the local dataset/output locations.

Then:

```bash
python train_levels.py
```

The script writes fold-level metrics and a run manifest to the configured output directory.

### 4. Run the multimodal workflow

Similarly configure the local `ROOT` and `OUT` paths in `train_multimodal_dl.py`, then:

```bash
python train_multimodal_dl.py
```

Environment variables supported by the current implementation include:

```bash
TASK=multiclass
FOLD_LIMIT=20
EPOCHS=18
```

For example, a one-fold pilot run can be performed with:

```bash
FOLD_LIMIT=1 EPOCHS=1 python train_multimodal_dl.py
```

A pilot run is intended only for checking the software environment and data interface; it is **not** a substitute for the reported experiment.

## Data leakage safeguards

The evaluation code follows several safeguards relevant to overlapping physiological time-series:

- folds are assigned before segment expansion at the trial level;
- normalization parameters are estimated from training data only;
- class weights are derived from training data only;
- test data are not used for parameter estimation;
- generated binary arrays and checkpoints are excluded from version control.

These details should be considered part of the experimental definition when comparing results.

## Metrics

The evaluation supports:

- accuracy;
- balanced accuracy;
- macro-F1;
- confusion matrices;
- fold-level reporting;
- segment-level predictions;
- trial identifiers;
- participant identifiers.

Balanced accuracy and macro-F1 are reported alongside accuracy because class imbalance and unequal class difficulty can make aggregate accuracy misleading.

## Reproducibility and artifact policy

This repository intentionally does **not** contain:

- raw KneE-PAD recordings;
- participant-identifiable information;
- generated `.dat` arrays;
- model checkpoints;
- local virtual environments;
- temporary training outputs.

The committed CSV files are lightweight result artifacts intended to make the reported summary independently inspectable.

## Limitations

1. The public dataset remains external to this repository.
2. The current training scripts contain paths inherited from the original experimental environment and require local path configuration before execution.
3. The stored results correspond to the exact protocol represented by the committed code and artifacts; changing windowing, folds, normalization, augmentation, or class definitions can produce different results.
4. High exercise-recognition performance should not be interpreted as equivalent to reliable execution-quality assessment.
5. This work does not establish clinical efficacy or diagnostic validity.

## Citation

When using this code or results, please cite both the associated manuscript and the original KneE-PAD dataset publication.

### Dataset citation

Kasnesis, P., Plavoukou, T., Syropoulou, A. C., Toumanidis, L., and Georgoudis, G.  
*A Knee Rehabilitation Exercises Dataset for Postural Assessment using Wearable Devices.*  
Scientific Data, 2025.  
https://doi.org/10.1038/s41597-025-04963-4

## Contact

For questions concerning the research artifact, please use the GitHub repository issue tracker or contact the corresponding author listed in the associated manuscript.
