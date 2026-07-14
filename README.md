# UAE License Plate Detection — Preprocessed Dataset

## Overview

This repository contains the **data-preprocessing handoff** for the UAE License Plate Detection course project.

Its purpose is to prepare, document, validate, and distribute a clean single-class object-detection dataset for detecting the **complete visible UAE license plate**. It includes:

- YOLO-format labels
- COCO-format annotations
- Dataset manifests and integrity hashes
- Preprocessing and validation scripts
- Dataset statistics and visualizations
- A documented training-only augmentation policy
- An executed preprocessing notebook

This is intentionally a **preprocessing sub-repository**. It does not contain model training, model evaluation, inference, or deployment code.

---

## Dataset source

- **Dataset name:** UAE
- **Source:** https://universe.roboflow.com/addinguae/uae-zcfqj
- **Source dataset license:** CC BY 4.0

The original dataset contains 51 annotation classes, including digits, letters, expiration marks, emirate/style categories, and a complete-plate class.

For this project, only the source class named `plate` is retained. It is remapped to:

```yaml
0: license_plate
```

Digits, letters, expiration marks, emirate identifiers, and plate-style annotations are excluded because the project target is the **complete visible license plate as one object**.

---

## Final dataset summary

| Split | Images | Label files | Bounding boxes |
|---|---:|---:|---:|
| Train | 6,738 | 6,738 | 9,415 |
| Validation | 1,440 | 1,440 | 1,525 |
| Test | 1,432 | 1,432 | 1,511 |
| **Total** | **9,610** | **9,610** | **12,451** |

All final labels use one class:

```yaml
nc: 1
names:
  0: license_plate
```

---

## Data-cleaning summary

| Item | Count |
|---|---:|
| Original images | 9,985 |
| Original annotation boxes | 86,294 |
| Images without a complete `plate` box removed | 359 |
| Plate-only candidate images | 9,626 |
| Images excluded by release decisions | 16 |
| Final images | 9,610 |
| Non-target annotation boxes removed | 73,826 |
| Original complete-plate boxes | 12,468 |
| Complete-plate boxes belonging to excluded images | 17 |
| Final complete-plate boxes | 12,451 |

The detailed accounting is stored in:

- `reports/preprocessing_audit.csv`
- `reports/class_mapping.csv`
- `reports/excluded_images.csv`

Eleven images were conservatively excluded following cross-split scene-similarity review. Five additional images remain excluded to preserve the previously validated release because their original omission reasons could not be recovered from the available project evidence.

---

## Preprocessing workflow

The preprocessing pipeline performs the following steps:

1. Reads the original class names from the source `data.yaml`.
2. Validates each YOLO annotation row.
3. Confirms that every row contains:
   - exactly five numeric values
   - a valid integer class ID
   - finite coordinates
   - positive width and height
   - a bounding box within normalized image boundaries
4. Selects the unique source class named `plate`.
5. Remaps the retained class to `0 license_plate`.
6. Removes all character-level and plate-style annotations.
7. Removes images that contain no complete-plate annotation.
8. Preserves the accepted train, validation, and test membership.
9. Converts the final YOLO labels to COCO format.
10. Generates manifests, hashes, statistics, reports, and figures.
11. Checks for exact and perceptual cross-split duplication candidates.
12. Validates the complete preprocessing release.

The stored images are not permanently resized, normalized, or expanded through offline augmentation.

---

## Train, validation, and test policy

The repository preserves an accepted project-controlled split of approximately 70/15/15.

| Split | Percentage of final images |
|---|---:|
| Train | 70.1% |
| Validation | 15.0% |
| Test | 14.9% |

The active membership is recorded in:

```text
reports/dataset_manifest.csv
```

The original random seed and exact historical split-generation procedure are not independently reconstructable from the committed evidence. Therefore, the repository validates the accepted membership instead of generating a new split.

- Training data is used for model learning.
- Validation data is reserved for model selection and tuning.
- Test data is reserved for final evaluation.
- Exact SHA-256 matches across different splits are treated as validation failures.

---

## Augmentation policy

The proposed augmentation settings are stored in:

```text
configs/augmentation_policy.yaml
```

The policy is intended for **training data only**.

| Transformation | Setting |
|---|---|
| Rotation | -5° to +5° |
| Scale | 0.90 to 1.10 |
| Translation | Up to 5% |
| Brightness | 0.80 to 1.20 |
| Contrast | 0.80 to 1.20 |
| Gaussian blur probability | 10% |
| Gaussian blur radius | 0.10 to 1.50 |
| Constrained crop probability | 10% |
| Minimum retained box visibility | 70% |
| Horizontal flip | Disabled |
| Vertical flip | Disabled |
| Random seed | 486 |

Horizontal and vertical flipping are disabled because mirrored or inverted UAE license plates would be unrealistic.

Validation and test images do not receive random augmentation.

The repository includes an augmentation preview, but it does not claim that augmentation has already been integrated into any model-training pipeline.

---

## Repository structure

```text
.
├── .gitignore
├── README.md
├── data.yaml
├── dataset_release.json
├── requirements.txt
├── annotations/
│   └── coco/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── configs/
│   └── augmentation_policy.yaml
├── datasets/
│   └── uae_lp_v2_yolo/
│       ├── data.yaml
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── notebooks/
│   └── 01_data_preprocessing.ipynb
├── reports/
│   ├── class_mapping.csv
│   ├── dataset_manifest.csv
│   ├── dataset_stats.csv
│   ├── excluded_images.csv
│   ├── preprocessing_audit.csv
│   ├── split_leakage_candidates.csv
│   ├── validation_report.md
│   └── figures/
└── scripts/
    ├── build_notebook.py
    ├── check_split_leakage.py
    ├── generate_reports.py
    ├── preprocess_dataset.py
    ├── preprocessing_utils.py
    ├── validate_dataset.py
    └── visualize_dataset.py
```

Dataset image files are intentionally excluded from GitHub and are shared separately with the project team.

The YOLO labels, COCO annotations, metadata, reports, scripts, and notebook remain in the repository.

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/azimaus970/UAE-License-Plate-Detection-Preprocessed.git
cd UAE-License-Plate-Detection-Preprocessed
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

#### macOS or Linux

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The validated notebook environment used Python 3.14.5.

---

## Dataset image placement

The separately shared image folders must be placed at:

```text
datasets/uae_lp_v2_yolo/images/train
datasets/uae_lp_v2_yolo/images/val
datasets/uae_lp_v2_yolo/images/test
```

Expected image counts:

| Split | Expected images |
|---|---:|
| Train | 6,738 |
| Validation | 1,440 |
| Test | 1,432 |
| **Total** | **9,610** |

These image directories are excluded through `.gitignore` and must not be committed to GitHub.

---

## Dataset configuration files

### Root configuration

The root `data.yaml` is intended for commands executed from the repository root:

```yaml
train: datasets/uae_lp_v2_yolo/images/train
val: datasets/uae_lp_v2_yolo/images/val
test: datasets/uae_lp_v2_yolo/images/test

nc: 1

names:
  0: license_plate
```

### Nested dataset configuration

The file `datasets/uae_lp_v2_yolo/data.yaml` uses paths relative to the dataset directory:

```yaml
train: images/train
val: images/val
test: images/test

nc: 1

names:
  0: license_plate
```

---

## Validation

### Repository validation

Repository mode validates the committed image-free preprocessing package:

```bash
python scripts/validate_dataset.py --mode repository
```

This checks:

- required repository files
- repository packaging hygiene
- both dataset YAML files
- all 9,610 YOLO label files
- all 12,451 complete-plate boxes
- label syntax, class IDs, dimensions, and normalized boundaries
- manifest membership
- committed label SHA-256 hashes
- recorded image SHA-256 uniqueness across splits
- COCO categories, references, dimensions, and areas
- YOLO-to-COCO membership
- YOLO-to-COCO parity within 0.01 pixels
- exclusion records
- preprocessing accounting
- release metadata and hashes
- augmentation-policy rules

Image-dependent checks are reported as not run when the images are absent.

### Full-dataset validation

After placing all separately shared images into the expected folders, run:

```bash
python scripts/validate_dataset.py --mode full
```

Full mode additionally checks:

- exact image membership
- successful decoding of all 9,610 images
- image dimensions
- image byte sizes
- recomputed image SHA-256 hashes
- current exact cross-split duplicates
- current perceptual-hash candidates
- regeneration of visual contact sheets

The committed full validation evidence reports:

- 9,610 images successfully decoded
- 12,451 valid bounding boxes
- 0 exact cross-split duplicates
- 0 perceptual-hash candidates at threshold 8
- YOLO-to-COCO parity within 0.01 pixels
- overall validation status: `PASS`

> **Note:** The current validator updates `reports/validation_report.md`. Run the intended validation mode before committing a regenerated report.

---

## Leakage checking

Repository mode checks recorded SHA-256 values:

```bash
python scripts/check_split_leakage.py --mode repository
```

Full mode recomputes image hashes and performs the difference-hash scan:

```bash
python scripts/check_split_leakage.py --mode full
```

The leakage script reports candidates only. It does not automatically delete, move, or reassign images.

---

## Raw-source audit and reconstruction

### Audit the accepted release

With the full accepted dataset available:

```bash
python scripts/preprocess_dataset.py \
  --audit-existing \
  --source-root path/to/raw/export
```

This audits the accepted release and regenerates:

- class mapping
- dataset manifest
- COCO annotations
- preprocessing accounting
- release metadata

### Reconstruct the release from the original multiclass export

```bash
python scripts/preprocess_dataset.py \
  --build-from-raw \
  --source-root path/to/raw/export \
  --staging-root release_artifacts/reconstructed
```

The raw source is expected to contain:

- 51 classes
- source class ID 50 named `plate`
- 9,985 image/label pairs
- 86,294 total annotation boxes
- 12,468 source `plate` boxes

The reconstruction process:

- keeps only the complete-plate class
- remaps it to class 0
- rebuilds labels using normalized YOLO coordinates
- copies the accepted images into a separate staging directory
- compares reconstructed image and label hashes with the manifest
- verifies YOLO-to-COCO parity
- never automatically replaces the accepted dataset

---

## Reports and figures

### Machine-readable reports

| File | Purpose |
|---|---|
| `reports/class_mapping.csv` | Original 51-class mapping and keep/remove decisions |
| `reports/dataset_manifest.csv` | Final membership, dimensions, counts, and hashes |
| `reports/dataset_stats.csv` | Dataset and bounding-box statistics |
| `reports/excluded_images.csv` | The 16 project-controlled exclusions |
| `reports/preprocessing_audit.csv` | Source-to-final cleaning accounting |
| `reports/split_leakage_candidates.csv` | Current duplicate/leakage candidates |
| `reports/validation_report.md` | Repository and full-validation evidence |

### Generated figures

The repository includes:

- split image and box counts
- boxes-per-image distribution
- relative bounding-box area distribution
- bounding-box aspect-ratio distribution
- train, validation, and test sample sheets
- smallest-box examples
- largest-box examples
- multi-plate examples
- edge-touching boxes
- training-only augmentation preview

To regenerate statistics and plots with the complete dataset:

```bash
python scripts/generate_reports.py
python scripts/visualize_dataset.py
```

---

## Preprocessing notebook

The executed notebook is located at:

```text
notebooks/01_data_preprocessing.ipynb
```

It presents:

- final dataset counts
- cleaning accounting
- source-to-target class mapping
- committed figures
- a YOLO-to-COCO conversion example
- exclusion accounting
- augmentation and model-format handoff notes
- course-material mapping
- repository validation output

The notebook is designed to open successfully from an image-free clone. Full image decoding and regeneration require the separately shared image dataset.

The notebook source can be rebuilt using:

```bash
python scripts/build_notebook.py
```

---

## Model handoff formats

This repository prepares the dataset for the project team's model-development repository.

### YOLO-compatible format

Use:

```text
data.yaml
datasets/uae_lp_v2_yolo/labels/
```

### COCO-compatible format

Use:

```text
annotations/coco/train.json
annotations/coco/val.json
annotations/coco/test.json
```

The proposed project models are:

- YOLO as the baseline
- RT-DETR as a transformer comparison
- RF-DETR as the proposed main real-time transformer model

These models are not implemented or trained in this preprocessing repository.

---

## Course relevance

| Repository component | Course connection |
|---|---|
| Dataset cleaning and preprocessing | Course project requirements |
| Train/validation/test separation | CNN training and evaluation |
| Normalization policy | CNN data preprocessing |
| Rotation, scale, translation, crop, brightness, and contrast | Training-time data augmentation |
| Gaussian blur | Image filtering |
| Full-object bounding boxes | Object detection |
| YOLO labels | Object-detection annotation format |
| COCO conversion | Model-format interoperability |
| SHA-256, perceptual hashing, manifests, and semantic versioning | Project engineering and reproducibility |

SIFT, corner detection, blob detection, feature matching, optical flow, edge detection, and multiview geometry are not claimed as parts of this preprocessing pipeline.

---

## Known limitations

- Dataset images are not stored in GitHub and must be shared separately.
- The original split-generation seed and exact historical grouping procedure are unavailable.
- Five frozen-release omissions have no recoverable original omission reason.
- The augmentation policy is documented and previewed but is not claimed as integrated into model training.
- This repository does not contain model training, testing, evaluation, inference, or deployment code.

---

## AI assistance acknowledgment

Generative AI tools, including ChatGPT and Codex, assisted with portions of the code structure, debugging, validation design, and documentation.

All reported dataset counts, annotation checks, manifests, hashes, and format-conversion checks are produced by the repository's scripts and were verified before release.
