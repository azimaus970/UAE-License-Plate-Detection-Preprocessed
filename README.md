# UAE License Plate Detection: Preprocessing Handoff

### Drive Link : 

## 1. Project purpose

This folder is the dataset handoff for a course project on detecting complete visible UAE license plates. It supplies one accepted single-class dataset in YOLO and COCO formats. It does not train YOLO, RF-DETR, RT-DETR, or any other model.

The group’s final GitHub repository must separately provide the training and testing commands required by the project guidelines.

## 2. Preprocessing scope

The reproducible workflow reads the original local YOLO class schema and labels, keeps only complete plate boxes, preserves the accepted train/validation/test membership, validates every final image and annotation, converts final YOLO labels to COCO, generates release evidence, and creates visualization and augmentation previews. Saved source images are not permanently normalized or expanded with offline augmentations.

## 3. Dataset source 

- Dataset: UAE (<https://universe.roboflow.com/addinguae/uae-zcfqj>)

## 4. Target definition: full visible license plate only

The final variable is:

```text
0 license_plate
```

Only a bounding box around the complete visible plate is a target. OCR digits, letters, expiration marks, emirate names, and old/new plate-style boxes are removed. OCR and plate-text recognition are outside this preprocessing deliverable.

## 5. Exact final directory structure

```text
uae-license-plate-detection-final/
|-- .gitignore
|-- README.md
|-- data.yaml
|-- dataset_release.json
|-- requirements.txt
|-- annotations/
|   `-- coco/{train,val,test}.json
|-- configs/
|   `-- augmentation_policy.yaml
|-- datasets/
|   `-- uae_lp_v2_yolo/
|       |-- data.yaml
|       |-- images/{train,val,test}/
|       `-- labels/{train,val,test}/
|-- notebooks/
|   `-- 01_data_preprocessing.ipynb
|-- reports/
|   |-- class_mapping.csv
|   |-- dataset_manifest.csv
|   |-- dataset_stats.csv
|   |-- excluded_images.csv
|   |-- manual_inspection.csv
|   |-- preprocessing_audit.csv
|   |-- split_leakage_candidates.csv
|   |-- validation_report.md
|   |-- near_duplicate_review_decisions.csv
|   |-- near_duplicate_review_pairs/pair_001.jpg ... pair_043.jpg
|   `-- figures/
`-- scripts/
    |-- preprocess_dataset.py
    |-- validate_dataset.py
    |-- check_split_leakage.py
    |-- generate_reports.py
    |-- visualize_dataset.py
    `-- preprocessing_utils.py
```

`reports/final_dataset_manifest.csv` may also be retained as historical machine-generated evidence for reconstructing accepted-release omissions; the active manifest is `reports/dataset_manifest.csv`.

## 6. Raw-to-final preprocessing steps

1. Read source class IDs and names from the actual source `data.yaml`.
2. Validate every source YOLO row: five numeric finite values, integer class ID, positive size, normalized center, and a complete in-bounds box with a documented `1e-6` tolerance.
3. Keep only source class 50, `plate`; map it to `0 license_plate`.
4. Remove all non-target classes and drop images with no complete-plate box. Historical local evidence records 9,985 source images, 86,294 source boxes, 359 no-plate images, 73,826 non-target boxes removed, 9,626 plate-only images, and 12,468 plate boxes.
5. Apply the accepted project-controlled split and preserve current release membership. The current accepted release is 16 images and 17 boxes smaller than the 9,626-image split candidate.
6. Consume `reports/excluded_images.csv` deterministically. Eleven absent evaluation images are linked to the historical 43-pair CSV. Five additional accepted-release omissions have no reconstructable pair-specific reason and remain explicitly marked as such.
7. Decode final images, validate final class-0 labels, generate COCO directly from those labels, and generate the manifest, hashes, audit, statistics, and figures.

The pipeline never edits validation or test merely to make counts pass. An ambiguous source target class causes a failure before the accepted dataset is altered.

## 7. Source-to-target class mapping

The source distribution is established from actual labels, not this README:

| Source IDs | Actual names | Source boxes | Decision |
|---|---|---:|---|
| 0–9 | `0` through `9` | 53,935 | remove OCR digits |
| 10–35 | `A` through `Z` | 8,949 | remove OCR letters |
| 36 | `exp` | 3 | remove non-target mark; the exact full-plate class is separately named `plate` |
| 37–49 | `new_*` and `old_*` emirate/style classes | 10,939 | remove non-target emirate/style boxes |
| 50 | `plate` | 12,468 | keep and map to `0 license_plate`; 12,451 boxes remain in the accepted release |

The row-by-row mapping and counts are in `reports/class_mapping.csv`.

## 8. Train/validation/test methodology

The original export had train and test data but no populated validation directory. Historical project files prove that `scripts\v2\create_source_aware_split.py` created a deterministic 70/15/15 candidate split from the plate-only manifest with default split seed 486; the recorded rebuild command omitted `--seed` and therefore used that default. It placed crop-heavy and flagged samples in training when cleaner evaluation samples were available.

Validation remains separate for model selection and tuning. Test remains separate for the final unbiased evaluation. The same accepted membership must be used for YOLO and COCO/RF-DETR consumers.

## 9. Final counts

| Split | Images | Label files | Boxes |
|---|---:|---:|---:|
| Train | 6,738 | 6,738 | 9,415 |
| Validation | 1,440 | 1,440 | 1,525 |
| Test | 1,432 | 1,432 | 1,511 |
| Total | 9,610 | 9,610 | 12,451 |

These are release acceptance values. The validator must observe them from actual files before it reports `PASS`.

## 10. Scene/template leakage handling

SHA-256 checks exact byte duplicates across splits. The optional Pillow 16×16 difference hash and Hamming distance threshold 8 provide additional conservative QA. The historical 43 candidate pairs are best described as scene/template leakage: the same vehicle and background may be reused while plate text differs.

The 43 pair images are preserved under `reports/near_duplicate_review_pairs`. The scripts never automatically remove an image or fill a human reviewer. `NEEDS_HUMAN_REVIEW` remains in the ledgers until a real reviewer records a pair-specific decision.

Perceptual hashing and Hamming distance are useful project engineering checks, but they are not techniques taught in the supplied lectures.

## 11. Augmentation policy

`configs/augmentation_policy.yaml` defines mild training-only rotation, scale, translation, brightness, contrast, Gaussian blur, and constrained crop. Every geometric transformation updates every bounding box, and a sample is rejected if any plate box retains less than 70% visibility.

- Random transformations: training only.
- Validation random augmentation: disabled.
- Test random augmentation: disabled.
- Horizontal flip: disabled because mirrored UAE plates are unrealistic.
- Vertical flip: disabled because inverted road scenes and plates are unrealistic.
- Normalization and deterministic resizing: model-specific, not permanently applied to saved images.

The preview is evidence only and does not create a new dataset split.

### Augmentation ablation handoff

| Condition | Preprocessing condition | Split contract |
|---|---|---|
| `augmentation_off` | deterministic model preprocessing only | same accepted train, validation, and test split |
| `augmentation_on` | documented training-only augmentation policy | same accepted train, validation, and test split; validation/test are not duplicated or augmented |

Read-only training-code inspection found that the team YOLO script uses Ultralytics defaults unless `--no-augmentation` is selected; it does not load this exact policy. No RF-DETR training implementation was found. The available `train_rtdetr_ultralytics.py` is RT-DETR, not RF-DETR, and also uses framework defaults without loading this policy. Training code was not changed in this preprocessing task.

## 12. Validation checks

`scripts\validate_dataset.py` fails nonzero for missing/empty data, unreadable images, zero-byte files, image-label pairing errors, duplicate stems or filenames, invalid/nonfinite/out-of-bounds YOLO rows, nonzero class IDs, unexplained empty labels, invalid YAML, COCO category/ID/reference/dimension/box/area problems, YOLO–COCO membership or 0.01-pixel parity failures, excluded images still present, cross-split exact duplicates, empty splits, release hash/count mismatches, or accepted-count mismatches.

The complete executed command, timestamp, each check, observed counts, and final status are written only to `reports/validation_report.md`.

## 13. Reproducible Windows commands

Tested with Python 3.14.5 and the exact package versions in `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

.venv\Scripts\python.exe scripts\preprocess_dataset.py --source-root ..\datasets\raw_v2\addinguae_roboflow --dataset-root datasets\uae_lp_v2_yolo --package-root . --exclusions reports\excluded_images.csv --dry-run
.venv\Scripts\python.exe scripts\preprocess_dataset.py --source-root ..\datasets\raw_v2\addinguae_roboflow --dataset-root datasets\uae_lp_v2_yolo --package-root . --exclusions reports\excluded_images.csv
.venv\Scripts\python.exe scripts\validate_dataset.py --package-root . --dataset-root datasets\uae_lp_v2_yolo
.venv\Scripts\python.exe scripts\check_split_leakage.py --package-root . --dataset-root datasets\uae_lp_v2_yolo
.venv\Scripts\python.exe scripts\generate_reports.py --package-root . --dataset-root datasets\uae_lp_v2_yolo
.venv\Scripts\python.exe scripts\visualize_dataset.py --package-root . --dataset-root datasets\uae_lp_v2_yolo
.venv\Scripts\python.exe scripts\validate_dataset.py --package-root . --dataset-root datasets\uae_lp_v2_yolo
```

## 14. YOLO handoff instructions

Use root `data.yaml`, whose paths point to `datasets/uae_lp_v2_yolo/images/{train,val,test}`. A training program may perform its own deterministic resize and normalization at runtime. It must not merge validation with test or permanently normalize the stored images.

## 15. RF-DETR COCO handoff instructions

Use `annotations/coco/train.json`, `val.json`, and `test.json` with the corresponding images under `datasets/uae_lp_v2_yolo/images`. COCO category ID 1 is `license_plate`; every COCO image uses license ID 1. File names are stored as `images/train/...`, `images/val/...`, or `images/test/...` relative to the YOLO dataset root.

No RF-DETR training code is present in the inspected team release, so model-specific RF-DETR resizing, normalization, augmentation, and command-line wiring remain a separate training handoff task.
