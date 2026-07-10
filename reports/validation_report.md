# Dataset Validation Report

- Command: `C:\Users\kamsd\OneDrive\Desktop\uae-license-plate-detection\uae-license-plate-detection-final\.venv\Scripts\python.exe scripts\validate_dataset.py --package-root . --dataset-root datasets\uae_lp_v2_yolo`
- Execution date: 2026-07-10T14:06:06+04:00
- Dataset path: `C:\Users\kamsd\OneDrive\Desktop\uae-license-plate-detection\uae-license-plate-detection-final\datasets\uae_lp_v2_yolo`
- Floating-point bounds tolerance: `1e-06` normalized units
- YOLO-COCO parity tolerance: `0.01` pixel

## Check Results

| Check | Observed | Status |
|---|---|---|
| 1. Required directories exist | missing=0 | PASS |
| 2. Every image is fully decodable | decoded=9610, errors=0 | PASS |
| 3. No zero-byte images or labels | zero_byte_files=0 | PASS |
| 4. Every image has exactly one matching label | mismatches=0 | PASS |
| 5. Every label has exactly one matching image | mismatches=0 | PASS |
| 6. No duplicate image stems | duplicate_stem_issues=0 | PASS |
| 7. No duplicate filenames within or across splits | duplicate_filenames=0 | PASS |
| 8. Every YOLO row is numeric, valid, and finite | invalid_rows=0 | PASS |
| 9. Every class ID is zero | nonzero_rows=0 | PASS |
| 10. Every YOLO box stays within image bounds | tolerance=1e-06; violations=0 | PASS |
| 11. No unexplained empty labels | empty_labels=0 | PASS |
| 12. data.yaml paths, nc, and class name are correct | yaml_errors=0 | PASS |
| 13. COCO has exactly one license_plate category | category_issues=0 | PASS |
| 14. COCO image and annotation IDs are unique | image_ids=9610, annotation_ids=12451, issues=0 | PASS |
| 15. Every COCO annotation references an image | reference_issues=0 | PASS |
| 16. COCO dimensions equal decoded image dimensions | dimension_issues=0 | PASS |
| 17. Every COCO box is positive and inside the image | bbox_issues=0 | PASS |
| 18. Every COCO area equals width multiplied by height | area_issues=0 | PASS |
| 19. Every YOLO image appears once in the correct COCO split | membership_issues=0 | PASS |
| 20. Every YOLO box matches COCO within 0.01 pixel | parity_issues=0 | PASS |
| 21. No excluded image remains in a final split | excluded_rows=16, remaining=0 | PASS |
| 22. Exact SHA-256 image duplicates do not cross splits | images_hashed=9610, unique_hashes=9610, cross_split_groups=0 | PASS |
| 23. No split is empty | train=6738, val=1440, test=1432 | PASS |
| 24. Final release counts and hashes match dataset_release.json | release_issues=0 | PASS |
| 25. Accepted image and box counts match | train=6738/9415; val=1440/1525; test=1432/1511; total=9610/12451 | PASS |

## Observed Counts

| Split | Images | YOLO boxes | COCO images | COCO annotations |
|---|---:|---:|---:|---:|
| train | 6738 | 9415 | 6738 | 9415 |
| val | 1440 | 1525 | 1440 | 1525 |
| test | 1432 | 1511 | 1432 | 1511 |
| total | 9610 | 12451 | 9610 | 12451 |

## Split Leakage QA

- Historical review pairs: 43.
- Current perceptual/template candidates: 0.
- Unique flagged images in recorded candidates: 29; evaluation images: 11.
- Exact-duplicate candidate rows: 0.
- Difference-hash distance threshold: 8.
- The recorded issue is scene/template leakage: the vehicle and background may be reused while plate text differs.
- Difference hashing and Hamming distance are optional project QA and are not techniques taught in the supplied lectures.

## Overall Status

PASS
