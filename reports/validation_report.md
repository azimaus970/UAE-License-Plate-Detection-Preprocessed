# Dataset Validation Report

The GitHub source repository intentionally excludes dataset image files. Repository validation uses committed labels, manifests and COCO metadata; full validation uses the separately distributed images.

## Repository Validation

Command: `python scripts/validate_dataset.py --mode repository`

| Check | Observed | Status |
|---|---|---|
| Required repository files | missing=0 | PASS |
| Exactly one README.md | count=1 | PASS |
| Repository packaging hygiene | tracked_files=9648, issues=0 | PASS |
| Abandoned review artifacts are absent | remaining=0 | PASS |
| Legacy review language is absent | matches=0 | PASS |
| Absolute private paths are absent | matches=0 | PASS |
| Root and nested data.yaml semantics | issues=0 | PASS |
| Active manifest membership and split counts | rows=9610; counts={'train': {'images': 6738, 'boxes': 9415}, 'val': {'images': 1440, 'boxes': 1525}, 'test': {'images': 1432, 'boxes': 1511}, 'total': {'images': 9610, 'boxes': 12451}} | PASS |
| YOLO label syntax, class, bounds and nonempty rows | labels=9610, boxes=12451, issues=0 | PASS |
| Committed label SHA-256 values | checked=9610, issues=0 | PASS |
| Recorded image SHA-256 uniqueness across splits | cross_split_groups=0 | PASS |
| COCO category, IDs and references | category=0, ids=0, refs=0 | PASS |
| COCO dimensions, boxes and areas | issues=0 | PASS |
| YOLO-COCO membership | issues=0; counts={'train': {'images': 6738, 'boxes': 9415}, 'val': {'images': 1440, 'boxes': 1525}, 'test': {'images': 1432, 'boxes': 1511}} | PASS |
| YOLO-COCO 0.01-pixel parity | issues=0 | PASS |
| Sixteen project-owner exclusions | rows=16, active_overlap=0 | PASS |
| Preprocessing accounting | metrics=15, issues=0 | PASS |
| Source class mapping | rows=51, non_target_removed=73826 | PASS |
| Release counts and manifest/COCO hashes | manifest=f482a802f008ee434a97cc416b2db2ae769b3225bb46873048a8bc345d2ed9f4; coco={'train': '6a81fc7ebda7b2a768747c2ce6d756ea549bbb2aa92c44943d88f330240d594f', 'val': 'bcea301485e6f46861b20da7dd5a85522b4bc2be7db0176f63344428b469d0fb', 'test': 'ca2c8e962e8a756836bd40717156953b1d315054105937d23128eced2bd163f9'}; issues=0 | PASS |
| Training-only augmentation policy | issues=0 | PASS |
| Recorded exact cross-split duplicates | count=0 | PASS |

## Full Dataset Validation Evidence

Command: `python scripts/validate_dataset.py --mode full`

| Check | Observed | Status |
|---|---|---|
| Actual image membership | issues=0 | PASS |
| Actual image decoding | decoded=9610, issues=0 | PASS |
| Actual image-size verification | checked=9610, issues=0 | PASS |
| Actual image SHA-256 recomputation | checked=9610, issues=0 | PASS |
| Current exact cross-split duplicates | count=0 | PASS |
| Current perceptual-hash scan | candidates=0; threshold=8 | PASS |
| Visual contact-sheet regeneration | figures=8, issues=0 | PASS |

## Accepted Counts

| Split | Images | Boxes |
|---|---:|---:|
| train | 6738 | 9415 |
| val | 1440 | 1525 |
| test | 1432 | 1511 |
| total | 9610 | 12451 |

## Integrity and Leakage Summary

- Manifest SHA-256: `f482a802f008ee434a97cc416b2db2ae769b3225bb46873048a8bc345d2ed9f4` (PASS)
- COCO SHA-256 values: `{"test": "ca2c8e962e8a756836bd40717156953b1d315054105937d23128eced2bd163f9", "train": "6a81fc7ebda7b2a768747c2ce6d756ea549bbb2aa92c44943d88f330240d594f", "val": "bcea301485e6f46861b20da7dd5a85522b4bc2be7db0176f63344428b469d0fb"}` (PASS)
- Exact cross-split duplicates: 0 (PASS)
- Current perceptual scan: 0
- Crop-heavy training examples remain intentionally retained as valid full-plate training examples.

## Overall Status

PASS
