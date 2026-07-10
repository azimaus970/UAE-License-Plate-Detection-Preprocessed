"""Build the image-free, executable preprocessing evidence notebook."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build() -> dict:
    cells = [
        markdown(
            """# UAE full-plate preprocessing evidence

This executed notebook is designed for a clean, image-free GitHub clone. It reads committed labels, manifests, COCO annotations, release metadata, audit tables, and saved figures. Full image decoding and current perceptual hashing require the separately distributed full dataset."""
        ),
        code(
            r"""from pathlib import Path
import csv
import json
import subprocess
import sys
from collections import Counter
from IPython.display import Markdown, display

def discover_package_root():
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "dataset_release.json").is_file() and (candidate / "scripts" / "validate_dataset.py").is_file():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from committed marker files")

def read_csv(relative):
    with (PACKAGE_ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

PACKAGE_ROOT = discover_package_root()
release = json.loads((PACKAGE_ROOT / "dataset_release.json").read_text(encoding="utf-8"))
manifest = read_csv("reports/dataset_manifest.csv")
audit = read_csv("reports/preprocessing_audit.csv")
class_mapping = read_csv("reports/class_mapping.csv")
exclusions = read_csv("reports/excluded_images.csv")
print("Repository markers found; committed evidence loaded.")
print("Release version:", release["semantic_version"])
print("Package type:", release["package_type"])"""
        ),
        markdown("""## Frozen release membership

The split is frozen project-controlled approximately 70/15/15 membership recorded by `reports/dataset_manifest.csv`. The original membership-generation seed and exact grouping algorithm are not independently reconstructable from committed evidence, so this notebook verifies the active membership instead of resplitting it."""),
        code(
            r"""label_counts = Counter()
box_counts = Counter()
class_ids = Counter()
empty_labels = []

for row in manifest:
    split = row["split"]
    label_path = PACKAGE_ROOT / row["label_relative_path"]
    lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    label_counts[split] += 1
    box_counts[split] += len(lines)
    if not lines:
        empty_labels.append(row["label_relative_path"])
    for line in lines:
        tokens = line.split()
        assert len(tokens) == 5
        class_ids[int(float(tokens[0]))] += 1

observed = {
    split: {"images": label_counts[split], "boxes": box_counts[split]}
    for split in ("train", "val", "test")
}
observed["total"] = {
    "images": sum(item["images"] for item in observed.values()),
    "boxes": sum(item["boxes"] for item in observed.values()),
}
assert observed == {
    "train": {"images": 6738, "boxes": 9415},
    "val": {"images": 1440, "boxes": 1525},
    "test": {"images": 1432, "boxes": 1511},
    "total": {"images": 9610, "boxes": 12451},
}
assert class_ids == {0: 12451} and not empty_labels
display(Markdown("| Split | Images | Boxes |\n|---|---:|---:|\n" + "\n".join(
    f"| {split} | {values['images']:,} | {values['boxes']:,} |" for split, values in observed.items()
)))"""
        ),
        code(
            r"""audit_values = {row["metric"]: row["value"] for row in audit}
selected_metrics = [
    "source_images", "no_plate_images_removed", "plate_only_candidate_images",
    "release_decision_images_excluded", "final_images", "source_boxes",
    "non_target_boxes_removed", "source_plate_boxes",
    "plate_boxes_excluded_with_release_decisions", "final_plate_boxes",
    "total_boxes_not_in_final_release",
]
display(Markdown("## Committed cleaning accounting\n\n| Metric | Value |\n|---|---:|\n" + "\n".join(
    f"| {metric} | {int(audit_values[metric]):,} |" for metric in selected_metrics
)))

plate_mapping = [row for row in class_mapping if row["source_class_name"] == "plate"]
assert len(plate_mapping) == 1
print("Source plate class:", plate_mapping[0]["source_class_id"])
print("Source plate boxes:", plate_mapping[0]["source_box_count"])
print("Final class-0 boxes:", plate_mapping[0]["boxes_kept"])"""
        ),
        markdown("""## Committed figures

The statistical plots and contact sheets below are committed artifacts. Image-dependent regeneration is performed only in full validation."""),
        code(
            r"""figures = [
    ("Split counts", "split_counts.png"),
    ("Boxes per image", "boxes_per_image.png"),
    ("Relative box area", "relative_box_area.png"),
    ("Box aspect ratio", "box_aspect_ratio.png"),
    ("Train samples", "train_random_samples.jpg"),
    ("Validation samples", "val_random_samples.jpg"),
    ("Test samples", "test_random_samples.jpg"),
    ("Smallest boxes", "smallest_boxes.jpg"),
    ("Largest boxes", "largest_boxes.jpg"),
    ("Multi-plate images", "multi_plate_images.jpg"),
    ("Edge boxes", "edge_touching_boxes.jpg"),
    ("Training augmentation preview", "augmentation_preview.jpg"),
]
for _, filename in figures:
    assert (PACKAGE_ROOT / "reports" / "figures" / filename).is_file()
display(Markdown("\n\n".join(f"### {title}\n\n![{title}](../reports/figures/{filename})" for title, filename in figures)))"""
        ),
        markdown("""## YOLO-to-COCO example without image files

COCO stores the committed image dimensions, so normalized YOLO coordinates can be converted to pixel coordinates in an image-free clone."""),
        code(
            r"""example = manifest[0]
split = example["split"]
label_line = (PACKAGE_ROOT / example["label_relative_path"]).read_text(encoding="utf-8").splitlines()[0]
_, xc, yc, width, height = [float(value) for value in label_line.split()]
image_width, image_height = int(example["image_width"]), int(example["image_height"])
converted = [
    xc * image_width - width * image_width / 2,
    yc * image_height - height * image_height / 2,
    width * image_width,
    height * image_height,
]
coco = json.loads((PACKAGE_ROOT / "annotations" / "coco" / f"{split}.json").read_text(encoding="utf-8"))
coco_image = next(item for item in coco["images"] if Path(item["file_name"]).name == Path(example["image_relative_path"]).name)
coco_box = next(item["bbox"] for item in coco["annotations"] if item["image_id"] == coco_image["id"])
max_delta = max(abs(left - right) for left, right in zip(converted, coco_box))
assert max_delta <= 0.01
print("Example dimensions:", image_width, "x", image_height)
print("Converted COCO bbox:", converted)
print("Maximum YOLO-COCO delta in pixels:", max_delta)"""
        ),
        markdown("""## Sixteen project-owner exclusions

The accepted release excludes 16 images. Eleven are conservative cross-split scene-similarity exclusions; five remain omitted to preserve frozen membership because their original omission provenance is unavailable. These decisions do not assert corruption, duplication, unreadability, or mislabeling."""),
        code(
            r"""reason_counts = Counter(row["reason_category"] for row in exclusions)
status_counts = Counter(row["decision_status"] for row in exclusions)
authority_counts = Counter(row["decision_authority"] for row in exclusions)
assert len(exclusions) == 16
assert reason_counts == {
    "conservative_cross_split_scene_similarity_exclusion": 11,
    "frozen_release_omission_provenance_unavailable": 5,
}
assert status_counts == {"EXCLUDED_BY_PROJECT_DECISION": 16}
assert authority_counts == {"project_owner": 16}
print("Exclusion reasons:", dict(reason_counts))
print("Decision status:", dict(status_counts))
print("Decision authority:", dict(authority_counts))"""
        ),
        markdown("""## Proposed model handoff and augmentation

- YOLO is the proposed baseline and consumes the YOLO labels plus `data.yaml`.
- RT-DETR is a proposed detection-transformer comparison.
- RF-DETR is the proposed main real-time transformer model.
- Detection-transformer implementations can consume the COCO annotations.
- No model training is implemented in this preprocessing repository.
- Normalization and deterministic resizing are model/weight specific.
- Random augmentation is training-only; validation and test receive none.
- Horizontal and vertical flips are disabled.
- Crop-heavy training examples are intentionally retained as valid full-plate examples."""),
        markdown("""## Course mapping

| Topic | Course material |
|---|---|
| Data cleaning | Project Guidelines |
| Train/validation/test separation | 11-TrainingCNN |
| Normalization | 11-TrainingCNN and the professor's pretrained-model notebook |
| Rotation, scale, and crop augmentation | 11-TrainingCNN and professor 12-trainingCNN.ipynb |
| Gaussian blur | 03-Filtering |
| Full-object bounding boxes and YOLO | 13-Detection&Segmentation |
| SHA-256, difference hashing, COCO conversion, semantic versioning, manifest hashing | Project engineering, not directly taught lecture techniques |

SIFT, corners, blobs, optical flow, multiview geometry, and edge detection are not part of this preprocessing pipeline."""),
        markdown("""## AI assistance acknowledgment

Generative AI tools, including ChatGPT and Codex, assisted with portions of code structure, debugging, validation design, and documentation. AI use is permitted for the course project with acknowledgment. All reported dataset counts and annotation checks are produced by the repository's validation code and were independently verified before release."""),
        markdown("""## Repository validation

Repository mode validates the complete committed metadata and annotation contract without images. Full image decoding, actual image hash recomputation, current perceptual hashing, and visual regeneration require the separate full dataset."""),
        code(
            r"""command = [
    sys.executable,
    str(PACKAGE_ROOT / "scripts" / "validate_dataset.py"),
    "--package-root", str(PACKAGE_ROOT),
    "--mode", "repository",
]
completed = subprocess.run(command, cwd=PACKAGE_ROOT, text=True, capture_output=True)
print(completed.stdout.strip())
if completed.returncode:
    raise RuntimeError(completed.stderr.strip() or "Repository validation failed")
assert '"status": "PASS"' in completed.stdout
print("Notebook finished with repository validation PASS.")"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.5"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    output = package_root / "notebooks" / "01_data_preprocessing.ipynb"
    output.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {output.relative_to(package_root)}")


if __name__ == "__main__":
    main()
