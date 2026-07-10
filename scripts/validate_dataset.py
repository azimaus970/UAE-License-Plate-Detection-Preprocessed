"""Strictly validate the accepted YOLO/COCO dataset and write one complete report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image

from preprocessing_utils import (
    ACCEPTED_COUNTS,
    BOUND_TOLERANCE,
    COCO_PARITY_TOLERANCE_PX,
    SPLITS,
    TARGET_CLASS_ID,
    TARGET_CLASS_NAME,
    find_image_files,
    find_label_files,
    load_yaml,
    parse_yolo_line,
    read_csv,
    required_dataset_directories,
    sha256_file,
    yolo_to_coco_bbox,
)

EXCLUSION_COLUMNS = [
    "image_relative_path",
    "original_split",
    "reason_category",
    "related_pair_ids",
    "evidence_file",
    "decision_status",
    "human_reviewer",
    "review_date",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/uae_lp_v2_yolo"))
    return parser.parse_args()


def _resolve(package_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (package_root / value).resolve()


def _result(name: str, observed: str, errors: list[str]) -> dict[str, object]:
    return {"check": name, "observed": observed, "status": "FAIL" if errors else "PASS", "errors": errors}


def _validate_data_yaml(path: Path, package_root: Path, dataset_root: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = load_yaml(path)
    except Exception as exc:
        return [str(exc)]
    if data.get("nc") != 1:
        errors.append(f"{path}: nc must equal 1")
    names = data.get("names")
    if isinstance(names, dict):
        try:
            actual_names = [str(names[key]) for key in sorted(names, key=lambda key: int(key))]
        except Exception:
            actual_names = []
    elif isinstance(names, list):
        actual_names = [str(value) for value in names]
    else:
        actual_names = []
    if actual_names != [TARGET_CLASS_NAME]:
        errors.append(f"{path}: names must be exactly 0: {TARGET_CLASS_NAME}")
    for split in SPLITS:
        value = data.get(split)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{path}: missing {split} path")
            continue
        candidates = [(path.parent / value).resolve(), (package_root / value).resolve()]
        expected = (dataset_root / "images" / split).resolve()
        if expected not in candidates:
            errors.append(f"{path}: {split} path `{value}` does not resolve to {expected}")
    return errors


def _load_labels(path: Path) -> tuple[list, list[str]]:
    boxes = []
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return boxes, [f"{path}: cannot read label: {exc}"]
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            boxes.append(parse_yolo_line(line, path, line_number))
        except ValueError as exc:
            errors.append(str(exc))
    return boxes, errors


def _leakage_summary(package_root: Path) -> list[str]:
    path = package_root / "reports" / "split_leakage_candidates.csv"
    if not path.is_file():
        return ["Leakage script has not yet produced reports/split_leakage_candidates.csv."]
    rows = read_csv(path)
    historical = [row for row in rows if row.get("source") == "historical_review_pair"]
    current = [row for row in rows if row.get("source") == "current_difference_hash"]
    exact = [row for row in rows if row.get("exact_duplicate", "").casefold() == "true"]
    flagged = {
        row[key]
        for row in rows
        for key in ("image_a", "image_b")
        if row.get(key)
    }
    evaluation = {
        row[key]
        for row in rows
        for key, split_key in (("image_a", "split_a"), ("image_b", "split_b"))
        if row.get(key) and row.get(split_key) in {"val", "test"}
    }
    thresholds = sorted({row.get("distance_threshold", "") for row in rows if row.get("distance_threshold", "")})
    return [
        f"Historical review pairs: {len(historical)}.",
        f"Current perceptual/template candidates: {len(current)}.",
        f"Unique flagged images in recorded candidates: {len(flagged)}; evaluation images: {len(evaluation)}.",
        f"Exact-duplicate candidate rows: {len(exact)}.",
        f"Difference-hash distance threshold: {', '.join(thresholds) if thresholds else 'NOT_AVAILABLE'}.",
        "The recorded issue is scene/template leakage: the vehicle and background may be reused while plate text differs.",
        "Difference hashing and Hamming distance are optional project QA and are not techniques taught in the supplied lectures.",
    ]


def run_validation(package_root: Path, dataset_root: Path, command: str, *, write_report: bool = True) -> dict[str, object]:
    package_root = package_root.resolve()
    dataset_root = dataset_root.resolve()
    results: list[dict[str, object]] = []
    all_errors: list[str] = []

    missing_dirs = [str(path) for path in required_dataset_directories(dataset_root) if not path.is_dir()]
    errors = [f"Missing required directory: {path}" for path in missing_dirs]
    results.append(_result("1. Required directories exist", f"missing={len(missing_dirs)}", errors))
    all_errors.extend(errors)

    images_by_split = {split: find_image_files(dataset_root / "images" / split) for split in SPLITS}
    labels_by_split = {split: find_label_files(dataset_root / "labels" / split) for split in SPLITS}
    dimensions: dict[tuple[str, str], tuple[int, int]] = {}
    image_hashes: dict[tuple[str, str], str] = {}
    decode_errors: list[str] = []
    zero_errors: list[str] = []
    image_tasks = [(split, image_path) for split in SPLITS for image_path in images_by_split[split]]

    def decode_image(task: tuple[str, Path]):
        split, image_path = task
        if image_path.stat().st_size == 0:
            return split, image_path, None, None, f"Zero-byte image: {image_path}"
        try:
            payload = image_path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                size = image.size
            return split, image_path, size, digest, None
        except Exception as exc:
            return split, image_path, None, None, f"Unreadable image {image_path}: {exc}"

    workers = min(32, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for split, image_path, size, digest, error in executor.map(decode_image, image_tasks):
            if error:
                if error.startswith("Zero-byte"):
                    zero_errors.append(error)
                else:
                    decode_errors.append(error)
                continue
            dimensions[(split, image_path.name)] = size
            image_hashes[(split, image_path.name)] = digest
    for split in SPLITS:
        for label_path in labels_by_split[split]:
            if label_path.stat().st_size == 0:
                zero_errors.append(f"Zero-byte label: {label_path}")
    results.append(_result("2. Every image is fully decodable", f"decoded={len(dimensions)}, errors={len(decode_errors)}", decode_errors))
    results.append(_result("3. No zero-byte images or labels", f"zero_byte_files={len(zero_errors)}", zero_errors))
    all_errors.extend(decode_errors + zero_errors)

    image_label_errors: list[str] = []
    label_image_errors: list[str] = []
    stem_errors: list[str] = []
    for split in SPLITS:
        image_stems = Counter(path.stem.casefold() for path in images_by_split[split])
        label_stems = Counter(path.stem.casefold() for path in labels_by_split[split])
        for stem, count in image_stems.items():
            if label_stems[stem] != 1:
                image_label_errors.append(f"{split}: image stem {stem} has {label_stems[stem]} matching labels")
            if count != 1:
                stem_errors.append(f"{split}: duplicate image stem {stem} occurs {count} times")
        for stem, count in label_stems.items():
            if image_stems[stem] != 1:
                label_image_errors.append(f"{split}: label stem {stem} has {image_stems[stem]} matching images")
            if count != 1:
                stem_errors.append(f"{split}: duplicate label stem {stem} occurs {count} times")
    global_stems = Counter(path.stem.casefold() for split in SPLITS for path in images_by_split[split])
    for stem, count in global_stems.items():
        if count > 1:
            stem_errors.append(f"Across splits: image stem {stem} occurs {count} times")
    results.append(_result("4. Every image has exactly one matching label", f"mismatches={len(image_label_errors)}", image_label_errors))
    results.append(_result("5. Every label has exactly one matching image", f"mismatches={len(label_image_errors)}", label_image_errors))
    results.append(_result("6. No duplicate image stems", f"duplicate_stem_issues={len(stem_errors)}", stem_errors))
    all_errors.extend(image_label_errors + label_image_errors + stem_errors)

    filenames = Counter(path.name.casefold() for split in SPLITS for path in images_by_split[split])
    filename_errors = [f"Duplicate image filename within/across splits: {name} ({count})" for name, count in filenames.items() if count > 1]
    results.append(_result("7. No duplicate filenames within or across splits", f"duplicate_filenames={len(filename_errors)}", filename_errors))
    all_errors.extend(filename_errors)

    boxes_by_image: dict[tuple[str, str], list] = {}
    row_errors: list[str] = []
    class_errors: list[str] = []
    bounds_errors: list[str] = []
    empty_errors: list[str] = []
    for split in SPLITS:
        for image_path in images_by_split[split]:
            label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            boxes, label_errors = _load_labels(label_path)
            row_errors.extend(label_errors)
            bounds_errors.extend(error for error in label_errors if "bounds" in error)
            if not boxes:
                empty_errors.append(f"Unexplained empty label: {label_path}")
            for box in boxes:
                if box.class_id != TARGET_CLASS_ID:
                    class_errors.append(f"Nonzero class ID {box.class_id}: {label_path}")
            boxes_by_image[(split, image_path.name)] = boxes
    results.append(_result("8. Every YOLO row is numeric, valid, and finite", f"invalid_rows={len(row_errors)}", row_errors))
    results.append(_result("9. Every class ID is zero", f"nonzero_rows={len(class_errors)}", class_errors))
    results.append(_result("10. Every YOLO box stays within image bounds", f"tolerance={BOUND_TOLERANCE:g}; violations={len(bounds_errors)}", bounds_errors))
    results.append(_result("11. No unexplained empty labels", f"empty_labels={len(empty_errors)}", empty_errors))
    all_errors.extend(row_errors + class_errors + empty_errors)

    yaml_errors = _validate_data_yaml(package_root / "data.yaml", package_root, dataset_root)
    yaml_errors.extend(_validate_data_yaml(dataset_root / "data.yaml", package_root, dataset_root))
    results.append(_result("12. data.yaml paths, nc, and class name are correct", f"yaml_errors={len(yaml_errors)}", yaml_errors))
    all_errors.extend(yaml_errors)

    coco_category_errors: list[str] = []
    coco_id_errors: list[str] = []
    coco_reference_errors: list[str] = []
    coco_dimension_errors: list[str] = []
    coco_bounds_errors: list[str] = []
    coco_area_errors: list[str] = []
    coco_membership_errors: list[str] = []
    parity_errors: list[str] = []
    coco_counts: dict[str, dict[str, int]] = {}
    global_image_ids: set[int] = set()
    global_annotation_ids: set[int] = set()
    for split in SPLITS:
        coco_path = package_root / "annotations" / "coco" / f"{split}.json"
        if not coco_path.is_file():
            coco_category_errors.append(f"Missing COCO file: {coco_path}")
            coco_counts[split] = {"images": 0, "boxes": 0}
            continue
        try:
            data = json.loads(coco_path.read_text(encoding="utf-8"))
        except Exception as exc:
            coco_category_errors.append(f"Cannot parse {coco_path}: {exc}")
            coco_counts[split] = {"images": 0, "boxes": 0}
            continue
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        expected_category = [{"id": 1, "name": TARGET_CLASS_NAME, "supercategory": TARGET_CLASS_NAME}]
        if categories != expected_category:
            coco_category_errors.append(f"{coco_path}: categories={categories}, expected={expected_category}")
        image_ids: dict[int, dict] = {}
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for image in images:
            image_id = image.get("id")
            if not isinstance(image_id, int) or image_id in global_image_ids:
                coco_id_errors.append(f"{coco_path}: duplicate/invalid image ID {image_id}")
            else:
                global_image_ids.add(image_id)
                image_ids[image_id] = image
            if image.get("license") != 1:
                coco_reference_errors.append(f"{coco_path}: image {image_id} must use license ID 1")
        for annotation in annotations:
            annotation_id = annotation.get("id")
            if not isinstance(annotation_id, int) or annotation_id in global_annotation_ids:
                coco_id_errors.append(f"{coco_path}: duplicate/invalid annotation ID {annotation_id}")
            else:
                global_annotation_ids.add(annotation_id)
            image_id = annotation.get("image_id")
            if image_id not in image_ids:
                coco_reference_errors.append(f"{coco_path}: annotation {annotation_id} references missing image {image_id}")
                continue
            annotations_by_image[image_id].append(annotation)
        coco_names: Counter[str] = Counter()
        for image_id, image in image_ids.items():
            file_name = str(image.get("file_name", ""))
            expected_prefix = f"images/{split}/"
            if not file_name.startswith(expected_prefix):
                coco_membership_errors.append(f"{coco_path}: wrong file_name split path {file_name}")
            name = Path(file_name).name
            coco_names[name.casefold()] += 1
            key = (split, name)
            actual_size = dimensions.get(key)
            if actual_size is None:
                coco_membership_errors.append(f"{coco_path}: no matching YOLO image for {file_name}")
                continue
            width, height = actual_size
            if image.get("width") != width or image.get("height") != height:
                coco_dimension_errors.append(f"{coco_path}: dimensions differ for {name}")
            expected_boxes = boxes_by_image.get(key, [])
            actual_annotations = sorted(annotations_by_image.get(image_id, []), key=lambda ann: ann.get("id", -1))
            if len(expected_boxes) != len(actual_annotations):
                parity_errors.append(f"{coco_path}: {name} YOLO boxes={len(expected_boxes)}, COCO boxes={len(actual_annotations)}")
            for index, annotation in enumerate(actual_annotations):
                bbox = annotation.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    coco_bounds_errors.append(f"{coco_path}: annotation {annotation.get('id')} bbox must contain four values")
                    continue
                try:
                    x, y, box_width, box_height = [float(value) for value in bbox]
                except (TypeError, ValueError):
                    coco_bounds_errors.append(f"{coco_path}: nonnumeric bbox in annotation {annotation.get('id')}")
                    continue
                if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
                    coco_bounds_errors.append(f"{coco_path}: nonfinite bbox in annotation {annotation.get('id')}")
                pixel_bounds_tolerance = BOUND_TOLERANCE * max(width, height)
                if box_width <= 0 or box_height <= 0 or x < -pixel_bounds_tolerance or y < -pixel_bounds_tolerance or x + box_width > width + pixel_bounds_tolerance or y + box_height > height + pixel_bounds_tolerance:
                    coco_bounds_errors.append(f"{coco_path}: out-of-bounds/nonpositive bbox in annotation {annotation.get('id')}")
                area = annotation.get("area")
                try:
                    if not math.isclose(float(area), box_width * box_height, rel_tol=1e-9, abs_tol=1e-6):
                        coco_area_errors.append(f"{coco_path}: area mismatch in annotation {annotation.get('id')}")
                except (TypeError, ValueError):
                    coco_area_errors.append(f"{coco_path}: invalid area in annotation {annotation.get('id')}")
                if annotation.get("category_id") != 1:
                    coco_category_errors.append(f"{coco_path}: annotation {annotation.get('id')} category_id must be 1")
                if index < len(expected_boxes):
                    expected_bbox = yolo_to_coco_bbox(expected_boxes[index], width, height)
                    if any(abs(left - right) > COCO_PARITY_TOLERANCE_PX for left, right in zip(expected_bbox, (x, y, box_width, box_height))):
                        parity_errors.append(f"{coco_path}: YOLO/COCO bbox mismatch for {name}, box {index + 1}")
        yolo_names = Counter(path.name.casefold() for path in images_by_split[split])
        if coco_names != yolo_names:
            coco_membership_errors.append(f"{coco_path}: YOLO/COCO image membership differs")
        coco_counts[split] = {"images": len(images), "boxes": len(annotations)}

    results.append(_result("13. COCO has exactly one license_plate category", f"category_issues={len(coco_category_errors)}", coco_category_errors))
    results.append(_result("14. COCO image and annotation IDs are unique", f"image_ids={len(global_image_ids)}, annotation_ids={len(global_annotation_ids)}, issues={len(coco_id_errors)}", coco_id_errors))
    results.append(_result("15. Every COCO annotation references an image", f"reference_issues={len(coco_reference_errors)}", coco_reference_errors))
    results.append(_result("16. COCO dimensions equal decoded image dimensions", f"dimension_issues={len(coco_dimension_errors)}", coco_dimension_errors))
    results.append(_result("17. Every COCO box is positive and inside the image", f"bbox_issues={len(coco_bounds_errors)}", coco_bounds_errors))
    results.append(_result("18. Every COCO area equals width multiplied by height", f"area_issues={len(coco_area_errors)}", coco_area_errors))
    results.append(_result("19. Every YOLO image appears once in the correct COCO split", f"membership_issues={len(coco_membership_errors)}", coco_membership_errors))
    results.append(_result("20. Every YOLO box matches COCO within 0.01 pixel", f"parity_issues={len(parity_errors)}", parity_errors))
    all_errors.extend(coco_category_errors + coco_id_errors + coco_reference_errors + coco_dimension_errors + coco_bounds_errors + coco_area_errors + coco_membership_errors + parity_errors)

    excluded_errors: list[str] = []
    exclusions_path = package_root / "reports" / "excluded_images.csv"
    if not exclusions_path.is_file():
        excluded_errors.append(f"Missing exclusion ledger: {exclusions_path}")
        exclusions_count = 0
    else:
        rows = read_csv(exclusions_path)
        exclusions_count = len(rows)
        if rows and list(rows[0]) != EXCLUSION_COLUMNS:
            excluded_errors.append("excluded_images.csv columns are not exact")
        current_names = {path.name.casefold() for split in SPLITS for path in images_by_split[split]}
        for row in rows:
            if Path(row.get("image_relative_path", "")).name.casefold() in current_names:
                excluded_errors.append(f"Excluded image remains in final split: {row.get('image_relative_path')}")
    results.append(_result("21. No excluded image remains in a final split", f"excluded_rows={exclusions_count}, remaining={len(excluded_errors)}", excluded_errors))
    all_errors.extend(excluded_errors)

    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (split, name), digest in image_hashes.items():
        by_hash[digest].append((split, name))
    cross_duplicates = [group for group in by_hash.values() if len({split for split, _ in group}) > 1]
    hash_errors = ["Cross-split exact duplicate: " + "; ".join(f"{split}/{name}" for split, name in group) for group in cross_duplicates]
    results.append(_result("22. Exact SHA-256 image duplicates do not cross splits", f"images_hashed={len(image_hashes)}, unique_hashes={len(by_hash)}, cross_split_groups={len(cross_duplicates)}", hash_errors))
    all_errors.extend(hash_errors)

    empty_split_errors = [f"Empty split: {split}" for split in SPLITS if not images_by_split[split]]
    results.append(_result("23. No split is empty", ", ".join(f"{split}={len(images_by_split[split])}" for split in SPLITS), empty_split_errors))
    all_errors.extend(empty_split_errors)

    observed_counts = {
        split: {"images": len(images_by_split[split]), "boxes": sum(len(boxes_by_image.get((split, image.name), [])) for image in images_by_split[split])}
        for split in SPLITS
    }
    observed_counts["total"] = {
        "images": sum(observed_counts[split]["images"] for split in SPLITS),
        "boxes": sum(observed_counts[split]["boxes"] for split in SPLITS),
    }
    release_errors: list[str] = []
    release_path = package_root / "dataset_release.json"
    if not release_path.is_file():
        release_errors.append(f"Missing {release_path}")
    else:
        try:
            release = json.loads(release_path.read_text(encoding="utf-8"))
            for split in (*SPLITS, "total"):
                if release.get("split_counts", {}).get(split) != observed_counts[split]["images"]:
                    release_errors.append(f"dataset_release.json image count mismatch for {split}")
                if release.get("box_counts", {}).get(split) != observed_counts[split]["boxes"]:
                    release_errors.append(f"dataset_release.json box count mismatch for {split}")
            manifest_path = package_root / "reports" / "dataset_manifest.csv"
            if not manifest_path.is_file() or release.get("manifest_sha256") != sha256_file(manifest_path):
                release_errors.append("dataset_release.json manifest SHA-256 mismatch")
            for split in SPLITS:
                coco_path = package_root / "annotations" / "coco" / f"{split}.json"
                if not coco_path.is_file() or release.get("coco_annotation_sha256", {}).get(split) != sha256_file(coco_path):
                    release_errors.append(f"dataset_release.json COCO SHA-256 mismatch for {split}")
        except Exception as exc:
            release_errors.append(f"Invalid dataset_release.json: {exc}")
    results.append(_result("24. Final release counts and hashes match dataset_release.json", f"release_issues={len(release_errors)}", release_errors))
    all_errors.extend(release_errors)

    accepted_errors = []
    for split in (*SPLITS, "total"):
        if observed_counts[split] != ACCEPTED_COUNTS[split]:
            accepted_errors.append(f"{split}: observed={observed_counts[split]}, accepted={ACCEPTED_COUNTS[split]}")
    results.append(_result("25. Accepted image and box counts match", "; ".join(f"{split}={observed_counts[split]['images']}/{observed_counts[split]['boxes']}" for split in (*SPLITS, "total")), accepted_errors))
    all_errors.extend(accepted_errors)

    overall = "PASS" if not all_errors and all(result["status"] == "PASS" for result in results) else "FAIL"
    report = {
        "status": overall,
        "results": results,
        "counts": observed_counts,
        "coco_counts": coco_counts,
        "errors": all_errors,
        "execution_date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": command,
    }
    if write_report:
        report_path = package_root / "reports" / "validation_report.md"
        lines = [
            "# Dataset Validation Report",
            "",
            f"- Command: `{command}`",
            f"- Execution date: {report['execution_date']}",
            f"- Dataset path: `{dataset_root}`",
            f"- Floating-point bounds tolerance: `{BOUND_TOLERANCE:g}` normalized units",
            f"- YOLO-COCO parity tolerance: `{COCO_PARITY_TOLERANCE_PX:.2f}` pixel",
            "",
            "## Check Results",
            "",
            "| Check | Observed | Status |",
            "|---|---|---|",
        ]
        for result in results:
            observed = str(result["observed"]).replace("|", "\\|")
            lines.append(f"| {result['check']} | {observed} | {result['status']} |")
        lines.extend(["", "## Observed Counts", "", "| Split | Images | YOLO boxes | COCO images | COCO annotations |", "|---|---:|---:|---:|---:|"])
        for split in SPLITS:
            coco = coco_counts.get(split, {"images": 0, "boxes": 0})
            lines.append(f"| {split} | {observed_counts[split]['images']} | {observed_counts[split]['boxes']} | {coco['images']} | {coco['boxes']} |")
        lines.append(f"| total | {observed_counts['total']['images']} | {observed_counts['total']['boxes']} | {sum(value['images'] for value in coco_counts.values())} | {sum(value['boxes'] for value in coco_counts.values())} |")
        lines.extend(["", "## Split Leakage QA", ""])
        lines.extend(f"- {line}" for line in _leakage_summary(package_root))
        if all_errors:
            lines.extend(["", "## Failures", ""])
            lines.extend(f"- {error}" for error in all_errors[:500])
            if len(all_errors) > 500:
                lines.append(f"- Additional failures omitted: {len(all_errors) - 500}")
        lines.extend(["", "## Overall Status", "", overall, ""])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    dataset_root = _resolve(package_root, args.dataset_root)
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    try:
        report = run_validation(package_root, dataset_root, command)
    except Exception as exc:
        report_path = package_root / "reports" / "validation_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "# Dataset Validation Report\n\n"
            f"- Command: `{command}`\n"
            f"- Execution date: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"- Dataset path: `{dataset_root}`\n\n"
            "## Overall Status\n\nFAIL\n\n"
            f"Validation aborted before completion: {exc}\n",
            encoding="utf-8",
            newline="\n",
        )
        raise
    print(f"Validation {report['status']}: {report['counts']}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
