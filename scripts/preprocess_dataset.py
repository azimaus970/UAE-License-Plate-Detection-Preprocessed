"""Single raw-to-clean audit and final YOLO-to-COCO release pipeline.

The accepted train/validation/test membership is immutable. The pipeline validates
the actual raw class schema and rows, confirms the cleaned final labels, consumes
the exclusion ledger, and deterministically regenerates derived release evidence.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date
from pathlib import Path

from preprocessing_utils import (
    ACCEPTED_COUNTS,
    COCO_CATEGORY_ID,
    LICENSE_ID,
    LICENSE_NAME,
    SOURCE_URL,
    SPLITS,
    TARGET_CLASS_ID,
    TARGET_CLASS_NAME,
    collect_dataset_records,
    count_source_labels,
    discover_source_splits,
    find_image_files,
    find_label_files,
    read_class_names,
    read_csv,
    require_dataset_layout,
    sha256_file,
    split_counts,
    write_csv,
    write_json,
    yolo_to_coco_bbox,
)

CLASS_MAPPING_COLUMNS = [
    "source_class_id",
    "source_class_name",
    "source_box_count",
    "decision",
    "target_class_id",
    "target_class_name",
    "boxes_kept",
    "boxes_removed",
    "reason",
]
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
MANIFEST_COLUMNS = [
    "split",
    "image_relative_path",
    "label_relative_path",
    "image_width",
    "image_height",
    "image_size_bytes",
    "label_size_bytes",
    "box_count",
    "image_sha256",
    "label_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Extracted original YOLO source root.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Accepted final YOLO dataset root.")
    parser.add_argument("--package-root", type=Path, required=True, help="Preprocessing handoff package root.")
    parser.add_argument("--exclusions", type=Path, required=True, help="Deterministic excluded_images.csv ledger.")
    parser.add_argument("--dry-run", action="store_true", help="Validate all inputs without writing derived artifacts.")
    return parser.parse_args()


def resolve_from(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def validate_exclusions(path: Path, dataset_root: Path, package_root: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Exclusion ledger is empty: {path}")
    if list(rows[0]) != EXCLUSION_COLUMNS:
        raise ValueError(f"{path} must have exactly these columns: {', '.join(EXCLUSION_COLUMNS)}")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    final_names = {
        image.name.casefold()
        for split in SPLITS
        for image in find_image_files(dataset_root / "images" / split)
    }
    for row in sorted(rows, key=lambda item: item["image_relative_path"].casefold()):
        relative = row["image_relative_path"].replace("\\", "/")
        key = relative.casefold()
        if key in seen:
            raise ValueError(f"Duplicate excluded image row: {relative}")
        seen.add(key)
        if Path(relative).name.casefold() in final_names:
            raise ValueError(f"Excluded image remains in the accepted dataset: {relative}")
        if row["decision_status"] != "NEEDS_HUMAN_REVIEW":
            raise ValueError(f"Unverified exclusion must remain NEEDS_HUMAN_REVIEW: {relative}")
        if row["human_reviewer"].strip():
            raise ValueError(f"Human reviewer must remain blank without factual reviewer evidence: {relative}")
        evidence = package_root / row["evidence_file"]
        if not evidence.is_file():
            raise FileNotFoundError(f"Missing exclusion evidence: {evidence}")
        row["image_relative_path"] = relative
        normalized.append(row)
    return normalized


def build_class_mapping(names: list[str], counts: Counter[int], final_box_count: int) -> list[dict[str, object]]:
    exact_plate_ids = [class_id for class_id, name in enumerate(names) if name.casefold() == "plate"]
    exact_license_plate_ids = [
        class_id for class_id, name in enumerate(names) if name.casefold() in {"license_plate", "license plate"}
    ]
    if len(exact_plate_ids) == 1:
        target_source_id = exact_plate_ids[0]
    elif len(names) == 1 and len(exact_license_plate_ids) == 1:
        target_source_id = exact_license_plate_ids[0]
    else:
        unresolved = exact_plate_ids + exact_license_plate_ids
        raise ValueError(
            "Source target class is ambiguous or absent; mark it unresolved before altering the accepted dataset. "
            f"Candidate IDs: {unresolved}; names: {names}"
        )
    rows: list[dict[str, object]] = []
    for class_id, name in enumerate(names):
        source_count = counts[class_id]
        if class_id == target_source_id:
            removed = source_count - final_box_count
            if removed < 0:
                raise ValueError("Accepted final dataset has more boxes than the source plate class")
            rows.append(
                {
                    "source_class_id": class_id,
                    "source_class_name": name,
                    "source_box_count": source_count,
                    "decision": "KEEP_AND_MAP",
                    "target_class_id": TARGET_CLASS_ID,
                    "target_class_name": TARGET_CLASS_NAME,
                    "boxes_kept": final_box_count,
                    "boxes_removed": removed,
                    "reason": "Actual source data.yaml identifies this exact class as the complete plate box; accepted split membership is preserved.",
                }
            )
        else:
            rows.append(
                {
                    "source_class_id": class_id,
                    "source_class_name": name,
                    "source_box_count": source_count,
                    "decision": "REMOVE_NON_TARGET",
                    "target_class_id": "",
                    "target_class_name": "",
                    "boxes_kept": 0,
                    "boxes_removed": source_count,
                    "reason": "Not the exact source `plate` class; OCR, character, emirate, style, and other non-target boxes are outside the full-plate target.",
                }
            )
    return rows


def validate_final_records(records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError("Accepted dataset has zero images")
    counts = split_counts(records)
    for record in records:
        boxes = list(record["boxes"])
        if not boxes:
            raise ValueError(f"Unexplained empty label: {record['label_path']}")
        for box in boxes:
            if box.class_id != TARGET_CLASS_ID:
                raise ValueError(f"Nonzero final class ID in {record['label_path']}")
    for split in (*SPLITS, "total"):
        observed = counts[split]
        expected = ACCEPTED_COUNTS[split]
        if observed["images"] != expected["images"] or observed["boxes"] != expected["boxes"]:
            raise ValueError(f"Accepted release count mismatch for {split}: observed={observed}, expected={expected}")


def write_manifest(package_root: Path, records: list[dict[str, object]]) -> Path:
    path = package_root / "reports" / "dataset_manifest.csv"
    rows = []
    for record in records:
        rows.append(
            {
                "split": record["split"],
                "image_relative_path": Path(record["image_path"]).relative_to(package_root).as_posix(),
                "label_relative_path": Path(record["label_path"]).relative_to(package_root).as_posix(),
                "image_width": record["image_width"],
                "image_height": record["image_height"],
                "image_size_bytes": record["image_size_bytes"],
                "label_size_bytes": record["label_size_bytes"],
                "box_count": record["box_count"],
                "image_sha256": record["image_sha256"],
                "label_sha256": record["label_sha256"],
            }
        )
    write_csv(path, MANIFEST_COLUMNS, rows)
    return path


def generate_coco(package_root: Path, records: list[dict[str, object]], creation_date: str) -> dict[str, Path]:
    output_dir = package_root / "annotations" / "coco"
    paths: dict[str, Path] = {}
    image_id = 1
    annotation_id = 1
    for split in SPLITS:
        images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        split_records = [record for record in records if record["split"] == split]
        for record in split_records:
            width = int(record["image_width"])
            height = int(record["image_height"])
            image_path = Path(record["image_path"])
            images.append(
                {
                    "id": image_id,
                    "license": LICENSE_ID,
                    "file_name": f"images/{split}/{image_path.name}",
                    "width": width,
                    "height": height,
                }
            )
            for box in record["boxes"]:
                bbox = yolo_to_coco_bbox(box, width, height)
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": COCO_CATEGORY_ID,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                annotation_id += 1
            image_id += 1
        payload = {
            "info": {
                "description": "Accepted UAE license plate detection release converted directly from final YOLO labels",
                "version": "2.0.0",
                "date_created": creation_date,
                "source_url": SOURCE_URL,
                "license": LICENSE_NAME,
            },
            "licenses": [{"id": LICENSE_ID, "name": LICENSE_NAME, "url": "https://creativecommons.org/licenses/by/4.0/"}],
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": COCO_CATEGORY_ID, "name": TARGET_CLASS_NAME, "supercategory": TARGET_CLASS_NAME}
            ],
        }
        path = output_dir / f"{split}.json"
        write_json(path, payload)
        paths[split] = path
    return paths


def write_audit(
    package_root: Path,
    source_totals: dict[str, int],
    final_counts: dict[str, dict[str, int]],
    mapping_rows: list[dict[str, object]],
) -> Path:
    removed_by_class = sum(int(row["boxes_removed"]) for row in mapping_rows if row["decision"] == "REMOVE_NON_TARGET")
    rows = [
        {"metric": "source images", "before": source_totals["images"], "after": "NOT_APPLICABLE", "evidence": "actual source split directories"},
        {"metric": "source label files", "before": source_totals["labels"], "after": "NOT_APPLICABLE", "evidence": "actual source label directories"},
        {"metric": "source boxes", "before": source_totals["boxes"], "after": "NOT_APPLICABLE", "evidence": "actual parsed source labels"},
        {"metric": "final images", "before": "NOT_APPLICABLE", "after": final_counts["total"]["images"], "evidence": "accepted final dataset"},
        {"metric": "final label files", "before": "NOT_APPLICABLE", "after": final_counts["total"]["labels"], "evidence": "accepted final dataset"},
        {"metric": "final boxes", "before": "NOT_APPLICABLE", "after": final_counts["total"]["boxes"], "evidence": "actual parsed final labels"},
        {"metric": "images excluded", "before": source_totals["images"], "after": source_totals["images"] - final_counts["total"]["images"], "evidence": "source-minus-accepted release count; includes no-plate filtering and accepted exclusions"},
        {"metric": "boxes removed by class cleaning", "before": source_totals["boxes"], "after": removed_by_class, "evidence": "class_mapping.csv"},
        {"metric": "invalid labels removed", "before": source_totals["boxes"], "after": 0, "evidence": "all source rows parsed successfully; no invalid row was removed"},
        {"metric": "unreadable images", "before": "NOT_AVAILABLE", "after": 0, "evidence": "historical raw decode count unavailable; every final image decoded during this run"},
        {"metric": "orphan images", "before": source_totals["orphan_images"], "after": 0, "evidence": "actual image/label stem comparison"},
        {"metric": "orphan labels", "before": source_totals["orphan_labels"], "after": 0, "evidence": "actual image/label stem comparison"},
    ]
    path = package_root / "reports" / "preprocessing_audit.csv"
    write_csv(path, ["metric", "before", "after", "evidence"], rows)
    return path


def ensure_manual_inspection_ledger(package_root: Path) -> Path:
    path = package_root / "reports" / "manual_inspection.csv"
    fieldnames = ["inspection_item", "evidence_file", "decision_status", "human_reviewer", "review_date", "notes"]
    rows = [
        {"inspection_item": "random ground-truth samples", "evidence_file": "reports/figures/train_random_samples.jpg; reports/figures/val_random_samples.jpg; reports/figures/test_random_samples.jpg"},
        {"inspection_item": "smallest boxes", "evidence_file": "reports/figures/smallest_boxes.jpg"},
        {"inspection_item": "largest boxes", "evidence_file": "reports/figures/largest_boxes.jpg"},
        {"inspection_item": "multi-plate images", "evidence_file": "reports/figures/multi_plate_images.jpg"},
        {"inspection_item": "edge-touching boxes", "evidence_file": "reports/figures/edge_touching_boxes.jpg"},
        {"inspection_item": "scene/template leakage pairs", "evidence_file": "reports/near_duplicate_review_pairs/"},
        {"inspection_item": "training-only augmentation preview", "evidence_file": "reports/figures/augmentation_preview.jpg"},
    ]
    for row in rows:
        row.update({"decision_status": "NEEDS_HUMAN_REVIEW", "human_reviewer": "", "review_date": "", "notes": ""})
    write_csv(path, fieldnames, rows)
    return path


def main() -> None:
    args = parse_args()
    invocation_root = Path.cwd().resolve()
    package_root = resolve_from(invocation_root, args.package_root)
    source_root = resolve_from(invocation_root, args.source_root)
    dataset_root = resolve_from(invocation_root, args.dataset_root)
    exclusions_path = resolve_from(invocation_root, args.exclusions)
    if not package_root.is_dir():
        raise FileNotFoundError(f"Package root does not exist: {package_root}")
    require_dataset_layout(dataset_root)
    source_splits = discover_source_splits(source_root)
    source_names = read_class_names(source_root / "data.yaml")
    exclusions = validate_exclusions(exclusions_path, dataset_root, package_root)

    print("Scanning actual source label files...", flush=True)
    source_class_counts, source_totals = count_source_labels(source_splits, len(source_names))
    print("Decoding and validating accepted final files...", flush=True)
    records = collect_dataset_records(dataset_root, decode_images=True, include_hashes=not args.dry_run)
    validate_final_records(records)
    counts = split_counts(records)
    mapping_rows = build_class_mapping(source_names, source_class_counts, counts["total"]["boxes"])

    print(f"Source root: {source_root}")
    print(f"Source images: {source_totals['images']}; source boxes: {source_totals['boxes']}")
    print(f"Accepted dataset root: {dataset_root}")
    for split in SPLITS:
        print(f"{split}: {counts[split]['images']} images, {counts[split]['boxes']} boxes")
    print(f"Excluded-image ledger rows: {len(exclusions)}")
    if args.dry_run:
        print("DRY RUN PASS: source mapping, final membership, labels, images, exclusions, and accepted counts are valid; no files were written.")
        return

    reports_dir = package_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = reports_dir / "class_mapping.csv"
    write_csv(mapping_path, CLASS_MAPPING_COLUMNS, mapping_rows)
    manifest_path = write_manifest(package_root, records)
    existing_release = package_root / "dataset_release.json"
    creation_date = date.today().isoformat()
    if existing_release.is_file():
        import json

        previous = json.loads(existing_release.read_text(encoding="utf-8"))
        creation_date = str(previous.get("creation_date") or creation_date)
    coco_paths = generate_coco(package_root, records, creation_date)
    audit_path = write_audit(package_root, source_totals, counts, mapping_rows)
    ensure_manual_inspection_ledger(package_root)

    release = {
        "release_name": "uae_lp_v2_yolo",
        "semantic_version": "2.0.0",
        "creation_date": creation_date,
        "source_dataset_name": "UAE (uae-zcfqj-ffa7t)",
        "source_url": SOURCE_URL,
        "license": LICENSE_NAME,
        "target_class": {"id": TARGET_CLASS_ID, "name": TARGET_CLASS_NAME},
        "split_counts": {split: counts[split]["images"] for split in SPLITS} | {"total": counts["total"]["images"]},
        "box_counts": {split: counts[split]["boxes"] for split in SPLITS} | {"total": counts["total"]["boxes"]},
        "actual_split_origin": "Project-controlled deterministic 70/15/15 source-aware split from the plate-only manifest; accepted membership preserves subsequent evaluation-side omissions represented in excluded_images.csv.",
        "actual_split_seed": 486,
        "manifest_sha256": sha256_file(manifest_path),
        "coco_annotation_sha256": {split: sha256_file(coco_paths[split]) for split in SPLITS},
        "augmentation_policy_filename": "configs/augmentation_policy.yaml",
        "known_limitations": [
            "The 43 historical difference-hash scene/template pairs and 16 absent evaluation images still require documented human review.",
            "Historical raw unreadable-image totals could not be reconstructed from the available local evidence.",
            "The local raw data.yaml uses azim-mohamed/uae-zcfqj-ffa7t while the required course attribution URL is addinguae/uae-zcfqj; this metadata discrepancy remains documented.",
            "Crop-heavy and suspicious training samples were retained by the historical split policy and have not been claimed as manually approved.",
        ],
    }
    write_json(existing_release, release)
    print(f"Wrote {mapping_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {audit_path}")
    print(f"Wrote {existing_release}")
    print("Preprocessing pipeline completed without changing accepted image/label membership.")


if __name__ == "__main__":
    main()
