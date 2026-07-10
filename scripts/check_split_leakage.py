"""Check exact and optional difference-hash split leakage without removing images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from PIL import Image

from preprocessing_utils import SPLITS, find_image_files, read_csv, write_csv

OUTPUT_COLUMNS = [
    "pair_id",
    "source",
    "distance",
    "distance_threshold",
    "split_a",
    "image_a",
    "split_b",
    "image_b",
    "split_combination",
    "exact_duplicate",
    "issue_type",
    "current_membership",
    "decision_status",
    "human_reviewer",
    "review_date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/uae_lp_v2_yolo"))
    parser.add_argument("--historical-pairs", type=Path, default=Path("reports/near_duplicate_review_decisions.csv"))
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--skip-perceptual", action="store_true", help="Run SHA-256 only.")
    return parser.parse_args()


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        image.load()
        gray = image.convert("L").resize((17, 16))
        pixels = list(gray.tobytes())
    value = 0
    for row in range(16):
        offset = row * 17
        for column in range(16):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def _membership(package_root: Path, image_a: str, image_b: str) -> str:
    present_a = (package_root / image_a).is_file()
    present_b = (package_root / image_b).is_file()
    if present_a and present_b:
        return "both_present"
    if present_a:
        return "image_a_present_only"
    if present_b:
        return "image_b_present_only"
    return "neither_present"


def load_historical(package_root: Path, path: Path, threshold: int) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Historical 43-pair CSV is missing: {path}")
    source_rows = read_csv(path)
    if len(source_rows) != 43:
        raise ValueError(f"Expected the existing 43 historical pairs, found {len(source_rows)} in {path}")
    rows: list[dict[str, object]] = []
    for row in source_rows:
        reviewer = row.get("reviewer_name", "").strip()
        if reviewer.casefold() in {"dataset_preprocessing", "codex", "chatgpt"}:
            reviewer = ""
        image_a = row["image_a"].replace("\\", "/")
        image_b = row["image_b"].replace("\\", "/")
        rows.append(
            {
                "pair_id": row["pair_id"],
                "source": "historical_review_pair",
                "distance": row["distance"],
                "distance_threshold": threshold,
                "split_a": row["split_a"],
                "image_a": image_a,
                "split_b": row["split_b"],
                "image_b": image_b,
                "split_combination": "-".join(sorted((row["split_a"], row["split_b"]))),
                "exact_duplicate": "false",
                "issue_type": "scene/template_leakage_candidate",
                "current_membership": _membership(package_root, image_a, image_b),
                "decision_status": "NEEDS_HUMAN_REVIEW",
                "human_reviewer": reviewer,
                "review_date": row.get("review_date", "") if reviewer else "",
            }
        )
    return rows


def current_records(dataset_root: Path) -> list[dict[str, object]]:
    records = []
    for split in SPLITS:
        for path in find_image_files(dataset_root / "images" / split):
            records.append({"split": split, "path": path, "relative": f"datasets/uae_lp_v2_yolo/images/{split}/{path.name}"})
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_duplicate_rows(records: list[dict[str, object]], threshold: int) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    workers = min(32, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for record, digest in zip(records, executor.map(lambda item: sha256(Path(item["path"])), records)):
            groups[digest].append(record)
    rows: list[dict[str, object]] = []
    counter = 1
    for digest, group in sorted(groups.items()):
        if len({str(item["split"]) for item in group}) < 2:
            continue
        for left, right in combinations(group, 2):
            if left["split"] == right["split"]:
                continue
            rows.append(
                {
                    "pair_id": f"exact_{counter:03d}",
                    "source": "current_sha256",
                    "distance": 0,
                    "distance_threshold": threshold,
                    "split_a": left["split"],
                    "image_a": left["relative"],
                    "split_b": right["split"],
                    "image_b": right["relative"],
                    "split_combination": "-".join(sorted((str(left["split"]), str(right["split"])))),
                    "exact_duplicate": "true",
                    "issue_type": "exact_duplicate",
                    "current_membership": "both_present",
                    "decision_status": "NEEDS_HUMAN_REVIEW",
                    "human_reviewer": "",
                    "review_date": "",
                    "sha256": digest,
                }
            )
            counter += 1
    return rows


def perceptual_rows(records: list[dict[str, object]], threshold: int) -> list[dict[str, object]]:
    workers = min(32, max(4, (os.cpu_count() or 4) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for record, dhash in zip(records, executor.map(lambda item: difference_hash(Path(item["path"])), records)):
            record["dhash"] = dhash
    by_split = {split: [record for record in records if record["split"] == split] for split in SPLITS}
    rows: list[dict[str, object]] = []
    counter = 1
    for split_a, split_b in (("train", "val"), ("train", "test"), ("val", "test")):
        for left in by_split[split_a]:
            left_hash = int(left["dhash"])
            for right in by_split[split_b]:
                distance = (left_hash ^ int(right["dhash"])).bit_count()
                if distance > threshold:
                    continue
                rows.append(
                    {
                        "pair_id": f"current_{counter:03d}",
                        "source": "current_difference_hash",
                        "distance": distance,
                        "distance_threshold": threshold,
                        "split_a": split_a,
                        "image_a": left["relative"],
                        "split_b": split_b,
                        "image_b": right["relative"],
                        "split_combination": f"{split_a}-{split_b}",
                        "exact_duplicate": "false",
                        "issue_type": "scene/template_leakage_candidate",
                        "current_membership": "both_present",
                        "decision_status": "NEEDS_HUMAN_REVIEW",
                        "human_reviewer": "",
                        "review_date": "",
                    }
                )
                counter += 1
    return sorted(rows, key=lambda row: (int(row["distance"]), str(row["image_a"]), str(row["image_b"])))


def update_validation_report(package_root: Path, rows: list[dict[str, object]], threshold: int, perceptual_ran: bool) -> None:
    report_path = package_root / "reports" / "validation_report.md"
    text = report_path.read_text(encoding="utf-8") if report_path.is_file() else "# Dataset Validation Report\n\n"
    marker = "## Split Leakage QA"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    historical = [row for row in rows if row["source"] == "historical_review_pair"]
    current = [row for row in rows if row["source"] == "current_difference_hash"]
    exact = [row for row in rows if row["source"] == "current_sha256"]
    flagged = {str(row[key]) for row in rows for key in ("image_a", "image_b")}
    evaluation = {
        str(row[key])
        for row in rows
        for key, split_key in (("image_a", "split_a"), ("image_b", "split_b"))
        if row[split_key] in {"val", "test"}
    }
    combinations_found = sorted({str(row["split_combination"]) for row in rows})
    lines = [
        "",
        marker,
        "",
        f"- Total recorded pair count: {len(rows)} ({len(historical)} historical review pairs, {len(current)} current perceptual/template candidates, {len(exact)} current exact-duplicate pairs).",
        f"- Unique flagged image count: {len(flagged)}.",
        f"- Unique flagged evaluation-image count: {len(evaluation)}.",
        f"- Difference-hash distance threshold: {threshold}.",
        f"- Split combinations: {', '.join(combinations_found) if combinations_found else 'none'}.",
        f"- Exact cross-split duplicates: {len(exact)}.",
        f"- Current perceptual/template candidates: {len(current) if perceptual_ran else 'NOT_RUN'}.",
        "- The detected historical issue is scene/template leakage: the vehicle and background are reused while plate text may differ.",
        "- Perceptual hashing and Hamming distance are optional project engineering QA and are not techniques taught in the supplied lectures.",
        "- No image was automatically removed and no human-review field was filled by this script.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text.rstrip() + "\n" + "\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    dataset_root = args.dataset_root.resolve() if args.dataset_root.is_absolute() else (package_root / args.dataset_root).resolve()
    historical_path = args.historical_pairs.resolve() if args.historical_pairs.is_absolute() else (package_root / args.historical_pairs).resolve()
    if args.threshold < 0:
        raise ValueError("Distance threshold must be nonnegative")
    records = current_records(dataset_root)
    if not records:
        raise ValueError(f"Dataset has zero images: {dataset_root}")
    rows = load_historical(package_root, historical_path, args.threshold)
    rows.extend(exact_duplicate_rows(records, args.threshold))
    if not args.skip_perceptual:
        rows.extend(perceptual_rows(records, args.threshold))
    output_path = package_root / "reports" / "split_leakage_candidates.csv"
    write_csv(output_path, OUTPUT_COLUMNS, rows)
    update_validation_report(package_root, rows, args.threshold, not args.skip_perceptual)
    historical_count = sum(row["source"] == "historical_review_pair" for row in rows)
    exact_count = sum(row["source"] == "current_sha256" for row in rows)
    current_count = sum(row["source"] == "current_difference_hash" for row in rows)
    print(f"Historical scene/template review pairs: {historical_count}")
    print(f"Current exact cross-split duplicate pairs: {exact_count}")
    print(f"Current difference-hash candidates: {current_count if not args.skip_perceptual else 'NOT_RUN'}")
    print(f"Wrote {output_path}")
    if exact_count:
        raise SystemExit("Exact cross-split duplicates found")


if __name__ == "__main__":
    main()
