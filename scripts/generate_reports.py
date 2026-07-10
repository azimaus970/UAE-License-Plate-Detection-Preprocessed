"""Generate factual dataset statistics and distribution figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from preprocessing_utils import SPLITS, collect_dataset_records, find_label_files, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/uae_lp_v2_yolo"))
    return parser.parse_args()


def describe(values: list[float | int]) -> dict[str, float | str]:
    if not values:
        return {key: "NOT_AVAILABLE" for key in ("min", "q25", "median", "mean", "q75", "max")}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(median(values)),
        "mean": float(mean(values)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def flatten(prefix: str, values: list[float | int]) -> dict[str, float | str]:
    return {f"{prefix}_{key}": value for key, value in describe(values).items()}


def stats_row(split: str, records: list[dict[str, object]], dataset_root: Path) -> dict[str, object]:
    selected = records if split == "total" else [record for record in records if record["split"] == split]
    boxes = [box for record in selected for box in record["boxes"]]
    boxes_per_image = [int(record["box_count"]) for record in selected]
    widths = [int(record["image_width"]) for record in selected]
    heights = [int(record["image_height"]) for record in selected]
    label_files = sum(len(find_label_files(dataset_root / "labels" / current)) for current in SPLITS) if split == "total" else len(find_label_files(dataset_root / "labels" / split))
    row: dict[str, object] = {
        "split": split,
        "image_count": len(selected),
        "label_file_count": label_files,
        "box_count": len(boxes),
        "empty_label_count": sum(not record["boxes"] for record in selected),
    }
    row.update(flatten("boxes_per_image", boxes_per_image))
    row.update(flatten("image_width", widths))
    row.update(flatten("image_height", heights))
    row.update(flatten("normalized_box_width", [box.width for box in boxes]))
    row.update(flatten("normalized_box_height", [box.height for box in boxes]))
    row.update(flatten("normalized_box_area", [box.area for box in boxes]))
    row.update(flatten("box_aspect_ratio", [box.aspect_ratio for box in boxes]))
    return row


def save_split_counts(rows: list[dict[str, object]], figures_dir: Path) -> None:
    split_rows = rows[:3]
    labels = [str(row["split"]) for row in split_rows]
    images = [int(row["image_count"]) for row in split_rows]
    boxes = [int(row["box_count"]) for row in split_rows]
    x = np.arange(len(labels))
    fig, left = plt.subplots(figsize=(8, 5))
    right = left.twinx()
    bars_images = left.bar(x - 0.18, images, width=0.36, color="#2f6f9f", label="Images")
    bars_boxes = right.bar(x + 0.18, boxes, width=0.36, color="#e07a3f", label="Boxes")
    left.set_xticks(x, labels)
    left.set_ylabel("Images")
    right.set_ylabel("Boxes")
    left.set_title("Accepted split image and box counts")
    left.bar_label(bars_images, padding=3)
    right.bar_label(bars_boxes, padding=3)
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    fig.tight_layout()
    fig.savefig(figures_dir / "split_counts.png", dpi=180)
    plt.close(fig)


def save_histogram(values_by_split: dict[str, list[float]], output: Path, title: str, xlabel: str, *, log_x: bool = False, bins: int | list[float] = 40) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    colors = {"train": "#2f6f9f", "val": "#e07a3f", "test": "#40916c"}
    for split in SPLITS:
        values = values_by_split[split]
        if not values:
            raise ValueError(f"No valid values exist for {output.name}: {split}")
        axis.hist(values, bins=bins, density=True, alpha=0.45, label=split, color=colors[split])
    if log_x:
        axis.set_xscale("log")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    dataset_root = args.dataset_root.resolve() if args.dataset_root.is_absolute() else (package_root / args.dataset_root).resolve()
    records = collect_dataset_records(dataset_root, decode_images=False, include_hashes=False)
    if not records:
        raise ValueError(f"Dataset has zero images: {dataset_root}")
    rows = [stats_row(split, records, dataset_root) for split in (*SPLITS, "total")]
    stats_path = package_root / "reports" / "dataset_stats.csv"
    write_csv(stats_path, list(rows[0]), rows)
    figures_dir = package_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_split_counts(rows, figures_dir)

    boxes_by_split = {
        split: [int(record["box_count"]) for record in records if record["split"] == split] for split in SPLITS
    }
    save_histogram(boxes_by_split, figures_dir / "boxes_per_image.png", "Boxes per image", "Box count", bins=np.arange(0.5, 8.6, 1).tolist())
    areas = {split: [box.area for record in records if record["split"] == split for box in record["boxes"]] for split in SPLITS}
    aspects = {split: [box.aspect_ratio for record in records if record["split"] == split for box in record["boxes"]] for split in SPLITS}
    save_histogram(areas, figures_dir / "relative_box_area.png", "Normalized bounding-box area", "Relative area (width × height)", log_x=True)
    save_histogram(aspects, figures_dir / "box_aspect_ratio.png", "Bounding-box aspect ratio", "Width / height", log_x=True)
    print(f"Wrote {stats_path}")
    for name in ("split_counts.png", "boxes_per_image.png", "relative_box_area.png", "box_aspect_ratio.png"):
        print(f"Wrote {figures_dir / name}")


if __name__ == "__main__":
    main()
