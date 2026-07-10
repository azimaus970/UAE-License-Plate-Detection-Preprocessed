"""Generate deterministic ground-truth and training-augmentation evidence sheets."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preprocessing_utils import (
    BOUND_TOLERANCE,
    SPLITS,
    TARGET_CLASS_NAME,
    VISUALIZATION_SEED,
    augment_training_sample,
    collect_dataset_records,
    load_augmentation_policy,
    make_contact_sheet,
    pixel_xyxy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/uae_lp_v2_yolo"))
    parser.add_argument("--sample-count", type=int, default=12)
    return parser.parse_args()


def stable_key(record: dict[str, object]) -> tuple[str, str]:
    return str(record["split"]), Path(record["image_path"]).name.casefold()


def select_unique_images(ranked: list[tuple[float, dict[str, object]]], count: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[Path] = set()
    for _, record in ranked:
        path = Path(record["image_path"])
        if path in seen:
            continue
        seen.add(path)
        selected.append(record)
        if len(selected) == count:
            break
    return selected


def font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def augmented_tile(image: np.ndarray, boxes: list[tuple[float, float, float, float]], caption: str) -> Image.Image:
    tile_size = (480, 390)
    caption_height = 48
    source = Image.fromarray(image)
    width, height = source.size
    scale = min(tile_size[0] / width, (tile_size[1] - caption_height) / height)
    rendered = source.resize((max(1, round(width * scale)), max(1, round(height * scale))))
    tile = Image.new("RGB", tile_size, "white")
    offset_x = (tile_size[0] - rendered.width) // 2
    tile.paste(rendered, (offset_x, 0))
    draw = ImageDraw.Draw(tile)
    for x1, y1, x2, y2 in boxes:
        coords = (offset_x + x1 * scale, y1 * scale, offset_x + x2 * scale, y2 * scale)
        draw.rectangle(coords, outline=(255, 30, 30), width=4)
        label_x = max(0, min(coords[0], tile_size[0] - 100))
        label_y = max(0, coords[1] - 17)
        draw.rectangle((label_x, label_y, label_x + 100, label_y + 17), fill=(255, 30, 30))
        draw.text((label_x + 2, label_y), TARGET_CLASS_NAME, fill="white", font=font(14))
    draw.text((6, tile_size[1] - 38), caption[:78], fill="black", font=font(14))
    return tile


def write_augmentation_preview(records: list[dict[str, object]], policy: dict, output: Path, sample_count: int) -> None:
    train = sorted((record for record in records if record["split"] == "train"), key=stable_key)
    if not train:
        raise ValueError("No training images exist for augmentation_preview.jpg")
    rng = random.Random(int(policy["random_seed"]))
    candidates = list(train)
    rng.shuffle(candidates)
    tiles: list[Image.Image] = []
    for record in candidates:
        bgr = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"OpenCV could not decode training image: {record['image_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        boxes = [pixel_xyxy(box, width, height) for box in record["boxes"]]
        transformed = None
        for attempt in range(20):
            transformed = augment_training_sample(
                rgb,
                boxes,
                policy,
                rng,
                force_constrained_crop=(len(tiles) == 0 and attempt == 0),
            )
            if transformed is not None:
                break
        if transformed is None:
            continue
        augmented, transformed_boxes = transformed
        tiles.append(augmented_tile(augmented, transformed_boxes, f"train only | transformed boxes={len(transformed_boxes)}"))
        if len(tiles) == sample_count:
            break
    if not tiles:
        raise ValueError("No augmentation preview sample met minimum_bbox_visibility=0.70")
    columns = 4
    rows = math.ceil(len(tiles) / columns)
    title_height = 42
    sheet = Image.new("RGB", (480 * columns, 390 * rows + title_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), "Training-only augmentation preview (seed 486; boxes transformed consistently)", fill="black", font=font(22))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 480, (index // columns) * 390 + title_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive")
    package_root = args.package_root.resolve()
    dataset_root = args.dataset_root.resolve() if args.dataset_root.is_absolute() else (package_root / args.dataset_root).resolve()
    records = collect_dataset_records(dataset_root, decode_images=False, include_hashes=False)
    if not records:
        raise ValueError(f"Dataset has zero images: {dataset_root}")
    figures = package_root / "reports" / "figures"
    rng = random.Random(VISUALIZATION_SEED)
    for split in SPLITS:
        pool = sorted((record for record in records if record["split"] == split), key=stable_key)
        if len(pool) < args.sample_count:
            raise ValueError(f"Not enough {split} images for requested random sample sheet")
        selected = rng.sample(pool, args.sample_count)
        make_contact_sheet(
            selected,
            figures / f"{split}_random_samples.jpg",
            title=f"{split} random ground truth | fixed seed {VISUALIZATION_SEED}",
        )

    box_rank = [(box.area, record) for record in records for box in record["boxes"]]
    smallest = select_unique_images(sorted(box_rank, key=lambda item: (item[0], stable_key(item[1]))), args.sample_count)
    largest = select_unique_images(sorted(box_rank, key=lambda item: (-item[0], stable_key(item[1]))), args.sample_count)
    make_contact_sheet(smallest, figures / "smallest_boxes.jpg", title="Deterministic smallest normalized boxes")
    make_contact_sheet(largest, figures / "largest_boxes.jpg", title="Deterministic largest normalized boxes")

    multi = sorted((record for record in records if int(record["box_count"]) > 1), key=lambda record: (-int(record["box_count"]), stable_key(record)))[: args.sample_count]
    make_contact_sheet(multi, figures / "multi_plate_images.jpg", title="Deterministic multi-plate images")

    edge = []
    for record in records:
        if any(
            min(box.normalized_xyxy()[0], box.normalized_xyxy()[1]) <= BOUND_TOLERANCE
            or box.normalized_xyxy()[2] >= 1 - BOUND_TOLERANCE
            or box.normalized_xyxy()[3] >= 1 - BOUND_TOLERANCE
            for box in record["boxes"]
        ):
            edge.append(record)
    edge = sorted(edge, key=stable_key)[: args.sample_count]
    make_contact_sheet(edge, figures / "edge_touching_boxes.jpg", title="Deterministic boxes touching an image edge")

    policy = load_augmentation_policy(package_root / "configs" / "augmentation_policy.yaml")
    write_augmentation_preview(records, policy, figures / "augmentation_preview.jpg", min(args.sample_count, 8))
    for name in (
        "train_random_samples.jpg",
        "val_random_samples.jpg",
        "test_random_samples.jpg",
        "smallest_boxes.jpg",
        "largest_boxes.jpg",
        "multi_plate_images.jpg",
        "edge_touching_boxes.jpg",
        "augmentation_preview.jpg",
    ):
        print(f"Wrote {figures / name}")


if __name__ == "__main__":
    main()
