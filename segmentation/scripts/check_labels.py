#!/usr/bin/env python3
"""Validate YOLO segmentation labels against images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from scripts.common import (
    DEFAULT_DATA_CONFIG,
    data_root,
    list_images,
    list_labels,
    load_data_config,
    split_dirs,
)

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 200, 255),
    2: (255, 128, 0),
    3: (255, 0, 128),
}


def parse_label_line(line: str, num_classes: int) -> tuple[int, list[tuple[float, float]]]:
    parts = line.split()
    if len(parts) < 7 or len(parts) % 2 == 0:
        raise ValueError(f"expected class_id plus >=3 coordinate pairs, got {len(parts)} values")

    class_id = int(parts[0])
    if class_id < 0 or class_id >= num_classes:
        raise ValueError(f"class_id {class_id} out of range 0..{num_classes - 1}")

    coords = [float(v) for v in parts[1:]]
    for value in coords:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"normalized coordinate {value} outside [0, 1]")

    points = list(zip(coords[::2], coords[1::2], strict=True))
    return class_id, points


def validate_label_file(label_path: Path, num_classes: int) -> list[str]:
    errors: list[str] = []
    text = label_path.read_text().strip()
    if not text:
        return errors

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parse_label_line(line, num_classes)
        except ValueError as exc:
            errors.append(f"{label_path}:{line_no}: {exc}")
    return errors


def check_split(
    split: str,
    data_dir: Path,
    config: dict,
    num_classes: int,
) -> tuple[list[str], list[tuple[Path, Path]]]:
    images_dir, labels_dir = split_dirs(data_dir, config, split)
    images = list_images(images_dir)
    labels = list_labels(labels_dir)
    errors: list[str] = []
    pairs: list[tuple[Path, Path]] = []

    for stem, image_path in sorted(images.items()):
        label_path = labels.get(stem)
        if label_path is None:
            errors.append(f"[{split}] image without label: {image_path}")
            continue
        pairs.append((image_path, label_path))
        errors.extend(validate_label_file(label_path, num_classes))

    for stem, label_path in sorted(labels.items()):
        if stem not in images:
            errors.append(f"[{split}] label without image: {label_path}")

    return errors, pairs


def render_overlay(image_path: Path, label_path: Path, num_classes: int, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    height, width = image.shape[:2]
    for raw_line in label_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        class_id, points = parse_label_line(line, num_classes)
        polygon = np.array(
            [[int(x * width), int(y * height)] for x, y in points],
            dtype=np.int32,
        )
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [polygon], color=color, lineType=cv2.LINE_AA)
        image = cv2.addWeighted(image, 0.75, mask, 0.25, 0)
        cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)
        cx = int(np.mean(polygon[:, 0]))
        cy = int(np.mean(polygon[:, 1]))
        cv2.putText(
            image,
            str(class_id),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YOLO segmentation labels.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Path to dataset YAML config.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="Render up to N train/val overlay previews.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("runs/label_preview"),
        help="Directory for preview images.",
    )
    args = parser.parse_args()

    config = load_data_config(args.config)
    num_classes = int(config["nc"])
    root = data_root(config, args.config)

    all_errors: list[str] = []
    preview_pairs: list[tuple[str, Path, Path]] = []

    for split in ("train", "val"):
        errors, pairs = check_split(split, root, config, num_classes)
        all_errors.extend(errors)
        for image_path, label_path in pairs:
            preview_pairs.append((split, image_path, label_path))

    if not preview_pairs:
        print("No labeled image pairs found in train/ or val/.")
        print("Add paired files under data/images/{train,val} and data/labels/{train,val}.")
        return 1 if all_errors else 0

    print(f"Checked {len(preview_pairs)} image/label pairs.")
    if all_errors:
        print(f"Found {len(all_errors)} error(s):")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print("All labels passed validation.")

    if args.preview > 0:
        rendered = 0
        for split, image_path, label_path in preview_pairs:
            if rendered >= args.preview:
                break
            output_path = args.preview_dir / split / f"{image_path.stem}.jpg"
            render_overlay(image_path, label_path, num_classes, output_path)
            print(f"Wrote preview: {output_path}")
            rendered += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
