#!/usr/bin/env python3
"""Create a derived YOLO segmentation dataset with deterministic preprocessing."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import yaml

from scripts.common import DEFAULT_DATA_CONFIG, data_root, list_images, load_data_config, split_dirs
from scripts.preprocessing import (
    DEFAULT_PREPROCESS_CONFIG,
    CropTransform,
    apply_preprocessing,
    load_preprocess_preset,
)

Point = tuple[float, float]


def _format_number(value: float) -> str:
    return f"{value:.6f}"


def _clip_boundary(
    polygon: list[Point],
    inside,
    intersection,
) -> list[Point]:
    if not polygon:
        return []
    output: list[Point] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def _vertical_intersection(start: Point, end: Point, x: float) -> Point:
    dx = end[0] - start[0]
    if abs(dx) < 1e-12:
        return x, start[1]
    ratio = (x - start[0]) / dx
    return x, start[1] + ratio * (end[1] - start[1])


def _horizontal_intersection(start: Point, end: Point, y: float) -> Point:
    dy = end[1] - start[1]
    if abs(dy) < 1e-12:
        return start[0], y
    ratio = (y - start[1]) / dy
    return start[0] + ratio * (end[0] - start[0]), y


def clip_polygon_to_crop(polygon: list[Point], transform: CropTransform) -> list[Point]:
    """Clip source-pixel polygon coordinates to the rectangular crop."""
    left = float(transform.x)
    top = float(transform.y)
    right = float(transform.x + transform.width)
    bottom = float(transform.y + transform.height)
    clipped = _clip_boundary(
        polygon,
        lambda point: point[0] >= left,
        lambda start, end: _vertical_intersection(start, end, left),
    )
    clipped = _clip_boundary(
        clipped,
        lambda point: point[0] <= right,
        lambda start, end: _vertical_intersection(start, end, right),
    )
    clipped = _clip_boundary(
        clipped,
        lambda point: point[1] >= top,
        lambda start, end: _horizontal_intersection(start, end, top),
    )
    clipped = _clip_boundary(
        clipped,
        lambda point: point[1] <= bottom,
        lambda start, end: _horizontal_intersection(start, end, bottom),
    )
    deduplicated: list[Point] = []
    for point in clipped:
        if not deduplicated or (
            abs(point[0] - deduplicated[-1][0]) > 1e-9
            or abs(point[1] - deduplicated[-1][1]) > 1e-9
        ):
            deduplicated.append(point)
    if len(deduplicated) > 1 and (
        abs(deduplicated[0][0] - deduplicated[-1][0]) <= 1e-9
        and abs(deduplicated[0][1] - deduplicated[-1][1]) <= 1e-9
    ):
        deduplicated.pop()
    return deduplicated


def _polygon_area(polygon: list[Point]) -> float:
    return abs(
        sum(
            start[0] * end[1] - end[0] * start[1]
            for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True)
        )
    ) / 2.0


def transform_segment_line(line: str, transform: CropTransform) -> str | None:
    parts = line.split()
    if len(parts) < 7 or len(parts) % 2 == 0:
        raise ValueError(
            f"expected class_id plus at least 3 coordinate pairs, got {len(parts)} values"
        )
    class_id = int(parts[0])
    coordinates = list(map(float, parts[1:]))
    if any(value < 0.0 or value > 1.0 for value in coordinates):
        raise ValueError("normalized polygon coordinate outside [0, 1]")

    polygon = [
        (
            coordinates[index] * transform.source_width,
            coordinates[index + 1] * transform.source_height,
        )
        for index in range(0, len(coordinates), 2)
    ]
    clipped = clip_polygon_to_crop(polygon, transform)
    if len(clipped) < 3 or _polygon_area(clipped) <= 1e-9:
        return None

    output = [str(class_id)]
    for x, y in clipped:
        normalized_x = max(0.0, min(1.0, (x - transform.x) / transform.width))
        normalized_y = max(0.0, min(1.0, (y - transform.y) / transform.height))
        output.extend([_format_number(normalized_x), _format_number(normalized_y)])
    return " ".join(output)


def transform_label_file(
    source: Path,
    destination: Path,
    transform: CropTransform,
) -> None:
    if transform.is_identity:
        shutil.copy2(source, destination)
        return

    transformed_lines = []
    for line_number, raw_line in enumerate(source.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            transformed = transform_segment_line(line, transform)
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc
        if transformed:
            transformed_lines.append(transformed)
    destination.write_text("\n".join(transformed_lines) + ("\n" if transformed_lines else ""))


def prepare_dataset(
    dataset_config: Path,
    preprocess_config: Path,
    preset_name: str,
    output_root: Path,
    output_config: Path,
    overwrite: bool = False,
) -> tuple[int, int]:
    config = load_data_config(dataset_config)
    source_root = data_root(config, dataset_config)
    preset = load_preprocess_preset(preset_name, preprocess_config)

    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(f"output path is not a directory: {output_root}")
        if any(output_root.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"output directory is not empty: {output_root}; "
                    "pass --overwrite to replace files"
                )
            shutil.rmtree(output_root)

    image_count = 0
    missing_labels = 0
    for split in ("train", "val"):
        source_images, source_labels = split_dirs(source_root, config, split)
        if not source_images.is_dir():
            raise FileNotFoundError(f"source image directory not found: {source_images}")

        destination_images = output_root / "images" / split
        destination_labels = output_root / "labels" / split
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)

        for stem, image_path in sorted(list_images(source_images).items()):
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"failed to read image: {image_path}")
            result = apply_preprocessing(image, preset)
            output_image = destination_images / image_path.name
            if preset_name == "original":
                shutil.copy2(image_path, output_image)
            elif not cv2.imwrite(str(output_image), result.image):
                raise RuntimeError(f"failed to write image: {output_image}")

            label_path = source_labels / f"{stem}.txt"
            if label_path.exists():
                transform_label_file(
                    label_path,
                    destination_labels / label_path.name,
                    result.transform,
                )
            else:
                missing_labels += 1
            image_count += 1

    generated_config = dict(config)
    generated_config["path"] = str(output_root.resolve())
    generated_config["train"] = "images/train"
    generated_config["val"] = "images/val"
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(generated_config, sort_keys=False))
    provenance = {
        "source_dataset_config": str(dataset_config.resolve()),
        "preprocess_config": str(preprocess_config.resolve()),
        "preset": preset_name,
    }
    (output_root / "preprocessing.yaml").write_text(yaml.safe_dump(provenance, sort_keys=False))
    return image_count, missing_labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a preprocessed YOLO segmentation dataset."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument(
        "--preprocess-config", type=Path, default=DEFAULT_PREPROCESS_CONFIG
    )
    parser.add_argument("--preset", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Derived dataset directory. Defaults to data_preprocessed/<preset>.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Generated dataset YAML. Defaults to configs/generated/forceps_seg_<preset>.yaml.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root or Path("data_preprocessed") / args.preset
    output_config = args.output_config or (
        Path("configs") / "generated" / f"forceps_seg_{args.preset}.yaml"
    )
    try:
        image_count, missing_labels = prepare_dataset(
            args.config,
            args.preprocess_config,
            args.preset,
            output_root,
            output_config,
            args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc)
        return 1

    print(f"Prepared {image_count} images with preset '{args.preset}'.")
    print(f"Dataset config: {output_config}")
    if missing_labels:
        print(f"Warning: {missing_labels} image(s) had no label file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
