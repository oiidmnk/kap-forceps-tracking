#!/usr/bin/env python3
"""Run YOLO segmentation inference on images or folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

from scripts.preprocessing import (
    DEFAULT_PREPROCESS_CONFIG,
    apply_preprocessing,
    find_images,
    load_preprocess_preset,
)

DEFAULT_WEIGHTS = Path("runs/segment/forceps/weights/best.pt")
CLASS_COLORS = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 128, 0),
    (255, 0, 180),
    (180, 80, 255),
    (255, 180, 0),
]


def _class_name(names: dict | list, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def render_box_result(result) -> object:
    """Render class-colored boxes and a class legend without masks or inline labels."""
    image = result.orig_img.copy()
    image_height, image_width = image.shape[:2]
    thickness = max(2, round(min(image_height, image_width) / 500))

    boxes = result.boxes
    if boxes is not None and boxes.xyxy is not None:
        coordinates = boxes.xyxy.cpu().numpy()
        class_ids = (
            boxes.cls.cpu().numpy().astype(int)
            if boxes.cls is not None
            else [0] * len(coordinates)
        )
        for (x1, y1, x2, y2), class_id in zip(
            coordinates, class_ids, strict=True
        ):
            color = CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]
            top_left = (
                max(0, min(image_width - 1, round(float(x1)))),
                max(0, min(image_height - 1, round(float(y1)))),
            )
            bottom_right = (
                max(0, min(image_width - 1, round(float(x2)))),
                max(0, min(image_height - 1, round(float(y2)))),
            )
            cv2.rectangle(
                image,
                top_left,
                bottom_right,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

    names = result.names
    class_ids = (
        sorted(int(class_id) for class_id in names)
        if isinstance(names, dict)
        else list(range(len(names)))
    )
    if not class_ids:
        return image

    row_height = 28
    panel_height = 12 + row_height * len(class_ids)
    rendered = cv2.copyMakeBorder(
        image,
        0,
        panel_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    for index, class_id in enumerate(class_ids):
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        center_y = image_height + 12 + index * row_height + row_height // 2
        cv2.rectangle(
            rendered,
            (12, center_y - 7),
            (26, center_y + 7),
            color,
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            _class_name(names, class_id),
            (36, center_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return rendered


def save_box_result(result, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), render_box_result(result))


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict with YOLO segmentation model.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to trained weights.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image path, directory, or glob pattern.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="predict")
    parser.add_argument(
        "--preprocess-config",
        type=Path,
        default=DEFAULT_PREPROCESS_CONFIG,
        help="Preprocessing presets YAML.",
    )
    parser.add_argument(
        "--preprocess-preset",
        default=None,
        help="Apply a named preprocessing preset before inference.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save prediction images.",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"Weights not found: {args.weights}")
        print("Train first with: python scripts/train.py")
        return 1

    model = YOLO(str(args.weights))
    if not args.preprocess_preset:
        results = model.predict(
            source=args.source,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            save=False,
            verbose=True,
        )
        if not args.no_save:
            output_dir = Path(args.project) / args.name
            for result in results:
                output_path = output_dir / Path(result.path).name
                if not save_box_result(result, output_path):
                    print(f"Failed to write prediction: {output_path}")
                    return 1
                print(f"Wrote box prediction: {output_path}")
        return 0

    try:
        preset = load_preprocess_preset(args.preprocess_preset, args.preprocess_config)
    except ValueError as exc:
        print(exc)
        return 1
    source_paths = find_images(args.source)
    if not source_paths:
        print(f"No readable image paths found for source: {args.source}")
        return 1

    output_dir = Path(args.project) / args.name
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_paths:
        image = cv2.imread(str(source_path))
        if image is None:
            print(f"Skipping unreadable image: {source_path}")
            continue
        processed = apply_preprocessing(image, preset).image
        results = model.predict(
            source=processed,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            save=False,
            verbose=True,
        )
        if not args.no_save:
            for result in results:
                output_path = output_dir / source_path.name
                if not save_box_result(result, output_path):
                    print(f"Failed to write prediction: {output_path}")
                    return 1
                print(f"Wrote box prediction: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
