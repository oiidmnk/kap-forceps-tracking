#!/usr/bin/env python3
"""Run YOLO segmentation inference on images or folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

from scripts.preprocessing import (
    CropTransform,
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
POSE_KEYPOINT_NAMES = {
    0: ("tip_left", "tip_right"),
    1: ("shadow_left", "shadow_right"),
}


def _class_name(names: dict | list, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _transform_points(points: np.ndarray, transform: CropTransform) -> list[list[float]]:
    restored = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    restored[:, 0] += transform.x
    restored[:, 1] += transform.y
    return restored.tolist()


def _normalize_points(points: list[list[float]], width: int, height: int) -> list[list[float]]:
    if width <= 0 or height <= 0:
        return []
    return [[float(x) / width, float(y) / height] for x, y in points]


def _transform_box(box: np.ndarray, transform: CropTransform) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1 + transform.x, y1 + transform.y, x2 + transform.x, y2 + transform.y]


def _normalize_box(box: list[float], width: int, height: int) -> list[float]:
    if width <= 0 or height <= 0:
        return []
    x1, y1, x2, y2 = box
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def serialize_segmentation_result(
    result: Any,
    transform: CropTransform | None = None,
) -> dict[str, Any]:
    """Convert one Ultralytics segmentation result into API-friendly JSON data."""
    image_height, image_width = result.orig_img.shape[:2]
    transform = transform or CropTransform(
        source_width=image_width,
        source_height=image_height,
        x=0,
        y=0,
        width=image_width,
        height=image_height,
    )
    source_width = int(transform.source_width or image_width)
    source_height = int(transform.source_height or image_height)

    boxes = getattr(result, "boxes", None)
    box_xyxy = (
        _to_numpy(boxes.xyxy).astype(float)
        if boxes is not None and getattr(boxes, "xyxy", None) is not None
        else np.empty((0, 4), dtype=float)
    )
    confidences = (
        _to_numpy(boxes.conf).astype(float)
        if boxes is not None and getattr(boxes, "conf", None) is not None
        else np.full(len(box_xyxy), np.nan, dtype=float)
    )
    class_ids = (
        _to_numpy(boxes.cls).astype(int)
        if boxes is not None and getattr(boxes, "cls", None) is not None
        else np.zeros(len(box_xyxy), dtype=int)
    )

    masks = getattr(result, "masks", None)
    mask_polygons = getattr(masks, "xy", None) or []
    instance_count = max(len(box_xyxy), len(mask_polygons))

    instances = []
    for index in range(instance_count):
        class_id = int(class_ids[index]) if index < len(class_ids) else 0
        confidence = float(confidences[index]) if index < len(confidences) else None
        box = _transform_box(box_xyxy[index], transform) if index < len(box_xyxy) else None
        points = (
            _transform_points(np.asarray(mask_polygons[index]), transform)
            if index < len(mask_polygons)
            else []
        )
        instances.append(
            {
                "class_id": class_id,
                "class_name": _class_name(result.names, class_id),
                "confidence": None if confidence is None or np.isnan(confidence) else confidence,
                "box": None
                if box is None
                else {
                    "xyxy": box,
                    "normalized_xyxy": _normalize_box(box, source_width, source_height),
                },
                "segments": [
                    {
                        "points": points,
                        "normalized_points": _normalize_points(
                            points, source_width, source_height
                        ),
                    }
                ]
                if points
                else [],
            }
        )

    return {
        "image": {
            "width": source_width,
            "height": source_height,
            "inference_width": int(image_width),
            "inference_height": int(image_height),
        },
        "transform": {
            "x": int(transform.x),
            "y": int(transform.y),
            "width": int(transform.width),
            "height": int(transform.height),
            "is_identity": transform.is_identity,
        },
        "instances": instances,
    }


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


def render_pose_result(result) -> object:
    """Render pose keypoints with compact labels."""
    image = result.orig_img.copy()
    image_height, image_width = image.shape[:2]
    thickness = max(2, round(min(image_height, image_width) / 500))

    boxes = result.boxes
    keypoints = result.keypoints
    class_ids = np.empty((0,), dtype=int)
    if boxes is not None and boxes.xyxy is not None:
        coordinates = boxes.xyxy.cpu().numpy()
        class_ids = (
            boxes.cls.cpu().numpy().astype(int)
            if boxes.cls is not None
            else np.zeros(len(coordinates), dtype=int)
        )
        for (x1, y1, x2, y2), class_id in zip(coordinates, class_ids, strict=True):
            color = CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]
            cv2.rectangle(
                image,
                (max(0, round(float(x1))), max(0, round(float(y1)))),
                (min(image_width - 1, round(float(x2))), min(image_height - 1, round(float(y2)))),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                image,
                _class_name(result.names, int(class_id)),
                (max(0, round(float(x1))), max(14, round(float(y1)) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    if keypoints is None or keypoints.xy is None:
        return image

    coordinates = keypoints.xy.cpu().numpy()
    confidences = (
        keypoints.conf.cpu().numpy()
        if keypoints.conf is not None
        else np.ones(coordinates.shape[:2], dtype=float)
    )
    if len(class_ids) != len(coordinates):
        class_ids = np.zeros(len(coordinates), dtype=int)
    for detection_points, detection_confidences, class_id in zip(
        coordinates,
        confidences,
        class_ids,
        strict=True,
    ):
        keypoint_names = POSE_KEYPOINT_NAMES.get(int(class_id), ("kp0", "kp1"))
        for keypoint_index, (x, y) in enumerate(detection_points[: len(keypoint_names)]):
            if detection_confidences[keypoint_index] <= 0 or (x == 0 and y == 0):
                continue
            point = (
                max(0, min(image_width - 1, round(float(x)))),
                max(0, min(image_height - 1, round(float(y)))),
            )
            color = CLASS_COLORS[int(class_id) % len(CLASS_COLORS)]
            cv2.circle(image, point, radius=7, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(image, point, radius=5, color=color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(
                image,
                keypoint_names[keypoint_index],
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return image


def save_prediction_result(result, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(result, "keypoints", None) is not None:
        rendered = render_pose_result(result)
    else:
        rendered = render_box_result(result)
    return cv2.imwrite(str(output_path), rendered)


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
                if not save_prediction_result(result, output_path):
                    print(f"Failed to write prediction: {output_path}")
                    return 1
                print(f"Wrote prediction: {output_path}")
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
                if not save_prediction_result(result, output_path):
                    print(f"Failed to write prediction: {output_path}")
                    return 1
                print(f"Wrote prediction: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
