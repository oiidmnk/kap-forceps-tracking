#!/usr/bin/env python3
"""Predict one image and write preprocessor input JSON for 3D visualization."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from scripts.common import REPO_ROOT
from scripts.preprocessing import (
    DEFAULT_PREPROCESS_CONFIG,
    CropTransform,
    apply_preprocessing,
    load_preprocess_preset,
)

WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_WEIGHTS = Path("runs/pose/forceps/weights/best.pt")
DEFAULT_BASE_INPUT = WORKSPACE_ROOT / "preprocessing" / "input_example.json"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "preprocessing" / "predicted_input.json"

POSE_OBJECTS = {
    0: ("forceps", ("left_tip_px", "right_tip_px")),
    1: ("shadow", ("left_shadow_px", "right_shadow_px")),
}


class PredictionExtractionError(ValueError):
    """Raised when a YOLO result cannot produce all required preprocessor points."""


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def restore_original_point(
    point: tuple[float, float], transform: CropTransform
) -> list[float]:
    """Map a point from the preprocessed image back to the source image frame."""
    return [float(point[0] + transform.x), float(point[1] + transform.y)]


def extract_preprocessor_points(
    result: Any,
    transform: CropTransform | None = None,
    kpt_conf: float = 0.0,
) -> dict[str, list[float]]:
    """Extract preprocessor pixel fields from forceps and shadow pose detections."""
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or getattr(keypoints, "xy", None) is None:
        raise PredictionExtractionError("prediction result does not contain pose keypoints")

    keypoint_xy = _to_numpy(keypoints.xy)
    if keypoint_xy.ndim != 3 or keypoint_xy.shape[1] < 2:
        raise PredictionExtractionError(
            "prediction result must contain at least 2 keypoints per detection"
        )

    boxes = getattr(result, "boxes", None)
    if len(keypoint_xy) == 0:
        raise PredictionExtractionError("model returned no pose detections")
    if boxes is None or getattr(boxes, "cls", None) is None:
        raise PredictionExtractionError("prediction result does not contain pose class IDs")

    class_ids = _to_numpy(boxes.cls).astype(int)
    if len(class_ids) != len(keypoint_xy):
        raise PredictionExtractionError("prediction boxes and keypoints differ in length")

    box_conf = (
        _to_numpy(boxes.conf).astype(float)
        if getattr(boxes, "conf", None) is not None
        else np.ones(len(keypoint_xy), dtype=float)
    )
    if len(box_conf) != len(keypoint_xy):
        raise PredictionExtractionError("prediction confidences and keypoints differ in length")

    keypoint_conf = (
        _to_numpy(keypoints.conf).astype(float)
        if getattr(keypoints, "conf", None) is not None
        else np.ones(keypoint_xy.shape[:2], dtype=float)
    )
    if keypoint_conf.shape[0] != keypoint_xy.shape[0]:
        raise PredictionExtractionError("keypoint coordinates and confidences differ in length")
    if keypoint_conf.shape[1] < 2:
        raise PredictionExtractionError(
            "prediction result must contain at least 2 keypoint confidences per detection"
        )

    missing = []
    extracted: dict[str, list[float]] = {}
    transform = transform or CropTransform(
        source_width=0,
        source_height=0,
        x=0,
        y=0,
        width=0,
        height=0,
    )

    for class_id, (class_name, input_keys) in POSE_OBJECTS.items():
        candidate_indices = np.flatnonzero(class_ids == class_id)
        if len(candidate_indices) == 0:
            missing.append(class_name)
            continue

        detection_index = int(candidate_indices[np.argmax(box_conf[candidate_indices])])
        for keypoint_index, input_key in enumerate(input_keys):
            x, y = [float(value) for value in keypoint_xy[detection_index][keypoint_index]]
            confidence = float(keypoint_conf[detection_index][keypoint_index])
            if confidence < kpt_conf or (x == 0.0 and y == 0.0):
                missing.append(input_key)
                continue
            extracted[input_key] = restore_original_point((x, y), transform)

    if missing:
        raise PredictionExtractionError("missing required pose data for: " + ", ".join(missing))
    return extracted


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def build_preprocessor_input(
    base_input: Path,
    sidecar: Path | None,
    predicted_points: dict[str, list[float]],
) -> dict[str, Any]:
    payload = load_json_object(base_input)
    if sidecar is not None:
        payload.update(load_json_object(sidecar))
    payload.update(predicted_points)
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_name = temp_file.name
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")
        os.replace(temp_name, path)
    except Exception:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        raise


def predict_one_image(args: argparse.Namespace) -> dict[str, Any]:
    if not args.source.is_file():
        raise FileNotFoundError(f"source image not found: {args.source}")
    if not args.weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {args.weights}. Train first with: python scripts/train.py"
        )

    image = cv2.imread(str(args.source))
    if image is None:
        raise RuntimeError(f"failed to read image: {args.source}")

    source_height, source_width = image.shape[:2]
    transform = CropTransform(source_width, source_height, 0, 0, source_width, source_height)
    model_input = image
    if args.preprocess_preset:
        preset = load_preprocess_preset(args.preprocess_preset, args.preprocess_config)
        preprocessed = apply_preprocessing(image, preset)
        model_input = preprocessed.image
        transform = preprocessed.transform

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    results = model.predict(
        source=model_input,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=False,
        verbose=True,
    )
    if not results:
        raise PredictionExtractionError("model returned no prediction results")

    predicted_points = extract_preprocessor_points(results[0], transform, args.kpt_conf)
    return build_preprocessor_input(args.base_input, args.sidecar, predicted_points)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one image and write JSON for the preprocessing websocket."
    )
    parser.add_argument("--source", type=Path, required=True, help="Single image path.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to trained YOLO weights.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--kpt-conf",
        type=float,
        default=0.0,
        help="Minimum keypoint confidence required for all four pose points.",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
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
        "--base-input",
        type=Path,
        default=DEFAULT_BASE_INPUT,
        help="Base preprocessor JSON with calibrated geometry values.",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Optional JSON object overriding non-predicted geometry values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Preprocessor input JSON to write atomically.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = predict_one_image(args)
        atomic_write_json(args.output, payload)
    except Exception as exc:  # noqa: BLE001 - keep CLI failure readable
        print(exc, file=sys.stderr)
        return 1

    print(f"Wrote preprocessor input: {args.output}")
    print("Run the websocket server with:")
    print(f"  cd {WORKSPACE_ROOT / 'preprocessing'}")
    print(f"  python3 ws_server.py --input {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
