"""Scene-aware filtering for forceps pose predictions."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _point_is_on_visible_image(
    image: np.ndarray,
    point: np.ndarray,
    *,
    black_threshold: float,
    sample_radius: int,
) -> bool:
    height, width = image.shape[:2]
    x, y = (float(point[0]), float(point[1]))
    if not np.isfinite(x) or not np.isfinite(y) or (x == 0.0 and y == 0.0):
        return False
    if x < 0 or x >= width or y < 0 or y >= height:
        return False

    center_x = int(round(x))
    center_y = int(round(y))
    x1 = max(0, center_x - sample_radius)
    y1 = max(0, center_y - sample_radius)
    x2 = min(width, center_x + sample_radius + 1)
    y2 = min(height, center_y + sample_radius + 1)
    patch = image[y1:y2, x1:x2]
    return bool(patch.size and float(np.max(patch)) > black_threshold)


def select_expected_pose_indices(
    result: Any,
    *,
    max_per_class: int = 1,
    black_threshold: float = 8.0,
    sample_radius: int = 3,
) -> list[int]:
    """Keep the best valid pose detection per class.

    Synthetic and real frames contain one forceps and one projected shadow.
    Predictions whose keypoints land on the black microscope border are invalid.
    Remaining candidates are ranked by box and keypoint confidence.
    """

    if max_per_class < 1:
        raise ValueError("max_per_class must be at least 1")
    if sample_radius < 0:
        raise ValueError("sample_radius must be non-negative")

    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if (
        boxes is None
        or getattr(boxes, "cls", None) is None
        or keypoints is None
        or getattr(keypoints, "xy", None) is None
    ):
        return []

    class_ids = _to_numpy(boxes.cls).astype(int).reshape(-1)
    box_confidences = (
        _to_numpy(boxes.conf).astype(float).reshape(-1)
        if getattr(boxes, "conf", None) is not None
        else np.ones(len(class_ids), dtype=float)
    )
    coordinates = _to_numpy(keypoints.xy).astype(float)
    keypoint_confidences = (
        _to_numpy(keypoints.conf).astype(float)
        if getattr(keypoints, "conf", None) is not None
        else np.ones(coordinates.shape[:2], dtype=float)
    )
    image = getattr(result, "orig_img", None)
    if (
        image is None
        or coordinates.ndim != 3
        or coordinates.shape[2] != 2
        or len(coordinates) != len(class_ids)
        or len(box_confidences) != len(class_ids)
        or keypoint_confidences.shape[:2] != coordinates.shape[:2]
    ):
        return []

    candidates: dict[int, list[tuple[float, int]]] = {}
    for index, class_id in enumerate(class_ids):
        points = coordinates[index]
        point_confidences = keypoint_confidences[index]
        required_count = min(3, len(points))
        if required_count < 2:
            continue
        required_points = points[:required_count]
        required_confidences = point_confidences[:required_count]
        if np.any(required_confidences <= 0):
            continue
        if not all(
            _point_is_on_visible_image(
                image,
                point,
                black_threshold=black_threshold,
                sample_radius=sample_radius,
            )
            for point in required_points
        ):
            continue

        score = float(box_confidences[index]) * float(
            np.mean(np.clip(required_confidences, 0.0, 1.0))
        )
        candidates.setdefault(int(class_id), []).append((score, index))

    selected = []
    for class_id in sorted(candidates):
        ranked = sorted(candidates[class_id], reverse=True)
        selected.extend(index for _, index in ranked[:max_per_class])
    return sorted(selected)


def filter_expected_pose_result(
    result: Any,
    *,
    max_per_class: int = 1,
    black_threshold: float = 8.0,
    sample_radius: int = 3,
) -> Any:
    """Return a result containing only valid scene-constrained pose detections."""

    if getattr(result, "keypoints", None) is None:
        return result
    indices = select_expected_pose_indices(
        result,
        max_per_class=max_per_class,
        black_threshold=black_threshold,
        sample_radius=sample_radius,
    )
    return result[indices]
