"""Confidence-aware temporal tracking for forceps pose video predictions."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from scripts.prediction_filtering import select_expected_pose_indices


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class _NumpyTensor:
    """Small tensor adapter for the existing pose renderer."""

    def __init__(self, values: Any) -> None:
        self._values = np.asarray(values)

    def cpu(self) -> "_NumpyTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._values


@dataclass
class _Observation:
    points: np.ndarray
    point_confidences: np.ndarray
    box: np.ndarray
    quality: float


@dataclass
class _Track:
    class_id: int
    points: np.ndarray
    velocity: np.ndarray
    point_confidences: np.ndarray
    box: np.ndarray
    quality: float
    misses: int = 0


class TemporalPoseTracker:
    """Track one forceps and one shadow pose with an alpha-beta filter.

    Candidates must first satisfy the regular scene constraints. Established
    tracks then accept only candidates close to their constant-velocity
    prediction, which prevents a single distant false positive from replacing
    a stable pose. Short gaps are bridged with decaying confidence.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.65,
        beta: float = 0.15,
        max_gap: int = 3,
        max_jump_fraction: float = 0.12,
        max_candidates_per_class: int = 5,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        if max_gap < 0:
            raise ValueError("max_gap must be non-negative")
        if max_jump_fraction <= 0:
            raise ValueError("max_jump_fraction must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_gap = int(max_gap)
        self.max_jump_fraction = float(max_jump_fraction)
        self.max_candidates_per_class = int(max_candidates_per_class)
        self._tracks: dict[int, _Track] = {}
        self._previous_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._previous_gray = None

    @staticmethod
    def _canonicalize(points: np.ndarray, confidences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keep the image-frame left endpoint before the right endpoint."""
        points = points.copy()
        confidences = confidences.copy()
        if len(points) >= 2 and points[0, 0] > points[1, 0]:
            points[[0, 1]] = points[[1, 0]]
            confidences[[0, 1]] = confidences[[1, 0]]
        return points, confidences

    def _observations(self, result: Any) -> dict[int, list[_Observation]]:
        indices = select_expected_pose_indices(
            result,
            max_per_class=self.max_candidates_per_class,
        )
        boxes = result.boxes
        keypoints = result.keypoints
        class_ids = _to_numpy(boxes.cls).astype(int).reshape(-1)
        box_confidences = _to_numpy(boxes.conf).astype(float).reshape(-1)
        box_coordinates = (
            _to_numpy(boxes.xyxy).astype(float)
            if getattr(boxes, "xyxy", None) is not None
            else None
        )
        points = _to_numpy(keypoints.xy).astype(float)
        point_confidences = (
            _to_numpy(keypoints.conf).astype(float)
            if getattr(keypoints, "conf", None) is not None
            else np.ones(points.shape[:2], dtype=float)
        )

        observations: dict[int, list[_Observation]] = {}
        for index in indices:
            candidate_points, candidate_confidences = self._canonicalize(
                points[index], point_confidences[index]
            )
            if box_coordinates is not None:
                box = box_coordinates[index].copy()
            else:
                minimum = np.min(candidate_points, axis=0)
                maximum = np.max(candidate_points, axis=0)
                box = np.concatenate([minimum, maximum])
            quality = float(box_confidences[index]) * float(
                np.mean(np.clip(candidate_confidences, 0.0, 1.0))
            )
            observations.setdefault(int(class_ids[index]), []).append(
                _Observation(candidate_points, candidate_confidences, box, quality)
            )
        return observations

    @staticmethod
    def _new_track(class_id: int, observation: _Observation) -> _Track:
        return _Track(
            class_id=class_id,
            points=observation.points.copy(),
            velocity=np.zeros_like(observation.points),
            point_confidences=observation.point_confidences.copy(),
            box=observation.box.copy(),
            quality=observation.quality,
        )

    @staticmethod
    def _distance(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape:
            return float("inf")
        return float(np.mean(np.linalg.norm(first - second, axis=1)))

    def _choose_candidate(
        self,
        track: _Track,
        candidates: list[_Observation],
        diagonal: float,
        predicted: np.ndarray,
    ) -> _Observation | None:
        gate = self.max_jump_fraction * diagonal * (1.0 + 0.35 * track.misses)
        eligible = [
            candidate
            for candidate in candidates
            if self._distance(candidate.points, predicted) <= gate
        ]
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda candidate: (
                self._distance(candidate.points, predicted) / max(diagonal, 1.0)
                - 0.05 * candidate.quality
            ),
        )

    def _correct(
        self,
        track: _Track,
        observation: _Observation,
        predicted: np.ndarray,
    ) -> None:
        previous_points = track.points.copy()
        residual = observation.points - predicted
        confidence_scale = 0.5 + 0.5 * float(np.clip(observation.quality, 0.0, 1.0))
        effective_alpha = self.alpha * confidence_scale
        track.points = predicted + effective_alpha * residual
        measured_velocity = observation.points - previous_points
        track.velocity = (1.0 - self.beta) * track.velocity + self.beta * measured_velocity
        track.point_confidences = observation.point_confidences.copy()
        track.box = track.box + effective_alpha * (observation.box - track.box)
        track.quality = observation.quality
        track.misses = 0

    def _coast(
        self,
        track: _Track,
        predicted: np.ndarray,
        width: int,
        height: int,
    ) -> None:
        previous_points = track.points.copy()
        track.points = predicted.copy()
        track.points[:, 0] = np.clip(track.points[:, 0], 0, max(0, width - 1))
        track.points[:, 1] = np.clip(track.points[:, 1], 0, max(0, height - 1))
        measured_velocity = track.points - previous_points
        track.velocity = (1.0 - self.beta) * track.velocity + self.beta * measured_velocity
        average_velocity = np.mean(measured_velocity, axis=0)
        track.box[[0, 2]] += average_velocity[0]
        track.box[[1, 3]] += average_velocity[1]
        track.point_confidences *= 0.6
        track.quality *= 0.6
        track.misses += 1

    def _flow_predictions(self, current_gray: np.ndarray) -> dict[int, np.ndarray]:
        predictions = {
            class_id: track.points + track.velocity
            for class_id, track in self._tracks.items()
        }
        if self._previous_gray is None or self._previous_gray.shape != current_gray.shape:
            return predictions

        height, width = current_gray.shape[:2]
        for class_id, track in self._tracks.items():
            next_points, status, errors = cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                current_gray,
                track.points.astype(np.float32).reshape(-1, 1, 2),
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if next_points is None or status is None:
                continue
            flowed = next_points.reshape(-1, 2).astype(float)
            valid = status.reshape(-1).astype(bool)
            if errors is not None:
                valid &= errors.reshape(-1) <= 30.0
            valid &= np.isfinite(flowed).all(axis=1)
            valid &= (flowed[:, 0] >= 0) & (flowed[:, 0] < width)
            valid &= (flowed[:, 1] >= 0) & (flowed[:, 1] < height)
            displacement = np.linalg.norm(flowed - track.points, axis=1)
            valid &= displacement <= self.max_jump_fraction * float(np.hypot(width, height))
            predictions[class_id][valid] = flowed[valid]
        return predictions

    def update(self, result: Any) -> Any:
        """Update tracks and return a renderer-compatible pose result."""
        image = result.orig_img
        height, width = image.shape[:2]
        diagonal = float(np.hypot(width, height))
        current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        predictions = self._flow_predictions(current_gray)
        observations = self._observations(result)
        class_ids = sorted(set(self._tracks) | set(observations))

        for class_id in class_ids:
            candidates = observations.get(class_id, [])
            track = self._tracks.get(class_id)
            if track is None:
                if candidates:
                    self._tracks[class_id] = self._new_track(
                        class_id, max(candidates, key=lambda candidate: candidate.quality)
                    )
                continue

            predicted = predictions[class_id]
            candidate = self._choose_candidate(track, candidates, diagonal, predicted)
            if candidate is not None:
                self._correct(track, candidate, predicted)
            elif track.misses < self.max_gap:
                self._coast(track, predicted, width, height)
            elif candidates:
                self._tracks[class_id] = self._new_track(
                    class_id, max(candidates, key=lambda item: item.quality)
                )
            else:
                del self._tracks[class_id]

        active = [self._tracks[class_id] for class_id in sorted(self._tracks)]
        keypoint_count = max((len(track.points) for track in active), default=0)
        if active:
            points = np.stack([track.points for track in active])
            confidences = np.stack([track.point_confidences for track in active])
            boxes = np.stack([track.box for track in active])
        else:
            points = np.empty((0, keypoint_count, 2), dtype=float)
            confidences = np.empty((0, keypoint_count), dtype=float)
            boxes = np.empty((0, 4), dtype=float)

        self._previous_gray = current_gray
        return SimpleNamespace(
            orig_img=image,
            names=result.names,
            boxes=SimpleNamespace(
                xyxy=_NumpyTensor(boxes),
                cls=_NumpyTensor([track.class_id for track in active]),
                conf=_NumpyTensor([track.quality for track in active]),
            ),
            keypoints=SimpleNamespace(
                xy=_NumpyTensor(points),
                conf=_NumpyTensor(confidences),
            ),
        )
