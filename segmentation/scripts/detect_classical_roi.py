#!/usr/bin/env python3
"""Find coarse forceps and shadow boxes using classical OpenCV heuristics."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

Box = tuple[int, int, int, int]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


@dataclass(frozen=True)
class ClassicalRoiResult:
    forceps_box: Box | None
    shadow_box: Box | None
    roi_box: Box | None
    forceps_mask: np.ndarray
    edge_mask: np.ndarray

    def as_dict(self) -> dict[str, list[int] | None]:
        return {
            "forceps_box": list(self.forceps_box) if self.forceps_box else None,
            "shadow_box": list(self.shadow_box) if self.shadow_box else None,
            "roi_box": list(self.roi_box) if self.roi_box else None,
        }


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _pad_box(box: Box, padding: int, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width - 1, x2 + padding),
        min(height - 1, y2 + padding),
    )


def _union_boxes(boxes: list[Box], padding: int, width: int, height: int) -> Box | None:
    if not boxes:
        return None
    return _pad_box(
        (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ),
        padding,
        width,
        height,
    )


def _touches_border(box: Box, width: int, height: int, margin: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= width - 1 - margin or y2 >= height - 1 - margin


def detect_forceps_box(
    image: np.ndarray,
    *,
    max_saturation: int = 190,
    max_value: int = 165,
    min_area_fraction: float = 0.0025,
    padding_fraction: float = 0.035,
) -> tuple[Box | None, np.ndarray]:
    """Find the broad, weakly saturated instrument component connected to an image edge."""

    height, width = image.shape[:2]
    scale = min(width, height)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] <= max_saturation) & (hsv[:, :, 2] <= max_value)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    close_size = _odd(round(scale * 0.03))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(40, int(round(width * height * min_area_fraction)))
    border_margin = max(2, int(round(scale * 0.025)))
    candidates: list[tuple[float, int, Box]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        if area < min_area:
            continue
        box = (x, y, x + component_width - 1, y + component_height - 1)
        longest = max(component_width, component_height)
        shortest = max(1, min(component_width, component_height))
        elongation = min(4.0, longest / shortest)
        border_bonus = 2.0 if _touches_border(box, width, height, border_margin) else 1.0
        candidates.append((area * border_bonus * (1.0 + 0.12 * elongation), label, box))

    if not candidates:
        return None, np.zeros_like(mask)
    _, selected_label, box = max(candidates, key=lambda item: item[0])
    selected_mask = np.where(labels == selected_label, 255, 0).astype(np.uint8)
    padding = max(4, int(round(scale * padding_fraction)))
    return _pad_box(box, padding, width, height), selected_mask


def _line_angle(line: np.ndarray) -> float:
    x1, y1, x2, y2 = map(int, line)
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)


def _midpoint_inside(line: np.ndarray, box: Box | None) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = map(float, line)
    middle_x, middle_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return box[0] <= middle_x <= box[2] and box[1] <= middle_y <= box[3]


def detect_shadow_box(
    image: np.ndarray,
    forceps_box: Box | None,
    *,
    angle_bin_degrees: float = 10.0,
    canny_low: int = 25,
    canny_high: int = 70,
    padding_fraction: float = 0.04,
) -> tuple[Box | None, np.ndarray]:
    """Find the strongest non-forceps cluster of long, similarly oriented shadow edges."""

    height, width = image.shape[:2]
    scale = min(width, height)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    min_line_length = max(24, int(round(scale * 0.12)))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360.0,
        threshold=max(18, int(round(scale * 0.08))),
        minLineLength=min_line_length,
        maxLineGap=max(8, int(round(scale * 0.045))),
    )
    if lines is None:
        return None, edges

    clusters: dict[int, list[np.ndarray]] = {}
    bin_count = max(1, int(round(180.0 / angle_bin_degrees)))
    for line in lines.reshape(-1, 4):
        if _midpoint_inside(line, forceps_box):
            continue
        angle = _line_angle(line)
        bucket = int(round(angle / angle_bin_degrees)) % bin_count
        clusters.setdefault(bucket, []).append(line)

    border_margin = max(3, int(round(scale * 0.025)))
    scored: list[tuple[float, Box]] = []
    for cluster_lines in clusters.values():
        if len(cluster_lines) < 2:
            continue
        xs = [int(value) for line in cluster_lines for value in (line[0], line[2])]
        ys = [int(value) for line in cluster_lines for value in (line[1], line[3])]
        box = (min(xs), min(ys), max(xs), max(ys))
        total_length = sum(float(np.hypot(line[2] - line[0], line[3] - line[1])) for line in cluster_lines)
        span = float(np.hypot(box[2] - box[0], box[3] - box[1]))
        if total_length < scale * 0.45 or span < scale * 0.3:
            continue
        border_bonus = 1.7 if _touches_border(box, width, height, border_margin) else 1.0
        score = total_length * border_bonus * (1.0 + min(1.0, span / max(width, height)))
        scored.append((score, box))

    if not scored:
        return None, edges
    _, box = max(scored, key=lambda item: item[0])
    padding = max(4, int(round(scale * padding_fraction)))
    return _pad_box(box, padding, width, height), edges


def detect_classical_roi(
    image: np.ndarray,
    *,
    forceps_max_saturation: int = 190,
    forceps_max_value: int = 165,
    canny_low: int = 25,
    canny_high: int = 70,
) -> ClassicalRoiResult:
    if image is None or image.size == 0:
        raise ValueError("cannot detect objects in an empty image")
    height, width = image.shape[:2]
    forceps_box, forceps_mask = detect_forceps_box(
        image,
        max_saturation=forceps_max_saturation,
        max_value=forceps_max_value,
    )
    shadow_box, edge_mask = detect_shadow_box(
        image,
        forceps_box,
        canny_low=canny_low,
        canny_high=canny_high,
    )
    roi_padding = max(4, int(round(min(width, height) * 0.04)))
    roi_box = _union_boxes(
        [box for box in (forceps_box, shadow_box) if box is not None],
        roi_padding,
        width,
        height,
    )
    return ClassicalRoiResult(forceps_box, shadow_box, roi_box, forceps_mask, edge_mask)


def _smooth_box(previous: Box | None, current: Box | None, alpha: float) -> Box | None:
    if current is None:
        return previous
    if previous is None:
        return current
    return tuple(
        int(round(alpha * old + (1.0 - alpha) * new))
        for old, new in zip(previous, current, strict=True)
    )


@dataclass
class TemporalBoxes:
    alpha: float = 0.65
    max_gap: int = 5
    forceps_box: Box | None = None
    shadow_box: Box | None = None
    forceps_gap: int = 0
    shadow_gap: int = 0

    def _update_one(self, previous: Box | None, current: Box | None, gap: int) -> tuple[Box | None, int]:
        if current is not None:
            return _smooth_box(previous, current, self.alpha), 0
        gap += 1
        return (previous, gap) if gap <= self.max_gap else (None, gap)

    def update(self, result: ClassicalRoiResult, width: int, height: int) -> ClassicalRoiResult:
        self.forceps_box, self.forceps_gap = self._update_one(
            self.forceps_box, result.forceps_box, self.forceps_gap
        )
        self.shadow_box, self.shadow_gap = self._update_one(
            self.shadow_box, result.shadow_box, self.shadow_gap
        )
        padding = max(4, int(round(min(width, height) * 0.04)))
        roi_box = _union_boxes(
            [box for box in (self.forceps_box, self.shadow_box) if box is not None],
            padding,
            width,
            height,
        )
        return ClassicalRoiResult(
            self.forceps_box,
            self.shadow_box,
            roi_box,
            result.forceps_mask,
            result.edge_mask,
        )


def annotate(image: np.ndarray, result: ClassicalRoiResult) -> np.ndarray:
    output = image.copy()
    styles = (
        (result.forceps_box, (60, 220, 60), "forceps"),
        (result.shadow_box, (255, 170, 40), "shadow"),
        (result.roi_box, (210, 80, 230), "pose ROI"),
    )
    for box, color, label in styles:
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(output, label, (x1 + 4, max(18, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output


def _make_browser_compatible_video(output_path: Path, source_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        print("FFmpeg not available; leaving video in OpenCV MPEG-4 format", flush=True)
        return
    temporary = output_path.with_name(f"{output_path.stem}.h264.tmp.mp4")
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(output_path), "-i", str(source_path),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            "-shortest", str(temporary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        print(f"Warning: H.264 encoding failed: {detail}", flush=True)
        return
    temporary.replace(output_path)


def process_image(
    source: Path,
    output_dir: Path,
    **detection_options: int,
) -> tuple[Path, dict[str, object]]:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = detect_classical_roi(image, **detection_options)
    output_path = output_dir / f"{source.stem}_classical_roi.png"
    if not cv2.imwrite(str(output_path), annotate(image, result)):
        raise RuntimeError(f"could not write output: {output_path}")
    (output_dir / "detections.json").write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    cv2.imwrite(str(output_dir / "forceps_mask.png"), result.forceps_mask)
    cv2.imwrite(str(output_dir / "shadow_edges.png"), result.edge_mask)
    summary: dict[str, object] = {
        "frames": 1,
        "forceps_detections": int(result.forceps_box is not None),
        "shadow_detections": int(result.shadow_box is not None),
        "media_type": "image",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(result.as_dict()), flush=True)
    return output_path, summary


def process_video(
    source: Path,
    output_dir: Path,
    *,
    temporal_smoothing: bool = True,
    temporal_alpha: float = 0.65,
    temporal_max_gap: int = 5,
    browser_compatible: bool = True,
    **detection_options: int,
) -> tuple[Path, dict[str, object]]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 30.0
    total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = output_dir / f"{source.stem}_classical_roi.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create output video: {output_path}")

    tracker = TemporalBoxes(alpha=temporal_alpha, max_gap=temporal_max_gap) if temporal_smoothing else None
    completed = forceps_detections = shadow_detections = 0
    started = time.monotonic()
    detections_path = output_dir / "detections.jsonl"
    try:
        with detections_path.open("w") as detections_file:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = detect_classical_roi(frame, **detection_options)
                if result.forceps_box is not None:
                    forceps_detections += 1
                if result.shadow_box is not None:
                    shadow_detections += 1
                if tracker is not None:
                    result = tracker.update(result, width, height)
                writer.write(annotate(frame, result))
                detections_file.write(
                    json.dumps({"frame": completed, **result.as_dict()}, separators=(",", ":")) + "\n"
                )
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == total:
                    elapsed = time.monotonic() - started
                    rate = completed / elapsed if elapsed else 0.0
                    print(
                        f"Classical ROI video: {completed}/{total or '?'} frames ({rate:.1f} fps)",
                        flush=True,
                    )
    finally:
        capture.release()
        writer.release()
    if completed == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"video contains no readable frames: {source}")
    if browser_compatible:
        _make_browser_compatible_video(output_path, source)
    summary = {
        "frames": completed,
        "forceps_detections": forceps_detections,
        "shadow_detections": shadow_detections,
        "media_type": "video",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote classical ROI video: {output_path}", flush=True)
    return output_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect coarse forceps and shadow boxes without machine learning."
    )
    parser.add_argument("--source", type=Path, required=True, help="Input image or video.")
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", type=Path, help="Annotated image output (legacy mode).")
    output_group.add_argument("--output-dir", type=Path, help="Directory for media and JSON outputs.")
    parser.add_argument("--json", type=Path, help="Optional JSON file containing the boxes.")
    parser.add_argument("--debug-dir", type=Path, help="Optionally write forceps and edge masks.")
    parser.add_argument("--forceps-max-saturation", type=int, default=190)
    parser.add_argument("--forceps-max-value", type=int, default=165)
    parser.add_argument("--canny-low", type=int, default=25)
    parser.add_argument("--canny-high", type=int, default=70)
    parser.add_argument("--temporal-alpha", type=float, default=0.65)
    parser.add_argument("--temporal-max-gap", type=int, default=5)
    parser.add_argument("--no-temporal-smoothing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = args.source.suffix.lower()
    if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        raise SystemExit(f"Unsupported media type: {suffix or 'none'}")
    options = {
        "forceps_max_saturation": args.forceps_max_saturation,
        "forceps_max_value": args.forceps_max_value,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
    }
    if args.output_dir:
        if suffix in VIDEO_SUFFIXES:
            process_video(
                args.source,
                args.output_dir,
                temporal_smoothing=not args.no_temporal_smoothing,
                temporal_alpha=args.temporal_alpha,
                temporal_max_gap=args.temporal_max_gap,
                **options,
            )
        else:
            process_image(args.source, args.output_dir, **options)
        return

    image = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.source}")
    result = detect_classical_roi(image, **options)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotate(image, result)):
        raise SystemExit(f"Could not write output: {args.output}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.debug_dir / "forceps_mask.png"), result.forceps_mask)
        cv2.imwrite(str(args.debug_dir / "shadow_edges.png"), result.edge_mask)
    print(json.dumps(result.as_dict()))


if __name__ == "__main__":
    main()
