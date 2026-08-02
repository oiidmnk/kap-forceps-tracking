#!/usr/bin/env python3
"""Crop a mono scope video to its tracked circular retinal content."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

INNER_MAX_FRAC = 0.85
INNER_SWEEP = (25, 30, 35, 40, 45, 50, 55, 60, 70, 80)


def detect_circle(bgr: np.ndarray, threshold: int):
    """Fit the largest bright external contour, matching the reference eyeproc logic."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.01 * bgr.shape[0] * bgr.shape[1]:
        return None
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    return (float(cx), float(cy), float(radius)) if radius >= 8 else None


def fit(bgr: np.ndarray, threshold: int, max_radius: float | None):
    circle = detect_circle(bgr, threshold)
    if circle is None or (max_radius is not None and circle[2] >= max_radius):
        return None
    return circle


def sample_frames(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if total <= 0:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame.copy())
    else:
        for index in range(count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index * (total - 1) / max(1, count - 1)))
            ok, frame = capture.read()
            if ok:
                frames.append(frame.copy())
    capture.release()
    return frames


def calibrate_inner(frames: list[np.ndarray], outer_radius: float):
    max_radius = INNER_MAX_FRAC * outer_radius
    required = max(3, len(frames) // 2)
    best = None
    for threshold in INNER_SWEEP:
        radii = sorted(
            circle[2]
            for circle in (detect_circle(frame, threshold) for frame in frames)
            if circle and circle[2] < max_radius
        )
        if len(radii) < required:
            continue
        median = statistics.median(radii)
        spread = radii[int(0.95 * (len(radii) - 1))] - radii[int(0.05 * (len(radii) - 1))]
        score = spread / max(1.0, median)
        print(
            f"[mask] calibrate thresh={threshold:3d} hits {len(radii):3d}/{len(frames)} "
            f"r={median:6.1f} spread={spread:5.1f} score={score:.4f}",
            flush=True,
        )
        if best is None or score < best[0]:
            best = (score, threshold, median)
    if best is None:
        return None, None
    _, threshold, median = best
    print(
        f"[mask] inner circle: thresh={threshold} r={median:.1f} "
        f"(outer r={outer_radius:.0f}, cutoff {max_radius:.0f})",
        flush=True,
    )
    return threshold, median


def scan(path: Path, threshold: int, total: int, max_radius: float | None = None):
    capture = cv2.VideoCapture(str(path))
    circles = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        circles.append(fit(frame, threshold, max_radius))
        print(f"[mask] scan {len(circles)}/{total or len(circles)}", flush=True)
    capture.release()
    return circles


def fill_centres(circles, fallback: tuple[float, float]):
    centres = [None if circle is None else (circle[0], circle[1]) for circle in circles]
    last = None
    for index, centre in enumerate(centres):
        if centre is None:
            centres[index] = last
        else:
            last = centre
    following = None
    for index in range(len(centres) - 1, -1, -1):
        if centres[index] is None:
            centres[index] = following
        else:
            following = centres[index]
    return [centre if centre is not None else fallback for centre in centres]


def smooth_track(centres, window: int):
    if window <= 1 or len(centres) < 2:
        return list(centres)
    half = window // 2
    smoothed = []
    for index in range(len(centres)):
        chunk = centres[max(0, index - half) : min(len(centres), index + half + 1)]
        smoothed.append(
            (sum(c[0] for c in chunk) / len(chunk), sum(c[1] for c in chunk) / len(chunk))
        )
    return smoothed


def cut(frame: np.ndarray, center_x: float, center_y: float, side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    output = np.zeros((side, side, 3), np.uint8)
    x0, y0 = int(round(center_x)) - side // 2, int(round(center_y)) - side // 2
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x0 + side), min(height, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        output[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return output


def circle_mask(side: int, radius: int) -> np.ndarray:
    mask = np.zeros((side, side), np.uint8)
    cv2.circle(mask, (side // 2, side // 2), max(1, radius), 255, -1, lineType=cv2.LINE_AA)
    return (mask.astype(np.float32) / 255.0)[:, :, None]


def open_encoder(output: Path, side: int, fps: float, crf: int, error_file):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{side}x{side}",
        "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", str(output),
    ]
    print("$ " + " ".join(command), flush=True)
    return subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_file
    )


def mask_video(
    source: Path,
    output_dir: Path,
    *,
    circle: str = "outer",
    inner_threshold: int = 0,
    probe: int = 24,
    radius: int = 0,
    track: bool = True,
    smooth: int = 9,
    erode: int = 2,
    threshold: int = 12,
    size: int = 0,
    crf: int = 18,
) -> dict:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    print(f"[mask] {source.name} {width}x{height} fps={fps:.3f} frames={total or 'unknown'}", flush=True)

    used_threshold, max_radius, outer_radius, inner_radius = threshold, None, None, None
    if circle == "inner":
        frames = sample_frames(source, max(4, probe))
        outer_fits = [fit[2] for fit in (detect_circle(frame, threshold) for frame in frames) if fit]
        if not outer_fits:
            raise ValueError("cannot find the outer aperture on probe frames")
        outer_radius = statistics.median(outer_fits)
        if inner_threshold > 0:
            used_threshold = inner_threshold
        else:
            used_threshold, inner_radius = calibrate_inner(frames, outer_radius)
            if used_threshold is None:
                raise ValueError("no threshold fit an inner circle consistently")
        max_radius = INNER_MAX_FRAC * outer_radius

    circles = scan(source, used_threshold, total, max_radius)
    if not circles:
        raise ValueError("no frames decoded")
    found = [item for item in circles if item is not None]
    misses = len(circles) - len(found)
    detected_radius = statistics.median(item[2] for item in found) if found else min(width, height) / 2
    output_radius = radius if radius > 0 else int(round(detected_radius))
    side = 2 * output_radius
    if side % 2:
        side += 1
    if side < 16:
        raise ValueError(f"radius {output_radius} px is too small")

    centres = fill_centres(circles, (width / 2, height / 2))
    if track:
        centres = smooth_track(centres, smooth)
    else:
        fixed = (
            statistics.median(center[0] for center in centres),
            statistics.median(center[1] for center in centres),
        )
        centres = [fixed] * len(circles)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "masked.mp4"
    preview_path = output_dir / "preview.png"
    mask = circle_mask(side, output_radius - max(0, erode))
    output_size = size if size > 0 else side
    if output_size % 2:
        output_size += 1

    capture = cv2.VideoCapture(str(source))
    error_file = tempfile.TemporaryFile("w+")
    encoder = open_encoder(output_path, output_size, fps, crf, error_file)
    written, last = 0, None
    try:
        for index, center in enumerate(centres):
            ok, frame = capture.read()
            if not ok:
                break
            rendered = (cut(frame, center[0], center[1], side) * mask).astype(np.uint8)
            if output_size != side:
                rendered = cv2.resize(rendered, (output_size, output_size), interpolation=cv2.INTER_AREA)
            if index == len(centres) // 2:
                cv2.imwrite(str(preview_path), rendered)
            last = rendered
            try:
                encoder.stdin.write(np.ascontiguousarray(rendered).tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError("FFmpeg stopped while receiving frames") from exc
            written += 1
            print(f"[mask] frame {written}/{len(centres)}", flush=True)
    finally:
        capture.release()
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()
        code = encoder.wait()
    if code != 0:
        error_file.seek(0)
        detail = "\n".join(error_file.read().strip().splitlines()[-15:])
        raise RuntimeError(f"FFmpeg failed with status {code}: {detail}")
    error_file.close()
    if not preview_path.exists() and last is not None:
        cv2.imwrite(str(preview_path), last)

    metadata = {
        "source": str(source), "source_size": [width, height], "frames": written,
        "fps": round(fps, 4), "radius": output_radius,
        "radius_source": "manual" if radius > 0 else "median-detected",
        "radius_detected_median": round(detected_radius, 2), "detections": len(found),
        "detection_misses": misses, "track": "on" if track else "off", "smooth": smooth,
        "erode": erode, "circle": circle, "thresh": used_threshold,
        "outer_thresh": threshold, "outer_radius": None if outer_radius is None else round(outer_radius, 2),
        "inner_radius_probe": None if inner_radius is None else round(inner_radius, 2),
        "out_size": output_size, "centres_first": [round(value, 2) for value in centres[0]],
        "centres_last": [round(value, 2) for value in centres[-1]],
    }
    (output_dir / "mask.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[mask] wrote {output_path} {output_size}x{output_size} frames={written} r={output_radius}", flush=True)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Mask and crop one circular scope video.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--circle", choices=("outer", "inner"), default="outer")
    parser.add_argument("--inner-thresh", type=int, default=0)
    parser.add_argument("--probe", type=int, default=24)
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument("--no-track", dest="track", action="store_false")
    parser.add_argument("--smooth", type=int, default=9)
    parser.add_argument("--erode", type=int, default=2)
    parser.add_argument("--thresh", type=int, default=12)
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--crf", type=int, default=18)
    parser.set_defaults(track=True)
    args = parser.parse_args()
    try:
        mask_video(
            args.source, args.output_dir, circle=args.circle, inner_threshold=args.inner_thresh,
            probe=args.probe, radius=args.radius, track=args.track, smooth=args.smooth,
            erode=args.erode, threshold=args.thresh, size=args.size, crf=args.crf,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[mask] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
