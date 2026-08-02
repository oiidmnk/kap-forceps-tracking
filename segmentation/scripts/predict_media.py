#!/usr/bin/env python3
"""Run one trained model on an image or video and save annotated media."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

from scripts.predict import render_box_result, render_pose_result
from scripts.prediction_filtering import filter_expected_pose_result
from scripts.preprocessing import DEFAULT_PREPROCESS_CONFIG, apply_preprocessing, load_preprocess_preset
from scripts.temporal_pose import TemporalPoseTracker

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def make_browser_compatible_video(output_path: Path, source_path: Path) -> None:
    """Atomically replace an OpenCV MP4 with H.264 and optional source audio."""
    if shutil.which("ffmpeg") is None:
        print("FFmpeg not available; leaving video in OpenCV MPEG-4 format", flush=True)
        return
    temporary = output_path.with_name(f"{output_path.stem}.h264.tmp.mp4")
    print("Encoding browser-compatible H.264 video…", flush=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(output_path),
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(temporary),
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


def render_result(result, scene_filter: bool):
    if scene_filter:
        result = filter_expected_pose_result(result)
    return render_pose_result(result) if getattr(result, "keypoints", None) is not None else render_box_result(result)


def predict_result(model, frame, args, preset):
    source = apply_preprocessing(frame, preset).image if preset is not None else frame
    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        save=False,
        verbose=False,
    )
    if not results:
        raise RuntimeError("model returned no prediction result")
    return results[0]


def track_result(model, frame, args, preset):
    """Track one frame while preserving state in Ultralytics between calls."""
    source = apply_preprocessing(frame, preset).image if preset is not None else frame
    results = model.track(
        source=source,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        save=False,
        verbose=False,
    )
    if not results:
        raise RuntimeError("model returned no tracking result")
    return results[0]


def predict_frame(model, frame, args, preset):
    return render_result(predict_result(model, frame, args, preset), args.scene_filter)


def predict_image(model, source: Path, output_dir: Path, args, preset) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"could not read image: {source}")
    started = time.monotonic()
    rendered = predict_frame(model, image, args, preset)
    output_path = output_dir / f"{source.stem}_prediction{source.suffix.lower()}"
    if not cv2.imwrite(str(output_path), rendered):
        raise RuntimeError(f"failed to write prediction: {output_path}")
    print(f"Predicted 1 frame in {time.monotonic() - started:.2f}s")
    print(f"Wrote prediction: {output_path}")
    return output_path


def predict_video(model, source: Path, output_dir: Path, args, preset) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    output_path = output_dir / f"{source.stem}_prediction.mp4"
    writer = None
    completed = 0
    started = time.monotonic()
    temporal_filter = getattr(args, "temporal_filter", False)
    tracker = (
        TemporalPoseTracker(
            alpha=args.temporal_alpha,
            beta=args.temporal_beta,
            max_gap=args.temporal_max_gap,
            max_jump_fraction=args.temporal_max_jump,
        )
        if temporal_filter
        else None
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if tracker is None:
                rendered = predict_frame(model, frame, args, preset)
            else:
                result = track_result(model, frame, args, preset)
                rendered = (
                    render_pose_result(tracker.update(result))
                    if getattr(result, "keypoints", None) is not None
                    else render_box_result(result)
                )
            if writer is None:
                height, width = rendered.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not create output video: {output_path}")
            writer.write(rendered)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == total:
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0
                print(f"Predicting video: {completed}/{total or '?'} frames ({rate:.1f} fps)", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if completed == 0:
        raise ValueError(f"video contains no readable frames: {source}")
    make_browser_compatible_video(output_path, source)
    print(f"Wrote prediction video: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict one image or video.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--preprocess-config", type=Path, default=DEFAULT_PREPROCESS_CONFIG)
    parser.add_argument("--preprocess-preset", default=None)
    parser.add_argument("--no-scene-filter", dest="scene_filter", action="store_false")
    parser.add_argument(
        "--no-temporal-filter",
        dest="temporal_filter",
        action="store_false",
        help="Disable temporal pose smoothing and short-gap recovery for videos.",
    )
    parser.add_argument("--temporal-alpha", type=float, default=0.65)
    parser.add_argument("--temporal-beta", type=float, default=0.15)
    parser.add_argument("--temporal-max-gap", type=int, default=3)
    parser.add_argument("--temporal-max-jump", type=float, default=0.12)
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        help="Ultralytics tracker config used for video temporal filtering.",
    )
    parser.set_defaults(scene_filter=True)
    parser.set_defaults(temporal_filter=True)
    args = parser.parse_args()

    if not args.weights.is_file() or not args.source.is_file():
        print("Weights or source file not found", file=sys.stderr)
        return 1
    suffix = args.source.suffix.lower()
    if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        print(f"Unsupported media type: {suffix or 'none'}", file=sys.stderr)
        return 1
    try:
        preset = load_preprocess_preset(args.preprocess_preset, args.preprocess_config) if args.preprocess_preset else None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(args.weights))
        if suffix in VIDEO_SUFFIXES:
            predict_video(model, args.source, args.output_dir, args, preset)
        else:
            predict_image(model, args.source, args.output_dir, args, preset)
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
