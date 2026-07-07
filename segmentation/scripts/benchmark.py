#!/usr/bin/env python3
"""Benchmark YOLO segmentation inference, optionally including preprocessing."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

from scripts.predict import DEFAULT_WEIGHTS
from scripts.preprocessing import (
    DEFAULT_PREPROCESS_CONFIG,
    apply_preprocessing,
    find_images,
    load_preprocess_preset,
)

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def sync_device(device: str | None) -> None:
    if not device or not str(device).startswith(("cuda", "0", "1", "2", "3")):
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_frames(
    source: str,
    max_frames: int | None,
    stride: int,
) -> list[tuple[str, object]]:
    path = Path(source)
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        frames: list[tuple[str, object]] = []
        capture = cv2.VideoCapture(source)
        frame_index = 0
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                frames.append((f"{source}#{frame_index}", frame))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            frame_index += 1
        capture.release()
        return frames

    frames = []
    for image_path in find_images(source):
        if max_frames is not None and len(frames) >= max_frames:
            break
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        frames.append((str(image_path), image))
    return frames


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["frame", "iteration", "duration_ms", "fps"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO segmentation inference speed."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--preprocess-config", type=Path, default=DEFAULT_PREPROCESS_CONFIG
    )
    parser.add_argument("--preprocess-preset", default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"Weights not found: {args.weights}")
        return 1
    if args.repeat < 1 or args.warmup < 0 or args.stride < 1:
        print("--repeat and --stride must be >= 1; --warmup must be >= 0")
        return 1

    frames = load_frames(args.source, args.max_frames, args.stride)
    if not frames:
        print(f"No readable frames found for source: {args.source}")
        return 1
    preset = None
    if args.preprocess_preset:
        try:
            preset = load_preprocess_preset(
                args.preprocess_preset, args.preprocess_config
            )
        except ValueError as exc:
            print(exc)
            return 1

    def model_input(frame):
        return apply_preprocessing(frame, preset).image if preset is not None else frame

    model = YOLO(str(args.weights))
    for _ in range(args.warmup):
        model.predict(
            model_input(frames[0][1]),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            save=False,
            verbose=False,
        )
    sync_device(args.device)

    rows: list[dict[str, float | int | str]] = []
    durations_ms: list[float] = []
    started = time.perf_counter()
    for iteration in range(args.repeat):
        for frame_name, frame in frames:
            sync_device(args.device)
            frame_started = time.perf_counter()
            model.predict(
                model_input(frame),
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                save=False,
                verbose=False,
            )
            sync_device(args.device)
            duration_ms = (time.perf_counter() - frame_started) * 1000.0
            durations_ms.append(duration_ms)
            rows.append(
                {
                    "frame": frame_name,
                    "iteration": iteration + 1,
                    "duration_ms": duration_ms,
                    "fps": 1000.0 / duration_ms if duration_ms else 0.0,
                }
            )

    total_seconds = time.perf_counter() - started
    print(f"Weights:        {args.weights}")
    print(f"Preprocessing:  {args.preprocess_preset or 'none'}")
    print(f"Timed frames:   {len(durations_ms)}")
    print(f"Throughput:     {len(durations_ms) / total_seconds:.2f} frames/s")
    print(f"Mean duration:  {statistics.fmean(durations_ms):.2f} ms/frame")
    print(f"Median:         {statistics.median(durations_ms):.2f} ms/frame")
    print(f"P95:            {percentile(durations_ms, 0.95):.2f} ms/frame")
    print(f"Min/Max:        {min(durations_ms):.2f} / {max(durations_ms):.2f} ms/frame")
    if args.csv:
        write_csv(args.csv, rows)
        print(f"Wrote timings:  {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
