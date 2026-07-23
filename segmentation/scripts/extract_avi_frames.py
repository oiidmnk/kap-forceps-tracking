#!/usr/bin/env python3
"""Extract image frames from an AVI video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def output_params(
    cv2_module: Any, extension: str, quality: int, png_compression: int
) -> list[int]:
    ext = extension.lower()
    if ext in {".jpg", ".jpeg"}:
        return [cv2_module.IMWRITE_JPEG_QUALITY, quality]
    if ext == ".png":
        return [cv2_module.IMWRITE_PNG_COMPRESSION, png_compression]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from a .avi video into image files."
    )
    parser.add_argument("video", type=Path, help="Path to the input .avi video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("frames"),
        help="Directory where extracted frames are written.",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Output filename prefix, e.g. frame_000001.jpg.",
    )
    parser.add_argument(
        "--ext",
        choices=(".jpg", ".jpeg", ".png"),
        default=".jpg",
        help="Output image extension.",
    )
    parser.add_argument(
        "--every",
        type=positive_int,
        default=1,
        help="Extract every Nth frame.",
    )
    parser.add_argument(
        "--start",
        type=non_negative_int,
        default=0,
        help="First zero-based frame index to extract.",
    )
    parser.add_argument(
        "--end",
        type=non_negative_int,
        help="Last zero-based frame index to extract, inclusive.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing image files.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        choices=range(1, 101),
        metavar="[1-100]",
        help="JPEG quality for .jpg/.jpeg output.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=3,
        choices=range(0, 10),
        metavar="[0-9]",
        help="PNG compression level for .png output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        print(f"Input video does not exist: {args.video}", file=sys.stderr)
        return 1
    if not args.video.is_file():
        print(f"Input path is not a file: {args.video}", file=sys.stderr)
        return 1
    if args.video.suffix.lower() != ".avi":
        print(f"Expected a .avi video, got: {args.video.name}", file=sys.stderr)
        return 1
    if args.end is not None and args.end < args.start:
        print("--end must be greater than or equal to --start", file=sys.stderr)
        return 1

    try:
        import cv2
    except ImportError:
        print(
            "OpenCV is required. Install the segmentation package with `pip install -e .`.",
            file=sys.stderr,
        )
        return 1

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"Could not open video: {args.video}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = output_params(cv2, args.ext, args.jpeg_quality, args.png_compression)

    frame_index = 0
    extracted = 0
    skipped_existing = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index < args.start:
                frame_index += 1
                continue
            if args.end is not None and frame_index > args.end:
                break
            if (frame_index - args.start) % args.every != 0:
                frame_index += 1
                continue

            output_path = args.output_dir / f"{args.prefix}_{frame_index:06d}{args.ext}"
            if output_path.exists() and not args.overwrite:
                skipped_existing += 1
                frame_index += 1
                continue

            if not cv2.imwrite(str(output_path), frame, params):
                print(f"Failed to write frame: {output_path}", file=sys.stderr)
                return 1
            extracted += 1
            frame_index += 1
    finally:
        capture.release()

    print(
        f"Extracted {extracted} frame(s) to {args.output_dir}"
        + (
            f" ({skipped_existing} existing file(s) skipped; use --overwrite to replace them)"
            if skipped_existing
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
