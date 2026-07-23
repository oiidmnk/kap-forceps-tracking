#!/usr/bin/env python3
"""Apply an adaptive-threshold mask to source images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from scripts.adaptive_threshold_forceps import compute_raw_adaptive_mask
from scripts.preprocessing import find_images


def load_mask(mask_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cannot read mask image: {mask_path}")
    if mask.shape[:2] != image_shape:
        raise ValueError(
            f"mask size {mask.shape[1]}x{mask.shape[0]} does not match "
            f"image size {image_shape[1]}x{image_shape[0]}"
        )
    return cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image, image, mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the raw adaptive-threshold mask to image pixels."
    )
    parser.add_argument("--source", required=True, help="Image path, directory, or glob.")
    parser.add_argument(
        "--mask",
        type=Path,
        help="Optional saved mask to apply. If omitted, the raw adaptive mask is computed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/adaptive_threshold_masked"),
    )
    parser.add_argument("--block-size", type=int, default=61)
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument(
        "--write-mask",
        action="store_true",
        help="Also write the mask used for masking.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        paths = find_images(args.source)
        if not paths:
            print(f"No readable image paths found for source: {args.source}")
            return 1
        if args.mask is not None and len(paths) != 1:
            print("--mask can only be used with a single source image")
            return 1

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"Skipping unreadable image: {path}")
                continue

            if args.mask is None:
                mask = compute_raw_adaptive_mask(
                    image,
                    block_size=args.block_size,
                    c=args.c,
                )
            else:
                mask = load_mask(args.mask, image.shape[:2])

            masked = apply_mask(image, mask)
            masked_path = args.output_dir / f"{path.stem}_mask_applied.png"
            if not cv2.imwrite(str(masked_path), masked):
                raise RuntimeError(f"failed to write masked image: {masked_path}")
            print(f"Wrote masked image: {masked_path}")

            if args.write_mask:
                mask_path = args.output_dir / f"{path.stem}_raw_mask.png"
                if not cv2.imwrite(str(mask_path), mask):
                    raise RuntimeError(f"failed to write mask: {mask_path}")
                print(f"Wrote mask: {mask_path}")
    except ValueError as exc:
        print(exc)
        return 1
    except RuntimeError as exc:
        print(exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
