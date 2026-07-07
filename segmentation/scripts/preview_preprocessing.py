#!/usr/bin/env python3
"""Render side-by-side previews of preprocessing presets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from scripts.preprocessing import (
    DEFAULT_PREPROCESS_CONFIG,
    apply_preprocessing,
    find_images,
    load_preprocess_preset,
)


def titled_image(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] != target_height:
        scale = target_height / image.shape[0]
        image = cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), target_height),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.copyMakeBorder(
        image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)
    )


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    output = image.copy()
    cv2.putText(
        output,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview preprocessing presets side by side.")
    parser.add_argument("--source", required=True, help="Image path, directory, or glob.")
    parser.add_argument(
        "--presets",
        default="original,roi,clahe,roi_clahe,roi_clahe_gamma,full",
        help="Comma-separated preprocessing preset names.",
    )
    parser.add_argument(
        "--preprocess-config",
        type=Path,
        default=DEFAULT_PREPROCESS_CONFIG,
        help="Preprocessing presets YAML.",
    )
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/preprocessing_preview")
    )
    args = parser.parse_args()

    if args.max_images < 1:
        print("--max-images must be >= 1")
        return 1

    preset_names = [name.strip() for name in args.presets.split(",") if name.strip()]
    try:
        presets = [
            (name, load_preprocess_preset(name, args.preprocess_config))
            for name in preset_names
        ]
    except ValueError as exc:
        print(exc)
        return 1

    paths = find_images(args.source)[: args.max_images]
    if not paths:
        print(f"No readable image paths found for source: {args.source}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue
        target_height = image.shape[0]
        panels = [
            add_title(
                titled_image(apply_preprocessing(image, preset).image, target_height),
                name,
            )
            for name, preset in presets
        ]
        output_path = args.output_dir / f"{path.stem}_comparison.jpg"
        if not cv2.imwrite(str(output_path), cv2.hconcat(panels)):
            print(f"Failed to write preview: {output_path}")
            return 1
        print(f"Wrote preview: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
