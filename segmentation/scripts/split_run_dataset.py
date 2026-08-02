#!/usr/bin/env python3
"""Copy a dataset run into a new reproducible train/validation split."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import IMAGE_EXTENSIONS


def collect_run_pairs(dataset_root: Path) -> dict[str, tuple[Path, Path]]:
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise ValueError(f"dataset must contain images/ and labels/: {dataset_root}")

    images = {
        path.stem: path
        for path in images_root.glob("**/*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {
        path.stem: path
        for path in labels_root.glob("**/*.txt")
        if path.is_file()
    }
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        details = []
        if missing_labels:
            details.append(f"{len(missing_labels)} image(s) have no label")
        if missing_images:
            details.append(f"{len(missing_images)} label(s) have no image")
        raise ValueError("; ".join(details))
    if not images:
        raise ValueError(f"dataset contains no paired samples: {dataset_root}")
    return {stem: (images[stem], labels[stem]) for stem in sorted(images)}


def split_run_dataset(
    source_root: Path,
    output_root: Path,
    train_ratio: float,
    seed: int,
) -> tuple[int, int]:
    if not 0 < train_ratio <= 1:
        raise ValueError("train ratio must be greater than 0 and at most 1")
    pairs = collect_run_pairs(source_root)
    stems = list(pairs)
    random.Random(seed).shuffle(stems)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    if len(stems) == 1 or train_ratio == 1:
        train_count = len(stems)
    else:
        train_count = min(len(stems) - 1, max(1, round(len(stems) * train_ratio)))

    for index, stem in enumerate(stems):
        split = "train" if index < train_count else "val"
        image, label = pairs[stem]
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_dir / image.name)
        shutil.copy2(label, label_dir / label.name)

    val_count = len(stems) - train_count
    print(f"Split {len(stems)} paired samples: {train_count} train, {val_count} val")
    print(f"Dataset written under {output_root}")
    return train_count, val_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Split the output of a dataset run.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        split_run_dataset(args.source_root, args.out_dir, args.train_ratio, args.seed)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
