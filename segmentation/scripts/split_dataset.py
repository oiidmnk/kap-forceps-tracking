#!/usr/bin/env python3
"""Split raw paired images and labels into train/val folders by session."""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import IMAGE_EXTENSIONS, REPO_ROOT

DEFAULT_RAW_IMAGES = REPO_ROOT / "data" / "raw" / "images"
DEFAULT_RAW_LABELS = REPO_ROOT / "data" / "raw" / "labels"
DEFAULT_TRAIN_IMAGES = REPO_ROOT / "data" / "images" / "train"
DEFAULT_VAL_IMAGES = REPO_ROOT / "data" / "images" / "val"
DEFAULT_TRAIN_LABELS = REPO_ROOT / "data" / "labels" / "train"
DEFAULT_VAL_LABELS = REPO_ROOT / "data" / "labels" / "val"


def session_key(stem: str, pattern: str | None) -> str:
    if pattern:
        match = re.match(pattern, stem)
        if match and match.groupdict().get("session"):
            return match.group("session")
        if match and match.lastindex:
            return match.group(1)
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def collect_pairs(images_dir: Path, labels_dir: Path) -> dict[str, tuple[Path, Path]]:
    images = {
        p.stem: p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {
        p.stem: p for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt"
    }

    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        if missing_labels:
            print("Images without labels:")
            for stem in missing_labels:
                print(f"  - {images[stem]}")
        if missing_images:
            print("Labels without images:")
            for stem in missing_images:
                print(f"  - {labels[stem]}")
        raise SystemExit(1)

    return {stem: (images[stem], labels[stem]) for stem in sorted(images)}


def move_pair(image_path: Path, label_path: Path, images_dir: Path, labels_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, images_dir / image_path.name)
    shutil.copy2(label_path, labels_dir / label_path.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split paired raw images/labels into train and val by session."
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_RAW_IMAGES)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_RAW_LABELS)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--session-pattern",
        default=None,
        help=(
            "Regex with a 'session' named group, e.g. "
            "'(?P<session>.+)_frame_\\d+'."
        ),
    )
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--val-images", type=Path, default=DEFAULT_VAL_IMAGES)
    parser.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--val-labels", type=Path, default=DEFAULT_VAL_LABELS)
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying.",
    )
    args = parser.parse_args()

    if not args.images_dir.is_dir() or not args.labels_dir.is_dir():
        print("Raw image/label directories not found.")
        print(f"Expected images: {args.images_dir}")
        print(f"Expected labels: {args.labels_dir}")
        return 1

    pairs = collect_pairs(args.images_dir, args.labels_dir)
    sessions: dict[str, list[str]] = defaultdict(list)
    for stem in pairs:
        sessions[session_key(stem, args.session_pattern)].append(stem)

    session_ids = sorted(sessions)
    rng = random.Random(args.seed)
    rng.shuffle(session_ids)

    if len(session_ids) == 1:
        train_sessions = {session_ids[0]}
        val_sessions: set[str] = set()
        print("Only one session found; all samples go to train.")
    else:
        train_count = max(1, int(round(len(session_ids) * args.train_ratio)))
        train_count = min(train_count, len(session_ids) - 1)
        train_sessions = set(session_ids[:train_count])
        val_sessions = set(session_ids[train_count:])

    copy_fn = shutil.move if args.move else shutil.copy2

    def place(stem: str, images_dir: Path, labels_dir: Path) -> None:
        image_path, label_path = pairs[stem]
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        copy_fn(image_path, images_dir / image_path.name)
        copy_fn(label_path, labels_dir / label_path.name)

    train_stems = [stem for session in sorted(train_sessions) for stem in sessions[session]]
    val_stems = [stem for session in sorted(val_sessions) for stem in sessions[session]]

    for stem in train_stems:
        place(stem, args.train_images, args.train_labels)
    for stem in val_stems:
        place(stem, args.val_images, args.val_labels)

    print(f"Sessions: {len(session_ids)} total, {len(train_sessions)} train, {len(val_sessions)} val")
    print(f"Samples:  {len(train_stems)} train, {len(val_stems)} val")
    return 0


if __name__ == "__main__":
    sys.exit(main())
