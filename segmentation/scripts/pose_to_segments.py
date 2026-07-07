#!/usr/bin/env python3
"""Convert YOLO pose labels into tiny YOLO segmentation boxes."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT

DEFAULT_LABELS_ROOT = REPO_ROOT / "data" / "labels"
DEFAULT_BACKUP_ROOT = REPO_ROOT / "data" / "labels_pose_backup"
KEYPOINT_CLASSES = ("tip_left", "tip_right", "shadow_left", "shadow_right")


def same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def format_value(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def square_polygon(cx: float, cy: float, size: float) -> list[float]:
    half = size / 2.0
    x1 = clamp(cx - half)
    y1 = clamp(cy - half)
    x2 = clamp(cx + half)
    y2 = clamp(cy + half)
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def parse_pose_line(line: str, label_path: Path, line_no: int) -> list[float]:
    parts = line.split()
    expected_values = 5 + len(KEYPOINT_CLASSES) * 3
    if len(parts) != expected_values:
        raise ValueError(
            f"{label_path}:{line_no}: expected {expected_values} pose values, got {len(parts)}"
        )

    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{label_path}:{line_no}: non-numeric value") from exc

    return values


def convert_pose_text(
    text: str,
    label_path: Path,
    box_size: float,
    min_visibility: float,
) -> tuple[str, int]:
    output_lines: list[str] = []
    objects_written = 0

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        values = parse_pose_line(line, label_path, line_no)
        keypoints = values[5:]
        for class_id in range(len(KEYPOINT_CLASSES)):
            x, y, visibility = keypoints[class_id * 3 : class_id * 3 + 3]
            if visibility < min_visibility:
                continue
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError(
                    f"{label_path}:{line_no}: keypoint {KEYPOINT_CLASSES[class_id]} "
                    f"has coordinates outside [0, 1]: {x}, {y}"
                )

            polygon = square_polygon(x, y, box_size)
            coords = " ".join(format_value(value) for value in polygon)
            output_lines.append(f"{class_id} {coords}")
            objects_written += 1

    if not output_lines:
        return "", objects_written
    return "\n".join(output_lines) + "\n", objects_written


def copy_backup(input_root: Path, backup_root: Path, splits: list[str]) -> None:
    if backup_root.exists():
        raise FileExistsError(
            f"backup directory already exists: {backup_root}\n"
            "Move it away or pass --backup-root to use a new backup location."
        )

    for split in splits:
        src = input_root / split
        dst = backup_root / split
        if src.exists():
            shutil.copytree(src, dst)


def clear_caches(labels_root: Path, splits: list[str]) -> int:
    removed = 0
    for split in splits:
        cache_path = labels_root / f"{split}.cache"
        if cache_path.exists():
            cache_path.unlink()
            removed += 1
    return removed


def convert_split(
    split: str,
    input_root: Path,
    output_root: Path,
    box_size: float,
    min_visibility: float,
    dry_run: bool,
) -> tuple[int, int]:
    input_dir = input_root / split
    output_dir = output_root / split
    if not input_dir.is_dir():
        raise FileNotFoundError(f"labels split not found: {input_dir}")

    files_written = 0
    objects_written = 0
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for label_path in sorted(input_dir.glob("*.txt")):
        converted, object_count = convert_pose_text(
            label_path.read_text(),
            label_path,
            box_size,
            min_visibility,
        )
        files_written += 1
        objects_written += object_count

        if not dry_run:
            (output_dir / label_path.name).write_text(converted)

    return files_written, objects_written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert YOLO pose/keypoint labels into YOLO segmentation labels. "
            "Each keypoint becomes a small square polygon with class IDs: "
            "0 tip_left, 1 tip_right, 2 shadow_left, 3 shadow_right."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Destination labels root. Defaults to --input-root, which makes "
            "the dataset immediately usable with configs/forceps_seg.yaml."
        ),
    )
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--box-size",
        type=float,
        default=0.02,
        help="Normalized square side length around each keypoint. 0.02 is about 20 px at 1024 px.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=1.0,
        help="Minimum YOLO keypoint visibility to convert. Use 2 for visible-only.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not back up source labels before in-place conversion.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not remove train.cache/val.cache after conversion.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.box_size <= 0.0 or args.box_size > 1.0:
        print("--box-size must be in the range (0, 1].")
        return 1

    input_root = args.input_root.resolve()
    output_root = (args.output_root or args.input_root).resolve()
    in_place = same_path(input_root, output_root)

    try:
        if in_place and not args.no_backup and not args.dry_run:
            copy_backup(input_root, args.backup_root.resolve(), args.splits)

        total_files = 0
        total_objects = 0
        for split in args.splits:
            files, objects = convert_split(
                split,
                input_root,
                output_root,
                args.box_size,
                args.min_visibility,
                args.dry_run,
            )
            total_files += files
            total_objects += objects
            print(f"{split}: converted {files} label files, wrote {objects} segment boxes")

        removed_caches = 0
        if not args.keep_cache and not args.dry_run:
            removed_caches = clear_caches(output_root, args.splits)

    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    action = "Would convert" if args.dry_run else "Converted"
    print(f"{action} {total_files} files into {total_objects} total segment boxes.")
    if in_place and not args.no_backup and not args.dry_run:
        print(f"Backed up original pose labels to: {args.backup_root.resolve()}")
    if removed_caches:
        print(f"Removed {removed_caches} stale Ultralytics cache file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
