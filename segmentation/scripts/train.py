#!/usr/bin/env python3
"""Train a YOLO segmentation model on forceps tip/shadow data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from scripts.common import DEFAULT_DATA_CONFIG, REPO_ROOT
from scripts.prepare_preprocessed_dataset import prepare_dataset
from scripts.preprocessing import DEFAULT_PREPROCESS_CONFIG


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Dataset YAML config.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n-seg.pt",
        help="Pretrained checkpoint or model YAML.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="forceps")
    parser.add_argument(
        "--preprocess-preset",
        default=None,
        help="Train on a derived dataset created with this preprocessing preset.",
    )
    parser.add_argument(
        "--preprocess-config",
        type=Path,
        default=DEFAULT_PREPROCESS_CONFIG,
        help="Preprocessing presets YAML.",
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=None,
        help="Derived dataset root. Defaults to data_preprocessed/<preset>.",
    )
    parser.add_argument(
        "--preprocessed-data-config",
        type=Path,
        default=None,
        help="Generated dataset YAML path.",
    )
    parser.add_argument(
        "--rebuild-preprocessed",
        action="store_true",
        help="Regenerate existing preprocessed images before training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the most recent training run.",
    )
    args = parser.parse_args()

    if args.preprocess_preset:
        preprocessed_root = args.preprocessed_root or (
            REPO_ROOT / "data_preprocessed" / args.preprocess_preset
        )
        preprocessed_config = args.preprocessed_data_config or (
            REPO_ROOT
            / "configs"
            / "generated"
            / f"forceps_seg_{args.preprocess_preset}.yaml"
        )
        if args.rebuild_preprocessed or not preprocessed_config.exists():
            try:
                image_count, missing_labels = prepare_dataset(
                    args.config,
                    args.preprocess_config,
                    args.preprocess_preset,
                    preprocessed_root,
                    preprocessed_config,
                    overwrite=args.rebuild_preprocessed,
                )
            except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
                print(exc)
                return 1
            print(
                f"Prepared {image_count} images with preset "
                f"'{args.preprocess_preset}' at {preprocessed_root}."
            )
            if missing_labels:
                print(f"Warning: {missing_labels} image(s) had no label file.")
        args.config = preprocessed_config

    model = YOLO(args.model)
    model.train(
        data=str(args.config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
