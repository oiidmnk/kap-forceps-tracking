#!/usr/bin/env python3
"""Validate a trained YOLO segmentation model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from scripts.common import DEFAULT_DATA_CONFIG

DEFAULT_WEIGHTS = Path("runs/segment/forceps/weights/best.pt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YOLO segmentation model.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to trained weights.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Dataset YAML config.",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"Weights not found: {args.weights}")
        print("Train first with: python scripts/train.py")
        return 1

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.config),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    print(f"Box mAP50-95:  {metrics.box.map:.4f}")
    print(f"Box mAP50:     {metrics.box.map50:.4f}")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
    print(f"Mask mAP50:    {metrics.seg.map50:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
