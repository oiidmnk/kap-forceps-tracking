#!/usr/bin/env python3
"""Detect and inpaint retinal vessels while protecting forceps regions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from scripts.preprocessing import find_images

DEFAULT_PROTECTION_POLYGONS = [
    [
        [66, 231],
        [80, 223],
        [83, 188],
        [93, 190],
        [145, 247],
        [214, 319],
        [210, 352],
        [172, 352],
        [112, 288],
    ],
    [
        [284, 118],
        [322, 100],
        [608, 183],
        [608, 289],
        [512, 246],
        [382, 190],
    ],
]
DEFAULT_PROTECTION_SIZE = (608, 406)


def _positive_odd(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value if value % 2 == 1 else value + 1


def load_polygons(path: Path) -> list[list[list[int]]]:
    with path.open() as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("protection polygon file must contain a list of polygons")

    polygons: list[list[list[int]]] = []
    for polygon in data:
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError("each protection polygon must contain at least three points")
        points: list[list[int]] = []
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("each protection point must be [x, y]")
            points.append([int(point[0]), int(point[1])])
        polygons.append(points)
    return polygons


def scale_polygons(
    polygons: list[list[list[int]]],
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[list[list[int]]]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source protection size must be positive")

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scaled: list[list[list[int]]] = []
    for polygon in polygons:
        scaled.append(
            [
                [int(round(x * scale_x)), int(round(y * scale_y))]
                for x, y in polygon
            ]
        )
    return scaled


def create_protection_mask(
    image: np.ndarray,
    *,
    polygons: list[list[list[int]]],
    dilation_kernel: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    if polygons and dilation_kernel > 0:
        kernel_size = _positive_odd(dilation_kernel, "protection_dilation_kernel")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def detect_vessels(
    image: np.ndarray,
    *,
    kernel_sizes: tuple[int, ...],
    clahe_clip_limit: float,
    clahe_tile_grid: int,
) -> np.ndarray:
    green = image[:, :, 1]
    green = cv2.GaussianBlur(green, (3, 3), 0)

    if clahe_tile_grid < 1:
        raise ValueError("clahe_tile_grid must be at least 1")
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_tile_grid, clahe_tile_grid),
    )
    enhanced = clahe.apply(green)

    responses = []
    for size in kernel_sizes:
        kernel_size = _positive_odd(size, "blackhat kernel size")
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        responses.append(cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel))

    vessel_response = np.maximum.reduce(responses)
    vessel_response = cv2.normalize(
        vessel_response,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    _, vessel_mask = cv2.threshold(
        vessel_response,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, opening_kernel)

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(vessel_mask, dilation_kernel, iterations=1)


def remove_small_components(binary_mask: np.ndarray, minimum_area: int) -> np.ndarray:
    if minimum_area < 1:
        raise ValueError("minimum_area must be at least 1")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary_mask)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= minimum_area:
            cleaned[labels == label] = 255
    return cleaned


def parse_kernel_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("kernel sizes must be comma-separated integers") from exc
    if not sizes:
        raise argparse.ArgumentTypeError("at least one kernel size is required")
    if any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("kernel sizes must be at least 1")
    return sizes


def inpaint_vessels(
    image: np.ndarray,
    *,
    polygons: list[list[list[int]]],
    protection_dilation_kernel: int,
    kernel_sizes: tuple[int, ...],
    clahe_clip_limit: float,
    clahe_tile_grid: int,
    minimum_area: int,
    inpaint_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError("cannot process an empty image")
    if inpaint_radius <= 0:
        raise ValueError("inpaint_radius must be greater than 0")

    protection_mask = create_protection_mask(
        image,
        polygons=polygons,
        dilation_kernel=protection_dilation_kernel,
    )
    vessel_mask = detect_vessels(
        image,
        kernel_sizes=kernel_sizes,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid=clahe_tile_grid,
    )
    vessel_mask = remove_small_components(vessel_mask, minimum_area)
    vessel_mask[protection_mask > 0] = 0

    result = cv2.inpaint(
        image,
        vessel_mask,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA,
    )
    return result, vessel_mask, protection_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and inpaint retinal vessels with optional forceps protection."
    )
    parser.add_argument("--source", required=True, help="Image path, directory, or glob.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/vessel_inpaint"),
    )
    parser.add_argument(
        "--protection-polygons",
        type=Path,
        help="JSON file containing polygons as [[[x, y], ...], ...].",
    )
    parser.add_argument(
        "--no-default-protection",
        action="store_true",
        help="Do not use the built-in approximate forceps/shadow protection polygons.",
    )
    parser.add_argument("--protection-dilation-kernel", type=int, default=9)
    parser.add_argument(
        "--blackhat-kernels",
        type=parse_kernel_sizes,
        default=(7, 11, 15, 21),
    )
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid", type=int, default=8)
    parser.add_argument("--minimum-area", type=int, default=15)
    parser.add_argument("--inpaint-radius", type=float, default=4.0)
    parser.add_argument("--write-mask", action="store_true", help="Also write the vessel mask.")
    parser.add_argument(
        "--write-protection-mask",
        action="store_true",
        help="Also write the protection mask.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        custom_polygons: list[list[list[int]]] = []
        if args.protection_polygons is not None:
            custom_polygons.extend(load_polygons(args.protection_polygons))

        paths = find_images(args.source)
        if not paths:
            print(f"No readable image paths found for source: {args.source}")
            return 1

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipping unreadable image: {path}")
                continue

            polygons: list[list[list[int]]] = []
            if not args.no_default_protection:
                polygons.extend(
                    scale_polygons(
                        DEFAULT_PROTECTION_POLYGONS,
                        source_size=DEFAULT_PROTECTION_SIZE,
                        target_size=(image.shape[1], image.shape[0]),
                    )
                )
            polygons.extend(custom_polygons)

            result, vessel_mask, protection_mask = inpaint_vessels(
                image,
                polygons=polygons,
                protection_dilation_kernel=args.protection_dilation_kernel,
                kernel_sizes=args.blackhat_kernels,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_grid=args.clahe_tile_grid,
                minimum_area=args.minimum_area,
                inpaint_radius=args.inpaint_radius,
            )

            output_path = args.output_dir / f"{path.stem}_without_vessels.png"
            if not cv2.imwrite(str(output_path), result):
                raise RuntimeError(f"failed to write result: {output_path}")
            print(f"Wrote result: {output_path}")

            if args.write_mask:
                mask_path = args.output_dir / f"{path.stem}_vessel_mask.png"
                if not cv2.imwrite(str(mask_path), vessel_mask):
                    raise RuntimeError(f"failed to write vessel mask: {mask_path}")
                print(f"Wrote vessel mask: {mask_path}")

            if args.write_protection_mask:
                protection_path = args.output_dir / f"{path.stem}_protection_mask.png"
                if not cv2.imwrite(str(protection_path), protection_mask):
                    raise RuntimeError(
                        f"failed to write protection mask: {protection_path}"
                    )
                print(f"Wrote protection mask: {protection_path}")
    except ValueError as exc:
        print(exc)
        return 1
    except RuntimeError as exc:
        print(exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
