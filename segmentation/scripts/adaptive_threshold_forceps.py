#!/usr/bin/env python3
"""Extract broad forceps and shadow regions with adaptive thresholding."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from scripts.preprocessing import find_images


@dataclass(frozen=True)
class ThresholdResult:
    raw_mask: np.ndarray
    vessel_mask: np.ndarray | None
    orientation_mask: np.ndarray | None
    seeds: np.ndarray
    final_mask: np.ndarray
    masked_image: np.ndarray


def _positive_odd(value: int, name: str) -> int:
    if value <= 1:
        raise ValueError(f"{name} must be greater than 1")
    if value % 2 == 0:
        value += 1
    return value


def _positive_kernel(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value if value % 2 == 1 else value + 1


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def reconstruct_from_seeds(
    seeds: np.ndarray,
    mask: np.ndarray,
    *,
    kernel_size: int = 5,
    max_iterations: int = 200,
) -> np.ndarray:
    """Dilate seeds while constraining growth to the adaptive-threshold mask."""

    reconstruction = seeds.copy()
    kernel = _ellipse(_positive_kernel(kernel_size, "reconstruction kernel size"))
    for _ in range(max_iterations):
        previous = reconstruction.copy()
        reconstruction = cv2.dilate(reconstruction, kernel)
        reconstruction = cv2.bitwise_and(reconstruction, mask)
        if np.array_equal(previous, reconstruction):
            break
    return reconstruction


def keep_largest_components(
    mask: np.ndarray,
    *,
    min_area: int,
    keep_components: int,
) -> np.ndarray:
    if min_area < 1:
        raise ValueError("min_area must be at least 1")
    if keep_components < 1:
        raise ValueError("keep_components must be at least 1")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))

    output = np.zeros_like(mask)
    for _, label in sorted(components, reverse=True)[:keep_components]:
        output[labels == label] = 255
    return output


def _component_solidity(component: np.ndarray) -> float:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(cv2.contourArea(contour) / hull_area)


def filter_broad_components(
    mask: np.ndarray,
    *,
    min_solidity: float,
    min_fill_ratio: float,
) -> np.ndarray:
    if not 0.0 <= min_solidity <= 1.0:
        raise ValueError("min_solidity must be within [0, 1]")
    if not 0.0 <= min_fill_ratio <= 1.0:
        raise ValueError("min_fill_ratio must be within [0, 1]")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if width <= 0 or height <= 0:
            continue

        fill_ratio = area / float(width * height)
        component = ((labels == label).astype(np.uint8)) * 255
        if fill_ratio >= min_fill_ratio and _component_solidity(component) >= min_solidity:
            output[labels == label] = 255
    return output


def remove_vessel_like_structures(
    mask: np.ndarray,
    *,
    vessel_max_thickness: float,
    min_solidity: float,
    min_fill_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if vessel_max_thickness <= 0:
        raise ValueError("vessel_max_thickness must be greater than 0")

    opening_size = _positive_kernel(
        max(3, int(round(vessel_max_thickness * 2 + 1))),
        "vessel opening kernel size",
    )
    broad_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(opening_size))
    broad_mask = filter_broad_components(
        broad_mask,
        min_solidity=min_solidity,
        min_fill_ratio=min_fill_ratio,
    )
    vessel_mask = cv2.bitwise_and(mask, cv2.bitwise_not(broad_mask))
    return broad_mask, vessel_mask


def compute_raw_adaptive_mask(
    image: np.ndarray,
    *,
    block_size: int = 61,
    c: float = 4.0,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("cannot threshold an empty image")

    block_size = _positive_odd(block_size, "block_size")
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    smooth = cv2.bilateralFilter(lightness, d=7, sigmaColor=45, sigmaSpace=45)
    raw_mask = cv2.adaptiveThreshold(
        smooth,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )
    return cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, _ellipse(3))


def _axial_angle_degrees(x1: int, y1: int, x2: int, y2: int) -> float:
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    if angle < 0:
        angle += 180.0
    return float(angle)


def _axial_angle_diff(first: float, second: float) -> float:
    return float(abs((first - second + 90.0) % 180.0 - 90.0))


def orientation_line_filter(
    mask: np.ndarray,
    *,
    force_angle: float | None,
    angle_tolerance: float,
    hough_threshold: int,
    min_line_length: int,
    max_line_gap: int,
    corridor_width: int,
    orientation_open_kernel: int,
    angle_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    if angle_tolerance <= 0:
        raise ValueError("angle_tolerance must be greater than 0")
    if hough_threshold < 1:
        raise ValueError("hough_threshold must be at least 1")
    if min_line_length < 1:
        raise ValueError("min_line_length must be at least 1")
    if max_line_gap < 0:
        raise ValueError("max_line_gap must be non-negative")
    if corridor_width < 1:
        raise ValueError("corridor_width must be at least 1")
    if angle_clusters < 1:
        raise ValueError("angle_clusters must be at least 1")

    line_source = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        _ellipse(_positive_kernel(orientation_open_kernel, "orientation_open_kernel")),
    )
    lines = cv2.HoughLinesP(
        line_source,
        1,
        np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return mask, np.zeros_like(mask)

    entries: list[tuple[int, int, int, int, float, float]] = []
    angle_bins: dict[int, float] = {}
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = _axial_angle_degrees(x1, y1, x2, y2)
        bucket = int(round(angle / 10.0) * 10) % 180
        angle_bins[bucket] = angle_bins.get(bucket, 0.0) + length
        entries.append((x1, y1, x2, y2, length, angle))

    if force_angle is None:
        selected_angles = [
            angle
            for angle, _ in sorted(
                angle_bins.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:angle_clusters]
        ]
    else:
        selected_angles = [force_angle % 180.0]
    if not selected_angles:
        return mask, np.zeros_like(mask)

    corridor = np.zeros_like(mask)
    for x1, y1, x2, y2, _, angle in entries:
        if any(_axial_angle_diff(angle, selected) <= angle_tolerance for selected in selected_angles):
            cv2.line(
                corridor,
                (x1, y1),
                (x2, y2),
                255,
                corridor_width,
                lineType=cv2.LINE_AA,
            )

    orientation_mask = cv2.bitwise_and(mask, corridor)
    return orientation_mask, orientation_mask


def adaptive_threshold_forceps(
    image: np.ndarray,
    *,
    block_size: int = 61,
    c: float = 4.0,
    seed_thickness: float = 7.0,
    min_area: int = 1500,
    keep_components: int = 2,
    close_kernel: int = 15,
    dilate_kernel: int = 5,
    remove_vessels: bool = False,
    vessel_max_thickness: float = 4.0,
    min_solidity: float = 0.18,
    min_fill_ratio: float = 0.04,
    orientation_filter: bool = False,
    force_angle: float | None = None,
    angle_tolerance: float = 15.0,
    hough_threshold: int = 35,
    min_line_length: int = 80,
    max_line_gap: int = 30,
    corridor_width: int = 24,
    orientation_open_kernel: int = 9,
    angle_clusters: int = 2,
) -> ThresholdResult:
    if image is None or image.size == 0:
        raise ValueError("cannot threshold an empty image")
    if seed_thickness <= 0:
        raise ValueError("seed_thickness must be greater than 0")

    block_size = _positive_odd(block_size, "block_size")
    close_kernel = _positive_kernel(close_kernel, "close_kernel")
    dilate_kernel = _positive_kernel(dilate_kernel, "dilate_kernel")

    raw_mask = compute_raw_adaptive_mask(image, block_size=block_size, c=c)

    threshold_mask = raw_mask
    vessel_mask = None
    orientation_mask = None
    if remove_vessels:
        threshold_mask, vessel_mask = remove_vessel_like_structures(
            raw_mask,
            vessel_max_thickness=vessel_max_thickness,
            min_solidity=min_solidity,
            min_fill_ratio=min_fill_ratio,
        )
    if orientation_filter:
        threshold_mask, orientation_mask = orientation_line_filter(
            threshold_mask,
            force_angle=force_angle,
            angle_tolerance=angle_tolerance,
            hough_threshold=hough_threshold,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap,
            corridor_width=corridor_width,
            orientation_open_kernel=orientation_open_kernel,
            angle_clusters=angle_clusters,
        )

    effective_seed_thickness = seed_thickness
    if remove_vessels:
        effective_seed_thickness = min(seed_thickness, vessel_max_thickness)
    if orientation_filter:
        effective_seed_thickness = min(effective_seed_thickness, 5.0)

    distance = cv2.distanceTransform((threshold_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    seeds = (distance >= effective_seed_thickness).astype(np.uint8) * 255
    reconstructed = reconstruct_from_seeds(seeds, threshold_mask)

    final_mask = cv2.morphologyEx(
        reconstructed,
        cv2.MORPH_CLOSE,
        _ellipse(close_kernel),
    )
    final_mask = cv2.dilate(final_mask, _ellipse(dilate_kernel), iterations=1)
    final_mask = keep_largest_components(
        final_mask,
        min_area=min_area,
        keep_components=keep_components,
    )

    return ThresholdResult(
        raw_mask=raw_mask,
        vessel_mask=vessel_mask,
        orientation_mask=orientation_mask,
        seeds=seeds,
        final_mask=final_mask,
        masked_image=cv2.bitwise_and(image, image, mask=final_mask),
    )


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def debug_panel(image: np.ndarray, result: ThresholdResult) -> np.ndarray:
    panels = [
        image,
        _to_bgr(result.raw_mask),
    ]
    if result.vessel_mask is not None:
        panels.append(_to_bgr(result.vessel_mask))
    if result.orientation_mask is not None:
        panels.append(_to_bgr(result.orientation_mask))
    panels.extend(
        [
            _to_bgr(result.seeds),
            _to_bgr(result.final_mask),
            result.masked_image,
        ]
    )
    return cv2.hconcat(panels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep broad forceps and forceps-shadow regions using adaptive thresholding."
    )
    parser.add_argument("--source", required=True, help="Image path, directory, or glob.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/adaptive_threshold"),
        help="Directory for masked images and optional masks/debug panels.",
    )
    parser.add_argument("--write-mask", action="store_true", help="Also write a binary mask.")
    parser.add_argument(
        "--write-vessel-mask",
        action="store_true",
        help="Also write the vessel-like pixels removed by --remove-vessels.",
    )
    parser.add_argument(
        "--write-orientation-mask",
        action="store_true",
        help="Also write pixels kept by --orientation-filter before reconstruction.",
    )
    parser.add_argument("--debug", action="store_true", help="Write a side-by-side debug panel.")
    parser.add_argument("--block-size", type=int, default=61)
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument("--seed-thickness", type=float, default=7.0)
    parser.add_argument("--min-area", type=int, default=1500)
    parser.add_argument("--keep-components", type=int, default=2)
    parser.add_argument("--close-kernel", type=int, default=15)
    parser.add_argument("--dilate-kernel", type=int, default=5)
    parser.add_argument(
        "--remove-vessels",
        action="store_true",
        help="Subtract thin vessel-like structures before seed reconstruction.",
    )
    parser.add_argument(
        "--vessel-max-thickness",
        type=float,
        default=4.0,
        help="Approximate maximum vessel half-thickness to remove.",
    )
    parser.add_argument("--min-solidity", type=float, default=0.18)
    parser.add_argument("--min-fill-ratio", type=float, default=0.04)
    parser.add_argument(
        "--orientation-filter",
        action="store_true",
        help="Keep pixels near the dominant Hough line-angle corridors.",
    )
    parser.add_argument(
        "--force-angle",
        type=float,
        help="Use this axial angle in degrees instead of auto-selecting Hough angle clusters.",
    )
    parser.add_argument("--angle-tolerance", type=float, default=15.0)
    parser.add_argument("--hough-threshold", type=int, default=35)
    parser.add_argument("--min-line-length", type=int, default=80)
    parser.add_argument("--max-line-gap", type=int, default=30)
    parser.add_argument("--corridor-width", type=int, default=24)
    parser.add_argument("--orientation-open-kernel", type=int, default=9)
    parser.add_argument("--angle-clusters", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        paths = find_images(args.source)
        if not paths:
            print(f"No readable image paths found for source: {args.source}")
            return 1

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"Skipping unreadable image: {path}")
                continue

            result = adaptive_threshold_forceps(
                image,
                block_size=args.block_size,
                c=args.c,
                seed_thickness=args.seed_thickness,
                min_area=args.min_area,
                keep_components=args.keep_components,
                close_kernel=args.close_kernel,
                dilate_kernel=args.dilate_kernel,
                remove_vessels=args.remove_vessels,
                vessel_max_thickness=args.vessel_max_thickness,
                min_solidity=args.min_solidity,
                min_fill_ratio=args.min_fill_ratio,
                orientation_filter=args.orientation_filter,
                force_angle=args.force_angle,
                angle_tolerance=args.angle_tolerance,
                hough_threshold=args.hough_threshold,
                min_line_length=args.min_line_length,
                max_line_gap=args.max_line_gap,
                corridor_width=args.corridor_width,
                orientation_open_kernel=args.orientation_open_kernel,
                angle_clusters=args.angle_clusters,
            )

            masked_path = args.output_dir / f"{path.stem}_masked.png"
            if not cv2.imwrite(str(masked_path), result.masked_image):
                raise RuntimeError(f"failed to write masked image: {masked_path}")
            print(f"Wrote masked image: {masked_path}")

            if args.write_mask:
                mask_path = args.output_dir / f"{path.stem}_mask.png"
                if not cv2.imwrite(str(mask_path), result.final_mask):
                    raise RuntimeError(f"failed to write mask: {mask_path}")
                print(f"Wrote mask: {mask_path}")

            if args.write_vessel_mask and result.vessel_mask is not None:
                vessel_path = args.output_dir / f"{path.stem}_vessel_mask.png"
                if not cv2.imwrite(str(vessel_path), result.vessel_mask):
                    raise RuntimeError(f"failed to write vessel mask: {vessel_path}")
                print(f"Wrote vessel mask: {vessel_path}")

            if args.write_orientation_mask and result.orientation_mask is not None:
                orientation_path = args.output_dir / f"{path.stem}_orientation_mask.png"
                if not cv2.imwrite(str(orientation_path), result.orientation_mask):
                    raise RuntimeError(
                        f"failed to write orientation mask: {orientation_path}"
                    )
                print(f"Wrote orientation mask: {orientation_path}")

            if args.debug:
                debug_path = args.output_dir / f"{path.stem}_debug.png"
                if not cv2.imwrite(str(debug_path), debug_panel(image, result)):
                    raise RuntimeError(f"failed to write debug panel: {debug_path}")
                print(f"Wrote debug panel: {debug_path}")
    except ValueError as exc:
        print(exc)
        return 1
    except RuntimeError as exc:
        print(exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
