#!/usr/bin/env python3
"""Generate synthetic forceps pose training data.

The generated labels follow YOLO pose format with two objects per image:

    0 cx cy w h tip_left_x tip_left_y v tip_right_x tip_right_y v
    1 cx cy w h shadow_left_x shadow_left_y v shadow_right_x shadow_right_y v
"""

from __future__ import annotations

import argparse
import os
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_labels import CLASS_COLORS

FORCEPS_KEYPOINT_NAMES = ("tip_left", "tip_right")
SHADOW_KEYPOINT_NAMES = ("shadow_left", "shadow_right")
FORCEPS_CLASS_ID = 0
SHADOW_CLASS_ID = 1


@dataclass(frozen=True)
class Pose:
    tip_polygons: list[np.ndarray]
    shadow_polygons: list[np.ndarray]


@dataclass(frozen=True)
class GenerationTask:
    index: int
    seed: int
    count: int
    out_dir: Path
    width: int
    height: int
    background: Optional[Path]
    background_rotation: float
    axis_roll: float
    prefix: str
    start_index: int
    val_fraction: float
    preview: int
    preview_dir: Path
    image_ext: str


def unit(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)


def ellipse_polygon(
    center: np.ndarray,
    radius_x: float,
    radius_y: float,
    angle: float = 0.0,
    points: int = 14,
) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    coords: list[list[float]] = []
    for theta in np.linspace(0, math.tau, points, endpoint=False):
        x = radius_x * math.cos(theta)
        y = radius_y * math.sin(theta)
        coords.append(
            [
                float(center[0] + x * cos_a - y * sin_a),
                float(center[1] + x * sin_a + y * cos_a),
            ]
        )
    return np.array(coords, dtype=np.float32)


def oriented_box(center: np.ndarray, direction: np.ndarray, length: float, width: float) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    half_l = length / 2.0
    half_w = width / 2.0
    return np.array(
        [
            center - direction * half_l - normal * half_w,
            center + direction * half_l - normal * half_w,
            center + direction * half_l + normal * half_w,
            center - direction * half_l + normal * half_w,
        ],
        dtype=np.float32,
    )


def keypoint_polygon(center: np.ndarray, radius: float = 1.8) -> np.ndarray:
    return np.array(
        [
            center + [-radius, -radius],
            center + [radius, -radius],
            center + [radius, radius],
            center + [-radius, radius],
        ],
        dtype=np.float32,
    )


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    mix = (mask.astype(np.float32) / 255.0)[:, :, None] * alpha
    color_arr = np.array(color, dtype=np.float32)
    image[:] = np.clip(image.astype(np.float32) * (1.0 - mix) + color_arr * mix, 0, 255).astype(np.uint8)


def draw_soft_line(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    alpha: float,
    blur: int,
) -> None:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.round(points).astype(np.int32)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    if blur > 0:
        blur = blur + 1 if blur % 2 == 0 else blur
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    blend_mask(image, mask, color, alpha)


def draw_soft_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
    blur: int = 0,
) -> None:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], color=255, lineType=cv2.LINE_AA)
    if blur > 0:
        blur = blur + 1 if blur % 2 == 0 else blur
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    blend_mask(image, mask, color, alpha)


def motion_blur_mask(mask: np.ndarray, direction: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return mask
    size = size + 1 if size % 2 == 0 else size
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    center = size // 2
    endpoint = direction * center
    kernel_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.line(
        kernel_mask,
        (int(round(center - endpoint[0])), int(round(center - endpoint[1]))),
        (int(round(center + endpoint[0])), int(round(center + endpoint[1]))),
        255,
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    kernel = kernel_mask.astype(np.float32) / 255.0
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0:
        return mask
    return cv2.filter2D(mask, -1, kernel / kernel_sum)


def tapered_segment_polygon(
    start: np.ndarray,
    end: np.ndarray,
    start_width: float,
    end_width: float,
) -> np.ndarray:
    direction = end - start
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    return np.array(
        [
            start - normal * start_width / 2.0,
            end - normal * end_width / 2.0,
            end + normal * end_width / 2.0,
            start + normal * start_width / 2.0,
        ],
        dtype=np.float32,
    )


def draw_soft_tapered_segment(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_width: float,
    end_width: float,
    color: tuple[int, int, int],
    alpha: float,
    blur: int = 0,
    motion_blur: int = 0,
) -> None:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    polygon = tapered_segment_polygon(start, end, start_width, end_width)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], color=255, lineType=cv2.LINE_AA)
    if motion_blur > 1:
        mask = motion_blur_mask(mask, end - start, motion_blur)
    if blur > 0:
        blur = blur + 1 if blur % 2 == 0 else blur
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    blend_mask(image, mask, color, alpha)


def draw_metal_segment(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    start_width: float,
    end_width: float,
    rng: np.random.Generator,
    highlight_side: float = -0.22,
) -> None:
    direction = end - start
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    dark_edge = (31, 29, 29)
    base = (72, 65, 60)
    highlight = (127, 116, 103)

    draw_soft_tapered_segment(image, start, end, start_width, end_width, dark_edge, 0.88, blur=7)
    draw_soft_tapered_segment(
        image,
        start + normal * start_width * 0.03,
        end + normal * end_width * 0.03,
        start_width * 0.78,
        end_width * 0.78,
        base,
        0.92,
        blur=5,
    )
    draw_soft_line(
        image,
        np.vstack([start + normal * start_width * 0.42, end + normal * end_width * 0.42]),
        (21, 20, 21),
        thickness=max(1, round(min(start_width, end_width) * 0.12)),
        alpha=0.58,
        blur=5,
    )
    draw_soft_line(
        image,
        np.vstack([start - normal * start_width * 0.42, end - normal * end_width * 0.42]),
        (38, 34, 33),
        thickness=max(1, round(min(start_width, end_width) * 0.10)),
        alpha=0.42,
        blur=5,
    )
    draw_soft_line(
        image,
        np.vstack(
            [
                start + normal * start_width * highlight_side + direction * start_width * 0.35,
                end + normal * end_width * highlight_side - direction * end_width * 0.35,
            ]
        ),
        highlight,
        thickness=max(1, round(min(start_width, end_width) * 0.12)),
        alpha=0.33,
        blur=5,
    )

    for _ in range(3):
        t0 = rng.uniform(0.18, 0.82)
        t1 = min(0.96, t0 + rng.uniform(0.08, 0.20))
        p0 = start * (1.0 - t0) + end * t0
        p1 = start * (1.0 - t1) + end * t1
        offset = normal * rng.uniform(-0.18, 0.18) * min(start_width, end_width)
        draw_soft_line(
            image,
            np.vstack([p0 + offset, p1 + offset]),
            (103, 95, 88),
            thickness=1,
            alpha=0.16,
            blur=3,
        )


def noisy_retina_background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = width * rng.uniform(0.46, 0.53)
    cy = height * rng.uniform(0.46, 0.53)
    radius = min(width, height) * rng.uniform(0.42, 0.47)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius

    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, :, 2] = rng.uniform(132, 176) - 13 * rr + 20 * (xx / width)
    base[:, :, 1] = rng.uniform(50, 78) - 7 * rr + 10 * (yy / height)
    base[:, :, 0] = rng.uniform(19, 35) - 2 * rr + 4 * (xx / width)

    noise = rng.normal(0, 5.5, size=(height, width, 1)).astype(np.float32)
    low_freq = rng.normal(0, 10, size=(max(2, height // 64), max(2, width // 64), 1)).astype(np.float32)
    low_freq = cv2.resize(low_freq, (width, height), interpolation=cv2.INTER_CUBIC)
    if low_freq.ndim == 2:
        low_freq = low_freq[:, :, None]
    warm_mottle = rng.normal(0, 10, size=(max(2, height // 34), max(2, width // 34), 1)).astype(np.float32)
    warm_mottle = cv2.resize(warm_mottle, (width, height), interpolation=cv2.INTER_CUBIC)
    if warm_mottle.ndim == 2:
        warm_mottle = warm_mottle[:, :, None]
    base[:, :, 1:3] += warm_mottle * np.array([[[0.35, 0.78]]], dtype=np.float32)
    fine_texture = rng.normal(0, 2.2, size=(height, width, 1)).astype(np.float32)
    image = np.clip(base + noise + low_freq + fine_texture, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (0, 0), 0.35)

    draw_retina_details(image, (cx, cy, radius), rng)
    sharpened = cv2.addWeighted(image, 1.16, cv2.GaussianBlur(image, (0, 0), 1.6), -0.16, 0)
    image[:] = np.clip(sharpened, 0, 255).astype(np.uint8)
    apply_scope_border(image, (cx, cy, radius), rng)
    return image


def draw_retina_details(
    image: np.ndarray,
    roi: tuple[float, float, float],
    rng: np.random.Generator,
) -> None:
    height, width = image.shape[:2]
    cx, cy, radius = roi
    disc = np.array(
        [
            cx - radius * rng.uniform(0.50, 0.62),
            cy - radius * rng.uniform(0.10, 0.25),
        ],
        dtype=np.float32,
    )
    disc_poly = ellipse_polygon(
        disc,
        radius * rng.uniform(0.085, 0.13),
        radius * rng.uniform(0.10, 0.15),
        rng.uniform(-0.45, 0.45),
        points=28,
    )
    draw_soft_polygon(image, disc_poly, (28, 118, 185), alpha=rng.uniform(0.42, 0.58), blur=13)
    draw_soft_polygon(
        image,
        ellipse_polygon(disc + np.array([radius * 0.025, radius * 0.02], dtype=np.float32), radius * 0.045, radius * 0.055, points=20),
        (51, 153, 211),
        alpha=0.24,
        blur=7,
    )

    for _ in range(rng.integers(14, 22)):
        start_angle = rng.uniform(-2.2, 2.0)
        length = radius * rng.uniform(0.45, 1.05)
        thickness = int(rng.integers(1, 4))
        pts = wavy_vessel(disc, start_angle, length, rng)
        draw_soft_line(image, pts, (15, 21, 74), thickness=thickness, alpha=rng.uniform(0.34, 0.58), blur=1)
        draw_soft_line(image, pts, (11, 14, 45), thickness=max(1, thickness - 1), alpha=0.25, blur=0)
        if rng.random() < 0.9:
            branch_start = pts[rng.integers(max(2, len(pts) // 4), len(pts) - 1)]
            branch = wavy_vessel(
                branch_start,
                start_angle + rng.choice([-1, 1]) * rng.uniform(0.35, 0.8),
                length * rng.uniform(0.28, 0.52),
                rng,
            )
            draw_soft_line(image, branch, (14, 22, 65), thickness=max(1, thickness - 1), alpha=0.30, blur=1)

    for _ in range(rng.integers(8, 15)):
        angle = rng.uniform(0, math.tau)
        dist = radius * math.sqrt(rng.uniform(0.03, 0.92))
        center = np.array([cx + math.cos(angle) * dist, cy + math.sin(angle) * dist], dtype=np.float32)
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            continue
        spot = ellipse_polygon(
            center,
            rng.uniform(4, 18),
            rng.uniform(3, 15),
            rng.uniform(0, math.tau),
            points=12,
        )
        color = [(21, 39, 68), (27, 54, 91), (15, 25, 47)][int(rng.integers(0, 3))]
        draw_soft_polygon(image, spot, color, alpha=rng.uniform(0.05, 0.13), blur=7)

    for _ in range(rng.integers(10, 18)):
        angle = rng.uniform(-0.35, 0.65)
        direction = unit(angle)
        start = np.array(
            [rng.uniform(-width * 0.05, width * 1.05), rng.uniform(0, height)],
            dtype=np.float32,
        )
        end = start + direction * rng.uniform(width * 0.18, width * 0.42)
        color = [(28, 45, 83), (23, 36, 70), (34, 58, 90)][int(rng.integers(0, 3))]
        draw_soft_line(
            image,
            np.vstack([start, end]),
            color,
            thickness=int(rng.integers(1, 3)),
            alpha=rng.uniform(0.08, 0.17),
            blur=5,
        )


def wavy_vessel(
    start: np.ndarray,
    angle: float,
    length: float,
    rng: np.random.Generator,
    segments: int = 16,
) -> np.ndarray:
    direction = unit(angle)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    points: list[np.ndarray] = []
    drift = rng.uniform(-0.45, 0.45)
    for i in range(segments):
        t = i / (segments - 1)
        curve = math.sin(t * math.pi * rng.uniform(0.8, 2.0) + drift) * length * rng.uniform(0.015, 0.035)
        points.append(start + direction * length * t + normal * curve)
    return np.array(points, dtype=np.float32)


def apply_scope_border(
    image: np.ndarray,
    roi: tuple[float, float, float],
    rng: np.random.Generator,
) -> None:
    height, width = image.shape[:2]
    cx, cy, radius = roi
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    outside = np.clip((dist - radius * 1.06) / (radius * 0.18), 0, 1)
    outside = cv2.GaussianBlur(outside, (0, 0), radius * 0.022)
    dark = np.array((20, 22, 27), dtype=np.float32)
    image[:] = np.clip(image.astype(np.float32) * (1.0 - outside[:, :, None]) + dark * outside[:, :, None], 0, 255)

    rim_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(rim_mask, (int(cx), int(cy)), int(radius * 1.08), 255, thickness=int(radius * 0.026), lineType=cv2.LINE_AA)
    rim_mask = cv2.GaussianBlur(rim_mask, (0, 0), radius * 0.035)
    blend_mask(image, rim_mask, (15, 17, 22), alpha=0.10)

    corner = np.zeros((height, width), dtype=np.uint8)
    corner_center = (int(width * rng.uniform(1.02, 1.12)), int(height * rng.uniform(-0.08, 0.05)))
    cv2.circle(corner, corner_center, int(min(width, height) * rng.uniform(0.20, 0.32)), 255, -1, lineType=cv2.LINE_AA)
    corner = cv2.GaussianBlur(corner, (0, 0), min(width, height) * 0.035)
    blend_mask(image, corner, (5, 7, 10), alpha=rng.uniform(0.28, 0.46))

    for _ in range(rng.integers(0, 2)):
        angle = rng.uniform(0, math.tau)
        center = np.array(
            [cx + math.cos(angle) * radius * rng.uniform(1.02, 1.2), cy + math.sin(angle) * radius * rng.uniform(1.02, 1.2)],
            dtype=np.float32,
        )
        glint = ellipse_polygon(center, radius * 0.09, radius * 0.018, angle + math.pi / 2, points=18)
        draw_soft_polygon(image, glint, (59, 84, 119), alpha=0.08, blur=25)


def load_background(background: Path, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    image = cv2.imread(str(background), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read background image: {background}")

    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(image, (math.ceil(src_w * scale), math.ceil(src_h * scale)), interpolation=cv2.INTER_AREA)
    max_x = resized.shape[1] - width
    max_y = resized.shape[0] - height
    x0 = int(rng.integers(0, max(1, max_x + 1)))
    y0 = int(rng.integers(0, max(1, max_y + 1)))
    crop = resized[y0 : y0 + height, x0 : x0 + width].copy()
    crop = cv2.GaussianBlur(crop, (0, 0), rng.uniform(0.25, 1.4))
    gain = rng.uniform(0.90, 1.10)
    bias = rng.uniform(-7, 7)
    return np.clip(crop.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def rotate_background(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    if abs(angle_degrees) < 1e-6:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    radians = math.radians(angle_degrees)
    cos_a = abs(math.cos(radians))
    sin_a = abs(math.sin(radians))
    cover_scale = max(
        (width * cos_a + height * sin_a) / width,
        (width * sin_a + height * cos_a) / height,
    )
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, cover_scale)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def render_forceps(image: np.ndarray, rng: np.random.Generator, axis_roll: float) -> Pose:
    height, width = image.shape[:2]
    for _ in range(200):
        tip_center = np.array(
            [
                rng.uniform(width * 0.35, width * 0.70),
                rng.uniform(height * 0.34, height * 0.68),
            ],
            dtype=np.float32,
        )
        direction = unit(rng.uniform(math.radians(188), math.radians(235)))
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        jaw_len = rng.uniform(width * 0.105, width * 0.18)
        jaw_open = rng.uniform(width * 0.045, width * 0.095)
        roll_angle = math.radians(rng.uniform(-axis_roll, axis_roll)) if axis_roll > 0 else 0.0
        roll_projection = 0.18 + 0.82 * abs(math.cos(roll_angle))
        shadow_roll_projection = 0.06 + 0.94 * abs(math.cos(roll_angle))
        projected_open = jaw_open * roll_projection
        projected_shadow_open = jaw_open * shadow_roll_projection
        roll_depth_shift = math.sin(roll_angle) * jaw_open * 0.22
        base = tip_center - direction * jaw_len
        tip_a = (
            base
            + direction * (jaw_len + roll_depth_shift / 2.0)
            + normal * projected_open / 2.0
        )
        tip_b = (
            base
            + direction * (jaw_len - roll_depth_shift / 2.0)
            - normal * projected_open / 2.0
        )
        shadow_offset = np.array(
            [rng.uniform(width * 0.05, width * 0.20), rng.uniform(-height * 0.11, height * 0.04)],
            dtype=np.float32,
        )
        shadow_center = base + direction * jaw_len + shadow_offset
        shadow_a = (
            shadow_center
            + direction * (roll_depth_shift / 2.0)
            + normal * projected_shadow_open / 2.0
        )
        shadow_b = (
            shadow_center
            - direction * (roll_depth_shift / 2.0)
            - normal * projected_shadow_open / 2.0
        )
        all_centers = np.vstack([tip_a, tip_b, shadow_a, shadow_b])
        if np.all((all_centers[:, 0] > 8) & (all_centers[:, 0] < width - 8) & (all_centers[:, 1] > 8) & (all_centers[:, 1] < height - 8)):
            break
    else:
        raise RuntimeError("could not sample a valid forceps pose")

    entry = base - direction * (max(width, height) * 0.72)
    shaft_thickness = int(rng.integers(28, 42))
    jaw_root_offset = normal * (shaft_thickness * (0.22 + 0.12 * roll_projection))
    jaw_root_a = base + jaw_root_offset
    jaw_root_b = base - jaw_root_offset
    shadow_base = base + shadow_offset
    shadow_entry = entry + shadow_offset
    shadow_root_a = jaw_root_a + shadow_offset
    shadow_root_b = jaw_root_b + shadow_offset

    tip_radius_l = rng.uniform(6, 11)
    tip_radius_w = rng.uniform(3, 6)
    visual_tip_polys = [
        oriented_box(tip_a, direction, tip_radius_l * 2.0, tip_radius_w * 2.0),
        oriented_box(tip_b, direction, tip_radius_l * 2.0, tip_radius_w * 2.0),
    ]
    tip_polys = [
        keypoint_polygon(tip_a + direction * tip_radius_l * 0.92),
        keypoint_polygon(tip_b + direction * tip_radius_l * 0.92),
    ]
    shadow_polys = [
        ellipse_polygon(shadow_a, rng.uniform(7, 14), rng.uniform(4, 9), math.atan2(direction[1], direction[0]), points=14),
        ellipse_polygon(shadow_b, rng.uniform(7, 14), rng.uniform(4, 9), math.atan2(direction[1], direction[0]), points=14),
    ]

    shadow_color = (18, 23, 35)
    shadow_strength = rng.uniform(0.85, 1.18)
    draw_soft_tapered_segment(
        image,
        shadow_entry,
        shadow_base,
        shaft_thickness * rng.uniform(0.92, 1.16),
        shaft_thickness * rng.uniform(0.48, 0.68),
        shadow_color,
        alpha=min(1.0, 0.20 * shadow_strength),
        blur=17,
        motion_blur=27,
    )
    draw_soft_tapered_segment(
        image,
        shadow_root_a,
        shadow_a,
        rng.uniform(9, 14),
        rng.uniform(5, 8),
        shadow_color,
        alpha=min(1.0, 0.24 * shadow_strength),
        blur=11,
        motion_blur=17,
    )
    draw_soft_tapered_segment(
        image,
        shadow_root_b,
        shadow_b,
        rng.uniform(9, 14),
        rng.uniform(5, 8),
        shadow_color,
        alpha=min(1.0, 0.24 * shadow_strength),
        blur=11,
        motion_blur=17,
    )
    for poly in shadow_polys:
        draw_soft_polygon(image, poly, shadow_color, alpha=min(1.0, 0.11 * shadow_strength), blur=11)

    shaft_start_width = shaft_thickness * rng.uniform(1.10, 1.26)
    shaft_end_width = shaft_thickness * rng.uniform(0.72, 0.92)
    draw_metal_segment(image, entry, base, shaft_start_width, shaft_end_width, rng)

    collar = oriented_box(
        base - direction * shaft_thickness * 0.05,
        direction,
        shaft_thickness * 0.62,
        shaft_thickness * (0.70 + 0.20 * roll_projection),
    )
    draw_soft_polygon(image, collar, (40, 36, 34), alpha=0.82, blur=5)
    draw_soft_polygon(
        image,
        oriented_box(base + direction * shaft_thickness * 0.08, direction, shaft_thickness * 0.42, shaft_thickness * 0.48),
        (92, 82, 74),
        alpha=0.35,
        blur=5,
    )

    jaw_width_root = rng.uniform(7, 10)
    jaw_width_tip = rng.uniform(2.4, 4.0)
    draw_metal_segment(
        image,
        jaw_root_a,
        tip_a,
        jaw_width_root,
        jaw_width_tip,
        rng,
        highlight_side=0.16,
    )
    draw_metal_segment(
        image,
        jaw_root_b,
        tip_b,
        jaw_width_root,
        jaw_width_tip,
        rng,
        highlight_side=-0.16,
    )

    for tip_poly in visual_tip_polys:
        draw_soft_polygon(image, tip_poly, (27, 26, 26), alpha=0.72, blur=3)
        tip_center = np.mean(tip_poly, axis=0)
        draw_soft_line(
            image,
            np.vstack([tip_center - direction * tip_radius_l * 0.45, tip_center + direction * tip_radius_l * 0.35]),
            (140, 130, 116),
            thickness=1,
            alpha=0.32,
            blur=3,
        )

    tip_polys = sorted(tip_polys, key=lambda p: float(np.mean(p[:, 0])))
    shadow_polys = sorted(shadow_polys, key=lambda p: float(np.mean(p[:, 0])))
    return Pose(tip_polygons=tip_polys, shadow_polygons=shadow_polys)


def format_yolo_value(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".")


def normalized_center(polygon: np.ndarray, width: int, height: int) -> tuple[float, float]:
    center = np.mean(polygon, axis=0)
    return (
        float(np.clip(center[0] / width, 0.0, 1.0)),
        float(np.clip(center[1] / height, 0.0, 1.0)),
    )


def object_bbox(
    keypoint_polygons: list[np.ndarray],
    width: int,
    height: int,
    padding: float = 0.02,
) -> list[float]:
    all_points = np.vstack(keypoint_polygons)
    x1 = max(0.0, float(np.min(all_points[:, 0])) / width - padding)
    y1 = max(0.0, float(np.min(all_points[:, 1])) / height - padding)
    x2 = min(1.0, float(np.max(all_points[:, 0])) / width + padding)
    y2 = min(1.0, float(np.max(all_points[:, 1])) / height + padding)
    return [
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        x2 - x1,
        y2 - y1,
    ]


def pose_label_line(
    class_id: int,
    keypoint_polygons: list[np.ndarray],
    width: int,
    height: int,
    visibility: int = 2,
) -> str:
    bbox = object_bbox(keypoint_polygons, width, height)
    values: list[float | int] = [class_id, *bbox]
    for polygon in keypoint_polygons:
        x, y = normalized_center(polygon, width, height)
        values.extend([x, y, visibility])

    return " ".join(format_yolo_value(float(value)) for value in values)


def pose_label_lines(pose: Pose, width: int, height: int, visibility: int = 2) -> list[str]:
    forceps_polygons = [pose.tip_polygons[0], pose.tip_polygons[1]]
    shadow_polygons = [pose.shadow_polygons[0], pose.shadow_polygons[1]]
    return [
        pose_label_line(FORCEPS_CLASS_ID, forceps_polygons, width, height, visibility),
        pose_label_line(SHADOW_CLASS_ID, shadow_polygons, width, height, visibility),
    ]


def normalized_box_xyxy(
    keypoint_polygons: list[np.ndarray],
    width: int,
    height: int,
    padding: float = 0.02,
) -> tuple[int, int, int, int]:
    cx, cy, box_width, box_height = object_bbox(keypoint_polygons, width, height, padding)
    x1 = int(round((cx - box_width / 2.0) * width))
    y1 = int(round((cy - box_height / 2.0) * height))
    x2 = int(round((cx + box_width / 2.0) * width))
    y2 = int(round((cy + box_height / 2.0) * height))
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    )


def render_preview(image: np.ndarray, pose: Pose) -> np.ndarray:
    preview = image.copy()
    height, width = preview.shape[:2]
    objects = [
        (
            FORCEPS_CLASS_ID,
            "forceps",
            FORCEPS_KEYPOINT_NAMES,
            [pose.tip_polygons[0], pose.tip_polygons[1]],
            CLASS_COLORS[0],
        ),
        (
            SHADOW_CLASS_ID,
            "shadow",
            SHADOW_KEYPOINT_NAMES,
            [pose.shadow_polygons[0], pose.shadow_polygons[1]],
            CLASS_COLORS[2],
        ),
    ]
    for _class_id, class_name, keypoint_names, polygons, color in objects:
        x1, y1, x2, y2 = normalized_box_xyxy(polygons, width, height)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        cv2.putText(
            preview,
            class_name,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        for keypoint_name, polygon in zip(keypoint_names, polygons, strict=True):
            center = np.round(np.mean(polygon, axis=0)).astype(int)
            point = (int(center[0]), int(center[1]))
            cv2.circle(preview, point, radius=6, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(preview, point, radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(
                preview,
                keypoint_name,
                (point[0] + 7, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return preview


def split_for_index(index: int, count: int, val_fraction: float) -> str:
    if val_fraction <= 0:
        return "train"
    if val_fraction >= 1:
        return "val"
    val_count = round(count * val_fraction)
    return "val" if index >= count - val_count else "train"


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_generation_progress(
    completed: int,
    total: int,
    generated: dict[str, int],
    start_time: float,
    *,
    final: bool = False,
) -> None:
    elapsed = time.monotonic() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else math.inf
    percent = 100.0 * completed / total if total else 100.0
    message = (
        f"Generating synthetic images: {completed}/{total} "
        f"({percent:5.1f}%) train={generated['train']} val={generated['val']} "
        f"{rate:4.1f} img/s ETA {format_duration(eta)}"
    )
    if sys.stderr.isatty():
        print(f"\r{message}", end="\n" if final else "", file=sys.stderr, flush=True)
    else:
        print(message, file=sys.stderr, flush=True)


def resolve_worker_count(requested_workers: int, count: int) -> int:
    if requested_workers < 0:
        raise SystemExit("--workers must be non-negative")
    if requested_workers == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(count, cpu_count - 1 if cpu_count > 1 else 1))
    return max(1, min(count, requested_workers))


def build_image_seeds(seed: Optional[int], count: int) -> list[int]:
    seed_sequence = np.random.SeedSequence(seed)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in seed_sequence.spawn(count)]


def generate_one_image(task: GenerationTask) -> str:
    cv2.setNumThreads(1)
    rng = np.random.default_rng(task.seed)
    split = split_for_index(task.index, task.count, task.val_fraction)
    name = f"{task.prefix}_{task.start_index + task.index:06d}"
    if task.background:
        image = load_background(task.background, task.width, task.height, rng)
    else:
        image = noisy_retina_background(task.width, task.height, rng)
    if task.background_rotation > 0:
        image = rotate_background(image, rng.uniform(-task.background_rotation, task.background_rotation))

    pose = render_forceps(image, rng, task.axis_roll)
    image_path = task.out_dir / "images" / split / f"{name}.{task.image_ext}"
    label_path = task.out_dir / "labels" / split / f"{name}.txt"
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"failed to write image: {image_path}")
    label_path.write_text("\n".join(pose_label_lines(pose, task.width, task.height)) + "\n")

    if task.index < task.preview:
        preview = render_preview(image, pose)
        preview_path = task.preview_dir / f"{name}.jpg"
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"failed to write preview: {preview_path}")

    return split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic retinal forceps pose data.")
    parser.add_argument("--count", type=int, default=100, help="Number of images to generate.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="Dataset root containing images/ and labels/.")
    parser.add_argument("--width", type=int, default=820, help="Output image width.")
    parser.add_argument("--height", type=int, default=920, help="Output image height.")
    parser.add_argument("--background", type=Path, help="Optional clean background image to composite onto.")
    parser.add_argument(
        "--background-rotation",
        type=float,
        default=180.0,
        metavar="DEGREES",
        help="Randomly rotate each retina/background by +/- this many degrees before drawing forceps. Use 0 to disable.",
    )
    parser.add_argument(
        "--axis-roll",
        type=float,
        default=180.0,
        metavar="DEGREES",
        help="Randomly roll forceps and shadow around the forceps shaft axis by +/- this many degrees. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation.")
    parser.add_argument("--prefix", default="synthetic", help="Filename prefix.")
    parser.add_argument("--start-index", type=int, default=0, help="First numeric image index.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of generated images written to val/.")
    parser.add_argument("--preview", type=int, default=0, help="Also render N label-overlay preview images.")
    parser.add_argument("--preview-dir", type=Path, default=Path("runs/synthetic_preview"), help="Directory for previews.")
    parser.add_argument("--image-ext", choices=["jpg", "png"], default="jpg", help="Image file extension.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes. Use 0 for auto, 1 for serial generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if not 0 <= args.val_fraction <= 1:
        raise SystemExit("--val-fraction must be between 0 and 1")
    if args.background_rotation < 0:
        raise SystemExit("--background-rotation must be non-negative")
    if args.axis_roll < 0:
        raise SystemExit("--axis-roll must be non-negative")
    worker_count = resolve_worker_count(args.workers, args.count)

    image_dirs = {split: args.out_dir / "images" / split for split in ("train", "val")}
    label_dirs = {split: args.out_dir / "labels" / split for split in ("train", "val")}
    for path in [*image_dirs.values(), *label_dirs.values()]:
        path.mkdir(parents=True, exist_ok=True)
    if args.preview:
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    image_seeds = build_image_seeds(args.seed, args.count)
    tasks = [
        GenerationTask(
            index=i,
            seed=image_seeds[i],
            count=args.count,
            out_dir=args.out_dir,
            width=args.width,
            height=args.height,
            background=args.background,
            background_rotation=args.background_rotation,
            axis_roll=args.axis_roll,
            prefix=args.prefix,
            start_index=args.start_index,
            val_fraction=args.val_fraction,
            preview=args.preview,
            preview_dir=args.preview_dir,
            image_ext=args.image_ext,
        )
        for i in range(args.count)
    ]

    generated = {"train": 0, "val": 0}
    start_time = time.monotonic()
    progress_interval = max(1, min(100, args.count // 100))
    print(f"Using {worker_count} synthetic generation worker(s)", file=sys.stderr, flush=True)
    print_generation_progress(0, args.count, generated, start_time)

    completed = 0
    if worker_count == 1:
        for task in tasks:
            split = generate_one_image(task)
            generated[split] += 1
            completed += 1
            if completed == args.count or completed % progress_interval == 0:
                print_generation_progress(
                    completed,
                    args.count,
                    generated,
                    start_time,
                    final=completed == args.count,
                )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(generate_one_image, task) for task in tasks]
            for future in as_completed(futures):
                split = future.result()
                generated[split] += 1
                completed += 1
                if completed == args.count or completed % progress_interval == 0:
                    print_generation_progress(
                        completed,
                        args.count,
                        generated,
                        start_time,
                        final=completed == args.count,
                    )

    if completed < args.count:
        print_generation_progress(
            completed,
            args.count,
            generated,
            start_time,
            final=True,
        )

    print(
        "Generated "
        f"{generated['train']} train and {generated['val']} val synthetic images under {args.out_dir}"
    )
    if args.preview:
        print(f"Wrote {min(args.preview, args.count)} previews under {args.preview_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
