import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.generate_synthetic_dataset import (
    EXPECTED_POSE_LABEL_COLUMNS,
    GenerationTask,
    Pose,
    circular_png_image,
    circular_object_safety_margin,
    distal_pad_width_profile,
    generate_one_image,
    load_background,
    pose_label_lines,
    parse_args,
    pose_inside_circular_view,
    render_forceps,
    rotate_image_and_pose,
    sample_roll_pair,
    select_background,
    select_image_rotation,
    shadow_clears_forceps,
)


def test_pose_label_lines_write_forceps_and_shadow_objects() -> None:
    pose = Pose(
        tip_polygons=[
            np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32),
            np.array([[30, 10], [40, 10], [40, 20], [30, 20]], dtype=np.float32),
            np.array([[20, 30], [30, 30], [30, 40], [20, 40]], dtype=np.float32),
        ],
        shadow_polygons=[
            np.array([[50, 50], [60, 50], [60, 60], [50, 60]], dtype=np.float32),
            np.array([[70, 50], [80, 50], [80, 60], [70, 60]], dtype=np.float32),
            np.array([[60, 70], [70, 70], [70, 80], [60, 80]], dtype=np.float32),
        ],
    )

    lines = pose_label_lines(pose, width=100, height=100)

    assert len(lines) == 2
    assert [line.split()[0] for line in lines] == ["0", "1"]
    assert [len(line.split()) for line in lines] == [
        EXPECTED_POSE_LABEL_COLUMNS,
        EXPECTED_POSE_LABEL_COLUMNS,
    ]

    forceps_values = [float(value) for value in lines[0].split()]
    shadow_values = [float(value) for value in lines[1].split()]
    assert forceps_values[5:] == pytest.approx(
        [0.15, 0.15, 2, 0.35, 0.15, 2, 0.25, 0.35, 2]
    )
    assert shadow_values[5:] == pytest.approx(
        [0.55, 0.55, 2, 0.75, 0.55, 2, 0.65, 0.75, 2]
    )


def test_pose_label_lines_keep_coordinates_normalized() -> None:
    pose = Pose(
        tip_polygons=[
            np.array([[-10, -10], [5, -10], [5, 5], [-10, 5]], dtype=np.float32),
            np.array([[95, 95], [110, 95], [110, 110], [95, 110]], dtype=np.float32),
            np.array([[45, 45], [55, 45], [55, 55], [45, 55]], dtype=np.float32),
        ],
        shadow_polygons=[
            np.array([[20, 30], [24, 30], [24, 34], [20, 34]], dtype=np.float32),
            np.array([[40, 50], [44, 50], [44, 54], [40, 54]], dtype=np.float32),
            np.array([[60, 70], [64, 70], [64, 74], [60, 74]], dtype=np.float32),
        ],
    )

    lines = pose_label_lines(pose, width=100, height=100)

    for line in lines:
        values = [float(value) for value in line.split()]
        normalized_values = values[1:5] + values[5:7] + values[8:10] + values[11:13]
        assert all(0.0 <= value <= 1.0 for value in normalized_values)


def test_shadow_visibility_rejects_points_covered_by_forceps_segments() -> None:
    forceps_segments = [
        (
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([100.0, 0.0], dtype=np.float32),
        )
    ]

    covered_shadow = np.array(
        [[40.0, 3.0], [60.0, 4.0], [50.0, 2.0]],
        dtype=np.float32,
    )
    clear_shadow = np.array(
        [[40.0, 30.0], [60.0, 34.0], [50.0, 32.0]],
        dtype=np.float32,
    )

    assert not shadow_clears_forceps(covered_shadow, forceps_segments, min_distance=12.0)
    assert shadow_clears_forceps(clear_shadow, forceps_segments, min_distance=12.0)


def test_circular_png_image_adds_black_non_transparent_edges() -> None:
    image = np.full((40, 40, 3), (20, 80, 150), dtype=np.uint8)

    output = circular_png_image(image)

    assert output.shape == (40, 40, 3)
    assert np.all(output[0, 0] == 0)
    assert np.all(output[20, 20] == image[20, 20])


def test_generate_one_image_writes_black_circular_rgb_png_by_default(tmp_path) -> None:
    task = GenerationTask(
        index=0,
        seed=123,
        count=1,
        out_dir=tmp_path,
        width=256,
        height=256,
        background=None,
        background_rotation=0,
        axis_roll=0,
        prefix="synthetic",
        start_index=0,
        val_fraction=0,
        preview=1,
        preview_dir=tmp_path / "preview",
    )
    task.preview_dir.mkdir()
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)

    split = generate_one_image(task)

    assert split == "train"
    image_path = tmp_path / "images" / "train" / "synthetic_000000.png"
    preview_path = tmp_path / "preview" / "synthetic_000000.png"
    assert image_path.exists()
    assert preview_path.exists()
    saved = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    assert saved.shape == (256, 256, 3)
    assert np.all(saved[0, 0] == 0)
    assert np.all(saved[-1, -1] == 0)
    assert np.any(saved[128, 128] > 0)

    label_lines = (
        tmp_path / "labels" / "train" / "synthetic_000000.txt"
    ).read_text().splitlines()
    assert [line.split()[0] for line in label_lines] == ["0", "1"]
    center = np.array([(256 - 1) / 2.0, (256 - 1) / 2.0])
    radius = 256 / 2.0 - 1.0
    required_clearance = circular_object_safety_margin(256, 256, 18.0)
    for line in label_lines:
        values = [float(value) for value in line.split()]
        for offset in (5, 8, 11):
            point = np.array([values[offset] * 256, values[offset + 1] * 256])
            clearance = radius - float(np.linalg.norm(point - center))
            assert clearance >= required_clearance


def test_generate_one_image_can_opt_into_full_rectangular_view(tmp_path) -> None:
    task = GenerationTask(
        index=0,
        seed=123,
        count=1,
        out_dir=tmp_path,
        width=256,
        height=256,
        background=None,
        background_rotation=0,
        axis_roll=0,
        prefix="synthetic",
        start_index=0,
        val_fraction=0,
        preview=0,
        preview_dir=tmp_path / "preview",
        circular_mask=False,
    )
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)

    generate_one_image(task)

    saved = cv2.imread(
        str(tmp_path / "images" / "train" / "synthetic_000000.png"),
        cv2.IMREAD_UNCHANGED,
    )
    assert saved.shape == (256, 256, 3)
    assert np.any(saved[0, 0] > 0)
    assert np.any(saved[128, 128] > 0)


def test_roll_sampler_covers_forceps_only_shadow_only_and_both() -> None:
    rng = np.random.default_rng(9324)
    samples = [sample_roll_pair(rng, 180.0, 180.0) for _ in range(160)]

    assert any(abs(forceps) > 35 and abs(shadow) < 15 for forceps, shadow in samples)
    assert any(abs(forceps) < 15 and abs(shadow) > 35 for forceps, shadow in samples)
    assert any(abs(forceps) > 35 and abs(shadow) > 35 for forceps, shadow in samples)


def test_render_forceps_records_requested_large_tip_and_shadow_scales() -> None:
    image = np.full((640, 640, 3), (35, 80, 165), dtype=np.uint8)

    pose = render_forceps(
        image,
        np.random.default_rng(1701),
        axis_roll=180.0,
        shadow_axis_roll=180.0,
        shadow_scale_range=(1.75, 1.75),
        tip_scale_range=(1.90, 1.90),
        shadow_opacity_range=(0.62, 0.62),
        shadow_blur_range=(11.0, 11.0),
    )

    assert pose.variation is not None
    assert pose.variation.shadow_scale == pytest.approx(1.75)
    assert pose.variation.tip_scale == pytest.approx(1.90)
    assert pose.variation.shadow_opacity == pytest.approx(0.62)
    assert pose.variation.shadow_softness == pytest.approx(11.0)
    assert -180.0 <= pose.variation.forceps_roll_degrees <= 180.0
    assert -180.0 <= pose.variation.shadow_roll_degrees <= 180.0


def test_distal_pad_is_a_single_continuous_jaw_width_profile() -> None:
    widths = distal_pad_width_profile(
        point_count=18,
        root_width=12.0,
        jaw_tip_width=3.0,
        pad_width=8.0,
        pad_length_fraction=0.25,
    )

    assert widths.shape == (18,)
    assert widths[0] == pytest.approx(12.0)
    assert np.all(widths > 0)
    assert widths[-1] == pytest.approx(8.0 * 0.58)
    assert np.max(np.abs(np.diff(widths))) < 3.0


def test_select_background_samples_from_all_provided_images() -> None:
    backgrounds = tuple(Path(f"retina_{index}.png") for index in range(3))
    rng = np.random.default_rng(803)

    selected = {select_background(backgrounds, rng) for _ in range(80)}

    assert selected == set(backgrounds)


def test_parse_args_accepts_multiple_and_repeated_background_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_synthetic_dataset.py",
            "--background",
            "retina_a.png",
            "retina_b.png",
            "--background",
            "retina_c.png",
        ],
    )

    args = parse_args()

    assert args.background == [
        Path("retina_a.png"),
        Path("retina_b.png"),
        Path("retina_c.png"),
    ]


def test_parse_args_accepts_shadow_visibility_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_synthetic_dataset.py",
            "--shadow-visibility",
            "0.45",
            "0.75",
        ],
    )

    args = parse_args()

    assert args.shadow_opacity == pytest.approx([0.45, 0.75])


def test_parse_args_accepts_shadow_softness_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_synthetic_dataset.py",
            "--shadow-softness",
            "2",
            "24",
        ],
    )

    args = parse_args()

    assert args.shadow_blur == pytest.approx([2.0, 24.0])


def test_rotate_image_and_pose_uses_black_canvas_and_transforms_labels() -> None:
    image = np.full((80, 100, 3), 255, dtype=np.uint8)
    keypoint = np.array(
        [[68, 38], [72, 38], [72, 42], [68, 42]],
        dtype=np.float32,
    )
    pose = Pose(
        tip_polygons=[keypoint.copy(), keypoint.copy(), keypoint.copy()],
        shadow_polygons=[keypoint.copy(), keypoint.copy(), keypoint.copy()],
    )

    rotated, rotated_pose = rotate_image_and_pose(image, pose, 90.0)

    assert rotated.shape == image.shape
    assert np.all(rotated[0, 0] == 0)
    assert np.all(rotated[-1, -1] == 0)
    original_center = np.mean(keypoint, axis=0)
    rotated_center = np.mean(rotated_pose.tip_polygons[0], axis=0)
    assert rotated_center[0] == pytest.approx(49.5, abs=0.6)
    assert rotated_center[1] < original_center[1]


def test_load_background_composites_transparent_pixels_onto_black(tmp_path) -> None:
    source = np.full((40, 40, 4), 255, dtype=np.uint8)
    source[:, :20, 3] = 0
    path = tmp_path / "transparent_background.png"
    assert cv2.imwrite(str(path), source)

    loaded = load_background(path, 40, 40, np.random.default_rng(12))

    assert loaded.shape == (40, 40, 3)
    assert np.all(loaded[:, :18] == 0)
    assert np.any(loaded[:, 22:] > 0)


def test_load_background_preserves_brightness(tmp_path) -> None:
    source = np.full((40, 40, 3), (42, 103, 187), dtype=np.uint8)
    path = tmp_path / "constant_background.png"
    assert cv2.imwrite(str(path), source)

    loaded = load_background(path, 40, 40, np.random.default_rng(31))

    assert np.array_equal(loaded, source)


def test_select_image_rotation_uses_only_discrete_requested_angles() -> None:
    rotations = (0.0, 90.0, 180.0, 270.0)
    rng = np.random.default_rng(55)

    selected = {select_image_rotation(rotations, rng) for _ in range(100)}

    assert selected == set(rotations)


def test_parse_args_defaults_to_quarter_turn_rotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["generate_synthetic_dataset.py"])

    args = parse_args()

    assert args.image_rotations == pytest.approx([90.0, 180.0, 270.0])


def test_pose_inside_circular_view_checks_both_labeled_objects() -> None:
    center_polygon = np.array(
        [[47, 47], [53, 47], [53, 53], [47, 53]],
        dtype=np.float32,
    )
    outside_polygon = center_polygon + np.array([48, 0], dtype=np.float32)
    inside = Pose(
        tip_polygons=[center_polygon.copy() for _ in range(3)],
        shadow_polygons=[center_polygon.copy() for _ in range(3)],
    )
    shadow_outside = Pose(
        tip_polygons=[center_polygon.copy() for _ in range(3)],
        shadow_polygons=[center_polygon.copy(), center_polygon.copy(), outside_polygon],
    )

    assert pose_inside_circular_view(inside, 100, 100, margin=2)
    assert not pose_inside_circular_view(shadow_outside, 100, 100, margin=2)


def test_circular_object_safety_margin_accounts_for_view_and_blur() -> None:
    assert circular_object_safety_margin(820, 920, 18.0) == pytest.approx(57.4)
    assert circular_object_safety_margin(820, 920, 40.0) == pytest.approx(92.3)
