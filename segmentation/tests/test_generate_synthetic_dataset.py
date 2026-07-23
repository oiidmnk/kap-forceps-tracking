import cv2
import numpy as np
import pytest

from scripts.generate_synthetic_dataset import (
    EXPECTED_POSE_LABEL_COLUMNS,
    GenerationTask,
    Pose,
    circular_png_image,
    generate_one_image,
    pose_label_lines,
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


def test_circular_png_image_adds_transparent_non_black_edges() -> None:
    image = np.full((40, 40, 3), (20, 80, 150), dtype=np.uint8)

    output = circular_png_image(image)

    assert output.shape == (40, 40, 4)
    assert output[0, 0, 3] == 0
    assert output[20, 20, 3] == 255
    assert np.any(output[0, 0, :3] > 0)


def test_generate_one_image_always_writes_circular_png(tmp_path) -> None:
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
    assert saved.shape == (256, 256, 4)
    assert saved[0, 0, 3] == 0
    assert saved[128, 128, 3] == 255
    assert np.any(saved[0, 0, :3] > 0)
