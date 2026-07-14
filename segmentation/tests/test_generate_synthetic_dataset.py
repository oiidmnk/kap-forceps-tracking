import numpy as np
import pytest

from scripts.generate_synthetic_dataset import Pose, pose_label_lines


def test_pose_label_lines_write_forceps_and_shadow_objects() -> None:
    pose = Pose(
        tip_polygons=[
            np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32),
            np.array([[30, 10], [40, 10], [40, 20], [30, 20]], dtype=np.float32),
        ],
        shadow_polygons=[
            np.array([[50, 50], [60, 50], [60, 60], [50, 60]], dtype=np.float32),
            np.array([[70, 50], [80, 50], [80, 60], [70, 60]], dtype=np.float32),
        ],
    )

    lines = pose_label_lines(pose, width=100, height=100)

    assert len(lines) == 2
    assert [line.split()[0] for line in lines] == ["0", "1"]
    assert [len(line.split()) for line in lines] == [11, 11]

    forceps_values = [float(value) for value in lines[0].split()]
    shadow_values = [float(value) for value in lines[1].split()]
    assert forceps_values[5:] == pytest.approx([0.15, 0.15, 2, 0.35, 0.15, 2])
    assert shadow_values[5:] == pytest.approx([0.55, 0.55, 2, 0.75, 0.55, 2])
