import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.predict_preprocessor import (
    PredictionExtractionError,
    atomic_write_json,
    build_preprocessor_input,
    extract_preprocessor_points,
    parse_args,
    restore_original_point,
)
from scripts.preprocessing import CropTransform


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def fake_pose_result(keypoints, classes=None, box_conf=None, keypoint_conf=None):
    classes = classes if classes is not None else list(range(len(keypoints)))
    box_conf = box_conf if box_conf is not None else [0.5] * len(keypoints)
    keypoint_conf = keypoint_conf if keypoint_conf is not None else [
        [1.0] * len(detection_keypoints) for detection_keypoints in keypoints
    ]
    return SimpleNamespace(
        boxes=SimpleNamespace(cls=FakeTensor(classes), conf=FakeTensor(box_conf)),
        keypoints=SimpleNamespace(
            xy=FakeTensor(keypoints),
            conf=FakeTensor(keypoint_conf),
        ),
    )


def test_extracts_pose_keypoints_in_preprocessor_order() -> None:
    result = fake_pose_result(
        [
            [
                [10, 11],
                [20, 21],
            ],
            [
                [30, 31],
                [40, 41],
            ]
        ]
    )

    points = extract_preprocessor_points(result)

    assert points == {
        "left_tip_px": [10.0, 11.0],
        "right_tip_px": [20.0, 21.0],
        "left_shadow_px": [30.0, 31.0],
        "right_shadow_px": [40.0, 41.0],
    }


def test_extracts_pose_endpoints_and_ignores_root_keypoints() -> None:
    result = fake_pose_result(
        [
            [
                [10, 11],
                [20, 21],
                [15, 31],
            ],
            [
                [30, 31],
                [40, 41],
                [35, 51],
            ]
        ]
    )

    points = extract_preprocessor_points(result)

    assert points == {
        "left_tip_px": [10.0, 11.0],
        "right_tip_px": [20.0, 21.0],
        "left_shadow_px": [30.0, 31.0],
        "right_shadow_px": [40.0, 41.0],
    }


def test_uses_highest_confidence_pose_detection() -> None:
    result = fake_pose_result(
        [
            [
                [1, 1],
                [2, 2],
            ],
            [
                [10, 11],
                [20, 21],
            ],
            [
                [3, 3],
                [4, 4],
            ],
            [
                [30, 31],
                [40, 41],
            ],
        ],
        classes=[0, 0, 1, 1],
        box_conf=[0.1, 0.9, 0.2, 0.8],
    )

    points = extract_preprocessor_points(result)

    assert points["left_tip_px"] == [10.0, 11.0]
    assert points["right_shadow_px"] == [40.0, 41.0]


def test_missing_required_keypoint_raises_for_zero_coordinate() -> None:
    result = fake_pose_result(
        [
            [
                [10, 11],
                [0, 0],
            ],
            [
                [30, 31],
                [40, 41],
            ]
        ]
    )

    with pytest.raises(PredictionExtractionError, match="right_tip_px"):
        extract_preprocessor_points(result)


def test_keypoint_confidence_threshold_raises_when_required_point_is_low() -> None:
    result = fake_pose_result(
        [
            [
                [10, 11],
                [20, 21],
            ],
            [
                [30, 31],
                [40, 41],
            ]
        ],
        keypoint_conf=[[0.9, 0.9], [0.1, 0.9]],
    )

    with pytest.raises(PredictionExtractionError, match="left_shadow_px"):
        extract_preprocessor_points(result, kpt_conf=0.5)


def test_restores_cropped_coordinates_to_original_image_frame() -> None:
    transform = CropTransform(
        source_width=100,
        source_height=80,
        x=12,
        y=8,
        width=50,
        height=40,
    )

    assert restore_original_point((5.5, 7.25), transform) == [17.5, 15.25]


def test_extracts_pose_keypoints_after_crop_transform() -> None:
    transform = CropTransform(
        source_width=100,
        source_height=80,
        x=12,
        y=8,
        width=50,
        height=40,
    )
    result = fake_pose_result(
        [
            [
                [10, 11],
                [20, 21],
            ],
            [
                [30, 31],
                [40, 41],
            ]
        ]
    )

    points = extract_preprocessor_points(result, transform)

    assert points["left_tip_px"] == [22.0, 19.0]
    assert points["right_shadow_px"] == [52.0, 49.0]


def test_missing_forceps_detection_raises() -> None:
    result = fake_pose_result(
        [
            [
                [30, 31],
                [40, 41],
            ]
        ],
        classes=[1],
    )

    with pytest.raises(PredictionExtractionError, match="forceps"):
        extract_preprocessor_points(result)


def test_missing_shadow_detection_raises() -> None:
    result = fake_pose_result(
        [
            [
                [10, 11],
                [20, 21],
            ]
        ],
        classes=[0],
    )

    with pytest.raises(PredictionExtractionError, match="shadow"):
        extract_preprocessor_points(result)


def test_builds_merged_payload_and_writes_atomically(tmp_path) -> None:
    base = tmp_path / "base.json"
    sidecar = tmp_path / "sidecar.json"
    output = tmp_path / "predicted_input.json"
    base.write_text(
        json.dumps(
            {
                "light_rot_up": 0.2,
                "forceps_rot_clock": -1.2,
                "left_tip_px": [0, 0],
            }
        )
    )
    sidecar.write_text(json.dumps({"forceps_rot_clock": -1.0}))

    payload = build_preprocessor_input(
        base,
        sidecar,
        {
            "left_tip_px": [1.0, 2.0],
            "right_tip_px": [3.0, 4.0],
            "left_shadow_px": [5.0, 6.0],
            "right_shadow_px": [7.0, 8.0],
        },
    )
    atomic_write_json(output, payload)

    written = json.loads(output.read_text())
    assert written["forceps_rot_clock"] == -1.0
    assert written["left_tip_px"] == [1.0, 2.0]
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_parse_args_accepts_single_image_bridge_options() -> None:
    args = parse_args(
        [
            "--source",
            "frame.png",
            "--weights",
            "best.pt",
            "--kpt-conf",
            "0.25",
            "--base-input",
            "../preprocessing/input_example.json",
            "--output",
            "../preprocessing/predicted_input.json",
        ]
    )

    assert str(args.source) == "frame.png"
    assert str(args.weights) == "best.pt"
    assert args.kpt_conf == 0.25
