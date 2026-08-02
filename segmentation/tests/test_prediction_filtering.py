from types import SimpleNamespace

import numpy as np

from scripts.prediction_filtering import select_expected_pose_indices


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def fake_result(image, classes, box_conf, keypoints, keypoint_conf):
    return SimpleNamespace(
        orig_img=image,
        boxes=SimpleNamespace(
            cls=FakeTensor(classes),
            conf=FakeTensor(box_conf),
        ),
        keypoints=SimpleNamespace(
            xy=FakeTensor(keypoints),
            conf=FakeTensor(keypoint_conf),
        ),
    )


def test_selects_one_highest_quality_detection_per_class() -> None:
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    result = fake_result(
        image,
        classes=[0, 0, 1, 1],
        box_conf=[0.5, 0.9, 0.8, 0.6],
        keypoints=[
            [[10, 10], [20, 10], [15, 20]],
            [[30, 30], [40, 30], [35, 40]],
            [[50, 50], [60, 50], [55, 60]],
            [[70, 70], [80, 70], [75, 80]],
        ],
        keypoint_conf=np.ones((4, 3)),
    )

    assert select_expected_pose_indices(result) == [1, 2]


def test_rejects_high_confidence_detection_on_black_border() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:80, 20:80] = 120
    result = fake_result(
        image,
        classes=[0, 0, 1],
        box_conf=[0.99, 0.7, 0.8],
        keypoints=[
            [[5, 5], [8, 8], [10, 10]],
            [[30, 30], [40, 30], [35, 40]],
            [[50, 50], [60, 50], [55, 60]],
        ],
        keypoint_conf=np.ones((3, 3)),
    )

    assert select_expected_pose_indices(result) == [1, 2]
