from types import SimpleNamespace

import numpy as np

from scripts.predict import serialize_segmentation_result
from scripts.preprocessing import CropTransform


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_serializer_returns_boxes_and_segments_in_original_image_coordinates() -> None:
    result = SimpleNamespace(
        orig_img=np.zeros((80, 100, 3), dtype=np.uint8),
        names={0: "tip_left", 1: "tip_right"},
        boxes=SimpleNamespace(
            xyxy=FakeTensor([[10, 20, 40, 60]]),
            conf=FakeTensor([0.9]),
            cls=FakeTensor([1]),
        ),
        masks=SimpleNamespace(
            xy=[
                np.asarray(
                    [
                        [12, 22],
                        [38, 22],
                        [38, 58],
                        [12, 58],
                    ],
                    dtype=float,
                )
            ]
        ),
    )
    transform = CropTransform(
        source_width=200,
        source_height=160,
        x=50,
        y=30,
        width=100,
        height=80,
    )

    payload = serialize_segmentation_result(result, transform)

    assert payload["image"]["width"] == 200
    assert payload["image"]["inference_width"] == 100
    assert payload["transform"]["is_identity"] is False
    assert payload["instances"] == [
        {
            "class_id": 1,
            "class_name": "tip_right",
            "confidence": 0.9,
            "box": {
                "xyxy": [60.0, 50.0, 90.0, 90.0],
                "normalized_xyxy": [0.3, 0.3125, 0.45, 0.5625],
            },
            "segments": [
                {
                    "points": [
                        [62.0, 52.0],
                        [88.0, 52.0],
                        [88.0, 88.0],
                        [62.0, 88.0],
                    ],
                    "normalized_points": [
                        [0.31, 0.325],
                        [0.44, 0.325],
                        [0.44, 0.55],
                        [0.31, 0.55],
                    ],
                }
            ],
        }
    ]
