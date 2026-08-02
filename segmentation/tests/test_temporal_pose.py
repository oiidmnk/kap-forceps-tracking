from types import SimpleNamespace

import numpy as np

from scripts.temporal_pose import TemporalPoseTracker


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def fake_result(detections):
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    classes = [detection[0] for detection in detections]
    qualities = [detection[1] for detection in detections]
    points = [detection[2] for detection in detections]
    boxes = []
    for detection_points in points:
        coordinates = np.asarray(detection_points, dtype=float)
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        boxes.append([*minimum, *maximum])
    point_array = np.asarray(points, dtype=float).reshape(len(points), 3, 2)
    return SimpleNamespace(
        orig_img=image,
        names={0: "forceps", 1: "shadow"},
        boxes=SimpleNamespace(
            cls=FakeTensor(classes),
            conf=FakeTensor(qualities),
            xyxy=FakeTensor(np.asarray(boxes, dtype=float).reshape(len(boxes), 4)),
        ),
        keypoints=SimpleNamespace(
            xy=FakeTensor(point_array),
            conf=FakeTensor(np.ones((len(points), 3), dtype=float)),
        ),
    )


def rendered_points(result):
    return result.keypoints.xy.numpy()


def test_smooths_pose_motion_between_frames() -> None:
    tracker = TemporalPoseTracker(alpha=0.5, beta=0.0)
    tracker.update(fake_result([(0, 1.0, [[10, 10], [20, 10], [15, 20]])]))

    result = tracker.update(fake_result([(0, 1.0, [[20, 10], [30, 10], [25, 20]])]))

    np.testing.assert_allclose(
        rendered_points(result)[0],
        [[15, 10], [25, 10], [20, 20]],
    )


def test_prefers_nearby_candidate_over_distant_high_confidence_candidate() -> None:
    tracker = TemporalPoseTracker(alpha=1.0, beta=0.0, max_jump_fraction=0.2)
    tracker.update(fake_result([(0, 0.8, [[10, 10], [20, 10], [15, 20]])]))

    result = tracker.update(
        fake_result(
            [
                (0, 0.6, [[12, 10], [22, 10], [17, 20]]),
                (0, 0.99, [[70, 70], [80, 70], [75, 80]]),
            ]
        )
    )

    np.testing.assert_allclose(
        rendered_points(result)[0],
        [[11.6, 10], [21.6, 10], [16.6, 20]],
    )


def test_bridges_short_detection_gap_then_expires_track() -> None:
    tracker = TemporalPoseTracker(alpha=1.0, beta=0.5, max_gap=1)
    tracker.update(fake_result([(0, 1.0, [[10, 10], [20, 10], [15, 20]])]))
    tracker.update(fake_result([(0, 1.0, [[12, 10], [22, 10], [17, 20]])]))

    bridged = tracker.update(fake_result([]))
    expired = tracker.update(fake_result([]))

    assert len(rendered_points(bridged)) == 1
    assert rendered_points(bridged)[0, 0, 0] > 12
    assert len(rendered_points(expired)) == 0


def test_normalizes_left_and_right_endpoint_order() -> None:
    tracker = TemporalPoseTracker()

    result = tracker.update(
        fake_result([(0, 1.0, [[30, 10], [10, 10], [20, 20]])])
    )

    assert rendered_points(result)[0, 0, 0] == 10
    assert rendered_points(result)[0, 1, 0] == 30


def test_fuses_optical_flow_with_current_pose(monkeypatch) -> None:
    tracker = TemporalPoseTracker(alpha=0.5, beta=0.0)
    tracker.update(fake_result([(0, 1.0, [[10, 10], [20, 10], [15, 20]])]))

    def translated_flow(_previous, _current, points, _unused, **_kwargs):
        translated = points.copy()
        translated[:, :, 0] += 4
        count = len(points)
        return translated, np.ones((count, 1), dtype=np.uint8), np.zeros((count, 1))

    monkeypatch.setattr("scripts.temporal_pose.cv2.calcOpticalFlowPyrLK", translated_flow)
    result = tracker.update(
        fake_result([(0, 1.0, [[16, 10], [26, 10], [21, 20]])])
    )

    np.testing.assert_allclose(
        rendered_points(result)[0],
        [[15, 10], [25, 10], [20, 20]],
    )
