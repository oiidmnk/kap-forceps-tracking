from pathlib import Path

from scripts.pose_to_segments import convert_pose_text


def test_converts_three_keypoint_pose_objects_to_endpoint_segments() -> None:
    text = "\n".join(
        [
            "0 0.5 0.5 0.4 0.4 0.1 0.2 2 0.3 0.4 2 0.2 0.3 2",
            "1 0.6 0.6 0.4 0.4 0.5 0.6 2 0.7 0.8 2 0.6 0.7 2",
        ]
    )

    converted, object_count = convert_pose_text(
        text,
        Path("labels.txt"),
        box_size=0.02,
        min_visibility=1.0,
    )

    lines = converted.strip().splitlines()
    assert object_count == 4
    assert [line.split()[0] for line in lines] == ["0", "1", "2", "3"]
