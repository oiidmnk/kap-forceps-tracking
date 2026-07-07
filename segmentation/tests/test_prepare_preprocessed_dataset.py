from scripts.preprocessing import CropTransform
from scripts.prepare_preprocessed_dataset import transform_segment_line


TRANSFORM = CropTransform(
    source_width=100,
    source_height=100,
    x=20,
    y=20,
    width=60,
    height=60,
)


def test_polygon_inside_crop_is_renormalized() -> None:
    result = transform_segment_line(
        "2 0.30 0.30 0.50 0.30 0.50 0.50 0.30 0.50",
        TRANSFORM,
    )
    assert result == (
        "2 0.166667 0.166667 0.500000 0.166667 "
        "0.500000 0.500000 0.166667 0.500000"
    )


def test_polygon_crossing_crop_is_clipped() -> None:
    result = transform_segment_line(
        "1 0.10 0.30 0.30 0.30 0.30 0.50 0.10 0.50",
        TRANSFORM,
    )
    assert result == (
        "1 0.000000 0.166667 0.166667 0.166667 "
        "0.166667 0.500000 0.000000 0.500000"
    )


def test_polygon_outside_crop_is_removed() -> None:
    result = transform_segment_line(
        "0 0.01 0.01 0.10 0.01 0.10 0.10",
        TRANSFORM,
    )
    assert result is None
