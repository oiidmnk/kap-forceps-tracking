import numpy as np
import pytest

from detector.sources import rotate_frame


def test_rotate_180_is_flip_both_axes():
    f = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert (rotate_frame(f, 180) == f[::-1, ::-1]).all()
    assert rotate_frame(f, 0) is f and rotate_frame(f, 360) is f


def test_rotate_90_swaps_shape_and_rejects_odd_angles():
    f = np.zeros((3, 4, 3), np.uint8)
    assert rotate_frame(f, 90).shape == (4, 3, 3) and rotate_frame(f, -90).shape == (4, 3, 3)
    with pytest.raises(ValueError):
        rotate_frame(f, 45)
