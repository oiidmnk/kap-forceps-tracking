from types import SimpleNamespace

import numpy as np

from scripts.predict import CLASS_COLORS, render_box_result


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_renderer_draws_boxes_and_bottom_legend_without_filling_masks() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    result = SimpleNamespace(
        orig_img=image,
        boxes=SimpleNamespace(
            xyxy=FakeTensor([[10, 10, 50, 50]]),
            cls=FakeTensor([2]),
        ),
        names={0: "tip_left", 1: "tip_right", 2: "shadow_left", 3: "shadow_right"},
    )

    rendered = render_box_result(result)

    assert rendered.shape[0] > image.shape[0]
    assert tuple(rendered[10, 10]) == CLASS_COLORS[2]
    assert tuple(rendered[30, 30]) == (0, 0, 0)
    assert np.any(rendered[image.shape[0] :] != 0)
