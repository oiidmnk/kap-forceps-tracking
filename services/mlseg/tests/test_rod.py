"""Rod labels and decoding helpers (no checkpoint needed)."""
import numpy as np


def test_rod_shadow_is_paired_by_a_common_offset():
    from mlseg.detector import pair_rod_shadow
    ends = np.array([[0.0, 0.0], [100.0, 0.0]])
    shadow = np.array([[130.0, 40.0], [30.0, 40.0]])          # listed in the other order
    assert np.allclose(pair_rod_shadow(ends, shadow), [[30, 40], [130, 40]])


def test_rod_channels_share_one_heatmap_and_mask():
    from mlseg.data import channel_mask, heatmaps
    from mlseg.model import ROW_CH
    pts = np.zeros((10, 2), np.float32)
    pts[6], pts[7] = (10, 10), (30, 10)
    valid = np.zeros(10, bool)
    valid[6:8] = True
    hm = heatmaps(pts, 48, 2.0, valid, ROW_CH)
    assert hm.shape[0] == 8 and hm[6, 10, 10] == 1 and hm[6, 10, 30] == 1 and hm[7].max() == 0
    state = np.array([1] * 6 + [1, 1, -1, -1], np.int8)
    assert channel_mask(state, ROW_CH).tolist() == [1] * 7 + [0]


def test_forceps_shadow_is_held_briefly_then_dropped(monkeypatch):
    from mlseg import detector as D
    det = D.MLDetector.__new__(D.MLDetector)
    det.shadow_hold_s, det._offset = 1.0, None
    t = [100.0]
    monkeypatch.setattr(D.time, "monotonic", lambda: t[0])
    tips = D.MLToolPose(True, "", (10.0, 10.0), (20.0, 10.0))
    seen = det._hold_shadow(tips, D.MLToolPose(True, "", (40.0, 0.0), (50.0, 0.0)))
    assert seen.reason == ""
    gone = D.MLToolPose(False, "tool not visible")
    moved = D.MLToolPose(True, "", (15.0, 12.0), (25.0, 12.0))
    t[0] = 100.5
    held = det._hold_shadow(moved, gone)
    assert held.ok and held.reason == "held" and held.left_tip == (45.0, 2.0) and held.right_tip == (55.0, 2.0)
    t[0] = 101.6
    assert not det._hold_shadow(moved, gone).ok          # older than the hold time
