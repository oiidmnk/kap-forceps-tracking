from detector.contact import RodContact
from detector.live import RodFilter, RodHold, Smoother
from detector.pipeline import Detection, RodPose, ToolPose

ROD = RodPose(((10.0, 10.0), (50.0, 10.0)), ((12.0, 30.0), (52.0, 30.0)))
NONE = ToolPose(False, "tool not visible")


def test_rod_only_frame_is_pushed_without_forceps_keys():
    pts = Detection(NONE, NONE, rod=ROD).orchestrator_points()
    assert pts == {"rod_a_px": [10.0, 10.0], "rod_b_px": [50.0, 10.0],
                   "rod_a_shadow_px": [12.0, 30.0], "rod_b_shadow_px": [52.0, 30.0], "rod_contact": False}
    assert Detection(NONE, NONE, rod=RodPose(ROD.ends, None)).orchestrator_points() is None


def test_smoother_keeps_rod_ends_on_their_side_and_bridges_gaps():
    s = Smoother(0.5, hold=1)
    s({"rod_a_px": [0, 0], "rod_b_px": [100, 0], "rod_a_shadow_px": [0, 9], "rod_b_shadow_px": [100, 9]})
    out = s({"rod_a_px": [100, 0], "rod_b_px": [2, 0], "rod_a_shadow_px": [100, 9], "rod_b_shadow_px": [2, 9]})
    assert out["rod_a_px"] == [1.0, 0.0] and out["rod_b_shadow_px"] == [100.0, 9.0]   # swapped peaks re-paired
    held = s({"left_tip_px": [5, 5]})
    assert held["rod_a_px"] == [1.0, 0.0] and held["left_tip_px"] == [5, 5]           # one missed frame bridged
    assert "rod_a_px" not in s({"left_tip_px": [5, 5]})                              # then dropped


def test_smoother_passes_the_contact_flag_through():
    s = Smoother(0.5)
    s({"rod_a_px": [0, 0], "rod_contact": False})
    assert s({"rod_a_px": [2, 0], "rod_contact": True}) == {"rod_a_px": [1.0, 0.0], "rod_contact": True}


# rod 150 px long (width 20), lying along x at y=100; its shadow 60 px lower
RA, RB = (100.0, 100.0), (250.0, 100.0)
RSA, RSB = (100.0, 160.0), (250.0, 160.0)


def frame(tip_y, shadow_y=None, shift=0.0, held=False):
    """forceps tips around x=175 at height tip_y; its shadow tips at shadow_y (None: not seen)"""
    forceps = ToolPose(True, "", (170.0 + shift, tip_y), (180.0 + shift, tip_y))
    shadow = (ToolPose(True, "held" if held else "", (170.0 + shift, shadow_y), (180.0 + shift, shadow_y))
              if shadow_y is not None else NONE)
    rod = RodPose(((RA[0] + shift, RA[1]), (RB[0] + shift, RB[1])), ((RSA[0] + shift, RSA[1]), (RSB[0] + shift, RSB[1])))
    return Detection(forceps, shadow, rod=rod)


def run(c, frames):
    return [c(f) for f in frames]


def test_tip_on_the_rod_with_its_shadow_on_the_rod_shadow_is_contact():
    assert run(RodContact(), [frame(100, 160)] * 3) == [False, True, True]       # confirmed on the 2nd frame


def test_tip_above_the_rod_casts_its_shadow_elsewhere():
    assert not any(run(RodContact(), [frame(100, 260)] * 10))                   # on the rod in the image only


def test_contact_survives_the_shadow_merging_and_ends_when_the_tips_leave():
    c = RodContact()
    out = run(c, [frame(100, 160)] * 3 + [frame(100)] * 20 + [frame(100, 160, held=True)] * 5 + [frame(160, 220)] * 3)
    assert all(out[1:30]) and not out[-1]                                       # released after 3 frames off the rod


def test_one_noisy_frame_or_a_brief_miss_does_not_end_contact():
    c = RodContact()
    out = run(c, [frame(100, 160)] * 3 + [frame(100, 260)] + [Detection(NONE, NONE, rod=None)] * 4 + [frame(100, 160)] * 2)
    assert all(out[1:])
    assert not any(run(RodContact(), [Detection(NONE, NONE, rod=None)] * 3))


def test_tips_that_push_the_rod_are_in_contact_without_a_shadow():
    out = run(RodContact(), [frame(100, shift=3.0 * i) for i in range(6)])
    assert out[-1] and not out[0]


FORCEPS = {"left_tip_px": [0.0, 0.0], "right_tip_px": [10.0, 0.0], "left_shadow_px": [0.0, 9.0], "right_shadow_px": [10.0, 9.0]}
SEEN_ROD = {"rod_a_px": [20.0, 0.0], "rod_b_px": [60.0, 0.0], "rod_a_shadow_px": [20.0, 30.0], "rod_b_shadow_px": [60.0, 30.0]}


def moved(d, dx):
    return {k: [v[0] + dx, v[1]] for k, v in d.items()}


def test_a_lost_rod_stays_where_it_was_seen():
    h = RodHold()
    h({**FORCEPS, **SEEN_ROD, "rod_contact": False})
    out = h(moved(FORCEPS, 5.0))                     # forceps moves on, rod not found
    assert out["rod_a_px"] == [20.0, 0.0] and out["rod_contact"] is False
    assert h(None) is None                           # nothing found at all: still nothing pushed
    h.reset()
    assert "rod_a_px" not in h(FORCEPS)              # source changed: nothing to hold


def test_a_lost_rod_in_contact_moves_with_the_tips_then_stops():
    h = RodHold(follow=2)
    h({**FORCEPS, **SEEN_ROD, "rod_contact": True})
    out = h(moved(FORCEPS, 5.0))
    assert out["rod_a_px"] == [25.0, 0.0] and out["rod_b_shadow_px"] == [65.0, 30.0] and out["rod_contact"] is True
    h(moved(FORCEPS, 7.0))
    out = h(moved(FORCEPS, 20.0))                    # lost for longer than `follow`: it stays put
    assert out["rod_a_px"] == [27.0, 0.0] and out["rod_contact"] is False
    assert h({**FORCEPS, **moved(SEEN_ROD, 1.0)})["rod_a_px"] == [21.0, 0.0]   # found again: the detection wins


def test_rod_filter_steadies_a_resting_rod_and_ignores_a_jumping_shadow():
    f = RodFilter()
    f({**FORCEPS, **SEEN_ROD})
    out = f({**FORCEPS, **moved(SEEN_ROD, 2.0)})                 # 2 px of jitter counts 2/6 (move_px 6)
    assert abs(out["rod_a_px"][0] - 20.0 - 2.0 / 3) < 1e-9
    for _ in range(5):                                           # jitter back and forth stays near the rest pose
        out = f({**FORCEPS, **moved(SEEN_ROD, 1.0)})
        out = f({**FORCEPS, **moved(SEEN_ROD, -1.0)})
    assert abs(out["rod_a_px"][0] - 20.0) < 0.5
    jumped = {**SEEN_ROD, "rod_a_shadow_px": [20.0, 90.0]}       # shadow end jumps 60 px, the rod rests
    assert f({**FORCEPS, **jumped})["rod_a_shadow_px"] == out["rod_a_shadow_px"]


def test_rod_filter_follows_a_pushed_rod_and_keeps_its_end_order():
    f = RodFilter()
    f({**FORCEPS, **SEEN_ROD})
    assert f({**FORCEPS, **moved(SEEN_ROD, 10.0)})["rod_a_px"] == [30.0, 0.0]   # moved > move_px: no lag
    swapped = {"rod_a_px": [60.0, 0.0], "rod_b_px": [20.0, 0.0], "rod_a_shadow_px": [60.0, 30.0],
               "rod_b_shadow_px": [20.0, 30.0]}
    out = f({**FORCEPS, **swapped})                              # labels swapped by the detector, rod moved back
    assert out["rod_a_px"] == [20.0, 0.0] and out["rod_b_shadow_px"] == [60.0, 30.0]


def test_rod_filter_passes_contact_frames_through():
    f = RodFilter()
    f({**FORCEPS, **SEEN_ROD})
    assert f({**FORCEPS, **moved(SEEN_ROD, 2.0), "rod_contact": True})["rod_a_px"] == [22.0, 0.0]
