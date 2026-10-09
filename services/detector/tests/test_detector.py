import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from detector.background import Background
from detector.config import Settings
from detector.geometry import analyze
from detector.pipeline import Detector
from detector.service import create_app

ROOT = Path(__file__).resolve().parents[3]
FRAME = ROOT / "data/annotated/frame_000042.png"
BG = ROOT / "data/background.png"
needs_data = pytest.mark.skipif(not (FRAME.exists() and BG.exists()), reason="sample data missing")


def synthetic_fork():
    """A tool entering from the right with two jaws opening to the left (tip at ~ (300, 300))."""
    m = np.zeros((1080, 1080), np.uint8)
    cv2.fillConvexPoly(m, np.array([[420, 296], [420, 312], [900, 340], [900, 280]], np.int32), 1)   # shaft
    cv2.fillConvexPoly(m, np.array([[300, 280], [420, 296], [420, 300]], np.int32), 1)             # upper jaw
    cv2.fillConvexPoly(m, np.array([[300, 330], [420, 308], [420, 312]], np.int32), 1)             # lower jaw
    return m


def test_geometry_synthetic_fork():
    j = analyze(synthetic_fork())
    assert j.ok, j.reason
    assert abs(j.junction[0] - 420) < 6
    # image-y larger = the tool's "left" when heading towards -x
    assert j.left_tip[1] > j.right_tip[1]
    assert abs(j.left_tip[0] - 300) < 8 and abs(j.right_tip[0] - 300) < 8


def test_geometry_empty_mask():
    assert not analyze(np.zeros((1080, 1080), np.uint8)).ok


@needs_data
def test_detects_annotated_frame():
    gt = {s["label"]: np.array(s["points"][0]) for s in json.loads(FRAME.with_suffix(".json").read_text())["shapes"]}
    d = Detector(Settings(background_path=str(BG)))
    det = d.detect(cv2.imread(str(FRAME)))
    assert det.forceps.ok and det.shadow.ok
    for got, lab in [(det.forceps.left_tip, "left_tip"), (det.forceps.right_tip, "right_tip"),
                     (det.shadow.left_tip, "left_shadow"), (det.shadow.right_tip, "right_shadow")]:
        assert np.linalg.norm(np.array(got) - gt[lab]) < 15
    assert set(det.orchestrator_points()) == {"left_tip_px", "right_tip_px", "left_shadow_px", "right_shadow_px"}


@needs_data
def test_non_square_frame_maps_back_to_original_pixels():
    img = cv2.imread(str(FRAME))
    wide = cv2.copyMakeBorder(img, 0, 0, 200, 200, cv2.BORDER_CONSTANT)      # 1480 x 1080
    d = Detector(Settings(background_path=str(BG)))
    a, b = d.detect(img), d.detect(wide)
    assert np.allclose(np.array(a.forceps.left_tip) + [200, 0], b.forceps.left_tip, atol=3)


@needs_data
def test_segment_endpoint_matches_orchestrator_contract():
    client = TestClient(create_app(Settings(background_path=str(BG), source="", state_path="")))
    r = client.post("/segment", files={"image": ("f.png", FRAME.read_bytes(), "image/png")})
    assert r.status_code == 200
    names = {i["class_name"] for i in r.json()["instances"]}
    assert names == {"tip_left", "tip_right", "shadow_left", "shadow_right"}
    assert client.get("/health").json()["weights_available"] is True


def test_background_learning_from_samples():
    bg = Background(None, 64)
    frame = np.full((64, 64, 3), (40, 60, 140), np.uint8)
    for i in range(29):
        assert not bg.add_sample(frame)
    assert bg.add_sample(frame) and bg.state == "static"


def test_detect_without_background_does_not_crash():
    d = Detector(Settings(background_path=""))
    out = d.detect(np.zeros((1080, 1920, 3), np.uint8))
    assert not out.forceps.ok and not out.shadow.ok


def test_eye_disc_found_in_small_retina_circle():
    from detector.background import find_eye_disc
    img = np.zeros((1080, 1080, 3), np.uint8)
    cv2.circle(img, (600, 500), 300, (40, 70, 180), -1)
    cx, cy, r = find_eye_disc(img)
    assert abs(cx - 600) < 12 and abs(cy - 500) < 12 and abs(r - 300) < 12


def _frame_with_retina(w=1920, h=1080, cx=975, cy=555, r=340):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (cx, cy), r, (40, 70, 180), -1)
    cv2.circle(img, (cx - r - 40, cy), 60, (200, 200, 200), -1)        # something outside the circle
    return img


def test_normalize_is_square_black_outside_circle():
    d = Detector(Settings(background_path=""))
    d.roi = (975.0, 555.0, 340.0)
    out = d.normalize(_frame_with_retina())
    assert out.shape == (1080, 1080, 3)
    assert out[540, 540].tolist() != [0, 0, 0]                 # circle content kept
    assert out[5, 5].tolist() == [0, 0, 0] and out[1075, 1075].tolist() == [0, 0, 0]   # corners black
    assert out.max(axis=2)[540, :40].max() > 0                 # circle touches the left edge, content there


def test_normalize_handles_circle_partly_outside_frame():
    d = Detector(Settings(background_path=""))
    d.roi = (100.0, 100.0, 300.0)
    assert d.normalize(_frame_with_retina()).shape == (1080, 1080, 3)


def test_runtime_config_roundtrip(tmp_path):
    from detector.runtime import RuntimeConfig
    p = tmp_path / "c.json"
    rt = RuntimeConfig("a.mp4", str(p))
    rt.update(source="0", circle={"cx": 10, "cy": 20, "r": 50})
    again = RuntimeConfig("zzz", str(p))
    assert again.source == "0" and again.circle == (10.0, 20.0, 50.0)
    with pytest.raises(ValueError):
        rt.update(circle={"cx": 1, "cy": 1, "r": 5})


def test_config_endpoints(tmp_path):
    client = TestClient(create_app(Settings(background_path="", source="", state_path=str(tmp_path / "c.json"))))
    with client:
        assert client.get("/config").json()["circle"] is None
        r = client.post("/config", json={"circle": {"cx": 500, "cy": 400, "r": 300}, "source": "/data/video/x.mp4"})
        assert r.status_code == 200 and r.json()["circle"]["r"] == 300 and r.json()["normalized_size"] == 1080
        assert client.post("/config", json={"circle": {"cx": 1, "cy": 1, "r": 2}}).status_code == 400
        assert client.post("/config", json={"clear_circle": True}).json()["circle"] is None
        assert "files" in client.get("/sources").json()


def test_reset_endpoint_bumps_reset_counter(tmp_path):
    app = create_app(Settings(background_path="", source="", state_path=str(tmp_path / "c.json")))
    with TestClient(app) as client:
        before = app.state.runtime.reset_count
        assert client.post("/reset").status_code == 200
        assert app.state.runtime.reset_count == before + 1


def test_hard_reset_discards_background(tmp_path):
    from detector.live import LiveWorker
    from detector.runtime import RuntimeConfig
    cfg = Settings(background_path=str(BG) if BG.exists() else "", source="x", state_path="")
    det = Detector(cfg)
    rt = RuntimeConfig("x", None)
    w = LiveWorker(det, cfg, rt)
    w._apply(*rt.snapshot())                       # first apply keeps the loaded background
    if BG.exists():
        assert det.bg.state == "static"
    rt.restart()
    w._apply(*rt.snapshot())
    assert det.bg.state == "none" and det.bg.v is None
