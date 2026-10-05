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
    client = TestClient(create_app(Settings(background_path=str(BG), source="")))
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
