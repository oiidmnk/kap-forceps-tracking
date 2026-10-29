"""Needs a trained checkpoint (MLSEG_CKPT or weights/last.pt) and the sample frames; skipped otherwise."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[3])
CKPT = Path(os.environ.get("MLSEG_CKPT") or Path(__file__).resolve().parents[1] / "weights/last.pt")
FRAMES = sorted((ROOT / "material").glob("frame_*.png"))
pytestmark = pytest.mark.skipif(not (CKPT.exists() and FRAMES), reason="checkpoint or sample frames missing")

KEYS = [("forceps", "left_tip"), ("forceps", "right_tip"), ("shadow", "left_tip"), ("shadow", "right_tip")]
LABELS = ["left_tip", "right_tip", "left_shadow", "right_shadow"]


@pytest.fixture(scope="module")
def client():
    from mlseg.service import create_app
    with TestClient(create_app(str(CKPT))) as c:
        yield c


def labelled(png: Path):
    d = json.loads(png.with_suffix(".json").read_text())
    return {s["label"]: s["points"][0] for s in d["shapes"]}


def test_detect_matches_labels(client):
    png = FRAMES[3]
    r = client.post("/detect", files={"image": ("f.png", png.read_bytes(), "image/png")}).json()
    gt = labelled(png)
    for (tool, key), lab in zip(KEYS, LABELS):
        p = r[tool][key]
        assert p is not None
        assert np.hypot(p[0] - gt[lab][0], p[1] - gt[lab][1]) < 8       # labels are not pixel-perfect


def test_junction_is_between_tips_and_shaft(client):
    r = client.post("/detect", files={"image": ("f.png", FRAMES[3].read_bytes(), "image/png")}).json()["forceps"]
    if r["junction"] is None:
        pytest.skip("checkpoint has no junction channel")
    tip = np.mean([r["left_tip"], r["right_tip"]], 0)
    assert 20 < np.hypot(*(np.array(r["junction"]) - tip)) < 250


def test_segment_is_orchestrator_compatible(client):
    r = client.post("/segment", files={"image": ("f.png", FRAMES[0].read_bytes(), "image/png")}).json()
    assert {i["class_name"] for i in r["instances"]} == {"tip_left", "tip_right", "shadow_left", "shadow_right"}


def test_annotated_without_junction(client):
    r = client.post("/detect/annotated", files={"image": ("f.png", FRAMES[0].read_bytes(), "image/png")})
    assert r.status_code == 200 and cv2.imdecode(np.frombuffer(r.content, np.uint8), 1).shape[0] == 1080


def test_empty_retina_reports_nothing(client):
    bg = ROOT / "data/background.png"
    if not bg.exists():
        pytest.skip("no empty background image")
    d = client.post("/detect", files={"image": ("b.png", bg.read_bytes(), "image/png")}).json()
    assert not d["forceps"]["ok"] and not d["shadow"]["ok"]
