import json

import httpx
from fastapi.testclient import TestClient

from orchestrator.app import create_app

CAL = {"light_rot_up": 0.4, "light_rot_clock": 4.3, "light_depth_mm": 12.0, "forceps_rot_up": 0.9,
       "forceps_rot_clock": 1.5, "eye_center_px": [95.0, 95.0], "eye_radius_px": 30.0, "eye_radius_mm": 24.0,
       "jaw_length_mm": 1.5}


def make_client(tmp_path):
    calls = []
    state = {"config": {"source": "/data/video/a.mp4", "circle": None, "normalized_size": 1080,
                        "frame_size": [1920, 1080], "background": "static"}}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path == "/config" and request.method == "GET":
            return httpx.Response(200, json=state["config"])
        if path == "/config":
            body = json.loads(request.content)
            if body.get("circle") and body["circle"]["r"] < 20:
                return httpx.Response(400, json={"detail": "circle radius must be >= 20 px"})
            if body.get("circle"):
                state["config"]["circle"] = body["circle"]
            return httpx.Response(200, json=state["config"])
        if path == "/reset":
            return httpx.Response(200, json=state["config"])
        if path == "/sources":
            return httpx.Response(200, json={"current": "x", "files": ["/data/video/a.mp4"],
                                             "cameras": [{"index": 0, "width": 1920, "height": 1080}]})
        if path == "/preview.jpg":
            return httpx.Response(200, content=b"\xff\xd8jpeg", headers={"X-Frame-Width": "1920", "X-Frame-Height": "1080"})
        if path == "/autocircle":
            return httpx.Response(200, json={"cx": 975, "cy": 555, "r": 340})
        if path == "/inputs":
            return httpx.Response(200, json={"inputs": {**CAL, "left_tip_px": [1, 2], "right_tip_px": [3, 4],
                                                        "left_shadow_px": [5, 6], "right_shadow_px": [7, 8]}})
        return httpx.Response(404, json={"detail": "nope"})

    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(CAL))
    app = create_app(calibration_path=cal, default_calibration_source=None,
                     segmentation_url="http://detector:8000", stream_url="http://stream:8765",
                     segmentation_transport=httpx.MockTransport(handler),
                     stream_transport=httpx.MockTransport(handler),
                     detector_state_path=tmp_path / "detector.json")
    return TestClient(app), calls, cal


def test_source_overview_and_proxy(tmp_path):
    client, calls, _ = make_client(tmp_path)
    data = client.get("/api/source").json()
    assert data["detector_url"] == "http://detector:8000"
    assert data["sources"]["files"] == ["/data/video/a.mp4"] and data["config"]["source"].endswith("a.mp4")
    r = client.get("/api/source/preview.jpg")
    assert r.status_code == 200 and r.headers["x-frame-width"] == "1920"
    assert client.get("/api/source/autocircle").json()["r"] == 340


def test_circle_rebases_eye_calibration(tmp_path):
    client, calls, cal = make_client(tmp_path)
    res = client.post("/api/source/config", json={"circle": {"cx": 975, "cy": 555, "r": 340}})
    assert res.status_code == 200
    saved = json.loads(cal.read_text())
    assert saved["eye_center_px"] == [540.0, 540.0] and saved["eye_radius_px"] == 540.0
    assert ("PUT", "/inputs") in calls            # pushed onto the stream


def test_source_switch_does_not_touch_calibration(tmp_path):
    client, _, cal = make_client(tmp_path)
    client.post("/api/source/config", json={"source": "0"})
    assert json.loads(cal.read_text())["eye_radius_px"] == 30.0


def test_detector_errors_are_forwarded(tmp_path):
    client, _, _ = make_client(tmp_path)
    r = client.post("/api/source/config", json={"circle": {"cx": 1, "cy": 1, "r": 5}})
    assert r.status_code == 400 and "radius" in r.json()["detail"]


def test_detector_url_is_persisted_and_validated(tmp_path):
    client, _, _ = make_client(tmp_path)
    assert client.put("/api/source/detector-url", json={"url": "ftp://x"}).status_code == 400
    r = client.put("/api/source/detector-url", json={"url": "http://host.docker.internal:8001/"})
    assert r.json()["detector_url"] == "http://host.docker.internal:8001"
    assert client.get("/api/source").json()["detector_url"] == "http://host.docker.internal:8001"
    assert json.loads((tmp_path / "detector.json").read_text())["url"] == "http://host.docker.internal:8001"


def test_reset_is_proxied(tmp_path):
    client, calls, _ = make_client(tmp_path)
    assert client.post("/api/source/reset").status_code == 200
    assert ("POST", "/reset") in calls
