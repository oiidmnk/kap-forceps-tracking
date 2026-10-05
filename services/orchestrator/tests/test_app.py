import json
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from orchestrator.app import (
    OrchestratorError,
    create_app,
    decode_video_frame,
    select_preprocessor_points,
)


CALIBRATION = {
    "light_rot_up": 0.4,
    "light_rot_clock": 4.3,
    "light_depth_mm": 12.0,
    "forceps_rot_up": 0.8,
    "forceps_rot_clock": 1.5,
    "eye_center_px": [95.0, 95.0],
    "eye_radius_px": 210.0,
    "eye_radius_mm": 24.0,
    "jaw_length_mm": 3.0,
}


SEGMENTATION = {
    "instances": [
        {
            "class_name": "tip_left",
            "confidence": 0.2,
            "box": {"xyxy": [0, 0, 2, 2]},
            "segments": [],
        },
        {
            "class_name": "tip_left",
            "confidence": 0.9,
            "box": {"xyxy": [0, 0, 10, 10]},
            "segments": [{"points": [[2, 2], [6, 2], [6, 6], [2, 6]]}],
        },
        {
            "class_name": "tip_right",
            "confidence": 0.8,
            "box": {"xyxy": [10, 20, 14, 28]},
            "segments": [],
        },
        {
            "class_name": "shadow_left",
            "confidence": 0.8,
            "box": {"xyxy": [20, 20, 22, 22]},
            "segments": [{"points": [[20, 20], [24, 20], [24, 24], [20, 24]]}],
        },
        {
            "class_name": "shadow_right",
            "confidence": 0.8,
            "box": {"xyxy": [30, 30, 32, 32]},
            "segments": [{"points": [[30, 30], [34, 30], [34, 34], [30, 34]]}],
        },
    ]
}


def test_select_preprocessor_points_uses_top_confidence_polygon_and_box_fallback():
    points = select_preprocessor_points(SEGMENTATION)

    assert points["left_tip_px"] == [4.0, 4.0]
    assert points["right_tip_px"] == [12.0, 24.0]
    assert points["left_shadow_px"] == [22.0, 22.0]
    assert points["right_shadow_px"] == [32.0, 32.0]


def test_select_preprocessor_points_falls_back_to_class_ids():
    payload = {
        "instances": [
            {
                "class_id": 0,
                "class_name": "0",
                "confidence": 0.9,
                "box": {"xyxy": [0, 0, 2, 2]},
                "segments": [],
            },
            {
                "class_id": 1,
                "class_name": "1",
                "confidence": 0.9,
                "box": {"xyxy": [2, 2, 4, 4]},
                "segments": [],
            },
            {
                "class_id": 2,
                "class_name": "2",
                "confidence": 0.9,
                "box": {"xyxy": [4, 4, 6, 6]},
                "segments": [],
            },
            {
                "class_id": 3,
                "class_name": "3",
                "confidence": 0.9,
                "box": {"xyxy": [6, 6, 8, 8]},
                "segments": [],
            },
        ]
    }

    points = select_preprocessor_points(payload)

    assert points == {
        "left_tip_px": [1.0, 1.0],
        "right_tip_px": [3.0, 3.0],
        "left_shadow_px": [5.0, 5.0],
        "right_shadow_px": [7.0, 7.0],
    }


def test_select_preprocessor_points_rejects_missing_required_class():
    payload = {"instances": [item for item in SEGMENTATION["instances"] if item["class_name"] != "shadow_right"]}

    try:
        select_preprocessor_points(payload)
    except OrchestratorError as exc:
        assert "shadow_right" in str(exc)
        assert "Detected classes were" in str(exc)
    else:
        raise AssertionError("missing required class should fail")


def test_select_preprocessor_points_explains_empty_detections():
    try:
        select_preprocessor_points({"instances": []})
    except OrchestratorError as exc:
        assert "returned no detections" in str(exc)
        assert "SEGMENTATION_CONF" in str(exc)
    else:
        raise AssertionError("empty detections should fail")


def test_process_upload_calls_segmentation_and_stream(tmp_path):
    default_calibration = tmp_path / "default_calibration.json"
    calibration_path = tmp_path / "calibration.json"
    default_calibration.write_text(json.dumps(CALIBRATION), encoding="utf-8")
    stream_requests = []

    async def segmentation_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/segment"
        return httpx.Response(200, json=SEGMENTATION)

    async def stream_handler(request: httpx.Request) -> httpx.Response:
        stream_requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"positions": {"left_tip_forceps": [1, 2, 3]}})

    app = create_app(
        calibration_path=calibration_path,
        default_calibration_source=default_calibration,
        segmentation_url="http://segmentation.test",
        stream_url="http://stream.test",
        segmentation_transport=httpx.MockTransport(segmentation_handler),
        stream_transport=httpx.MockTransport(stream_handler),
    )
    client = TestClient(app)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    response = client.post(
        "/api/process",
        files={"file": ("frame.png", encoded.tobytes(), "image/png")},
        data={"frame_time_s": "0"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_points"]["left_tip_px"] == [4.0, 4.0]
    assert stream_requests[0]["eye_radius_mm"] == 24.0
    assert stream_requests[0]["right_tip_px"] == [12.0, 24.0]


def test_process_upload_explains_missing_segmentation_weights(tmp_path):
    default_calibration = tmp_path / "default_calibration.json"
    default_calibration.write_text(json.dumps(CALIBRATION), encoding="utf-8")

    async def segmentation_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"detail": "segmentation weights not found: runs/segment/forceps/weights/best.pt"},
        )

    app = create_app(
        calibration_path=tmp_path / "calibration.json",
        default_calibration_source=default_calibration,
        segmentation_url="http://segmentation.test",
        stream_url="http://stream.test",
        segmentation_transport=httpx.MockTransport(segmentation_handler),
    )
    client = TestClient(app)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    response = client.post(
        "/api/process",
        files={"file": ("frame.png", encoded.tobytes(), "image/png")},
        data={"frame_time_s": "0"},
    )

    assert response.status_code == 502
    assert "Put trained weights" in response.json()["detail"]


def test_apply_calibration_merges_onto_current_stream_inputs(tmp_path):
    default_calibration = tmp_path / "default_calibration.json"
    calibration_path = tmp_path / "calibration.json"
    default_calibration.write_text(json.dumps(CALIBRATION), encoding="utf-8")
    put_payloads = []

    current_inputs = {
        **CALIBRATION,
        "forceps_rot_up": 0.1,  # stale angle that the calibration must override
        "left_tip_px": [133.4, 112.9],
        "left_shadow_px": [209.2, 45.1],
        "right_tip_px": [157.2, 99.5],
        "right_shadow_px": [227.7, 44.9],
    }

    async def stream_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/inputs"
        if request.method == "GET":
            return httpx.Response(200, json={"inputs": current_inputs, "positions": {}})
        put_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"positions": {"left_tip_forceps": [1, 2, 3]}})

    client = TestClient(
        create_app(
            calibration_path=calibration_path,
            default_calibration_source=default_calibration,
            segmentation_url="http://segmentation.test",
            stream_url="http://stream.test",
            stream_transport=httpx.MockTransport(stream_handler),
        )
    )

    client.put("/api/calibration", json=dict(CALIBRATION, forceps_rot_up=1.23))
    response = client.post("/api/apply-calibration")

    assert response.status_code == 200
    # Calibration wins for the trocar angle; the last detections are preserved.
    assert put_payloads[-1]["forceps_rot_up"] == 1.23
    assert put_payloads[-1]["left_tip_px"] == [133.4, 112.9]


def test_manual_points_bypasses_segmentation_and_merges_calibration(tmp_path):
    default_calibration = tmp_path / "default_calibration.json"
    calibration_path = tmp_path / "calibration.json"
    default_calibration.write_text(json.dumps(CALIBRATION), encoding="utf-8")
    put_payloads = []

    async def stream_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        assert request.url.path == "/inputs"
        put_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"positions": {"left_tip_forceps": [1, 2, 3]}})

    client = TestClient(
        create_app(
            calibration_path=calibration_path,
            default_calibration_source=default_calibration,
            segmentation_url="http://segmentation.test",
            stream_url="http://stream.test",
            stream_transport=httpx.MockTransport(stream_handler),
        )
    )

    manual_points = {
        "left_tip_px": [133.4, 112.9],
        "right_tip_px": [157.2, 99.5],
        "left_shadow_px": [209.2, 45.1],
        "right_shadow_px": [227.7, 44.9],
    }
    response = client.post("/api/manual-points", json=manual_points)

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_points"] == manual_points
    assert put_payloads[-1]["left_tip_px"] == [133.4, 112.9]
    assert put_payloads[-1]["eye_radius_mm"] == 24.0  # calibration merged in


def test_manual_points_rejects_missing_keys(tmp_path):
    default_calibration = tmp_path / "default_calibration.json"
    default_calibration.write_text(json.dumps(CALIBRATION), encoding="utf-8")
    client = TestClient(
        create_app(
            calibration_path=tmp_path / "calibration.json",
            default_calibration_source=default_calibration,
            segmentation_url="http://segmentation.test",
            stream_url="http://stream.test",
        )
    )

    incomplete = {"left_tip_px": [1.0, 2.0], "right_tip_px": [3.0, 4.0]}
    response = client.post("/api/manual-points", json=incomplete)

    assert response.status_code == 400
    assert "left_shadow_px" in response.json()["detail"]
    assert "right_shadow_px" in response.json()["detail"]


def test_calibration_update_persists(tmp_path):
    default_calibration = tmp_path / "default_calibration.json"
    calibration_path = tmp_path / "calibration.json"
    default_calibration.write_text(json.dumps(CALIBRATION), encoding="utf-8")
    client = TestClient(
        create_app(
            calibration_path=calibration_path,
            default_calibration_source=default_calibration,
            segmentation_url="http://segmentation.test",
            stream_url="http://stream.test",
        )
    )

    updated = dict(CALIBRATION, jaw_length_mm=4.5)
    response = client.put("/api/calibration", json=updated)

    assert response.status_code == 200
    assert json.loads(calibration_path.read_text(encoding="utf-8"))["jaw_length_mm"] == 4.5


def test_index_renders_template_with_static_assets(tmp_path):
    default_calibration = tmp_path / "default_calibration.json"
    default_calibration.write_text(json.dumps(CALIBRATION), encoding="utf-8")
    client = TestClient(
        create_app(
            calibration_path=tmp_path / "calibration.json",
            default_calibration_source=default_calibration,
            segmentation_url="http://segmentation.test",
            stream_url="http://stream.test",
        )
    )

    response = client.get("/")

    assert response.status_code == 200
    assert "Forceps Orchestrator" in response.text
    assert "/static/app.css" in response.text
    assert "/static/app.js" in response.text


def test_status_reports_missing_weights(tmp_path):
    default_calibration = tmp_path / "default_calibration.json"
    default_calibration.write_text(json.dumps(CALIBRATION), encoding="utf-8")

    async def segmentation_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "weights": "runs/segment/forceps/weights/best.pt",
                "weights_available": False,
            },
        )

    async def stream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "inputs_valid": True})

    client = TestClient(
        create_app(
            calibration_path=tmp_path / "calibration.json",
            default_calibration_source=default_calibration,
            segmentation_url="http://segmentation.test",
            stream_url="http://stream.test",
            segmentation_transport=httpx.MockTransport(segmentation_handler),
            stream_transport=httpx.MockTransport(stream_handler),
        )
    )

    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert "Segmentation weights are not available" in response.json()["message"]


def test_decode_video_frame_reads_selected_frame(tmp_path):
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (12, 12),
    )
    if not writer.isOpened():
        return
    writer.write(np.full((12, 12, 3), 20, dtype=np.uint8))
    writer.write(np.full((12, 12, 3), 200, dtype=np.uint8))
    writer.release()

    frame = decode_video_frame(video_path.read_bytes(), 0.1)

    assert frame.shape[:2] == (12, 12)
    assert frame.mean() > 100
