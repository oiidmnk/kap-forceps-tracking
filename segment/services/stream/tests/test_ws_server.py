import asyncio
import json
import sys
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ws_server import load_inputs, put_inputs  # noqa: E402


VALID_INPUTS = {
    "light_rot_up": 0.4,
    "light_rot_clock": 4.3,
    "light_depth_mm": 12.0,
    "forceps_rot_up": 0.8,
    "forceps_rot_clock": 1.5,
    "left_tip_px": [133.4, 112.9],
    "left_shadow_px": [209.2, 45.1],
    "right_tip_px": [157.2, 99.5],
    "right_shadow_px": [227.7, 44.9],
    "eye_center_px": [95.0, 95.0],
    "eye_radius_px": 210.0,
    "eye_radius_mm": 24.0,
    "jaw_length_mm": 3.0,
}


class FakeRequest:
    def __init__(self, app, payload):
        self.app = app
        self.payload = payload

    async def json(self):
        return self.payload


def test_put_inputs_accepts_payload_writes_file_and_returns_positions(tmp_path):
    async def exercise():
        input_path = tmp_path / "predicted_input.json"
        input_path.write_text(json.dumps(VALID_INPUTS), encoding="utf-8")
        response = await put_inputs(FakeRequest({"input_path": input_path}, VALID_INPUTS))
        assert response.status == 200
        payload = json.loads(response.text)
        assert payload["inputs"]["left_tip_px"] == [133.4, 112.9]
        assert "left_tip_forceps" in payload["positions"]
        assert load_inputs(input_path)["right_shadow_px"] == [227.7, 44.9]

    asyncio.run(exercise())


def test_put_inputs_rejects_missing_required_fields(tmp_path):
    async def exercise():
        input_path = tmp_path / "predicted_input.json"
        input_path.write_text(json.dumps(VALID_INPUTS), encoding="utf-8")
        invalid = dict(VALID_INPUTS)
        invalid.pop("right_shadow_px")
        try:
            await put_inputs(FakeRequest({"input_path": input_path}, invalid))
        except web.HTTPBadRequest as exc:
            assert "right_shadow_px" in exc.text
        else:
            raise AssertionError("missing required field should fail")

    asyncio.run(exercise())
