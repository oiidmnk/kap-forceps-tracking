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


def test_viewer_and_websocket_routes_coexist(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from ws_server import build_app

    async def exercise():
        input_path = tmp_path / "inputs.json"
        input_path.write_text(json.dumps(VALID_INPUTS), encoding="utf-8")
        async with TestClient(TestServer(build_app(input_path, 0.01))) as client:
            response = await client.get('/viewer')
            assert response.status == 200
            assert response.content_type == 'text/html'
            assert 'Position stream' in await response.text()
            paths = {route.resource.canonical for route in client.server.app.router.routes()}
            assert {'/', '/ws', '/viewer'} <= paths

    asyncio.run(exercise())


def test_rod_is_reconstructed_from_its_shadow_like_a_tip():
    from ws_server import resilient_positions
    from tool_positions import _norm3, _sub3

    data = dict(VALID_INPUTS)
    # the rod's ends sit exactly where the forceps tips are, with the same shadows -> same 3D points
    data.update(rod_a_px=data["left_tip_px"], rod_a_shadow_px=data["left_shadow_px"],
                rod_b_px=data["right_tip_px"], rod_b_shadow_px=data["right_shadow_px"])
    positions, error = resilient_positions(data)
    assert error is None
    assert _norm3(_sub3(positions["rod_a"], positions["left_tip_forceps"])) < 1e-9
    assert _norm3(_sub3(positions["rod_b"], positions["right_tip_forceps"])) < 1e-9


def test_rod_is_optional_and_never_breaks_the_forceps():
    from ws_server import resilient_positions

    positions, error = resilient_positions(dict(VALID_INPUTS))
    assert error is None and positions["rod_a"] is None and positions["rod_b"] is None
    bad = dict(VALID_INPUTS, rod_a_px=[10, 10], rod_a_shadow_px=[10, 10], rod_b_px=[1, 1], rod_b_shadow_px="x")
    positions, error = resilient_positions(bad)
    assert positions["left_tip_forceps"] is not None and positions["rod_a"] is None and "rod" in error


def test_rod_contact_moves_the_rod_onto_the_forceps():
    from ws_server import resilient_positions
    from tool_positions import _norm3, _sub3

    data = dict(VALID_INPUTS, rod_a_px=[120.0, 140.0], rod_a_shadow_px=[200.0, 80.0],
                rod_b_px=[170.0, 140.0], rod_b_shadow_px=[240.0, 80.0])
    apart, _ = resilient_positions(dict(data))
    touching, error = resilient_positions(dict(data, rod_contact=True))
    assert error is None and apart["rod_contact"] is False and touching["rod_contact"] is True

    def to_axis(pos):
        a, b = pos["rod_a"], pos["rod_b"]
        mid = [(l + r) / 2 for l, r in zip(pos["left_tip_forceps"], pos["right_tip_forceps"])]
        v = _sub3(b, a)
        t = min(1.0, max(0.0, sum(x * y for x, y in zip(_sub3(mid, a), v)) / sum(x * x for x in v)))
        return _norm3(_sub3(mid, [p + t * c for p, c in zip(a, v)]))

    assert to_axis(apart) > 1e-3 and to_axis(touching) < 1e-9
    # moved as a whole: same length, same direction
    assert _norm3(_sub3(_sub3(touching["rod_b"], touching["rod_a"]), _sub3(apart["rod_b"], apart["rod_a"]))) < 1e-9


def test_rod_contact_without_forceps_changes_nothing():
    from ws_server import rod_positions, resilient_positions

    data = dict(VALID_INPUTS, rod_a_px=[120.0, 140.0], rod_a_shadow_px=[200.0, 80.0],
                rod_b_px=[170.0, 140.0], rod_b_shadow_px=[240.0, 80.0], rod_contact=True,
                left_tip_px=[1000.0, 1000.0])          # tips outside the eye -> no forceps
    positions, _ = resilient_positions(data)
    assert positions["left_tip_forceps"] is None and positions["rod_contact"] is False
    assert positions["rod_a"] == rod_positions(data)["rod_a"]


def _project(p, c, center, rpx):
    """Pinhole camera at (0, 0, -c): 3D point (eye radii) -> pixel."""
    k = (1 + c) / (p[2] + c)
    return (center[0] + rpx * k * p[0], center[1] + rpx * k * p[1])


def test_pinhole_camera_reconstructs_a_point_from_its_image_and_shadow():
    import math
    from tool_positions import light_tip_position, reconstruct_tip_from_shadow, _add3, _scale3, _sub3, _norm3

    c, center, rpx = 2.2, (540.0, 540.0), 600.0
    light = light_tip_position(0.6, -0.5, 0.005)
    tip = (0.15, -0.1, 0.55)
    # its shadow: continue the light ray through the tip to the far retina
    d = _sub3(tip, light)
    a, b, cc = sum(x * x for x in d), 2 * sum(x * y for x, y in zip(light, d)), sum(x * x for x in light) - 1
    shadow = _add3(light, _scale3(d, (-b + math.sqrt(b * b - 4 * a * cc)) / (2 * a)))
    got = reconstruct_tip_from_shadow(light, _project(tip, c, center, rpx), _project(shadow, c, center, rpx), center, rpx, c)
    assert _norm3(_sub3(got, tip)) < 1e-9


def test_camera_distance_defaults_to_the_orthographic_camera():
    from ws_server import resilient_positions
    from tool_positions import _norm3, _sub3

    data = dict(VALID_INPUTS, rod_a_px=[120.0, 140.0], rod_a_shadow_px=[200.0, 80.0],
                rod_b_px=[170.0, 140.0], rod_b_shadow_px=[240.0, 80.0])
    ortho, _ = resilient_positions(dict(data))
    assert resilient_positions(dict(data, camera_distance=0.0))[0] == ortho
    far, _ = resilient_positions(dict(data, camera_distance=1e6))     # a very distant pinhole is orthographic
    for key in ("left_tip_forceps", "right_tip_forceps", "rod_a", "rod_b"):
        assert _norm3(_sub3(far[key], ortho[key])) < 1e-4
