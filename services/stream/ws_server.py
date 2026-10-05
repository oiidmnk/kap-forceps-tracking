"""Stream tool-position calculations over WebSocket, with a live inputs API.

A single aiohttp app on one port serves both halves of the pipeline:

- ``GET /ws``     WebSocket feed: re-reads the inputs file every frame and
                  streams the reconstructed tool positions (consumed by the
                  viz, proxied through nginx at ``/ws``).
- ``PUT /inputs`` replace the current inputs with a full JSON payload
                  (validated), write them to disk, and return the freshly
                  computed positions. This is how the orchestrator pushes new
                  detections / calibration into the live stream.
- ``GET /inputs`` read back the current inputs and their positions (used by
                  the orchestrator to merge calibration onto the last frame).

Inputs are re-read from a JSON file on every frame so an external process can
update the file live (either by writing the file directly or via ``PUT
/inputs``) while the WebSocket feed keeps streaming.

Usage:
    pip install aiohttp
    python ws_server.py
    python ws_server.py --input path/to/inputs.json --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from aiohttp import web

from tool_positions import (
    compute_tool_positions,
    light_tip_position,
    trocar_position,
)

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

DEFAULT_INPUT = Path(__file__).with_name("input_example.json")

REQUIRED_KEYS = [
    "light_rot_up",
    "light_rot_clock",
    "light_depth_mm",
    "forceps_rot_up",
    "forceps_rot_clock",
    "left_tip_px",
    "left_shadow_px",
    "right_tip_px",
    "right_shadow_px",
    "eye_center_px",
    "eye_radius_px",
    "eye_radius_mm",
    "jaw_length_mm",
]


def validate_inputs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a raw inputs dict and fill optional defaults.

    ``light_aim_tilt``/``light_aim_clock`` are optional and default to ``0.0``
    (aiming straight at the eye center) so existing input files and payloads
    keep working unchanged.
    """
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"input file missing keys: {', '.join(missing)}")

    data.setdefault("light_aim_tilt", 0.0)
    data.setdefault("light_aim_clock", 0.0)
    return data


def load_inputs(path: Path) -> Dict[str, Any]:
    """Load and validate the input JSON file."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return validate_inputs(data)


def write_inputs(path: Path, data: Dict[str, Any]) -> None:
    """Persist inputs to disk so the WebSocket loop picks them up next frame."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _as_vec2(value: Any) -> Vec2:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"expected [x, y], got {value!r}")
    return (float(value[0]), float(value[1]))


def compute_positions(data: Dict[str, Any]) -> Dict[str, Any]:
    """Run the reconstruction on a validated inputs dict."""
    positions = compute_tool_positions(
        light_rot_up=float(data["light_rot_up"]),
        light_rot_clock=float(data["light_rot_clock"]),
        light_depth_mm=float(data["light_depth_mm"]),
        light_aim_tilt=float(data.get("light_aim_tilt", 0.0)),
        light_aim_clock=float(data.get("light_aim_clock", 0.0)),
        forceps_rot_up=float(data["forceps_rot_up"]),
        forceps_rot_clock=float(data["forceps_rot_clock"]),
        left_tip_px=_as_vec2(data["left_tip_px"]),
        left_shadow_px=_as_vec2(data["left_shadow_px"]),
        right_tip_px=_as_vec2(data["right_tip_px"]),
        right_shadow_px=_as_vec2(data["right_shadow_px"]),
        eye_center_px=_as_vec2(data["eye_center_px"]),
        eye_radius_px=float(data["eye_radius_px"]),
        eye_radius_mm=float(data["eye_radius_mm"]),
        jaw_length_mm=float(data["jaw_length_mm"]),
    )
    return {key: [value[0], value[1], value[2]] for key, value in positions.items()}


def anchor_positions(data: Dict[str, Any]) -> Dict[str, Any]:
    """Points that depend only on the calibration angles, not on the tips.

    The trocars and the light tip are fixed by the calibration alone, so they
    can always be placed even when the tip/shadow reconstruction is
    inconsistent (e.g. a freshly-clicked eye circle that doesn't match stale
    tip pixels). The forceps tip points are reported as ``None`` in that case.
    """
    r_mm = float(data["eye_radius_mm"])
    light_depth = float(data["light_depth_mm"]) / r_mm if r_mm else 0.0
    trocar_light = trocar_position(float(data["light_rot_up"]), float(data["light_rot_clock"]))
    tip_light = light_tip_position(
        float(data["light_rot_up"]),
        float(data["light_rot_clock"]),
        light_depth,
        float(data.get("light_aim_tilt", 0.0)),
        float(data.get("light_aim_clock", 0.0)),
    )
    trocar_forceps = trocar_position(
        float(data["forceps_rot_up"]), float(data["forceps_rot_clock"])
    )
    return {
        "trocar_light": list(trocar_light),
        "tip_light": list(tip_light),
        "trocar_forceps": list(trocar_forceps),
        "left_tip_forceps": None,
        "right_tip_forceps": None,
        "jaw_meet_forceps": None,
    }


def resilient_positions(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    """Compute positions without raising on inconsistent tip geometry.

    Returns ``(positions, error)``. On the happy path ``error`` is ``None`` and
    all six points are present. If the tip reconstruction is inconsistent, the
    trocars/light are still returned (tips ``None``) alongside the error text,
    so the stream degrades gracefully instead of returning a 500.
    """
    try:
        return compute_positions(data), None
    except Exception as exc:  # noqa: BLE001 - degrade to anchors, report the reason
        try:
            return anchor_positions(data), str(exc)
        except Exception as anchor_exc:  # noqa: BLE001
            return {}, f"{exc}; anchors failed: {anchor_exc}"


def compute_from_file(path: Path) -> Tuple[Dict[str, Any], Any]:
    """Read inputs from disk and run the (resilient) reconstruction."""
    return resilient_positions(load_inputs(path))


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

async def put_inputs(request: web.Request) -> web.Response:
    """Replace the current inputs with a full JSON payload and recompute."""
    input_path: Path = request.app["input_path"]
    payload = await request.json()
    try:
        validated = validate_inputs(dict(payload))
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))

    write_inputs(input_path, validated)
    positions, error = resilient_positions(validated)
    return web.json_response(
        {"inputs": validated, "positions": positions, "error": error}
    )


async def get_inputs(request: web.Request) -> web.Response:
    """Return the current inputs and their computed positions."""
    input_path: Path = request.app["input_path"]
    try:
        data = load_inputs(input_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise web.HTTPServiceUnavailable(text=str(exc))
    positions, error = resilient_positions(data)
    return web.json_response({"inputs": data, "positions": positions, "error": error})


async def health(request: web.Request) -> web.Response:
    """Report service liveness and whether the current inputs file is valid."""
    input_path: Path = request.app["input_path"]
    try:
        load_inputs(input_path)
        inputs_valid, error = True, None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        inputs_valid, error = False, str(exc)
    return web.json_response(
        {
            "status": "ok",
            "input_file": str(input_path.resolve()),
            "inputs_valid": inputs_valid,
            "error": error,
        }
    )


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    """Stream one calculation result per frame until the client disconnects."""
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)

    input_path: Path = request.app["input_path"]
    interval: float = request.app["interval"]
    print(f"client connected: {request.remote}")

    frame = 0
    try:
        while not ws.closed:
            try:
                positions, error = compute_from_file(input_path)
            except Exception as exc:  # noqa: BLE001 - file read / validation error
                positions, error = None, str(exc)

            try:
                await ws.send_json(
                    {
                        "frame": frame,
                        "timestamp": time.time(),
                        "input_file": str(input_path.resolve()),
                        "positions": positions,
                        "error": error,
                    }
                )
            except (ConnectionResetError, RuntimeError):
                break

            frame += 1
            await asyncio.sleep(interval)
    finally:
        print(f"client disconnected: {request.remote}")
    return ws


def build_app(input_path: Path, interval: float) -> web.Application:
    app = web.Application()
    app["input_path"] = input_path
    app["interval"] = interval
    app.add_routes(
        [
            web.get("/health", health),
            # WebSocket feed served at both the root (the viz connects directly
            # to ws://host:8765 via the baked VITE_WS_URL) and /ws (the nginx
            # proxy path), matching how the old websockets server accepted any
            # path.
            web.get("/", ws_handler),
            web.get("/ws", ws_handler),
            web.get("/inputs", get_inputs),
            web.put("/inputs", put_inputs),
        ]
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream tool positions over WebSocket")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"JSON input file (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="bind port (default: 8765)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1 / 30,
        help="seconds between streamed frames (default: 1/30)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input file not found: {args.input}")

    app = build_app(args.input, args.interval)
    print(
        f"streaming from {args.input.resolve()} on "
        f"ws://{args.host}:{args.port}/ws (inputs API at /inputs) "
        f"every {args.interval:.3f}s"
    )
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
