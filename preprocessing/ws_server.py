"""WebSocket server that streams tool-position calculations as fast as possible.

Inputs are re-read from a JSON file on every frame so an external process
can update the file live while you verify the communication path.

Usage:
    pip install websockets
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

import websockets
from websockets.server import WebSocketServerProtocol

from tool_positions import compute_tool_positions

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

DEFAULT_INPUT = Path(__file__).with_name("input_example.json")

def load_inputs(path: Path) -> Dict[str, Any]:
    """Load and validate the input JSON file.

    ``light_aim_tilt``/``light_aim_clock`` are optional and default to
    ``0.0`` (aiming straight at the eye center) so existing input files
    keep working unchanged.
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    required = [
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
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"input file missing keys: {', '.join(missing)}")

    data.setdefault("light_aim_tilt", 0.0)
    data.setdefault("light_aim_clock", 0.0)

    return data


def _as_vec2(value: Any) -> Vec2:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"expected [x, y], got {value!r}")
    return (float(value[0]), float(value[1]))


def compute_from_file(path: Path) -> Dict[str, Any]:
    """Read inputs from disk and run the reconstruction."""
    data = load_inputs(path)

    positions = compute_tool_positions(
        light_rot_up=float(data["light_rot_up"]),
        light_rot_clock=float(data["light_rot_clock"]),
        light_depth_mm=float(data["light_depth_mm"]),
        light_aim_tilt=float(data["light_aim_tilt"]),
        light_aim_clock=float(data["light_aim_clock"]),
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

    return {
        key: [value[0], value[1], value[2]]
        for key, value in positions.items()
    }


async def stream_positions(
    websocket: WebSocketServerProtocol, input_path: Path, interval: float
) -> None:
    """Send one calculation result per frame until the client disconnects."""
    frame = 0

    while True:
        try:
            positions = compute_from_file(input_path)
            payload = {
                "frame": frame,
                "timestamp": time.time(),
                "input_file": str(input_path.resolve()),
                "positions": positions,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - stream errors to client
            payload = {
                "frame": frame,
                "timestamp": time.time(),
                "input_file": str(input_path.resolve()),
                "positions": None,
                "error": str(exc),
            }

        await websocket.send(json.dumps(payload))
        frame += 1
        await asyncio.sleep(interval)


async def handler(
    websocket: WebSocketServerProtocol, input_path: Path, interval: float
) -> None:
    client = websocket.remote_address
    print(f"client connected: {client}")
    try:
        await stream_positions(websocket, input_path, interval)
    except websockets.exceptions.ConnectionClosed:
        print(f"client disconnected: {client}")


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

    async def serve() -> None:
        async with websockets.serve(
            lambda ws: handler(ws, args.input, args.interval),
            args.host,
            args.port,
        ):
            print(
                f"streaming from {args.input.resolve()} "
                f"on ws://{args.host}:{args.port} every {args.interval:.3f}s"
            )
            await asyncio.Future()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
