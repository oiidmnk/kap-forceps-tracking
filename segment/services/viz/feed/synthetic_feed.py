#!/usr/bin/env python3
"""Synthetic microforceps tracking feed.

Emits the same per-frame JSON the real tracking pipeline will send, over a
WebSocket, so the 3D dashboard can be developed against the *live* code path
before the real tracker exists. Swap this out for the real emitter later.

Message shape (units = mm, origin = eye-globe center, right-handed):
    {"t": float, "tip_left": [x,y,z], "tip_right": [x,y,z],
     "trocar": [x,y,z], "confidence": float}

Run:  python feed/synthetic_feed.py   (needs `pip install websockets`)
"""
import asyncio
import json
import math
import time

import websockets

HOST, PORT = "localhost", 8765
RATE_HZ = 30
EYE_RADIUS = 12.0  # mm


def normalize(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return [c / n for c in v]


def scale(v, s):
    return [c * s for c in v]


def add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


# +Y-up convention: instruments enter near the top (+Y); retina is the lower
# (-Y) hemisphere. Forceps trocar is fixed on the surface near the top.
TROCAR = scale(normalize([0.35, 0.9, 0.45]), EYE_RADIUS)

# Light pipe (illumination): a second, roughly static instrument entering on the
# opposite side near the top and aiming at the posterior pole.
LIGHT_TROCAR = scale(normalize([-0.5, 0.82, -0.28]), EYE_RADIUS)
LIGHT_AXIS = normalize(sub([1.5, -EYE_RADIUS, 1.0], LIGHT_TROCAR))
LIGHT_TIP = add(LIGHT_TROCAR, scale(LIGHT_AXIS, 9))


def perp_basis(d):
    ref = [0, 1, 0] if abs(d[1]) < 0.9 else [1, 0, 0]
    u = normalize(cross(d, ref))
    v = normalize(cross(d, u))
    return u, v


SPEED = 0.35  # overall pace of the synthetic motion; lower = slower


def sample_frame(t):
    s = t * SPEED
    inward = normalize(scale(TROCAR, -1))
    u, v = perp_basis(inward)
    shaft = normalize(
        add(inward, add(scale(u, 0.35 * math.sin(s * 0.7)),
                        scale(v, 0.35 * math.cos(s * 0.5))))
    )
    # Sweeps the tips from mid-vitreous to near the retina (exercises the
    # Distance-to-Retina safety states); stays inside the globe.
    depth = 12.5 + 8.0 * math.sin(s * 0.4)
    jaw_center = add(TROCAR, scale(shaft, depth))

    # Jaws open/close within a realistic microforceps range (~7-18 deg full).
    half = 0.06 + 0.1 * (0.5 + 0.5 * math.sin(s * 1.3))
    jaw_len = 1.2
    ju, _ = perp_basis(shaft)
    along = scale(shaft, jaw_len * math.cos(half))
    spread = scale(ju, jaw_len * math.sin(half))

    return {
        "t": round(t, 3),
        "tip_left": add(jaw_center, add(along, spread)),
        "tip_right": add(jaw_center, add(along, scale(spread, -1))),
        "trocar": TROCAR,
        "light_tip": LIGHT_TIP,
        "light_trocar": LIGHT_TROCAR,
        "confidence": 0.99,
    }


async def handler(websocket):
    print(f"client connected: {websocket.remote_address}")
    start = time.perf_counter()
    try:
        while True:
            t = time.perf_counter() - start
            await websocket.send(json.dumps(sample_frame(t)))
            await asyncio.sleep(1 / RATE_HZ)
    except websockets.ConnectionClosed:
        print("client disconnected")


async def main():
    print(f"synthetic feed on ws://{HOST}:{PORT}  ({RATE_HZ} Hz)")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped")
