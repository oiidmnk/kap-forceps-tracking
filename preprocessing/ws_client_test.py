"""Quick client to verify the WebSocket stream."""

import asyncio
import json

import websockets


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        for _ in range(3):
            msg = json.loads(await ws.recv())
            print(
                f"frame={msg['frame']} "
                f"error={msg['error']} "
                f"left_tip={msg['positions']['left_tip_forceps'] if msg['positions'] else None}"
            )


if __name__ == "__main__":
    asyncio.run(main())
