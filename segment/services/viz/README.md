# Microforceps 3D Tracking — Digital Twin

Real-time 3D visualization dashboard for surgical microforceps in robotic eye
microsurgery. Renders a digital twin of the eye globe, the trocar anchor, and a
pivoting/articulating forceps, with the critical **Distance to Retina** safety
readout. See `CLAUDE.md` for the full project spec.

This repo is the **visualization** layer. Tracking (computer vision) and shadow
triangulation are upstream Python components that feed 3D coordinates in.

## Architecture

```
Python tracking pipeline ──(WebSocket JSON)──▶  React + three.js dashboard
   (CV + triangulation)                            (this repo)
```

The frontend is decoupled from the tracker: it consumes a per-frame JSON stream.
During development a synthetic feed (or an in-browser mock) produces the same
message shape, so the real tracker drops in with no frontend changes.

Message shape (units = mm, origin = eye-globe center, right-handed):

```json
{ "t": 1234.5, "tip_left": [x,y,z], "tip_right": [x,y,z], "trocar": [x,y,z], "confidence": 0.99 }
```

## Run

Frontend (works standalone with the built-in mock feed):

```bash
npm install
npm run dev
```

Optional live feed (exercises the real WebSocket code path):

```bash
pip install -r feed/requirements.txt
python feed/synthetic_feed.py
```

Then click **"Use live feed"** in the dashboard (defaults to `ws://localhost:8765`,
configurable in `src/config.js`).

## Key files

| Path | Purpose |
|------|---------|
| `src/config.js` | Geometry constants + safety thresholds + WS URL |
| `src/geometry.js` | Distance-to-Retina, forceps pose math |
| `src/hooks/useTrackingFeed.js` | WebSocket / mock feed subscription |
| `src/scene/` | Eye globe, trocar, forceps 3D components |
| `src/components/HUD.jsx` | 2D overlay: safety readout + controls |
| `feed/synthetic_feed.py` | Stand-in Python live feed |
```
