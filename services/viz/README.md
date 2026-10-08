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

Then select **Live tracking** in the Data source selector. The default endpoint is
`/ws` on the current host (proxied by nginx in Docker). To connect directly to
the optional Python feed during development, run
`VITE_WS_URL=ws://localhost:8765 npm run dev`.

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

## UI versions

The **workstation is the active default** at `/`. The original visualization
is preserved independently in `src/classic/` and is available at `/?ui=classic`
in both development and production. Return to `/` for the workstation.
Only the selected version is loaded, keeping their styles isolated.

## Workstation controls

- **Data source:** Simulation, Live tracking, or Manual simulation. Simulated
  confidence is not presented as measured telemetry. Calibration is unknown
  unless the live source explicitly reports a `calibrated` boolean.
- **Views:** Overview, Surgeon, and Sagittal (keys 1–3); Reset view restores
  the selected framing. Display contains scene-layer and scope controls.
- **Depth profile:** toggle between 0–10 mm and a magnified 0–3 mm scale.
  The dashed markers use the configured 0.5 mm and 2 mm thresholds.
- **Proximity scope:** keeps its space when the tips are outside 5 mm, when
  disabled, or while tracking is unavailable. Drag the active close-up to rotate.
- **Live state:** only complete, finite forceps coordinates produce measurements.
  Missing or malformed tracking clears readouts. A 1.5-second interruption
  marks data stale; a complete new frame restores tracking automatically.
  Native millimeter frames and normalized upstream `positions` are supported.

Run `npm test` for feed validation, freshness, and threshold-boundary tests;
run `npm run build` for a production build.
