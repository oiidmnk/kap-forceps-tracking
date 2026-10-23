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

- **Instrument geometry:** the forceps are modelled on the Haag-Streit
  simulator instrument: a 23G tube (Ø 0.64 mm) with a rounded open end, and two
  flat jaws with inward hooks that emerge from the bore (exposed length
  1.65 mm, full opening ~0.7 mm; constants in `src/config.js`). Displayed tips
  use the original tracking coordinates, without amplifying jaw opening.
  Distance guides and projected shadows share those positions in both the
  overview and proximity scope; the tube end is an estimate based on the
  nominal jaw length. The forceps shaft and the 25G light pipe continue 4 mm
  outside the eye through their trocars; these outer parts cast no shadow.
  The light pipe ends in a glowing fiber.
- **Data source:** Simulation, Live tracking, or Manual simulation. The header
  shows calibration and confidence only for live tracking; simulated data has
  neither. Calibration is unknown unless the live source explicitly reports a
  `calibrated` boolean.
- **Styling:** all colours in `src/workstation.css` are CSS variables in
  `:root` (surfaces, lines, text levels, states). Text is at least 12 px.
- **Views:** Overview, Surgeon, Sagittal and Fundus (keys 1–4). Fundus looks
  from just below the pupil onto the posterior pole, roughly like the
  simulator camera. Reset view (R) restores the selected framing.
- **Navigation:** the controls float in the 3D view next to the gizmo
  (orbit pivot, zoom, reset, `?` for all shortcuts). Drag to orbit, right-drag to pan (the pivot stays inside the
  globe), scroll to zoom toward the cursor (down to 1 mm, also inside the eye).
  *Orbit* chooses the pivot: Globe center (G), Instrument (F — the camera
  follows the jaws), or Point — double-click the instrument or the retina to
  orbit around that spot.
- **Display:** scene layers and scope controls. *Instrument shadow* casts the
  forceps' real shadow from the light-pipe tip onto the retina (a shadow-only
  spot light aimed at the jaws, so the shadow stays sharp) (main view and
  proximity scope). *Shadow rays* shows the tip projection rays. *Globe
  cutaway* hides the half of the globe facing the camera so the interior is
  unobstructed.
- **Distance panel:** the minimum distance as the primary readout, and both
  tips on compact depth bars (square-root 0–10 mm scale, so the critical first
  millimetres get most of the room; zones at the 0.5 mm and 2 mm thresholds).
- **Approach:** the minimum distance over the last 10 s (same scale and
  zones), approach speed, and an estimated time to contact while the
  instrument is approaching (at most 10 s ahead).
- **Instrument state:** jaw opening (open/closed), insertion depth from the
  trocar, and the shaft's angle to the retina where its axis would land.
- **In-view safety cues:** the clearance is labelled next to the drop line, and
  in the warning or critical zone the whole 3D view gets a colored frame and a
  banner (critical pulses unless reduced motion is preferred).
- **Proximity scope:** keeps its space when the tips are outside 5 mm, when
  disabled, or while tracking is unavailable. Drag the active close-up to rotate.
- **Live state:** only complete, finite forceps coordinates produce measurements.
  Missing or malformed tracking clears readouts. A 1.5-second interruption
  marks data stale; a complete new frame restores tracking automatically.
  While tracking is lost, the 3D view keeps the last measured pose as a faded
  ghost (no shadows, reticle or close-up) with a "last known pose" banner.
  Native millimeter frames and normalized upstream `positions` are supported.
- **Readouts:** distance numbers refresh 10×/s (`READOUT_INTERVAL_MS`); the 3D
  views update every frame. Safety colors use one-sided hysteresis
  (`DIST_HYSTERESIS_MM`): a more critical level shows immediately, a safer one
  only once the threshold is cleared by 0.05 mm. All panels and both 3D views
  share the same level.
- **Live smoothing:** the 3D views blend live poses between measurements
  (about one sample interval behind, never extrapolated); readouts show the
  raw measurements.
- **Rendering:** all 3D views share one WebGL context — a fixed transparent
  canvas renders each view into its DOM placeholder (`src/scene/SceneView.jsx`).
  Tracking data lives in external stores (`src/feed/pipeline.js`), so only the
  components that display it re-render.

Run `npm test` for feed validation, freshness, and threshold-boundary tests;
run `npm run build` for a production build.
