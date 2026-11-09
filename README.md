# segment – forceps & shadow tracking

Finds the two jaw **tips** of the forceps and of its **shadow**, plus the **junction** where the jaws meet the shaft,
in retina-simulator footage – offline *and* live – and feeds them into the 3D depth pipeline.

> **3D stage: perspective camera** ([#1](https://github.com/oiidmnk/kap-forceps-tracking/issues/1)). The stream now
> reconstructs with a pinhole camera ~22 mm from the eye centre instead of an orthographic one. With the simulator at
> 6.5× / 6 mm ([Simulator settings](#simulator-settings)) and a camera measured once, the tip is **0.12–0.24 mm** off
> (median, three runs, each scored with the calibration of the other two) instead of 1.5–1.8 mm, and the height above the
> retina has no systematic bias any more (was −0.4 to −0.5 mm). The camera is stored in video px, so the eye circle
> only crops the detector input and does not enter the 3D. Open:
> 1. **Eye roll per session** – the eye still rolls by up to ~2° between runs and moves during a run. The trocar line in
>    the lens-off view measures the roll to ~0.1–0.3° against the log; not yet used by the pipeline.
> 2. **TODO: live input = exported video?** The camera (`camera_focal_px`, `camera_pp_px`) is measured on the simulator's
>    exported MP4s (1080 × 1080). It holds for the live input only if that has the same framing (same size, no extra crop
>    or scaling). Expected to be the same, not yet checked: compare a live frame with an exported one, e.g. the trocars
>    in the lens-off view.
> 3. **Reporting rate** – on the lev2 runs the stream reports the tip in only 65–84 % of the frames (shadow detection).

```
 video file / RTSP / MJPEG / camera
        │
   ┌────▼─────┐  4 points   ┌──────────────┐  inputs   ┌────────┐  ws   ┌─────┐
   │ detector │────────────►│ orchestrator │──────────►│ stream │──────►│ viz │
   │  :8000   │ /api/manual │    :8090     │ PUT/inputs│ :8765  │       │:8080│
   └────┬─────┘  -points    └──────────────┘           └────────┘       └─────┘
        └─► annotated MJPEG  http://localhost:8000
```

| service | what | port |
|---|---|---|
| `services/detector` | classical-CV detector (no training), live worker, annotated MJPEG, `/segment` compatible with the YOLO service the orchestrator used before | 8000 |
| `services/orchestrator` | calibration UI/API, merges points + calibration (from `kap-forceps-tracking`) | 8090 |
| `services/stream` | 3D position computation, websocket feed (from `kap-forceps-tracking/preprocessing`) | 8765 |
| `services/viz` | React/Three.js dashboard (from `kap-forceps-tracking/microforceps-viz`) | 8080 |

## Run

```bash
make up                      # docker compose up -d --build
open http://localhost:8000   # annotated live view + state
open http://localhost:8080   # 3D viz, driven live by the detector
open http://localhost:8090   # calibration workspace
open http://localhost:8765/viewer  # live 3D position diagnostics
```

By default the stack plays `data/video/forceps_lev1.mp4` in a loop at real-time speed – i.e. it behaves like a camera.
Calibrate eye/trocars in the orchestrator (`:8090`); defaults live in `config/default_calibration.json`
(eye centre/radius are set for the 1080² sample footage – adjust for other cameras).
Trocar angles, light tip (1.73 mm past its trocar, with its aim), the perspective camera (`camera_focal_px`,
`camera_pp_px` in video px, `camera_center_mm` in the eye frame), eye radius (12.05 mm) and jaw length (1.9 mm) there are measured
from the simulator's trajectory logs of the three runs of 2026-11-09 at 6.5× / 6 mm (`scripts/eval_sim_gt.py … --write-calibration`,
averaged). Without the `camera_*` fields the stream falls back to the orthographic model. The orchestrator converts the
camera to the detector's crop (its current eye circle) whenever it pushes the calibration to the stream, so the circle
only has to contain the retina: set it once (*Auto-detect* or by hand) and leave it. Re-detecting it per frame would
add noise – the auto-detected centre scatters by 19–33 px between frames while the eye moves by 5–12 px. The values hold
for the simulator's own orientation (Haag-Streit logo upright bottom left, instruments from the upper left, shadow down/left):
capture it with `ROT=0`. Footage turned by 180° against that (logo upside down, instruments from the lower right) needs `ROT=180`.
The orchestrator keeps its own copy in its state volume: `make calibration` (also run by `make ml-cam`) pushes the file
to the running orchestrator and stream without touching the eye circle.

### Simulator settings

Set these on the Eyesi before every session; both can be read off the simulator:

| setting | value |
|---|---|
| microscope zoom | **6.5×** |
| lens–cornea distance | **6 mm** |

They fix the camera's focal length and distance to the eye, so a camera measured once holds for every session: in
three runs at these settings the focal length differed by 0.45 % and the camera distance by 0.1 mm. The camera in
`config/default_calibration.json` is measured at 6.5× / 6 mm; the recordings of 2026-10-22 predate this rule.
To record a run for evaluation: export video + trajectory log, and show the lens-off view for a few seconds (trocars).

### Source & eye circle (orchestrator UI, `http://localhost:8090`)

The orchestrator shows only **Source & eye circle** and controls the detector at runtime:

* **Source** – pick a video file from `data/video` or scan local capture devices. Switching the source relearns
  the background (~8 s).
* **Video library** – choose any saved video in the source dropdown to switch processing immediately.
* **Add a video** – upload an MP4 (up to 512 MiB) using the compact upload row below source selection. It is saved
  in `data/video` with its original filename and selected automatically. Duplicate names get a numbered suffix
  instead of overwriting existing videos. The selected detector uses `VIDEO_DIR` for its video folder.
  Uploading clears the previous eye circle; draw and apply a new one for the uploaded footage.
* **Circle** – drag it around the retina (drag inside = move, drag the edge or scroll = resize, drag on empty area =
  new circle, *Auto-detect* fits the bright retina). *Apply circle* sends it to the detector.
* **What the pipeline gets** – only the circle content, everything outside black, as a **1:1 square** (1080², shown on
  the right). Detections, the live view, and the points sent to the stream are all in that normalised space, so the
  eye calibration is re-based automatically (centre = 540/540, radius = 540).
* **Detector** (collapsible) – which detector the UI controls: `http://detector:8000` (Docker) or
  `http://host.docker.internal:8001` for the native one started with `make live-cam` (needed for USB capture cards on
  macOS; stop the Docker detector first: `docker compose stop detector`).

Source and circle are persisted (`detector-state` volume / `.detector_state.json` natively).

The detector, calibration workspace, and stream viewer use the Viz workstation's
dark palette and link to the other services on the current hostname (default ports).
The orchestrator retains source selection, MP4 upload, and the eye-circle editor; single-frame processing,
manual point entry, and the additional calibration panels are no longer shown. Stream diagnostics clear
invalid or disconnected measurements and mark a feed stale after 1.5 seconds.

### Live footage

| source | how |
|---|---|
| network camera / encoder | `SOURCE=rtsp://…` or `SOURCE=http://…` (MJPEG) in `.env`, then `make up` |
| local webcam / capture card | Docker on macOS cannot see cameras: `make live-cam` (`CAM=1` for another index) runs the detector natively and pushes to the dockerised orchestrator; view at `:8001` |
| other file | put it under `data/` and set `SOURCE=/data/…` |

Without a circle, non-square frames are centre-cropped to a square and coordinates are in the **original** pixel
space; with a circle they are in the normalised square (see above).

`FORCEPS_ANGLE` (degrees, default `0` = off) adds a colour-direction cue for forceps that are as saturated as the
retina (seen on some capture cards). It is not validated across scenes – tune per setup, values below ~10 hurt the
simulator shadow detection.

### The background

Shadows are found as "darker than the tool-free retina", so the detector needs a reference image:

* **static** – `data/background.png` (median of the sample video). Default; most accurate.
* **learned** – set `BACKGROUND_PATH=` (empty). The detector collects ~30 frames over `BOOTSTRAP_SECONDS` (default 8 s)
  while tools move, builds a per-pixel bright percentile, then keeps adapting where no tool is. Use this for new
  cameras/scenes. `POST /background/reset` · `GET/POST /config` (source + circle) · `GET /sources` · `GET /preview.jpg` ·
`GET /autocircle` · `GET /background.jpg` relearns. Until a background exists a weaker single-frame fallback is used.

## How it works

1. **Forceps mask** – low saturation (grey) and dark inside the eye disc.
2. **Shadow mask** – `V / V_background < 0.88` while still red-saturated (vessels cancel out).
3. **Geometry** (`detector/geometry.py`) – largest blob → principal axis → rotate so the tool points left → scan columns
   from the tip: the stretch with **two runs** is the open jaw pair. Outer ends = **tips**, where the gap closes =
   **junction**. Left/right follow your labelling (left = lower side when heading towards the tip).
4. **Live** – per frame: detect → draw → MJPEG; points are EMA-smoothed (`SMOOTHING`) and posted to the orchestrator at
   `PUSH_HZ`. Only frames with all four tip points are pushed.

Throughput ≈ 20 ms/frame at 1080² (30 fps real-time with headroom).

## API (detector)

`GET /` viewer · `GET /live.mjpg` · `GET /state` · `GET /health` · `POST /detect` (image → JSON) ·
`POST /detect/annotated` (image → JPEG) · `POST /segment` (orchestrator contract) · `POST /background` (image) ·
`POST /background/reset` · `GET/POST /config` (source + circle) · `GET /sources` · `GET /preview.jpg` ·
`GET /autocircle` · `GET /background.jpg`

## Accuracy (`make eval`, 43 hand-labelled frames in `data/annotated`)

| | mean | max |
|---|---|---|
| forceps tips | 4.4 px | 7.9 px |
| shadow tips | 3.7 px | 11.9 px |

Junction points are not labelled, so they are not scored. Single-frame fallback without a background: shadow ≈ 30 px
(use a background). Closed jaws have no fork → reported as `ok:false` for that frame.

## 3D accuracy vs. the simulator (`make sim-eval`)

The simulator's trajectory log (`<video>_trajectories.csv`: instrument poses in mm) is the ground truth. `scripts/eval_sim_gt.py`
fits camera and log offset from the forceps tips, replays the video through eye circle -> detector -> calibration ->
`tool_positions.py` and compares the tip midpoint (median / p95 in mm).

**Perspective model, 6.5× / 6 mm** – three runs of `mapskap1_2026-11-09` (forceps training lev 2, ~3100–3300 frames with
the tip in view each). "Fixed camera": camera, trocars and light averaged from the *other two* runs, i.e. what a camera
measured once gives, stored in video px; "today": the orthographic model with the 2026-10-22 calibration.

| run | | today (orthographic) | fixed camera (perspective) | height, fixed camera | 3D found |
|---|---|---|---|---|---|
| 11-14-13 | limit of the 3D model | 1.86 / 6.1 | 0.14 / 0.35 | 0.05 / 0.16 | 100 % |
| | learned segmenter | 1.78 / 8.6 | 0.24 / 6.1 | 0.12 / 2.4 | 84 % |
| 11-25-32 | limit of the 3D model | 1.96 / 6.1 | 0.16 / 0.33 | 0.08 / 0.19 | 100 % |
| | learned segmenter | 1.46 / 6.4 | 0.16 / 0.60 | 0.09 / 0.48 | 65 % |
| 11-32-14 | limit of the 3D model | 2.16 / 10.9 | 0.14 / 0.55 | 0.06 / 0.33 | 100 % |
| | learned segmenter | 1.81 / 9.5 | 0.12 / 0.61 | 0.06 / 0.44 | 69 % |

Height bias with the fixed camera: −0.05 to +0.09 mm (orthographic: −0.38 to −0.53 mm). With each run's own camera the
segmenter reaches 0.12–0.17 mm, so a camera measured once loses little. Stored relative to the auto-detected eye circle
instead of in video px, the same camera gave 0.31 / 0.23 / 0.19 mm: the circle moved by up to 15 px between runs while
the camera did not. The p95 of 6 mm in 11-14-13 comes from detection outliers, not from the model.

**Orthographic model (before #1)** – three runs of `mapskap1_2026-10-22` (~3500–4000 frames with the tip in view each):

| run | | 3D | height above retina | 3D found |
|---|---|---|---|---|
| 11-00-43 | ground-truth tip + modelled shadow (limit of the 3D model) | 1.42 / 5.3 | 0.63 / 2.9 | 100 % |
| | learned segmenter (`DET=ml`) | 1.40 / 4.2 | 0.60 / 2.0 | 83 % |
| | classical detector, no background (`DET=classic`) | 1.82 / 7.5 | 1.11 / 5.5 | 49 % |
| 10-35-27 | limit of the 3D model | 2.18 / 11.7 | 0.91 / 7.7 | 100 % |
| | learned segmenter | 2.36 / 7.8 | 0.85 / 4.1 | 81 % |
| 10-48-23 | limit of the 3D model | 2.11 / 5.5 | 0.58 / 3.1 | 100 % |
| | learned segmenter | 1.76 / 4.7 | 0.50 / 3.7 | 65 % |

With the segmenter the error is the 3D model's own, not the detection's: forceps tips 0.4–0.6 px, found in 97–99 % of
the frames. The shadow is found in 86–88 % of the frames, except 10-48-23 (69 %, not investigated yet). The trocars are
identical in all three runs; the runs differ in the eye's tilt towards the camera.

`make sim-eval SIM=<video>` (trajectory CSV next to it), `DET=classic` for the classical detector; results in `artifacts/sim_gt_*`.
`scripts/eval_sim_gt.py <video> --reuse --calibration <json>` scores another calibration on the cached detections;
`--write-calibration <json>` writes the calibration the recording implies (trocars, light, camera).

## Layout

```
config/    default calibration, stream seed
data/      video/, annotated/ (LabelMe frames), background.png
scripts/   evaluate.py, render_video.py, make_background.py
services/  detector, orchestrator, stream, viz
```

Tests: `make setup && make test`. Das frühere YOLO-Segmentierungs-Setup (`segmentation/`) ist aus der Git-History weiterhin abrufbar (vor Commit "Make repo root the segment project").

Service UI regression checks (Node.js): `node --test tests/service-ui.test.cjs`.
