# segment – forceps & shadow tracking

Finds the two jaw **tips** of the forceps and of its **shadow**, plus the **junction** where the jaws meet the shaft,
in retina-simulator footage – offline *and* live – and feeds them into the 3D depth pipeline.

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
```

By default the stack plays `data/video/forceps_lev1.mp4` in a loop at real-time speed – i.e. it behaves like a camera.
Calibrate eye/trocars in the orchestrator (`:8090`); defaults live in `config/default_calibration.json`
(eye centre/radius are set for the 1080² sample footage – adjust for other cameras).

### Live footage

| source | how |
|---|---|
| network camera / encoder | `SOURCE=rtsp://…` or `SOURCE=http://…` (MJPEG) in `.env`, then `make up` |
| local webcam / capture card | Docker on macOS cannot see cameras: `make live-cam` (`CAM=1` for another index) runs the detector natively and pushes to the dockerised orchestrator; view at `:8001` |
| other file | put it under `data/` and set `SOURCE=/data/…` |

Non-square frames are centre-cropped to a square; returned coordinates are in the **original** pixel space.

### The background

Shadows are found as "darker than the tool-free retina", so the detector needs a reference image:

* **static** – `data/background.png` (median of the sample video). Default; most accurate.
* **learned** – set `BACKGROUND_PATH=` (empty). The detector collects ~30 frames over `BOOTSTRAP_SECONDS` (default 8 s)
  while tools move, builds a per-pixel bright percentile, then keeps adapting where no tool is. Use this for new
  cameras/scenes. `POST /background/reset` relearns. Until a background exists a weaker single-frame fallback is used.

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
`POST /background/reset`

## Accuracy (`make eval`, 43 hand-labelled frames in `data/annotated`)

| | mean | max |
|---|---|---|
| forceps tips | 4.4 px | 7.9 px |
| shadow tips | 3.7 px | 11.9 px |

Junction points are not labelled, so they are not scored. Single-frame fallback without a background: shadow ≈ 30 px
(use a background). Closed jaws have no fork → reported as `ok:false` for that frame.

## Layout

```
config/    default calibration, stream seed
data/      video/, annotated/ (LabelMe frames), background.png
scripts/   evaluate.py, render_video.py, make_background.py
services/  detector, orchestrator, stream, viz
```

Tests: `make setup && make test`. Das frühere YOLO-Segmentierungs-Setup (`segmentation/`) ist aus der Git-History weiterhin abrufbar (vor Commit "Make repo root the segment project").
