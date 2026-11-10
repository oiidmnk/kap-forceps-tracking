# segment – forceps & shadow tracking

Finds the two jaw **tips** of the forceps and of its **shadow** in retina-simulator (Eyesi) footage, live or from a
video, and computes the 3D tip position above the retina (0.12–0.24 mm median against the simulator's log, see
[docs/3d.md](docs/3d.md)).

```
 capture card / video file / RTSP / MJPEG
        │
   ┌────▼─────┐  4 points   ┌──────────────┐  inputs   ┌────────┐  ws   ┌─────┐
   │ detector │────────────►│ orchestrator │──────────►│ stream │──────►│ viz │
   │ / segm.  │ /api/manual │    :8090     │ PUT/inputs│ :8765  │       │:8080│
   └────┬─────┘  -points    └──────────────┘           └────────┘       └─────┘
        └─► annotated live view  :8000 (classical) / :8002 (learned segmenter)
```

| service | what | port |
|---|---|---|
| `services/mlseg` | learned segmenter (default for live use, native on the Apple GPU) | 8002 |
| `services/detector` | classical-CV detector, no training (docker default) | 8000 |
| `services/orchestrator` | source, eye circle, eye roll; merges points + calibration and pushes them to the stream | 8090 |
| `services/stream` | 3D position, websocket feed, diagnostics at `/viewer` | 8765 |
| `services/viz` | React/Three.js dashboard | 8080 |

## Workflow: one session

### 1. Simulator

Set on the Eyesi before every session (both can be read off the simulator):

| setting | value |
|---|---|
| microscope zoom | **6.5×** |
| lens–cornea distance | **6 mm** |

They fix the camera's focal length and distance to the eye; the camera in `config/default_calibration.json` is
measured at these settings and is wrong at any other. Keep the simulator's own image orientation: Haag-Streit logo
upright bottom left, instruments from the upper left, shadow down/left.

### 2. Start the stack

```bash
make cams                # list capture devices
make ml-cam CAM=<index>  # capture card -> learned segmenter -> orchestrator -> stream -> viz
```

`make ml-cam` starts orchestrator, stream and viz in Docker, the segmenter natively, and pushes
`config/default_calibration.json` (`make calibration`). `ROT=180` if the image stands on its head, `KEEP=1` keeps the
eye circle of the last session. Docker/OrbStack must be running. Stop with Ctrl+C and `make ml-down`.

Without a capture card: `make up` runs the classical detector on `data/video/forceps_lev1.mp4` in a loop.

### 3. Eye circle (orchestrator, http://localhost:8090)

*Auto-detect* (retina must be in view) or drag the circle around the retina (drag inside = move, edge or scroll =
resize, *Draw new circle*), then *Apply circle*. The pipeline gets only the circle content, as a 1:1 square.

* It must contain the **whole retina**, nothing more matters: the 3D does not depend on where exactly it sits.
* **Set it once and leave it.** Re-detecting it during the session only adds noise.
* Set it again after `make ml-cam` without `KEEP=1`, after switching the source, or after uploading a video.

### 4. Eye roll (orchestrator)

Once per session, after the eye is fixed: **lens off, instruments out**, *Mark trocars*, click where the light and the
forceps trocar enter the eye (either order). The readout shows the roll, e.g. `roll +1.5° · line 4.8° vs 3.3°`;
it is saved and turns both trocars in the 3D. More than 15° is rejected (wrong clicks or rotated capture).
*Reset roll* sets it back to 0. Between days the eye sat 4.5–6.6° rolled, which costs ~0.1–0.15 mm.

### 5. Check

* live view http://localhost:8002 (segmenter) or :8000 (classical): tips and shadow marked
* stream diagnostics http://localhost:8765/viewer: 3D position, stale after 1.5 s without points
* viz http://localhost:8080

### Checklist for good results

| what | why |
|---|---|
| 6.5× / 6 mm | the camera is measured there |
| original orientation, `ROT=0` (or `ROT=180` if upside down) | trocars and camera assume it |
| circle around the whole retina, set once | crops the detector input; changing it per frame adds noise |
| *Mark trocars* once per session | corrects the eye roll |
| learned segmenter (`make ml-cam`) | the classical detector loses the shadow far more often |

**Open** ([details](docs/3d.md#open-points)): whether the live input has the same framing as the exported MP4s the
camera is measured on is not yet checked; the stream reports the tip in only 65–84 % of the frames (shadow detection).

## Calibration

`config/default_calibration.json` holds trocars, light, camera (in video px), eye radius and jaw length, measured from
the simulator's trajectory logs of 2026-11-09 ([docs/3d.md](docs/3d.md)). The orchestrator keeps its own copy in its
state volume; `make calibration` pushes the file to the running orchestrator and stream without touching eye circle
and eye roll. API: `GET/PUT /api/calibration`, `POST /api/apply-calibration`, `GET/POST/DELETE /api/eye-roll`.

## Other sources

| source | how |
|---|---|
| video file | orchestrator: pick it from `data/video` or *Upload & use video* (MP4, up to 512 MiB) |
| network camera / encoder | `SOURCE=rtsp://…` or `SOURCE=http://…` (MJPEG) in `.env`, then `make up` |
| capture card with the classical detector | `make live-cam` (`CAM=1` for another index), native on :8001; in the orchestrator set *Detector* to `http://host.docker.internal:8001` |
| back to the classical detector in Docker | `make classic` |

## Classical detector

1. **Forceps mask** – low saturation (grey) and dark inside the eye disc.
2. **Shadow mask** – `V / V_background < 0.88` while still red-saturated (vessels cancel out). Needs a background:
   `data/background.png` (default) or learned from the feed with `BACKGROUND_PATH=` (~8 s while tools move;
   *Reset & relearn* in the orchestrator).
3. **Geometry** (`detector/geometry.py`) – largest blob → principal axis → the stretch with two runs is the open jaw
   pair; outer ends = tips, where the gap closes = junction.
4. **Live** – points EMA-smoothed (`SMOOTHING`) and pushed at `PUSH_HZ`, only frames with all four tips.

~20 ms/frame at 1080². On 43 hand-labelled frames (`make eval`): forceps tips 4.4 px mean / 7.9 px max, shadow tips
3.7 / 11.9 px. `FORCEPS_ANGLE` adds a colour cue for forceps as saturated as the retina (not validated).

Detector API: `GET /` · `/live.mjpg` · `/state` · `/health` · `POST /detect` · `/detect/annotated` · `/segment` ·
`/background` · `/background/reset` · `GET/POST /config` (source + circle) · `GET /sources` · `/preview.jpg` ·
`/autocircle` · `/background.jpg`

## Layout and tests

```
config/    default calibration, stream seed
data/      video/, annotated/ (LabelMe frames), background.png
docs/      3d.md (3D model, calibration, accuracy)
scripts/   evaluate.py, eval_sim_gt.py, render_video.py, make_background.py
services/  detector, mlseg, orchestrator, stream, viz
```

`make setup && make test`. The earlier YOLO segmentation setup (`segmentation/`) is in the git history (before
"Make repo root the segment project").
