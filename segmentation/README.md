# auge_segment

YOLO instance segmentation training pipeline for forceps tips and shadow points within eyeball imagery.

The model targets **4 classes per image**:

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | `tip_left` | Left forceps tip |
| 1 | `tip_right` | Right forceps tip |
| 2 | `shadow_left` | Left shadow point at tip |
| 3 | `shadow_right` | Right shadow point at tip |

Built on [Ultralytics YOLO segmentation](https://docs.ultralytics.com/tasks/segment#predict).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset layout

Place hand-segmented data in the train/val folders:

```
data/
├── images/
│   ├── train/          # frame_001.png, ...
│   └── val/
└── labels/
    ├── train/          # frame_001.txt, ...
    └── val/
```

For unsplit raw data, use `data/raw/images/` and `data/raw/labels/` and run `split_dataset.py` first.

### Label format

One `.txt` file per image (same stem). Each line is one polygon instance:

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

Coordinates are normalized to `[0, 1]` relative to image width and height.

See [`examples/sample_label.txt`](examples/sample_label.txt) for a documented example.

### Labeling conventions

- **Left/right** are defined relative to the **image frame**, not surgeon handedness.
- **Shadow points** are the visible shadow/reflection blob associated with each tip, not the tip itself.
- If tips overlap, annotate two separate polygons.
- Omit lines for parts that are not visible in a frame.

## Synthetic pose training data

Generate retinal-style frames with randomized forceps and shadow positions:

```bash
python scripts/generate_synthetic_dataset.py --count 500 --preview 10
```

This writes paired PNG images and YOLO pose labels into `data/images/{train,val}`
and `data/labels/{train,val}`. Synthetic images are always rectangular,
non-transparent RGB PNG files, but the default usable training view is the
circular microscope field with solid black corners. Both labeled objects are
constrained to remain inside that circle with a safety margin based on both the
view diameter and the maximum requested shadow blur. This keeps the forceps and
shadow endpoint/root regions visibly away from the mask boundary. Use
`--rectangular-view` only when a full-frame view is required. Each label has two
objects:

```text
0 <forceps_bbox> <tip_left_x> <tip_left_y> 2 <tip_right_x> <tip_right_y> 2 <jaw_root_x> <jaw_root_y> 2
1 <shadow_bbox> <shadow_left_x> <shadow_left_y> 2 <shadow_right_x> <shadow_right_y> 2 <shadow_root_x> <shadow_root_y> 2
```

The forceps and shadow are trained as separate pose detections, each with two
endpoint keypoints plus a root keypoint. Existing two-keypoint pose labels and
weights must be regenerated/retrained before using this config.

Use `--background` with one or more clean microscope images to sample a
background uniformly for every generated frame; otherwise the script creates a
procedural retina-like background. The option can also be repeated:

```bash
python scripts/generate_synthetic_dataset.py \
  --count 500 \
  --background backgrounds/retina_01.png backgrounds/retina_02.png \
  --background backgrounds/retina_03.png
```

Each selected background still receives an independent crop, rotation, and
blur. Its brightness is otherwise preserved; the generator does not apply
random gain or brightness offsets.

By default, each retina/background is randomly rotated before the forceps are
drawn. Use `--background-rotation 0` to disable it, or pass a smaller value such
as `--background-rotation 30` for mild rotation variants.

Completed images are also sampled from discrete quarter-turn rotations together
with their YOLO keypoints, varying the forceps entry side as well as the retinal
orientation. The default set is exactly `90`, `180`, and `270` degrees. The
circular field is rotation-invariant, so quarter-turn variants fill the usable
round view without internal letterboxing; everything outside the circle remains
solid black:

```bash
python scripts/generate_synthetic_dataset.py \
  --image-rotations 90 180 270
```

Use `--image-rotations 0` to keep the forceps entering from the original
bottom-right region.

Forceps roll and projected-shadow roll are sampled independently by default.
The generator deliberately mixes forceps-only, shadow-only, independently
rolled, and physically correlated roll cases. This changes jaw foreshortening,
which jaw appears nearer, metal highlights, and the projected shadow opening.
Use `--axis-roll 0` or `--shadow-axis-roll 0` to disable either source of roll,
or pass a smaller value such as `--axis-roll 45` for milder variants.

Shadow and gripping-tip scale also vary independently. The defaults include
large penumbras and enlarged distal tips. Tune their ranges explicitly:

```bash
python scripts/generate_synthetic_dataset.py \
  --count 500 \
  --axis-roll 180 \
  --shadow-axis-roll 180 \
  --shadow-scale 0.9 2.1 \
  --shadow-opacity 0.4 0.7 \
  --shadow-blur 2 22 \
  --forceps-opacity 0.5 1.0 \
  --forceps-blur 0 2.5 \
  --tip-scale 0.85 2.0 \
  --preview 12
```

Preview overlays print the sampled roll angles and scale factors at the bottom
of each image, making it easy to audit that a batch covers the intended cases.
`--shadow-opacity MIN MAX` (also available as `--shadow-visibility`) controls
how visible shadows are on a `0` to `1` scale. Every sampled opacity remains
inside the supplied bounds; use equal values such as `--shadow-opacity 0.6 0.6`
for constant shadow visibility.

`--shadow-blur MIN MAX` (also available as `--shadow-softness`) controls the
Gaussian blur sigma in output pixels. Use `--shadow-blur 0 3` for mostly
hard-edged shadows, `--shadow-blur 12 28` for soft shadows, or equal values for
a constant blur. Far shadows are biased toward the blurrier end and near
shadows toward the sharper end, without exceeding the requested range.

`--forceps-opacity MIN MAX` and `--forceps-blur MIN MAX` vary the instrument
appearance independently of the background. The default opacity range includes
pale, partially translucent instruments like real microscope footage, while
the blur range adds mild defocus without blurring the retina or shadow.

Train with a pose checkpoint and the pose config:

```bash
yolo pose train model=yolo11n-pose.pt data=configs/forceps_pose.yaml imgsz=1024
```

## Workflow

### Run Studio web UI

Launch the segmentation project's local run manager:

```bash
auge-webui
```

Then open `http://127.0.0.1:8010`. The UI can start and cancel model training
or synthetic dataset generation, follow live logs and progress, and preview the
resulting metrics and images. Every job is isolated under `runs/webui/<run-id>`;
its parameters and status are persisted in `run.json`, its console output in
`run.log`, and generated results under `artifacts/`.

Training runs can use a completed synthetic dataset run directly. The studio
creates a run-local YOLO pose config that points at the selected dataset's
immutable artifacts. A dataset can also pass through a reproducible **Dataset
split** run first; the child run copies and repartitions the paired images and
labels, records its source run, and can then be selected as training input.

The **Prediction** run type selects the `best.pt` weights from a completed
training run and applies them to one uploaded image or video (a local source
path can be used instead). Confidence, image size, device, preprocessing, scene
filtering, and maximum detections are configurable. Annotated frames and MP4
videos are saved as run artifacts and can be viewed directly in the studio.
Video pose predictions use a confidence-aware temporal tracker by default. It
smooths continuous keypoint motion, rejects candidates that jump too far from
the predicted trajectory, preserves image-frame left/right endpoint ordering,
and bridges up to three missing frames with decaying confidence. Disable it in
the Studio or pass `--no-temporal-filter` to `scripts/predict_media.py` when raw
frame-independent predictions are needed for comparison.

The **Video mask** run type implements the single-disc masking workflow from
`open-a-eye`: it fits the largest bright circular contour on every frame,
optionally calibrates the crisp inner retina boundary, fixes the radius to the
median detection, fills and smooths the tracked center, crops a constant square
with black padding, and applies an anti-aliased circular mask. It writes an H.264
`masked.mp4`, middle-frame `preview.png`, and `mask.json`. Completed masking runs
appear as selectable inputs in Prediction runs. Stereo side-by-side selection is
intentionally not exposed; this run type expects one retinal disc per frame.

You can also run it without installing the entry point:

```bash
python scripts/webui.py
```

### 1. Split raw data (optional)

If images and labels are still in `data/raw/`:

```bash
python scripts/split_dataset.py
```

Split is **by session** (prefix before the last `_` in the filename) to avoid train/val leakage from the same procedure. Customize grouping with `--session-pattern`.

### 2. Validate labels

```bash
python scripts/check_labels.py
```

Render a few overlay previews:

```bash
python scripts/check_labels.py --preview 5
```

### 3. Train

```bash
python scripts/train.py
```

Common options:

```bash
python scripts/train.py --model yolo11s-seg.pt --epochs 150 --imgsz 1024 --batch 4 --device mps
```

Weights are saved to `runs/segment/forceps/weights/best.pt`.

### 4. Validate

```bash
python scripts/validate.py
```

Reports box and **mask** mAP. Prioritize `metrics.seg.*` for this task.

### 5. Predict

```bash
python scripts/predict.py --source data/images/val/
```

Outputs land in `runs/segment/predict/`.

Pose inference applies the known scene constraints by default: keypoints on the
black microscope border are rejected, and only the strongest valid forceps and
shadow detection are retained. Pass `--no-scene-filter` only when inspecting
all raw model proposals.

### 6. Serve predictions over HTTP

The Docker Compose stack includes a `segmentation` service exposing
`POST /segment` on port `8000`. It accepts a multipart image upload and returns
JSON with image dimensions, preprocessing transform metadata, and detected
instances with class names, confidences, boxes, and segmentation polygons.

```bash
docker compose up segmentation
curl -F image=@data/images/val/frame_001.png http://localhost:8000/segment
```

By default, the container looks for weights at
`runs/segment/forceps/weights/best.pt` inside `segmentation/`. Override runtime
settings with environment variables:

```bash
SEGMENTATION_WEIGHTS=runs/segment/forceps/weights/best.pt \
SEGMENTATION_PREPROCESS_PRESET=roi_clahe \
docker compose up segmentation
```

The HTTP service enables the same pose scene filter by default. Set
`SEGMENTATION_SCENE_FILTER=0` to return all raw detections.

## Train on a remote GPU server

Configure the server in `~/.ssh/config`, then sync the project and launch a
detached training run:

```bash
scripts/remote_train.sh start \
  --host gpu-box \
  --epochs 150 \
  --batch 16 \
  --device 0
```

The controller creates the remote virtual environment, verifies CUDA, records a
PID and log, and keeps training alive after SSH disconnects. Monitor and fetch
results with:

```bash
scripts/remote_train.sh status --host gpu-box
scripts/remote_train.sh logs --host gpu-box
scripts/remote_train.sh fetch --host gpu-box
```

Use `--remote-dir` to change the default `kap-forceps-segmentation` directory and
`--help` for all options. A preprocessing experiment can be prepared remotely
before training:

```bash
scripts/remote_train.sh start \
  --host gpu-box \
  --preprocess-preset roi_clahe
```

## Preprocessing experiments

Presets in `configs/preprocessing.yaml` include the original input, circular ROI
masking or cropping, CLAHE, gamma correction, bilateral denoising, highlight
compression, sharpening, and `retina_reference`. Source images and labels are
never modified.

### Mask an MP4 or MOV to the retinal aperture

To turn everything outside the inner circular retinal view black:

```bash
python scripts/mask_retinal_video.py input.mov output.mp4 \
  --preview runs/retinal-mask-preview.jpg
```

The script gets an initial aperture from several frames near the start, then
detects and tracks it on every frame by default. Implausible jumps and failed
detections reuse the previous circle, while temporal smoothing prevents the
mask from flickering. It uses FFmpeg when available so audio is retained. If
detection needs adjustment, either move the detected boundary inward with
`--margin 5`, or specify the initial circle in full-resolution pixels:

```bash
python scripts/mask_retinal_video.py input.mov output.mp4 \
  --center-x 829 --center-y 926 --radius 450
```

Use `--feather 4` for a four-pixel soft inner edge. Existing output files are
left untouched unless `--overwrite` is passed. Use `--static-mask` when the
aperture does not move and per-frame detection is unnecessary. Tracking runs
every frame unless changed with `--detect-every N`; `--smoothing`,
`--max-center-shift`, and `--max-radius-change` control its stability.

For prediction with weights trained on the generated `retina.png` backgrounds,
use the reference-matching preset:

```bash
python scripts/predict.py \
  --source data/testing \
  --weights runs/pose/forceps/weights/best.pt \
  --preprocess-preset retina_reference
```

The preset robustly matches LAB median color and percentile contrast to
`retina.png`, but only on bright, saturated retinal tissue. Low-saturation
metal, dark forceps-shadow pixels, vessels, and black microscope borders are
protected by a dilated, feathered foreground mask and retain their original
colors. The same circular field used by synthetic training is then applied.
The pose-to-JSON prediction bridge supports the same flag:

```bash
python scripts/predict_preprocessor.py \
  --source frame.png \
  --weights runs/pose/forceps/weights/best.pt \
  --preprocess-preset retina_reference
```

For the HTTP service, set
`SEGMENTATION_PREPROCESS_PRESET=retina_reference`. Docker Compose mounts the
repository `retina.png` reference into the inference container.

Reference matching can reduce the retinal color-domain gap, but it does not
change the instrument. A model trained only on opaque synthetic forceps will
still miss pale real forceps. Generate a substantially larger replacement
dataset with the default translucent/defocus variation and retrain; for example:

```bash
python scripts/generate_synthetic_dataset.py \
  --count 5000 \
  --background ../retina.png \
  --forceps-opacity 0.35 1.0 \
  --forceps-blur 0 3.5 \
  --preview 20
```

Preview presets:

```bash
python scripts/preview_preprocessing.py \
  --source data/testing \
  --max-images 5
```

Create a derived dataset. Polygon labels are clipped and renormalized when an
ROI crop is used:

```bash
python scripts/prepare_preprocessed_dataset.py --preset roi_clahe
```

Or prepare it automatically when training:

```bash
python scripts/train.py \
  --preprocess-preset roi_clahe \
  --epochs 150
```

Use the same preset for inference and latency measurements:

```bash
python scripts/predict.py \
  --weights <best.pt> \
  --source data/testing \
  --preprocess-preset roi_clahe

python scripts/benchmark.py \
  --weights <best.pt> \
  --source data/testing \
  --preprocess-preset roi_clahe
```

## Send one prediction to the 3D preprocessor

The single-image bridge reads the endpoint pose keypoints in order
`tip_left`, `tip_right`, `shadow_left`, `shadow_right`, ignores the root
keypoints for now, merges those pixel
coordinates into calibrated geometry values, and writes the JSON file that
`preprocessing/ws_server.py` already knows how to stream.

Start the websocket preprocessor from the workspace root:

```bash
cd ../preprocessing
python3 ws_server.py --input predicted_input.json
```

Run the React dashboard in live mode, then update the watched JSON whenever you
want to visualize a new image:

```bash
cd ../segmentation
python3 scripts/predict_preprocessor.py \
  --weights /Users/luis.carilla/uni/kap/auge_segment/runs/remote/forceps_remote/runs/pose/forceps_remote/weights/best.pt \
  --source data/testing/frame_001.png \
  --preprocess-preset roi_clahe \
  --base-input ../preprocessing/input_example.json \
  --sidecar ../preprocessing/geometry_sidecar_example.json \
  --output ../preprocessing/predicted_input.json
```

`--sidecar` is optional and should contain only non-predicted calibration values
such as trocar angles, light depth, eye center/radius, and jaw length. Predicted
pixel fields always come from the pose keypoints. Use the same
`--preprocess-preset` that the pose model was trained with.

## CLI entry points

After `pip install -e .`:

```bash
auge-check-labels
auge-split-dataset
auge-train
auge-validate
auge-predict --source data/images/val/
auge-predict-preprocessor --source data/testing/frame_001.png
auge-benchmark --source data/images/val/
auge-preview-preprocessing --source data/testing
auge-prepare-preprocessing --preset roi_clahe
auge-mask-retinal-video input.mov output.mp4
auge-webui
```

## Training tips

- Start with `yolo11n-seg.pt` or `yolo11s-seg.pt` for fast iteration.
- Default `imgsz=1024` helps small tips and shadow points; reduce if GPU memory is limited.
- Use modest augmentation at first — heavy color jitter can distort subtle shadow cues.
- Expect class imbalance if shadows are faint; include frames where shadows are clearly visible.

## Annotation tools

This repo does not include an annotation UI. Recommended options:

- [CVAT](https://www.cvat.ai/) (polygon export, convert to YOLO seg format)
- [Label Studio](https://labelstud.io/)
- [Roboflow](https://roboflow.com/) (export YOLO segmentation directly)

Export polygons in YOLO segmentation format with the four class names above.

## Project structure

```
configs/forceps_seg.yaml    # dataset config
configs/preprocessing.yaml  # preprocessing presets
scripts/
  check_labels.py                   # label QA
  split_dataset.py                  # raw -> train/val split
  train.py                          # training
  validate.py                       # evaluation
  predict.py                        # inference
  predict_preprocessor.py           # single-image inference -> preprocessor JSON
  benchmark.py                      # preprocessing + inference latency
  preprocessing.py                  # shared image transforms
  preview_preprocessing.py          # side-by-side preset previews
  prepare_preprocessed_dataset.py   # derived segmentation dataset builder
  remote_train.sh                   # SSH/rsync remote training controller
examples/sample_label.txt           # label format reference
```
