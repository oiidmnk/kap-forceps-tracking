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
│   ├── train/          # frame_001.jpg, ...
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

This writes paired images and YOLO pose labels into `data/images/{train,val}` and
`data/labels/{train,val}`. Each label has one forceps object with four keypoints
in this order: `tip_left`, `tip_right`, `shadow_left`, `shadow_right`.

Use `--background path/to/clean_background.png` if you have a clean microscope
background to composite onto; otherwise the script creates a procedural
retina-like background.

By default, each retina/background is randomly rotated before the forceps are
drawn. Use `--background-rotation 0` to disable it, or pass a smaller value such
as `--background-rotation 30` for mild rotation variants.

Train with a pose checkpoint and the pose config:

```bash
yolo pose train model=yolo11n-pose.pt data=configs/forceps_pose.yaml imgsz=1024
```

## Workflow

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
compression, and sharpening. Source images and labels are never modified.

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

The single-image bridge reads the four pose keypoints in order
`tip_left`, `tip_right`, `shadow_left`, `shadow_right`, merges those pixel
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
