# mlseg – learned forceps/shadow keypoint model

Drop-in replacement for `services/detector` (same HTTP API, live worker, MJPEG, `/segment` for the orchestrator).
A ResNet34 + UNet predicts 6 heatmaps: `left_tip`, `right_tip`, `left_shadow`, `right_shadow`, `forceps_junction`,
`shadow_junction`. Gaussian targets make training tolerant of labels that are not pixel-perfect.

## Use
```bash
# weights: services/mlseg/weights/last.pt  (git-ignored, 86 MB)
docker compose --profile ml up -d --build segmenter      # :8002, then set the orchestrator detector URL to http://segmenter:8000
cd services/mlseg && MLSEG_CKPT=weights/last.pt pytest    # needs torch, timm, ../detector installed
```
Env: `MLSEG_CKPT`, `MLSEG_DEVICE` (cuda/mps/cpu, auto), `MLSEG_MIN_SCORE` (default 0.3: forceps tips/junction),
`MLSEG_SHADOW_MIN_SCORE` (default 0.1: shadow tips - low confidence there still means an accurate position, measured). Speed: ~10 ms/frame on GPU,
~500 ms on CPU (768 px input).

## Train (GPU server)
```bash
python scripts/gen_pseudo.py --ckpt <old.pt> --material material --detector-src ../detector/src --background data/background.png --out data/train6
python scripts/train.py --data data/train6 --out runs/x --pseudo 1 --epochs 60 [--fold 0 | --val-every 0]
python scripts/run_video.py --video v.mp4 --ckpt runs/x/last.pt --out runs/vid/v --detector-src ../detector/src --background data/background.png
```
`gen_pseudo.py` builds the set: hand labels (tips) + junctions from the classical detector where its tips agree within
3 px + confident model predictions on the videos + empty frames as negatives; uncertain keypoints are ignored in the loss.

## Results (41 hand-labelled frames, 1080² px)
| | mean tip error | note |
|---|---|---|
| 4-point model, 5-fold block CV | 1.1–1.5 px | PCK@5 = 100 % |
| 6-point model + pseudo-labels, folds 0 / 3 | 1.3–1.4 px | forceps junction ~1.6 px, shadow junction 2–5 px (vs detector) |

Container video (`forceps_lev1.mp4`, eye-circle normalised): the shadow was missed in 13.4 % of the frames where the
forceps was found; two rounds of hard-positive mining (model unsure, classical detector confirms the spot within 8 px,
`gen_pseudo.py --lo`) bring that to 4.9 % (3.4 % at score 0.3) with 0 false alarms on the 284 empty frames.

Further hard-positive sources (`gen_temporal.py`: confident before/after a gap; `gen_closed.py`: closed jaws, both tips on
the same point) and a plausibility filter on the detector's junction labels (a third were far down the shaft) lift the
shadow rate on `video.mp4` from 37.5 % to 48 % of forceps frames and on the container video to 98 %, 0 false alarms.
A ResNet50 was no better than the ResNet34 (kept: faster).

Shadow tips on the optic disc were the weak spot (the disc is bright yellow, few real examples). `relight.py` cuts the disc's
gain out of the empty background image and multiplies it onto training frames (a shadow darkens multiplicatively), mostly
at the shadow tips (`train.py --relight data/background.png`). Synthetic test (labelled frames, disc relit under the shadow
tips, held-out fold): shadow found 46 % -> 96 %, median error 11 -> 2.6 px; on the real videos, shadow centre < 100 px from
the disc: 92 -> 97 % (container video), 51 -> 71 % (`video.mp4`).

On `video.mp4` the model is correct where it disagrees with the classical detector (red objects, tool partly out of
view). Known gaps: closed jaws (only one tip visible -> no full pair), junction accuracy of the shadow, and
generalisation is only judged qualitatively outside the labelled video – label a few hard `video.mp4` frames to fix that.

## Active-learning loop (getting better with more data)
```bash
python scripts/run_video.py --video new.mp4 --ckpt runs/x/last.pt --out runs/vid/new --step 3 --detector-src ../detector/src --background data/background.png --clip 0
python scripts/select_uncertain.py --video new.mp4 --npz runs/vid/new.npz --out review/ --n 40   # LabelMe files, points pre-filled
# correct the points in labelme (move only what is wrong), copy the .json/.png next to material/frame_*.json, then
python scripts/gen_pseudo.py ... && python scripts/train.py ... --fold N      # compare against the CV numbers above
```
Keep a fixed, hand-labelled test set that is never trained on; only promote a new model if it beats the old one there.

## Shadow recall on `video.mp4` (hand-annotated presence set)
Without labels for `video.mp4` I annotated, by eye on raw frames, whether the forceps / its shadow is visible in every 60th
frame (`ann.py` kept on the server: 62 frames with a visible shadow, 8 tool frames without one, 9 empty). Measured recall of
"at least one shadow tip reported" (accuracy of the reported tips: 1-3 px vs. a few tips read off zoomed grid crops, n=5):

| model | shadow thr 0.3 | thr 0.1 |
|---|---|---|
| 6 pt model + hard positives (p12) | 56 % | 60 % |
| + disc relight + shadow relocation augmentation (p13) | 68 % | 79 % |
| + more verified hard positives, 3 rounds (p17) | 82 % | 90 % |
| + shadows of `video.mp4` itself in the relocation bank, 10 hand-read frames (p18, shipped) | 86 % (both tips 84 %) | 94 % (both tips 90 %) |

False alarms on empty frames stay at 0; on tool frames without a visible shadow 1-2 of 8. Held-out-block CV: ~1 px.
`shadow relocation` (`shadowreloc.py`): the real shadow of a labelled frame is divided out (empty background of that video)
and multiplied back next to the forceps tips, rotated/scaled, labels moved with it - this teaches "shadow tip a few px from the
forceps tip", the configuration of frames ~3000-4300 of `video.mp4`.

The service reports a shadow only when *both* tips pass the threshold (the "both tips" numbers are what it delivers).
Ten shadow tips were read off zoomed grid crops by hand (+-5 px) and added as labelled frames (`frame_m*`); they sit between the
test frames (frame ids = 30 mod 60), so the test set stays disjoint but its neighbours are in training.

Still missing (~6-10 %): very faint, thin shadow tips right next to the forceps tips in `video.mp4` frames ~3400-4300
(listed in `material/review_vid`, exported with `select_uncertain.py`). Self-training cannot reach them (the model gives no
trace there); a few corrected frames can. The classical detector is *not* a useful second opinion there (it confuses forceps
and shadow tips in about half of these frames).

## Known failure mode (open): lower shadow tip on the forceps prong
In some open-jaw frames of `video.mp4` (3150, 3210, 3330) the left (lower) shadow tip is reported ~35 px away: on the forceps'
own prong, or collapsed onto the right tip. The heatmap has the right peak too (p ~0.25) but a stronger wrong one. Hand-read labels
(`train19`, `frame_m*`, 24 frames, weight x20, `--real-weight 20 --repeat 20`) fix the frames the model has seen, but do not
carry to unseen ones (hold-out: 3 of 5 still ~35 px off), and a colour rule to tell prong from shadow does not separate them
(both reddish grey). Only more real labels of this configuration will fix it. `p20` (weight x20) is a candidate on the
server (`~/luis/segml/mlseg/runs/p20_full`): recall 87 % / 95 % (thr 0.3 / 0.1), less flicker, one extra false alarm on
`video.mp4`'s empty frames; its CV could not be compared cleanly (hand-read frames fall into one fold). Shipped: p18.
