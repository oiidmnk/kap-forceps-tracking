# mlseg – learned forceps/shadow keypoint model

Drop-in replacement for `services/detector` (same HTTP API, live worker, MJPEG, `/segment` for the orchestrator).
A ResNet34 + UNet predicts 6 heatmaps: `left_tip`, `right_tip`, `left_shadow`, `right_shadow`, `forceps_junction`,
`shadow_junction`, plus (rod models, see [Rod](#rod-red-capsule-and-its-shadow)) `rod_end` and `rod_shadow_end`.
Gaussian targets make training tolerant of labels that are not pixel-perfect.

## Use
```bash
# weights: services/mlseg/weights/last.pt  (git-ignored, 86 MB) - `make weights` fetches them from musicanalyzer.cc,
#          `make publish-weights M="note"` publishes a new version (see scripts/publish_model.sh)
docker compose --profile ml up -d --build segmenter      # :8002, then set the orchestrator detector URL to http://segmenter:8000
cd services/mlseg && MLSEG_CKPT=weights/last.pt pytest    # needs torch, timm, ../detector installed
```
Env: `MLSEG_CKPT`, `MLSEG_DEVICE` (cuda/mps/cpu, auto), `MLSEG_MIN_SCORE` (default 0.3: forceps tips/junction),
`MLSEG_SHADOW_MIN_SCORE` (default 0.1: shadow tips - low confidence there still means an accurate position, measured),
`MLSEG_ROD_MIN_SCORE` (0.3) / `MLSEG_ROD_SHADOW_MIN_SCORE` (0.2) for the rod, `MLSEG_SHADOW_HOLD_S` (1.0: forceps tips seen but
not their shadow -> keep the last tip-to-shadow offset this long, reported as shadow reason `held`, thin rings in the live
view). Speed: ~10 ms/frame on GPU,
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
`video.mp4`'s empty frames; its CV could not be compared cleanly (hand-read frames fall into one fold). Shipped before the rod: p18 (now r4, see Rod).

## Rod (red capsule) and its shadow
Rod models (8 channels) add `rod_end` and `rod_shadow_end`: the rod is a symmetric capsule, so both ends share one
heatmap (two peaks, decoded as the two strongest peaks >= 6 heatmap px apart), the same for the two ends of its shadow.
`MLDetector` pairs the shadow ends to the rod ends (both ends of a shadow are shifted by about the same offset; the wrong
pairing differs by ~2x the rod), the live worker pushes `rod_a_px`/`rod_b_px` + `rod_a_shadow_px`/`rod_b_shadow_px`,
and the stream reconstructs each end from its shadow like a forceps tip (`rod_a`/`rod_b`). 6-channel checkpoints still
work (no rod). Labels and checks for `data/video/rod.mp4` (7631 frames, 1080², rod in almost every frame):

* **rod ends** - colour finder (`rodcv.find_rod`: G/R < 0.13, the rod is the only saturated red), ignored when an end
  touches the forceps (it may continue underneath, a third of the frames).
* **rod-shadow ends** - a global search confuses the dark capsule with the forceps shaft/shadow, so: every 79th frame
  read by eye on grid crops (+-15 px), snapped by `rodcv.fit_shadow` (capsule template on a vessel-free darkness map,
  rod/forceps pixels unknown; cap apex = half-contrast point on the axis profile); 65 frames in `labels/rod_shadows.json`
  (shadow hidden or merged -> not labelled). Then self-training (`gen_rod_selftrain.py`): a predicted shadow becomes a
  label only if the independent capsule fit lands within 5 px of both predicted ends (52 -> 799 -> 1074 frames).
* **forceps rows** in rod frames from p18 (confident -> present / clearly nothing -> absent / else ignored); the
  forceps-only sets get the rod rows as absent.

```bash
python scripts/gen_rod.py --video rod.mp4 --ckpt p18.pt --shadows labels/rod_shadows.json [--extra extra.json] \
    --out data/rod --val-out data/rod_val --holdout 2400:3200 6000:6800
python scripts/train.py --data data/train18 --pseudo 1 ... --init p18.pt --rod data/rod --rod-val data/rod_val
python scripts/gen_rod_selftrain.py --ckpt runs/x/last.pt --video rod.mp4 --holdout 2400:3200 6000:6800 --out extra.json
python scripts/eval_rod.py --ckpt runs/x/last.pt --val data/rod_val --video rod.mp4 --holdout 2400:3200 6000:6800
```

Held out (never trained on, not even as pseudo labels): frames 2400-3200 and 6000-6800 of `rod.mp4`.
Rod accuracy vs. the colour finder (67 frames), rod-shadow accuracy vs. the 13 hand-read frames of the held-out blocks;
"rod + shadow" = share of every 5th held-out frame (320) with both found (the shadow is clearly visible in only ~65 % of
them, by eye on 20 annotated frames: otherwise under the rod/forceps or merged with the forceps shadow); "fit agrees" =
found shadows where an independent capsule fit lands within 5 px of both predicted ends (a lower bound, the fit fails
where the shadow runs under the rod/forceps).

| model | rod median / max | rod found (320 frames) | shadow found / median / <=5 px | rod + shadow thr 0.1 / 0.2 | fit agrees |
|---|---|---|---|---|---|
| r1: p18 + 52 hand-read shadows | 1.1 / 6.1 px | 82 % | 77 % / 3.5 px / 80 % | 63 / 46 % | 42 % |
| r2: + 799 self-trained shadows | 1.0 / 6.1 px | 85 % | 85 % / 2.9 px / 91 % | 70 / 57 % | 56 % |
| r3: + 1074 self-trained shadows | 1.2 / 5.3 px | 86 % | 85 % / 2.5 px / 86 % | 72 / 64 % | 63 % |
| r4: + overlap (below), shipped | 1.3 / 5.7 px | 84 % | 100 % / 2.3 px / 85 % | 76 / 69 % | 59 % |

"Rod found" misses are mostly a rod held by the forceps (one end hidden: no full rod is reported).

**Forceps shadow inside the rod shadow.** While the forceps grasps the rod, its shadow tips lie in the rod's shadow;
two shadows of one light merge, so the tips are not visible at all and nobody can label them. r3 lost the forceps
shadow there (and, without a hold, the whole forceps): on held-out rod frames with the forceps found, its shadow was
found in 49 %. Two fixes:
* `MLDetector` keeps the last tip-to-shadow offset for `MLSEG_SHADOW_HOLD_S` (1 s) when only the shadow is missing.
* r4 learns to infer the hidden tips from the visible shaft shadow: `--overlay 0.6` lays a real rod shadow
  (`rodshadow.py`, cut out of the darkness map, merged as the darker of both, labels stay at the true tips) across the
  forceps shadow tips of forceps-only frames; plus forceps-shadow labels across short gaps (`gen_rod_temporal.py`, 123).

`eval_overlap.py` on the held-out blocks - synthetic: rod shadows of held-out labelled frames laid over frames where
r3 sees the forceps shadow clearly (139, truth = r3 there; same kind of overlay as in training, so optimistic);
real: frames with the forceps tips within 60 px of a found rod shadow:

| model | synthetic found / median / <=5 px | real overlap: shadow found | all forceps frames: shadow found |
|---|---|---|---|
| r3 | 61 % / 22.5 px / 48 % | 46 % (84) | 49 % |
| r4a (`--overlay 0.3`) | 97 % / 1.7 px / 77 % | 69 % (122) | 63 % |
| r4b (`--overlay 0.6`, shipped) | 99 % / 1.6 px / 81 % | 74 % (123) | 64 % |

Forceps on the old footage (same scripts as above), p18 / r3 / r4b: `video.mp4` presence set thr 0.1 both shadow tips
0.90 / 0.87 / 0.92, thr 0.3 0.84 / 0.77 / 0.84, false shadow on empty frames 0.11 / 0.00 / 0.00; `video.mp4` shadow
found (thr 0.1) 0.79 / 0.77 / 0.82 with on/off switches 116 / 94 / 60; container video unchanged (0.996, 0 false
alarms). p18 and r3 stay available as `weights/p18.pt`, `weights/r3.pt`.

**3D check against the video and calibration** (`scripts/fit_calibration.py` -> `config/rod_calibration.json`).
*Superseded scale and camera, see "Scale and camera" below: the free fit shrinks the scene ~3x.*
With `config/default_calibration.json` (made for the forceps footage) the 3D scene of `rod.mp4` is wrong: only 3 % of the
rod ends inside the eye, the rigid rod's length varies 10-20 mm, and while the forceps visibly carries the rod (rod and
tips move together, frames 3040-3062 and 3968-4034) tips and rod are 1.8 mm apart. The script tracks the video with the
live detector, fits the light tip and the eye radius in px (constant rod length, everything inside the eye, contact
while carried) and the forceps trocar (the point all shaft axes pass through). Result: rod length +-5 %, all points inside
the eye, tip-to-rod 0.10 / 0.25 mm in the two carried stretches (calibration units, eye 24 mm); fitted on the first
half of the video only, the carried stretch of the second half still comes out at 0.25 mm. In the viz (eye 12 mm) the
tips are 0.05-0.09 mm from the rod axis there, i.e. touching it; far apart in the image -> ~9 mm apart in 3D.
Limits: the light pipe is hand-held and moves (its image position, from object->shadow lines, wanders by up to ~400 px
between 10 s windows), so one calibration is a compromise; the light's depth is only weakly determined (two fits differ,
contact barely changes); the stream's camera is orthographic, the simulator's wide-field optics is not - the fit absorbs
that in a large eye radius (1875 px for the 1080 px circle), so absolute sizes are smaller than the real instruments.

**Scale and camera.** Left free, the eye radius absorbs the wide-field optics (1875 px) and the reconstructed scene is
~3x too small next to the real-size instruments of the viz (rod 1.1 x 0.15 mm beside a 0.64 mm shaft). The optic disc
gives the image scale: ~85 px for ~1.8 mm -> 0.021 mm/px -> the 12 mm eye is ~600 px (`--eye-radius-px 600`). The
stream's orthographic camera does not fit a wide-angle fundus view: the rod's reconstructed length spreads +-8 % and the
forceps shaft axes meet outside the eye. A pinhole camera c eye radii from the eye centre (`camera_distance`, optional
in the stream and the orchestrator, 0 = orthographic as before) fits much better; scanning c on rod.mp4 (light at its
trocar, refitted for each c):

| c | rod length IQR | rod within +-30 % | light trocar | forceps trocar (shaft axes) |
|---|---|---|---|---|
| orthographic | +-8 % | 91 % | 57 deg | outside the eye |
| 2.0 | +-5 % | 94 % | 45 deg | 41 deg |
| **2.2** | **+-4 %** | **93 %** | **33 deg** | **52 deg** |
| 2.6 | +-3 % | 94 % | 39 deg | misses the eye |

(sampled frames; angles from the straight-down axis, pars plana ~45-50 deg). `config/rod_calibration.json` uses c = 2.2
(`--camera-distance 2.2 --light-at-trocar 0.3 1.5 --contact-weight 0`): both trocars come out at plausible places
without clamping, and over all 1783 tracked frames the rod is within +-30 % of its median length in 91 % (orthographic:
88 %), 0.3 % of its ends below the retina. The reconstructed rod is 2.9 mm long (`ROD_LENGTH_MM` in the viz): it floats
well above the retina, so the image (optic-disc scale, ~3.9 mm) shows it enlarged.

**Contact decided in the image** (`detector/contact.py`, independent of the calibration). The live worker flags
`rod_contact` when the forceps tips lie on the rod and either the forceps shadow meets the rod's shadow at the same place
along the rod (same image point + same shadow -> same 3D point; within 1.3 rod widths) or the rod moves along with the
tips (it floats still unless pushed). Two frames in a row start it; it is kept while the shadow is merged into the rod's
shadow (grasping) and ends when the tips leave the rod or the shadows separate (2.5 widths). On contact the stream moves
the rod (as a whole) so its axis passes through the tips' midpoint, and the viz makes the rod glow ("Forceps touches
rod" in the distance panel) - a hit in the video is a hit in the viz with any calibration. On `rod.mp4` it flags 32
stretches: every stretch in which the forceps pushes, turns or carries the rod (checked frame by frame), no isolated
frames; a tip that lies on the rod in the image but hovers above it (its shadow elsewhere) is not flagged.

## Orientation of the source (capture cards)
The 3D stage (`services/stream`) is calibrated for the simulator's own orientation (trocars measured from its trajectory log,
see the top-level README): Haag-Streit logo upright bottom left, instruments from the upper left, shadow down/left, optic disc
right of the centre. The capture card delivers exactly that, so run it with `ROT=0`. The model needs no rotation either: on
`mapskap1_…11-00-43` (native orientation) it finds the forceps tips in 99 % and the shadow in 86 % of the frames with the tip in
view (`make sim-eval`). Footage turned by 180° against that (logo upside down, instruments from the lower right, like the old
sample `forceps_lev1.mp4`) gives a tip *behind* the retina ("0.00 mm critical proximity") although the 2D points are right:
`make ml-cam ROT=180` (`ROTATE=180` for the detector/segmenter, applied before the eye circle, preview and autocircle), then
`make circle`.
