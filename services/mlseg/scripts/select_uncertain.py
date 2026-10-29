"""Pick the frames of a video the model is least sure about and export them as LabelMe files (points pre-filled
with the model's prediction) for review/correction.  Uses run_video.py output (--out .npz)."""
import argparse, json, os, sys
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=40); ap.add_argument("--gap", type=int, default=20, help="min frames between picks")
ap.add_argument("--prefix", default="rev")
a = ap.parse_args()
d = np.load(a.npz); idx, P, S, D = d["idx"], d["pts"][:, :4], d["score"][:, :4], d["det"]
tool = (S[:, 0] > .5) | (S[:, 1] > .5) | (S[:, 2] > .2) | (S[:, 3] > .2)           # something to label
unsure = ((S > .1) & (S < .7)).sum(1)                                                 # peaks present but not confident
dis = np.nan_to_num(np.nanmax(np.where(np.isnan(D[..., 0]), np.nan, np.linalg.norm(P - D, axis=-1)), axis=1))
jump = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=-1).max(1)]
jump = np.where(jump > 60, 1, 0) * (S.min(1) > .3)                                    # confident yet teleporting
score = tool * (2 * unsure + np.clip(dis / 15, 0, 3) + 2 * jump)
order = np.argsort(-score); pick = []
for i in order:
    if score[i] <= 0 or len(pick) >= a.n: break
    if all(abs(int(idx[i]) - int(idx[j])) >= a.gap for j in pick): pick.append(i)
os.makedirs(a.out, exist_ok=True)
cap = cv2.VideoCapture(a.video); L = ("left_tip", "right_tip", "left_shadow", "right_shadow")
for i in sorted(pick, key=lambda i: idx[i]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx[i])); ok, f = cap.read()
    name = f"{a.prefix}_{int(idx[i]):06d}"; cv2.imwrite(f"{a.out}/{name}.png", f)
    shapes = [{"label": L[j], "score": float(S[i, j]), "points": [[float(P[i, j, 0]), float(P[i, j, 1])]], "group_id": None,
               "description": "", "difficult": bool(S[i, j] < .5), "shape_type": "point", "flags": {}, "attributes": {}, "kie_linking": []}
              for j in range(4)]
    json.dump({"version": "4.0.6", "flags": {}, "checked": False, "shapes": shapes, "imagePath": name + ".png",
               "imageData": None, "imageHeight": f.shape[0], "imageWidth": f.shape[1]}, open(f"{a.out}/{name}.json", "w"), indent=1)
print(len(pick), "frames ->", a.out, [int(idx[i]) for i in sorted(pick, key=lambda i: idx[i])])
