"""Light + forceps-trocar calibration from a recording with the rod (no clicking).

The rod is rigid, so its 3D length (each end reconstructed from its shadow, as the stream does) must stay constant;
every point must lie inside the eye; while the forceps carries the rod (rod and forceps tips move together, found in
2D) the tips should touch it in 3D. Fitted: light tip position, and the eye radius in px unless --eye-radius-px fixes
it. Left free, the radius absorbs the simulator's wide-field optics and comes out far too large (the whole scene shrinks
~3x against the real-size instruments of the viz), so it is fixed from a known size in the image: the optic disc
(~1.8 mm, ~85 px in rod.mp4 -> 0.021 mm/px -> 12 mm eye = ~600 px). The forceps trocar is the point all shaft axes (tip
midpoint -> junction) pass through. --camera-distance uses the stream's pinhole camera (tool_positions.py) instead of
the orthographic one: on rod.mp4 the rod's length spread drops from +-8 % to +-4 % at c = 2.2, and the shaft axes then
meet on the eye (a straight line stays straight in a pinhole image), at the pars plana.

python scripts/fit_calibration.py --video rod.mp4 --ckpt weights/last.pt --base ../../config/default_calibration.json \
    --out ../../config/rod_calibration.json --eye-radius-px 600 --contact-weight 0 --light-at-trocar 0.3 1.5 \
    --camera-distance 2.2 [--track runs/track.npz]
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
from scipy.optimize import minimize

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "src"), os.path.join(here, "..", "..", "detector", "src")]

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--base", required=True, help="calibration to start from (eye_radius_mm, jaw_length_mm are kept)")
ap.add_argument("--out", required=True)
ap.add_argument("--track", default="", help="cache of the per-frame detections (.npz)")
ap.add_argument("--step", type=int, default=2)
ap.add_argument("--starts", type=int, default=40)
ap.add_argument("--eye-radius-px", type=float, default=0.0, help="fix the eye radius (px of the work frame); 0: fit it")
ap.add_argument("--light-at-trocar", nargs=2, type=float, metavar=("UP_MIN", "UP_MAX"),
                help="the light sits at its trocar (chandelier / pipe barely inserted): fit only the trocar's angles, "
                     "rot_up within this range (radians from the straight-down axis; pars plana ~0.6-0.9)")
ap.add_argument("--camera-distance", type=float, default=0.0,
                help="pinhole camera this many eye radii from the eye centre (0: orthographic)")
ap.add_argument("--contact-weight", type=float, default=10.0,
                help="weight of tip-rod contact while carried (0: contact comes from the image, detector/contact.py)")
a = ap.parse_args()


def track():
    from detector.background import find_eye_disc
    from mlseg.detector import MLDetector
    det = MLDetector(a.ckpt, shadow_hold_s=0.0, rod_shadow_min_score=0.1)
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
    det.roi = find_eye_disc(cap.read()[1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    nan2 = [np.nan, np.nan]
    out = {k: [] for k in ("idx", "tips", "fsh", "junc", "rod", "rsh")}
    k = 0
    while cap.grab():
        if k % a.step == 0:
            d = det.detect(cap.retrieve()[1])
            out["idx"].append(k)
            out["tips"].append([d.forceps.left_tip or nan2, d.forceps.right_tip or nan2])
            out["fsh"].append([d.shadow.left_tip or nan2, d.shadow.right_tip or nan2])
            out["junc"].append(d.forceps.junction or nan2)
            out["rod"].append(list(d.rod.ends) if d.rod.ends else [nan2, nan2])
            out["rsh"].append(list(d.rod.shadow) if d.rod.shadow else [nan2, nan2])
        k += 1
    return {k: np.array(v, np.float64) for k, v in out.items()}


if a.track and os.path.exists(a.track):
    D = dict(np.load(a.track))
else:
    D = track()
    if a.track:
        np.savez(a.track, **D)
idx, T, F, J, R, RS = (D[k] for k in ("idx", "tips", "fsh", "junc", "rod", "rsh"))
base = json.load(open(a.base))
C = np.array([540.0, 540.0])                      # eye-circle normalised work frame
okR = ~(np.isnan(R).any((1, 2)) | np.isnan(RS).any((1, 2)))
okT = ~(np.isnan(T).any((1, 2)) | np.isnan(F).any((1, 2)))


def seg_dist(p, a_, b_):
    v = b_ - a_
    t = np.clip(((p - a_) * v).sum(-1) / np.maximum((v * v).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(p - (a_ + t[..., None] * v), axis=-1)


# carried: forceps tips on the rod (2D) while rod and tips move alike, for >= 10 frames
tipm = T.mean(1)
on = okR & okT & (np.nan_to_num(seg_dist(tipm, R[:, 0], R[:, 1]), nan=1e9) < 12)
vr = np.r_[0, np.linalg.norm(np.diff(R.mean(1), axis=0), axis=-1)]
vt = np.r_[0, np.linalg.norm(np.diff(tipm, axis=0), axis=-1)]
moving = on & (vr > 3) & (np.abs(vr - vt) < 0.5 * vt + 2)
carried = np.zeros(len(idx), bool)
i = 0
while i < len(idx):
    j = i
    while j < len(idx) and on[j]:
        j += 1
    if j - i >= 5 and moving[i:j].mean() > 0.5:
        carried[i:j] = True
    i = max(j, i + 1)
print(f"frames {len(idx)}: rod {okR.mean():.2f}, forceps+shadow {okT.mean():.2f}, carried {carried.sum()}")


CAM = a.camera_distance


def cam_ray(p, rpx):
    """on-axis pinhole ray, vectorised: origin (0, 0, -c), direction per pixel (the stream maps c onto its perspective camera, ws_server.camera_kwargs)"""
    u = (p - C) / rpx
    return np.concatenate([u / (1 + CAM), np.ones(u.shape[:-1] + (1,))], -1)


def rec(L, p, s, rpx):
    """tool_positions.reconstruct_tip_perspective for the on-axis pinhole, vectorised"""
    if CAM <= 0:
        sxy = (s - C) / rpx
        S3 = np.concatenate([sxy, np.sqrt(np.clip(1 - (sxy ** 2).sum(-1), 0, None))[..., None]], -1)
        ray = S3 - L
        t = (((p - C) / rpx - L[:2]) * ray[..., :2]).sum(-1) / np.maximum((ray[..., :2] ** 2).sum(-1), 1e-9)
        return L + t[..., None] * ray
    origin = np.array([0.0, 0.0, -CAM])
    d = cam_ray(s, rpx)
    dd = (d * d).sum(-1)
    S3 = origin + ((CAM + np.sqrt(np.clip(CAM * CAM - dd * (CAM * CAM - 1), 0, None))) / dd)[..., None] * d
    u, v, w0 = S3 - L, cam_ray(p, rpx), L - origin
    A, B, Cv, Dv, Ev = (u * u).sum(-1), (u * v).sum(-1), (v * v).sum(-1), (u * w0).sum(-1), (v * w0).sum(-1)
    return L + ((B * Ev - Cv * Dv) / np.maximum(A * Cv - B * B, 1e-12))[..., None] * u


sel, ts, cs = np.where(okR)[0][::3], np.where(okT)[0][::3], np.where(okR & okT & carried)[0]


DEPTH = 0.005                                     # light at its trocar: just inside the wall (eye radii)


def light(x):
    if a.light_at_trocar:
        up, clock = x[0], x[1]
        return (1 - DEPTH) * np.array([math.sin(up) * math.cos(clock), math.sin(up) * math.sin(clock), -math.cos(up)])
    return np.asarray(x[:3])


def full(x):
    return list(x[:3]) + [a.eye_radius_px if a.eye_radius_px else x[3]]


def parts(x):
    L, rpx = light(x), full(x)[3]
    rod = rec(L, R[sel], RS[sel], rpx)
    ln = np.linalg.norm(rod[:, 0] - rod[:, 1], axis=-1)
    tip = rec(L, T[ts], F[ts], rpx)
    rr = np.concatenate([np.linalg.norm(rod, axis=-1).ravel(), np.linalg.norm(tip, axis=-1).ravel()])
    contact = seg_dist(rec(L, T[cs], F[cs], rpx).mean(1), *np.moveaxis(rec(L, R[cs], RS[cs], rpx), 1, 0)) if len(cs) else np.zeros(1)
    return ln, rr, contact


def loss(x):
    if a.light_at_trocar:
        if not a.light_at_trocar[0] <= x[0] <= a.light_at_trocar[1]:
            return 1e3
    elif np.linalg.norm(x[:3]) > 0.9:
        return 1e3
    if not 300 < full(x)[3] < 3000:
        return 1e3
    ln, rr, contact = parts(x)
    lnl = np.log(np.clip(ln, 1e-4, None))
    dev = np.abs(lnl - np.median(lnl))
    # median spread, plus the share of frames whose rod is >30 % off (a median alone accepts a light position where some
    # frames blow up) and of points outside the eye
    return (np.median(dev) * 4 + np.mean(dev > np.log(1.3)) * 2 + np.mean(np.clip(rr - 0.98, 0, None)) * 20
            + np.mean(rr > 1.0) * 2 + np.median(contact) * a.contact_weight)


rng = np.random.default_rng(0)
best = None
for _ in range(a.starts):
    if a.light_at_trocar:
        x0 = [rng.uniform(*a.light_at_trocar), rng.uniform(-math.pi, math.pi), 0.0, rng.uniform(450, 2000)]
    else:
        v = rng.normal(size=3)
        x0 = list(v / np.linalg.norm(v) * rng.uniform(0.1, 0.85)) + [rng.uniform(450, 2000)]
    r = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-6})
    best = r if best is None or r.fun < best.fun else best
L, rpx = light(best.x), float(full(best.x)[3])
mm = base["eye_radius_mm"]
ln, rr, contact = parts(best.x)
print(f"light {L.round(3)} eye_radius_px {rpx:.0f} | rod length {np.median(ln) * mm:.2f} mm "
      f"(IQR +-{(np.percentile(ln, 75) - np.percentile(ln, 25)) / 2 / np.median(ln) * 100:.0f} %), inside eye "
      f"{(rr < 1).mean():.3f}, rod off >30 % {(np.abs(np.log(ln / np.median(ln))) > np.log(1.3)).mean():.3f}, carried tip-rod distance {np.median(contact) * mm:.2f} mm")

# forceps trocar: least-squares point on all shaft axes (robust: drop the worst 20 % three times)
okJ = okT & ~np.isnan(J).any(1)
m, j = T[okJ].mean(1), J[okJ]
v = m - j
keep = np.linalg.norm(v, axis=1) > 25
n = np.stack([-v[keep, 1], v[keep, 0]], 1) / np.linalg.norm(v[keep], axis=1)[:, None]
b = (n * j[keep]).sum(1)
w = np.ones(len(b), bool)
for _ in range(4):
    p = np.linalg.lstsq(n[w], b[w], rcond=None)[0]
    res = np.abs(n @ p - b)
    w = res < np.percentile(res, 80)
xy = (p - C) / rpx
if CAM > 0:
    # the shaft axes meet at the trocar's image: its camera ray hits the eye's front (or passes it: tangent point)
    d = cam_ray(p, rpx)
    dd = float(d @ d)
    s_hit = (CAM - math.sqrt(max(CAM * CAM - dd * (CAM * CAM - 1), 0.0))) / dd
    P3 = np.array([0.0, 0.0, -CAM]) + s_hit * d
    P3 /= np.linalg.norm(P3)


def light_entry(L, rot_up):
    """Light trocar at angle rot_up on the side of the tip, aimed through the tip -> (rot_up, clock, depth, tilt, aim
    clock) as tool_positions.light_tip_position takes them (depth in eye radii)."""
    clock = math.atan2(L[1], L[0])
    T = np.array([math.sin(rot_up) * math.cos(clock), math.sin(rot_up) * math.sin(clock), -math.cos(rot_up)])
    d = L - T
    depth = float(np.linalg.norm(d))
    d /= depth
    tilt = math.acos(float(np.clip(d @ -T, -1, 1)))
    e_up = np.array([math.cos(rot_up) * math.cos(clock), math.cos(rot_up) * math.sin(clock), math.sin(rot_up)])
    e_clock = np.array([-math.sin(clock), math.cos(clock), 0.0])
    return rot_up, clock, depth, tilt, math.atan2(float(d @ e_clock), float(d @ e_up))


# orthographic camera: trocars sit at about the limbus (6 mm of the 12 mm eye radius); with a realistic eye radius the
# axes' meeting point lands far outside the eye (the simulator's optics is wide-angle), so only its direction is used
LIMBUS = 0.5
LIMBUS_UP = math.asin(LIMBUS)
# the light pipe also enters near the limbus (placing its trocar straight behind the tip would put it at the side)
if a.light_at_trocar:
    l_up, l_clock, l_depth, l_tilt, l_aim = float(best.x[0]), float(best.x[1]), DEPTH, 0.0, 0.0
else:
    l_up, l_clock, l_depth, l_tilt, l_aim = light_entry(L, LIMBUS_UP)
out = {**base, "light_rot_up": round(l_up, 5), "light_rot_clock": round(l_clock, 5), "light_depth_mm": round(l_depth * mm, 3),
       "light_aim_tilt": round(l_tilt, 5), "light_aim_clock": round(l_aim, 5),
       "forceps_rot_up": round(math.acos(-float(P3[2])) if CAM > 0 else math.asin(min(float(np.linalg.norm(xy)), LIMBUS)), 5),
       "forceps_rot_clock": round(float(np.arctan2(xy[1], xy[0])), 5), "camera_distance": CAM,
       "eye_center_px": [540.0, 540.0], "eye_radius_px": round(rpx, 1)}
json.dump(out, open(a.out, "w"), indent=2)
print(f"trocar at {p.round(0)} px (shaft-axis residual {np.median(res):.0f} px) -> {a.out}")
