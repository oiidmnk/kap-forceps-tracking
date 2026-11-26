"""3D accuracy on a simulator recording, with the simulator's trajectory log as ground truth.

    eval_sim_gt.py VIDEO [--csv LOG] [--detector ml|classic] [--ckpt W] [--out artifacts/sim_gt] [--reuse]

Replays VIDEO through the live chain - fixed eye circle -> 1080² normalised square -> detector -> calibration
(config/default_calibration.json, re-based to the circle like the orchestrator does) -> services/stream/tool_positions.py -
and compares the reconstructed forceps tip (midpoint of the two jaw tips) with the logged pose.

The log (`<video>_trajectories.csv`): `t_ms;TOPIC;16 floats` = 4x4 pose, column-major, mm, origin = eye centre.
Instrument poses live in the model-eye frame (the forceps axis, local z, passes through a fixed trocar); the forceps
origin is the midpoint of its jaw tips. Image point = pinhole(EYE pose · X). Neither the camera nor the video<->log
offset is assumed: both are fitted to the detected forceps tips (12 parameters, thousands of frames). The camera also
gives the image-aligned frame the pipeline works in, so GT and reconstruction are compared there.

Attribution: the same reconstruction is also fed (a) the GT tip + a modelled shadow (light at the lamp trocar, retina =
eye sphere) -> error of the 3D model alone, (b) the GT tip + the detected shadow -> adds shadow detection.
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("services/mlseg/src", "services/detector/src", "services/stream"):
    sys.path.insert(0, os.path.join(ROOT, p))
import tool_positions as tp  # noqa: E402
from detector.background import find_eye_disc  # noqa: E402
from detector.config import Settings  # noqa: E402
from detector.pipeline import Detector  # noqa: E402

WORK = 1080                       # detector work size (normalised square)
FORCEPS, EYE, LAMP = "VRTOPIC_ENDGRIPPINGFORCEPS", "VRTOPIC_EYE", "VRTOPIC_LAMP"


# --- trajectory log ---------------------------------------------------------------------------------------------
def load_log(path):
    tracks = {}
    for line in open(path):
        p = line.strip().split(";")
        if len(p) == 18:                                      # other rows: VROBJECT_INSERT / _REMOVE events
            tracks.setdefault(p[1], []).append([float(v) for v in [p[0]] + p[2:]])
    out = {}
    for k, rows in tracks.items():
        a = np.array(rows)[::-1]                              # duplicate time stamps: keep the last row
        t, i = np.unique(a[:, 0], return_index=True)
        out[k] = (t, a[i, 1:].reshape(-1, 4, 4).transpose(0, 2, 1))
    return out


def pose_at(track, T):
    """Poses at times T (ms): linear blend of the neighbours, rotation re-orthonormalised."""
    t, M = track
    T = np.clip(T, t[0], t[-1])
    j = np.clip(np.searchsorted(t, T), 1, len(t) - 1)
    w = ((T - t[j - 1]) / np.maximum(t[j] - t[j - 1], 1e-9))[:, None, None]
    P = (1 - w) * M[j - 1] + w * M[j]
    u, _, vt = np.linalg.svd(P[:, :3, :3])
    P[:, :3, :3] = u @ vt
    return P


def trocar(M):
    """Point closest to all instrument axes (local z) - the remote centre of motion."""
    d = M[:, :3, 2]
    Q = np.eye(3) - d[:, :, None] * d[:, None, :]
    return np.linalg.solve(Q.sum(0), (Q @ M[:, :3, 3, None]).sum(0)[:, 0])


def apply(M, X):
    return np.einsum("nij,nj->ni", M[:, :3, :3], X) + M[:, :3, 3]


# --- camera ----------------------------------------------------------------------------------------------------
def dlt(X, uv):
    """3x4 projection matrix (normalised DLT) with uv ~ P [X, 1]."""
    mx, sx, mu, su = X.mean(0), X.std(), uv.mean(0), uv.std()
    Xn, un = np.c_[(X - mx) / sx, np.ones(len(X))], (uv - mu) / su
    A = np.zeros((2 * len(X), 12))
    A[0::2, :4], A[1::2, 4:8] = Xn, Xn
    A[0::2, 8:], A[1::2, 8:] = -un[:, :1] * Xn, -un[:, 1:] * Xn
    Pn = np.linalg.svd(A, full_matrices=False)[2][-1].reshape(3, 4)
    T = np.diag([1 / sx] * 3 + [1.0])
    T[:3, 3] = -mx / sx
    S = np.array([[su, 0, mu[0]], [0, su, mu[1]], [0, 0, 1]])
    return S @ Pn @ T


def project(P, X):
    p = np.c_[X, np.ones(len(X))] @ P.T
    return p[:, :2] / p[:, 2:]


def fit_camera(X, uv, iters=6):
    keep = np.ones(len(X), bool)
    for _ in range(iters):
        P = dlt(X[keep], uv[keep])
        r = np.linalg.norm(project(P, X) - uv, axis=1)
        keep = r < max(3 * np.median(r), 4.0)
    return P, r


def decompose(P):
    """-> K, R (world->camera rotation), camera centre."""
    if np.linalg.det(P[:, :3]) < 0:
        P = -P
    q, r = np.linalg.qr(np.flipud(P[:, :3]).T)
    K, R = np.flipud(r.T)[:, ::-1], np.flipud(q.T)
    D = np.diag(np.sign(np.diag(K)))
    K, R = K @ D, D @ R
    t = np.linalg.solve(K, P[:, 3])
    return K / K[2, 2], R, -R.T @ t


# --- detection pass ----------------------------------------------------------------------------------------------
def make_detector(kind, ckpt):
    if kind == "classic":                                     # single-frame fallback, no background model
        return Detector(Settings(background_path=""))
    from mlseg.detector import MLDetector
    return MLDetector(ckpt, Settings(background_path=""), device=os.environ.get("MLSEG_DEVICE") or None,
                      min_score=float(os.environ.get("MLSEG_MIN_SCORE") or 0.3),
                      shadow_min_score=float(os.environ.get("MLSEG_SHADOW_MIN_SCORE") or 0.1))


def auto_circle(video, every=15):
    """What 'Auto-detect' in the orchestrator would give, made robust: median over the recording."""
    cap, found, i = cv2.VideoCapture(video), [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % every == 0:
            _, im = cap.retrieve()
            d = find_eye_disc(im)
            if d:
                found.append(d)
        i += 1
    return tuple(float(v) for v in np.median(found, 0))


def detect_all(video, det, circle):
    """-> fps, rows [frame, forceps l/r, shadow l/r (x, y) in the normalised square; NaN = not reported]."""
    det.roi = circle
    cap, rows, i, t0 = cv2.VideoCapture(video), [], 0, time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, im = cap.read()
        if not ok:
            break
        d = det.detect(im)
        rows.append([i] + [v for pose in (d.forceps, d.shadow) for k in ("left_tip", "right_tip")
                           for v in (getattr(pose, k) if pose.ok else (np.nan, np.nan))])
        if i % 500 == 0:
            print(f"  frame {i}  {time.time() - t0:.0f} s", flush=True)
        i += 1
    return fps, np.array(rows)


# --- evaluation --------------------------------------------------------------------------------------------------
def reconstruct(cal, tips_l, tips_r, sh_l, sh_r):
    """Stream reconstruction per frame -> tip midpoint in mm (NaN where the stream would fail)."""
    out = np.full((len(tips_l), 3), np.nan)
    keys = ("light_rot_up", "light_rot_clock", "light_depth_mm", "forceps_rot_up", "forceps_rot_clock",
            "eye_radius_mm", "jaw_length_mm")
    for i, (a, b, c, d) in enumerate(zip(tips_l, tips_r, sh_l, sh_r)):
        try:
            p = tp.compute_tool_positions(**{k: cal[k] for k in keys}, light_aim_tilt=cal.get("light_aim_tilt", 0.0),
                                          light_aim_clock=cal.get("light_aim_clock", 0.0), left_tip_px=tuple(a),
                                          right_tip_px=tuple(b), left_shadow_px=tuple(c), right_shadow_px=tuple(d),
                                          eye_center_px=(WORK / 2, WORK / 2), eye_radius_px=WORK / 2)
            out[i] = (np.array(p["left_tip_forceps"]) + p["right_tip_forceps"]) / 2 * cal["eye_radius_mm"]
        except ValueError:
            pass
    return out


def summary(name, rec, gt, r_eye):
    ok = np.isfinite(rec).all(1)
    rec, gt = rec[ok], gt[ok]
    e3, lat, dz = np.linalg.norm(rec - gt, axis=1), np.linalg.norm(rec[:, :2] - gt[:, :2], axis=1), rec[:, 2] - gt[:, 2]
    dh = (r_eye - np.linalg.norm(rec, axis=1)) - (r_eye - np.linalg.norm(gt, axis=1))
    q = lambda v: "%5.2f %5.2f %5.2f" % tuple(np.percentile(v, [50, 75, 95]))
    print(f"{name:34s} n={ok.sum():5d} | 3D {q(e3)} | lateral {q(lat)} | depth {q(np.abs(dz))} "
          f"| height {q(np.abs(dh))}  bias {np.median(dh):+5.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--csv", help="trajectory log (default: <video stem>_trajectories.csv)")
    ap.add_argument("--detector", choices=("ml", "classic"), default="ml")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "services/mlseg/weights/last.pt"))
    ap.add_argument("--calibration", default=os.path.join(ROOT, "config/default_calibration.json"))
    ap.add_argument("--circle", help="eye circle cx,cy,r in video px (default: auto-detected, median over the video)")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts/sim_gt"))
    ap.add_argument("--reuse", action="store_true", help="reuse <out>_<detector>_detections.csv from an earlier run")
    a = ap.parse_args()
    log = load_log(a.csv or os.path.splitext(a.video)[0] + "_trajectories.csv")
    cal = json.load(open(a.calibration))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # 1. detections (normalised square), eye circle
    det_path = f"{a.out}_{a.detector}_detections.csv"
    circle = tuple(map(float, a.circle.split(","))) if a.circle else None
    if a.reuse and os.path.isfile(det_path):
        with open(det_path) as f:
            head = f.readline().strip().split(",")
            fps, circle = float(head[1]), tuple(map(float, head[3:6]))
        rows = np.loadtxt(det_path, delimiter=",", skiprows=2)
    else:
        circle = circle or auto_circle(a.video)
        print(f"eye circle {circle[0]:.1f},{circle[1]:.1f},{circle[2]:.1f}  detector {a.detector} ...", flush=True)
        fps, rows = detect_all(a.video, make_detector(a.detector, a.ckpt), circle)
        with open(det_path, "w") as f:
            f.write(f"fps,{fps},circle,{circle[0]},{circle[1]},{circle[2]}\n")
            f.write("frame,forceps_left_x,forceps_left_y,forceps_right_x,forceps_right_y,"
                    "shadow_left_x,shadow_left_y,shadow_right_x,shadow_right_y\n")
            np.savetxt(f, rows, delimiter=",", fmt="%.2f")
    frame, fl, fr, sl, sr = rows[:, 0], rows[:, 1:3], rows[:, 3:5], rows[:, 5:7], rows[:, 7:9]
    cx, cy, r = circle
    x0, y0, side = round(cx - r), round(cy - r), max(2, round(2 * r))     # as Detector.normalize
    to_raw = lambda n: np.array([x0, y0]) + n * side / WORK
    to_norm = lambda u: (u - [x0, y0]) * WORK / side

    # 2. video <-> log offset and camera, from the detected forceps tip midpoints
    tf = log[FORCEPS][0]
    mid_raw = to_raw((fl + fr) / 2)

    def gt_tip(lag):
        T = frame * 1000 / fps + lag
        sel = (T > tf[0] + 300) & (T < tf[-1] - 300)
        return sel, T

    def residual(lag, iters=3):
        sel, T = gt_tip(lag)
        sel &= np.isfinite(mid_raw).all(1)
        X = apply(pose_at(log[EYE], T[sel]), pose_at(log[FORCEPS], T[sel])[:, :3, 3])
        P, res = fit_camera(X, mid_raw[sel], iters)
        return np.median(res), P, res

    coarse = min((residual(lag)[0], lag) for lag in np.arange(-1000, 1001, 1000 / fps))[1]
    lag = min((residual(lag)[0], lag) for lag in np.arange(coarse - 40, coarse + 41, 2))[1]
    med, P, res = residual(lag, 6)
    K, Rc, C = decompose(P)
    print(f"log offset {lag:+.0f} ms | camera f={K[0, 0]:.0f}/{K[1, 1]:.0f} px, centre ({K[0, 2]:.0f},{K[1, 2]:.0f}), "
          f"skew {K[0, 1]:.2f}, {np.linalg.norm(C):.1f} mm from the eye centre | forceps tip reprojection "
          f"median {med:.2f} px, p95 {np.percentile(res, 95):.2f} px (fit on {len(res)} frames)")

    # 3. ground truth per frame, in the pipeline's frame (image axes, origin = eye centre, mm)
    sel, T = gt_tip(lag)
    E, F = pose_at(log[EYE], T), pose_at(log[FORCEPS], T)
    tip_w = F[:, :3, 3]
    gt = np.einsum("ij,njk,nk->ni", Rc, E[:, :3, :3], tip_w)
    r_eye = float(np.linalg.norm(trocar(log[FORCEPS][1])))
    gt_raw = project(P, apply(E, tip_w))
    gt_n = to_norm(gt_raw)
    visible = sel & (np.linalg.norm(gt_n - WORK / 2, axis=1) < 0.94 * WORK / 2)
    sh_n = np.full_like(gt_n, np.nan)
    if LAMP in log:                                           # shadow model: light at the lamp trocar, retina = eye sphere
        L = trocar(log[LAMP][1])
        d = tip_w - L
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        b = d @ L
        sh_w = L + (-b + np.sqrt(np.maximum(b ** 2 - (L @ L - r_eye ** 2), 0)))[:, None] * d
        sh_n = to_norm(project(P, apply(E, sh_w)))

    # 4. report
    f_ok, s_ok = np.isfinite(fl).all(1), np.isfinite(sl).all(1)
    e_f = np.linalg.norm((fl + fr) / 2 - gt_n, axis=1)
    e_s = np.linalg.norm((sl + sr) / 2 - sh_n, axis=1)
    print(f"\nframes with the forceps tip in view: {visible.sum()}  (eye radius from the trocar sphere: {r_eye:.2f} mm)")
    print(f"forceps tips found {100 * (f_ok & visible).sum() / visible.sum():5.1f} %  tip-midpoint error "
          f"median {np.nanmedian(e_f[visible]):.1f} px, p95 {np.nanpercentile(e_f[visible], 95):.1f} px (normalised 1080²)")
    print(f"shadow tips  found {100 * (s_ok & visible).sum() / visible.sum():5.1f} %  vs. shadow model "
          f"median {np.nanmedian(e_s[visible]):.1f} px, p95 {np.nanpercentile(e_s[visible], 95):.1f} px (model is approximate)")
    print(f"\n{'mm (median p75 p95)':34s} {'':7s}   {'3D':17s}   {'lateral':21s}   {'depth':19s}   height above retina")
    full = visible & f_ok & s_ok
    rec_pipe = reconstruct(cal, fl[full], fr[full], sl[full], sr[full])
    rec_model = reconstruct(cal, gt_n[visible], gt_n[visible], sh_n[visible], sh_n[visible])
    rec_shadow = reconstruct(cal, gt_n[full], gt_n[full], sl[full], sr[full])
    summary("(a) GT tip + modelled shadow", rec_model, gt[visible], r_eye)
    summary("(b) GT tip + detected shadow", rec_shadow, gt[full], r_eye)
    summary(f"(c) pipeline ({a.detector})", rec_pipe, gt[full], r_eye)
    print(f"    stream would report tips in {100 * np.isfinite(rec_pipe).all(1).sum() / visible.sum():.1f} % "
          f"of the frames with the tip in view")

    out = f"{a.out}_{a.detector}_frames.csv"
    rec_all = np.full_like(gt, np.nan)
    rec_all[full] = rec_pipe
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_log_ms", "visible", "gt_x", "gt_y", "gt_z", "rec_x", "rec_y", "rec_z", "err_3d_mm",
                    "height_gt_mm", "height_rec_mm", "forceps_err_px", "shadow_err_px"])
        for i in np.flatnonzero(sel):
            w.writerow([int(frame[i]), round(T[i], 1), int(visible[i]), *np.round(gt[i], 3), *np.round(rec_all[i], 3),
                        round(float(np.linalg.norm(rec_all[i] - gt[i])), 3), round(r_eye - float(np.linalg.norm(gt[i])), 3),
                        round(r_eye - float(np.linalg.norm(rec_all[i])), 3), round(float(e_f[i]), 2), round(float(e_s[i]), 2)])
    print(f"\nper frame: {out}")


if __name__ == "__main__":
    main()
