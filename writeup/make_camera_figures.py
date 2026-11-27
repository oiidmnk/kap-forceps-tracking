"""Figures and numbers for the 3D sections of the report (report/sections/reconstruction.tex, eval_3d.tex).

Run from the repository root with the ML venv (no detector run: it reuses the cached detections of
scripts/eval_sim_gt.py in artifacts/sim_gt_2026-11-09_<run>_ml_detections.csv):

    .venv-ml/bin/python writeup/make_camera_figures.py

For each run of 2026-11-09 in RUNS it fits camera and log offset as eval_sim_gt.py does, measures the calibration
the run implies, and reconstructs the tip with the calibration averaged from the *other* runs (leave-one-run-out),
once with the perspective camera and once with the orthographic model (same trocars and light, camera fields removed).
Inputs: (a) ground-truth tip + modelled shadow (limit of the 3D model), (c) detected tips and shadows (pipeline).
Writes writeup/figures/camera/*.png and stats.json, and the same figures as PDF to report/figures/camera/.
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_sim_gt as E  # noqa: E402

# 11-14-13 is left out: its shadow detection fails in 21 % of the frames with a 3D result (3-4 % in the others)
RUNS = ["11-25-32", "11-32-14"]
DATA = os.path.join(ROOT, "TestRuns/2026_11_09-exported data")
OUT = os.path.join(ROOT, "writeup/figures/camera")
REPORT_FIGS = os.path.join(ROOT, "report/figures/camera")
CAMERA_KEYS = ("camera_focal_px", "camera_pp_px", "camera_center_mm")

LABEL = {"perspective": "perspective", "orthographic": "orthographic (eye circle)"}
PERSP, ORTHO, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def log_path(run, suffix):
    stem = f"mapskap1_2026-11-09_{run}+0100_forcepstraining_lev2"
    return os.path.join(DATA, stem, stem + suffix)


def evaluate_run(run, base):
    """Ground truth, detections and the measured calibration of one run (steps 1-3 of eval_sim_gt.main)."""
    log = E.load_log(log_path(run, "_trajectories.csv"))
    det_path = os.path.join(ROOT, f"artifacts/sim_gt_2026-11-09_{run}_ml_detections.csv")
    with open(det_path) as f:
        head = f.readline().strip().split(",")
    fps, circle = float(head[1]), tuple(map(float, head[3:6]))
    rows = np.loadtxt(det_path, delimiter=",", skiprows=2)
    frame, fl, fr, sl, sr = rows[:, 0], rows[:, 1:3], rows[:, 3:5], rows[:, 5:7], rows[:, 7:9]
    cx, cy, r = circle
    x0, y0, side = round(cx - r), round(cy - r), max(2, round(2 * r))
    to_raw = lambda n: np.array([x0, y0]) + n * side / E.WORK
    to_norm = lambda u: (u - [x0, y0]) * E.WORK / side

    tf = log[E.FORCEPS][0]
    mid_raw = to_raw((fl + fr) / 2)

    def gt_tip(lag):
        T = frame * 1000 / fps + lag
        return (T > tf[0] + 300) & (T < tf[-1] - 300), T

    def residual(lag, iters=3):
        sel, T = gt_tip(lag)
        sel &= np.isfinite(mid_raw).all(1)
        X = E.apply(E.pose_at(log[E.EYE], T[sel]), E.pose_at(log[E.FORCEPS], T[sel])[:, :3, 3])
        P, res = E.fit_camera(X, mid_raw[sel], iters)
        return np.median(res), P, res

    coarse = min((residual(lag)[0], lag) for lag in np.arange(-1000, 1001, 1000 / fps))[1]
    lag = min((residual(lag)[0], lag) for lag in np.arange(coarse - 40, coarse + 41, 2))[1]
    med, P, res = residual(lag, 6)
    K, Rc, C = E.decompose(P)

    sel, T = gt_tip(lag)
    Ep, F = E.pose_at(log[E.EYE], T), E.pose_at(log[E.FORCEPS], T)
    tip_w = F[:, :3, 3]
    gt = np.einsum("ij,njk,nk->ni", Rc, Ep[:, :3, :3], tip_w)
    r_eye = float(np.linalg.norm(E.trocar(log[E.FORCEPS][1])))
    gt_n = to_norm(E.project(P, E.apply(Ep, tip_w)))
    visible = sel & (np.linalg.norm(gt_n - E.WORK / 2, axis=1) < 0.94 * E.WORK / 2)
    L = E.pose_at(log[E.LAMP], T)[:, :3, 3]
    d = tip_w - L
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    b = (d * L).sum(1)
    sh_w = L + (-b + np.sqrt(np.maximum(b ** 2 - ((L * L).sum(1) - r_eye ** 2), 0)))[:, None] * d
    sh_n = to_norm(E.project(P, E.apply(Ep, sh_w)))

    measured = {**base, **E.measured_calibration(log, T[sel], Rc, C, K, r_eye)}
    full = visible & np.isfinite(fl).all(1) & np.isfinite(sl).all(1)
    camera = {"lag_ms": float(lag), "fx": float(K[0, 0]), "fy": float(K[1, 1]), "skew": float(K[0, 1]),
              "focal_mean": float((K[0, 0] + K[1, 1]) / 2), "pp": [float(K[0, 2]), float(K[1, 2])],
              "distance_mm": float(np.linalg.norm(C)),
              "distance_eye_mm": float(np.linalg.norm(measured["camera_center_mm"])),
              "center_mm": measured["camera_center_mm"], "reproj_median_px": float(med),
              "reproj_p95_px": float(np.percentile(res, 95)), "fit_frames": int(len(res))}
    return dict(run=run, circle=circle, T=T, gt=gt, gt_n=gt_n, sh_n=sh_n, fl=fl, fr=fr, sl=sl, sr=sr,
                visible=visible, full=full, r_eye=r_eye, measured=measured, camera=camera)


def average(cals):
    out = dict(cals[0])
    for k, v in cals[0].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(np.mean([c[k] for c in cals]))
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            out[k] = [float(x) for x in np.mean([c[k] for c in cals], 0)]
    return out


def orthographic(cal):
    return {k: v for k, v in cal.items() if k not in CAMERA_KEYS}


def rolled(cal, deg):
    return {**cal, "light_rot_clock": cal["light_rot_clock"] + math.radians(deg),
            "forceps_rot_clock": cal["forceps_rot_clock"] + math.radians(deg)}


def variants(R, cal, xform=lambda p: p, inputs=("model", "pipe"), step=1):
    """Per-frame reconstruction (mm) on the visible frames: 'model' input and 'pipe' input (NaN where missing).

    xform maps the points (normalised square) before the reconstruction; step subsamples the frames.
    """
    c = E.camera_in_square(cal, R["circle"])
    v, f = R["visible"], R["full"]
    out = {}
    if "model" in inputs:
        g, sh = xform(R["gt_n"][v]), xform(R["sh_n"][v])
        out["model"] = np.full((v.sum(), 3), np.nan)
        out["model"][::step] = E.reconstruct(c, g[::step], g[::step], sh[::step], sh[::step])
    if "pipe" in inputs:
        out["pipe"] = np.full((v.sum(), 3), np.nan)
        out["pipe"][f[v]] = E.reconstruct(c, xform(R["fl"][f]), xform(R["fr"][f]), xform(R["sl"][f]), xform(R["sr"][f]))
    return out


def ortho_xform(k, dx, dy):
    """Orthographic image scale k times the eye circle's, centre shifted by (dx, dy) px of the normalised square.
    Only for the lower bound in the table: the fit needs the trajectory log."""
    h = E.WORK / 2
    return lambda p: h + (np.asarray(p) - h - np.array([dx, dy])) / k


def fit_orthographic(Rs, cal):
    """Best orthographic scale and centre for the ideal input of the recordings Rs (median 3D error), by a coarse and
    a fine coordinate search. The orthographic model at its best, not as the pipeline used it (circle = equator)."""
    oc = orthographic(cal)

    def cost(x):
        e = np.concatenate([errors(R, variants(R, oc, ortho_xform(*x), ("model",), step=8)["model"])[0] for R in Rs])
        e = e[np.isfinite(e)]
        return float(np.median(e)) if len(e) > 100 else np.inf

    x = [1.0, 0.0, 0.0]
    coarse = (np.arange(0.6, 1.61, 0.04), np.arange(-160, 41, 10), np.arange(-80, 61, 10))
    for grids in (coarse, None):
        for _ in range(2):
            for d in range(3):
                vals = grids[d] if grids else x[d] + np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]) * (0.01, 2, 2)[d]
                x = min(((cost(x[:d] + [float(v)] + x[d + 1:]), x[:d] + [float(v)] + x[d + 1:]) for v in vals))[1]
    return x


def errors(R, rec):
    gt = R["gt"][R["visible"]]
    h_gt = R["r_eye"] - np.linalg.norm(gt, axis=1)
    h_rec = R["r_eye"] - np.linalg.norm(rec, axis=1)
    return np.linalg.norm(rec - gt, axis=1), h_rec - h_gt, h_gt


def stats(R, rec):
    e3, dh, _ = errors(R, rec)
    ok = np.isfinite(e3)
    return {"n": int(ok.sum()), "found_pct": round(100 * ok.mean(), 1),
            "err3d_median": round(float(np.median(e3[ok])), 3), "err3d_p95": round(float(np.percentile(e3[ok], 95)), 3),
            "height_abs_median": round(float(np.median(np.abs(dh[ok]))), 3),
            "height_abs_p95": round(float(np.percentile(np.abs(dh[ok]), 95)), 3),
            "height_bias": round(float(np.median(dh[ok])), 3)}


# --- figures ------------------------------------------------------------------------------------------------------
def save(fig, path):
    """PNG for the markdown draft, vector PDF for the LaTeX report."""
    fig.savefig(path)
    fig.savefig(os.path.join(REPORT_FIGS, os.path.basename(path)[:-4] + ".pdf"))


def fig_geometry(path, tan_circle, R_eye=12.05, d_cam=21.9):
    """Side cut through the eye: camera rays (perspective) against parallel rays (orthographic).

    Left: the whole geometry. Right: magnified around tip and shadow. The orthographic model maps the eye circle
    (half-angle atan(tan_circle) seen from the camera) onto the eye's equator, as the pipeline did.
    """
    from matplotlib.patches import ConnectionPatch, Rectangle
    from matplotlib.lines import Line2D

    L = np.array([-6.5, -6.0])                      # light tip (x, z), z towards the retina
    P = np.array([3.0, 6.0])                        # forceps tip, ~5.3 mm above the retina
    d = (P - L) / np.linalg.norm(P - L)
    b = d @ L
    S = L + (-b + math.sqrt(b * b - (L @ L - R_eye ** 2))) * d      # shadow on the retina
    cam = np.array([0.0, -d_cam])
    u = lambda X: X[0] / (X[1] + d_cam)             # image coordinate of the pinhole
    scale = R_eye / tan_circle                      # orthographic: image coordinate -> mm, eye circle = equator
    xs = u(S) * scale
    S_o = np.array([xs, math.sqrt(R_eye ** 2 - xs ** 2)])
    xp = u(P) * scale
    d_o = S_o - L
    P_o = L + (xp - L[0]) / d_o[0] * d_o
    retina_z = lambda x: math.sqrt(R_eye ** 2 - x ** 2)

    def scene(ax, lw):
        ax.add_patch(plt.Circle((0, 0), R_eye, facecolor="#f4f3ef", edgecolor=MUTED, lw=0.9 * lw, zorder=0))
        ax.plot([L[0], S[0]], [L[1], S[1]], color=INK, lw=0.9 * lw, zorder=2)                  # light ray
        ax.plot([L[0], S_o[0]], [L[1], S_o[1]], color=ORTHO, lw=0.9 * lw, ls=(0, (1.5, 1.5)), zorder=2)
        for X in (P, S):                                                                         # camera rays
            ax.plot([cam[0], X[0]], [cam[1], X[1]], color=PERSP, lw=0.9 * lw, zorder=1)
        for X in (P_o, S_o):                                                                     # parallel rays
            ax.plot([X[0], X[0]], [-d_cam, X[1]], color=ORTHO, lw=0.9 * lw, ls=(0, (4, 2.5)), zorder=1)
        ms = 4.5 * lw
        ax.plot(*P, "o", ms=ms, color=INK, zorder=4)
        ax.plot(*S, "o", ms=ms, color=INK, mfc="white", mew=1.2, zorder=4)
        ax.plot(*P_o, "o", ms=ms, color=ORTHO, zorder=4)
        ax.plot(*S_o, "o", ms=ms, color=ORTHO, mfc="white", mew=1.2, zorder=4)
        ax.set_aspect("equal")
        ax.axis("off")

    fig = plt.figure(figsize=(6.6, 4.6))
    ax = fig.add_axes([0.0, 0.08, 0.42, 0.9])
    zoom = fig.add_axes([0.47, 0.2, 0.52, 0.7])
    scene(ax, 1.0)
    scene(zoom, 1.5)
    ax.set_xlim(-13, 13)
    ax.set_ylim(R_eye + 1.0, -d_cam - 2.5)

    # whole geometry: camera, light, eye centre, camera distance, scale bar
    ax.plot(*cam, marker="v", ms=9, color=PERSP, zorder=4)
    ax.text(cam[0] + 1.2, cam[1] - 0.2, "camera", fontsize=8, color=INK, va="center")
    ax.plot(*L, marker="*", ms=10, color=INK, zorder=4)
    ax.text(L[0], L[1] - 1.3, "light $L$", fontsize=8, color=INK, ha="center")
    ax.plot(0, 0, "+", color=MUTED, ms=5)
    ax.annotate("", (-11.5, -d_cam), (-11.5, 0), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.7))
    ax.text(-11.0, -d_cam / 2 - 2, f"{d_cam:.1f} mm\nto the eye\ncentre (+)", fontsize=7.5, color=MUTED, va="center")
    ax.plot([-11.8, -11.2], [0, 0], color=MUTED, lw=0.7)
    ax.plot([6, 11], [R_eye + 0.6] * 2, color=INK, lw=1.2)
    ax.text(8.5, R_eye + 0.4, "5 mm", fontsize=7, color=INK, ha="center", va="bottom")

    # magnified region and its frame on the left
    x0, x1, z0, z1 = 1.6, 8.6, 4.4, 11.4
    zoom.set_xlim(x0, x1)
    zoom.set_ylim(z1, z0)
    ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec=MUTED, lw=0.7, zorder=5))
    zoom.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec=MUTED, lw=0.7, clip_on=False))
    for (xa, za) in ((x1, z0), (x1, z1)):
        fig.add_artist(ConnectionPatch((xa, za), (x0 if xa == x1 else xa, za), "data", "data", axesA=ax, axesB=zoom,
                                       color=GRID, lw=0.7))

    # displacements, true -> orthographic
    arrow = dict(arrowstyle="-|>", color=ORTHO, lw=1.0, mutation_scale=8, shrinkA=5, shrinkB=5)
    zoom.annotate("", P_o, P, arrowprops=arrow, zorder=5)
    zoom.annotate("", S_o, S, arrowprops=arrow, zorder=5)
    mid = (P + P_o) / 2
    tag = dict(fontsize=7.5, color=ORTHO, ha="center", va="center", zorder=6,
               bbox=dict(boxstyle="round,pad=0.15", fc="#f4f3ef", ec="none"))
    zoom.text(mid[0], mid[1] - 0.38, f"{np.linalg.norm(P_o - P):.1f} mm", **tag)
    zoom.text(S_o[0] - 0.35, S_o[1] - 0.45, f"{np.linalg.norm(S_o - S):.1f} mm", **{**tag, "ha": "right"})
    zoom.text(7.9, retina_z(7.9) + 0.7, "retina", fontsize=7.5, color=MUTED, va="top")

    # key
    key = [Line2D([], [], color=PERSP, lw=1.2, label="camera ray, perspective"),
           Line2D([], [], color=ORTHO, lw=1.2, ls=(0, (4, 2.5)), label="camera ray, orthographic"),
           Line2D([], [], color=INK, lw=1.2, label="light ray"),
           Line2D([], [], color=ORTHO, lw=1.2, ls=(0, (1.5, 1.5)), label="light ray, orthographic"),
           Line2D([], [], color=INK, ls="", marker="o", ms=6, label="tip $P$"),
           Line2D([], [], color=INK, ls="", marker="o", ms=6, mfc="white", mew=1.2, label="shadow $S$"),
           Line2D([], [], color=ORTHO, ls="", marker="o", ms=6, label="tip, orthographic"),
           Line2D([], [], color=ORTHO, ls="", marker="o", ms=6, mfc="white", mew=1.2, label="shadow, orthographic")]
    fig.legend(handles=key, loc="lower left", bbox_to_anchor=(0.47, -0.01), ncol=2, fontsize=8, handlelength=2.6,
               columnspacing=1.5)
    save(fig, path)
    plt.close(fig)
    return {"tip_shift_mm": float(np.linalg.norm(P_o - P)), "shadow_shift_mm": float(np.linalg.norm(S_o - S)),
            "tip_height_mm": float(R_eye - np.linalg.norm(P)), "tip_height_ortho_mm": float(R_eye - np.linalg.norm(P_o))}


def binned(x, y, edges):
    mids, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b) & np.isfinite(y)
        if m.sum() >= 30:
            mids.append((a + b) / 2)
            q = np.percentile(y[m], [25, 50, 75])
            lo.append(q[0]), med.append(q[1]), hi.append(q[2])
    return np.array(mids), np.array(med), np.array(lo), np.array(hi)


def height_bins(pooled):
    """Median height error per 0.5 mm of true height (bins with >= 30 frames), for the text."""
    edges = np.arange(2.0, 10.01, 0.5)
    return {f"{cam}_{inp}": [[round(float(a), 2), round(float(b), 3)]
                             for a, b in zip(*binned(pooled["h_gt"], pooled[cam]["dh_" + inp], edges)[:2])]
            for cam in ("perspective", "orthographic") for inp in ("model", "pipe")}


def fig_height(path, pooled, inp="pipe"):
    """Reconstructed against true height, one dot per frame with a 3D result; frames beyond the y range are drawn as
    triangles on the edge. Returns the numbers in the panels."""
    out = {}
    h = pooled["h_gt"]
    xlim, ylim = (0, 11.5), (-2, 11.5)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6), sharex=True, sharey=True)
    for ax, cam, color in zip(axes, ("orthographic", "perspective"), (ORTHO, PERSP)):
        dh = pooled[cam]["dh_" + inp]
        ok = np.isfinite(dh)
        x, y = h[ok], h[ok] + dh[ok]
        low, high = y < ylim[0], y > ylim[1]
        inside = ~(low | high)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.plot(xlim, xlim, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.scatter(x[inside], y[inside], s=1.5, color=color, alpha=0.3, lw=0, rasterized=True, zorder=2)
        for m, edge, marker in ((low, ylim[0], "v"), (high, ylim[1], "^")):
            ax.scatter(x[m], np.full(m.sum(), edge), s=9, marker=marker, color=color, alpha=0.5, lw=0,
                       clip_on=False, zorder=3)
        med_abs, bias = float(np.median(np.abs(dh[ok]))), float(np.median(dh[ok]))
        out[cam] = {"n": int(ok.sum()), "height_abs_median": round(med_abs, 3), "height_bias": round(bias, 3),
                    "below_axis": int(low.sum()), "above_axis": int(high.sum()), "below_retina": int((y < 0).sum())}
        note = f"median |error| {med_abs:.2f} mm\nmedian bias {bias:.2f} mm"
        if low.sum():
            note += f"\n\u25be {low.sum()} frames below {ylim[0]} mm"
        ax.text(0.04, 0.96, note.replace("-", "\u2212"), transform=ax.transAxes, va="top", fontsize=7.5, color=INK,
                linespacing=1.4)
        ax.set_title(LABEL[cam], fontsize=9, color=INK)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xticks(range(0, 12, 2))
        ax.set_yticks(range(-2, 12, 2))
        ax.set_xlabel("true height above the retina (mm)")
    axes[0].set_ylabel("reconstructed height (mm)")
    save(fig, path)
    plt.close(fig)
    return out


def radial(pooled):
    """Median 3D error (ideal input) per 0.1 of the eye-circle radius, for the text."""
    edges = np.arange(0, 0.95, 0.1)
    return {cam: [[round(float(a), 2), round(float(b), 3)] for a, b in zip(*binned(pooled["radius"], pooled[cam]["model"], edges)[:2])]
            for cam in ("perspective", "orthographic")}


def detection_tail(R, rec):
    """Detected input, perspective: frames above 1 mm and how far their detected shadow is from the modelled one."""
    e, _, _ = errors(R, rec)
    v = R["visible"]
    shadow_px = np.linalg.norm((R["sl"][v] + R["sr"][v]) / 2 - R["sh_n"][v], axis=1)   # normalised 1080² square
    ok = np.isfinite(e)
    big = ok & (e > 1.0)
    return {"eligible": int(v.sum()), "with_result": int(ok.sum()), "within_1mm": int((ok & ~big).sum()),
            "above_1mm": int(big.sum()),
            "shadow_offset_px_median_above_1mm": round(float(np.median(shadow_px[big])), 1),
            "shadow_offset_px_median_within_1mm": round(float(np.median(shadow_px[ok & ~big])), 1)}


def height_tail(R, rec):
    """Detected input, perspective: frames with |height error| > 1 mm and where their detected shadow lies."""
    _, dh, h = errors(R, rec)
    v = R["visible"]
    ok = np.isfinite(dh)
    bad, good = ok & (np.abs(dh) > 1.0), ok & (np.abs(dh) <= 1.0)
    shadow, tip = (R["sl"][v] + R["sr"][v]) / 2, (R["fl"][v] + R["fr"][v]) / 2
    to_tip = np.linalg.norm(shadow - tip, axis=1)                                          # normalised 1080² square
    model_to_tip = np.linalg.norm(R["sh_n"][v] - R["gt_n"][v], axis=1)
    spread = np.linalg.norm(R["sl"][v] - R["sr"][v], axis=1)
    shadow_r = np.linalg.norm(R["sh_n"][v] - E.WORK / 2, axis=1) / (E.WORK / 2)
    med = lambda x, m: round(float(np.median(x[m])), 1)
    return {"above_1mm": int(bad.sum()), "too_low": int((ok & (dh < -1.0)).sum()), "with_result": int(ok.sum()),
            "detected_shadow_to_tip_px": [med(to_tip, bad), med(to_tip, good)],
            "modelled_shadow_to_true_tip_px": [med(model_to_tip, bad), med(model_to_tip, good)],
            "shadow_tip_spread_px": [med(spread, bad), med(spread, good)],
            "modelled_shadow_outside_circle": int((bad & (shadow_r > 1)).sum()),
            "true_height_median": med(h, bad), "reconstructed_height_median": med(h + dh, bad)}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REPORT_FIGS, exist_ok=True)
    base = json.load(open(os.path.join(ROOT, "config/default_calibration.json")))
    runs = {r: evaluate_run(r, base) for r in RUNS}
    out = {"camera": {r: runs[r]["camera"] for r in RUNS}, "runs": {}}
    cams = ("perspective", "orthographic", "orthographic_fitted")
    pooled = {"h_gt": [], "radius": [], **{cam: {} for cam in cams}}
    out["ortho_fit"] = {}
    for r in RUNS:
        R = runs[r]
        others = [runs[o] for o in RUNS if o != r]
        loo = average([o["measured"] for o in others])
        fit = fit_orthographic(others, loo)
        out["ortho_fit"][r] = {"scale_factor": fit[0], "centre_shift_px": fit[1:]}
        res = {"perspective": variants(R, loo), "orthographic": variants(R, orthographic(loo)),
               "orthographic_fitted": variants(R, orthographic(loo), ortho_xform(*fit)),
               "own": variants(R, R["measured"]), "roll5": variants(R, rolled(loo, 5.0))}
        out["runs"][r] = {name: {inp: stats(R, rec) for inp, rec in v.items()} for name, v in res.items()}
        _, _, h_gt = errors(R, res["perspective"]["model"])
        pooled["h_gt"].append(h_gt)
        pooled["radius"].append(np.linalg.norm(R["gt_n"][R["visible"]] - E.WORK / 2, axis=1) / (E.WORK / 2))
        for cam in cams:
            for inp in ("model", "pipe"):
                e3, dh, _ = errors(R, res[cam][inp])
                pooled[cam].setdefault(inp, []).append(e3)
                pooled[cam].setdefault("dh_" + inp, []).append(dh)
        out.setdefault("detection_tail", {})[r] = detection_tail(R, res["perspective"]["pipe"])
        out.setdefault("height_tail", {})[r] = height_tail(R, res["perspective"]["pipe"])
    pooled["h_gt"] = np.concatenate(pooled["h_gt"])
    pooled["radius"] = np.concatenate(pooled["radius"])
    for cam in cams:
        pooled[cam] = {k: np.concatenate(v) for k, v in pooled[cam].items()}
    out["pooled"] = {cam: {inp: {"median": round(float(np.nanmedian(pooled[cam][inp])), 3),
                                 "p95": round(float(np.nanpercentile(pooled[cam][inp], 95)), 3),
                                 "height_bias": round(float(np.nanmedian(pooled[cam]["dh_" + inp])), 3)}
                           for inp in ("model", "pipe")} for cam in cams}
    out["height_range_mm"] = [round(float(np.percentile(pooled["h_gt"], q)), 2) for q in (5, 50, 95)]
    # eye circle seen from the camera: circle radius / focal length (both video px)
    tan_circle = float(np.mean([runs[r]["circle"][2] / runs[r]["camera"]["fx"] for r in RUNS]))
    d_cam = float(np.mean([runs[r]["camera"]["distance_eye_mm"] for r in RUNS]))
    out["geometry"] = {"tan_circle": tan_circle, "camera_distance_mm": d_cam,
                       **fig_geometry(os.path.join(OUT, "geometry.png"), tan_circle, d_cam=d_cam)}
    out["height_scatter"] = fig_height(os.path.join(OUT, "height_error.png"), pooled)
    out["height_bins"] = height_bins(pooled)
    out["radial"] = radial(pooled)
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
