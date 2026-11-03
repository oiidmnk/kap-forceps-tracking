"""Figures for the trajectory-log learnings (writeup/learnings_trajectories.md).

    .venv-ml/bin/python writeup/figures/make_figures.py        (needs matplotlib in .venv-ml)

Input: the cached detections of `scripts/eval_sim_gt.py` for each run, i.e. first
    .venv-ml/bin/python scripts/eval_sim_gt.py TestRuns/<run>/<run>.mp4 --out artifacts/sim_gt_<HH-MM-SS>
for the three runs below. Output: fig1_error_by_model.png, fig2_height_bias.png and a printed table.

The perspective reconstruction here is a prototype (issue #1), not the stream's code: shadow = camera ray through the
shadow pixel hits the eye sphere (far side), tip = point on the line light -> shadow closest to the camera ray through
the tip pixel. Light and trocar angles come from config/default_calibration.json, exactly as for the orthographic model.
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "scripts"), os.path.join(ROOT, "services/stream")]
import eval_sim_gt as g  # noqa: E402
import tool_positions as tp  # noqa: E402

RUNS = ["10-35-27", "10-48-23", "11-00-43"]
LAG_MS = 167                                      # video lags the log; eval_sim_gt.py finds +167 ms in all three runs
CAL = json.load(open(os.path.join(ROOT, "config/default_calibration.json")))

# reference palette (dataviz skill), light mode
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def load(run):
    stem = f"mapskap1_2026-10-22_{run}+0200_forcepstraining_lev1"
    det = os.path.join(ROOT, f"artifacts/sim_gt_{run}_ml_detections.csv")
    head = open(det).readline().strip().split(",")
    fps, (cx, cy, r) = float(head[1]), map(float, head[3:6])
    rows = np.loadtxt(det, delimiter=",", skiprows=2)
    x0, y0, side = round(cx - r), round(cy - r), round(2 * r)
    to_raw = lambda n: np.array([x0, y0]) + n * side / g.WORK
    log = g.load_log(os.path.join(ROOT, "TestRuns", stem, stem + "_trajectories.csv"))
    T = rows[:, 0] * 1000 / fps + LAG_MS
    sel = (T > log[g.FORCEPS][0][0] + 300) & (T < log[g.FORCEPS][0][-1] - 300)
    E, F = g.pose_at(log[g.EYE], T), g.pose_at(log[g.FORCEPS], T)
    fl, fr, sl, sr = rows[:, 1:3], rows[:, 3:5], rows[:, 5:7], rows[:, 7:9]
    tip_px, sh_px = to_raw((fl + fr) / 2), to_raw((sl + sr) / 2)
    ok = sel & np.isfinite(tip_px).all(1)
    P, _ = g.fit_camera(g.apply(E[ok], F[ok][:, :3, 3]), tip_px[ok])
    K, Rc, C = g.decompose(P)
    r_eye = float(np.linalg.norm(g.trocar(log[g.FORCEPS][1])))
    tip_w = F[:, :3, 3]
    gt = np.einsum("ij,njk,nk->ni", Rc, E[:, :3, :3], tip_w)              # pipeline frame: camera axes, eye centre, mm
    gt_px = g.project(P, g.apply(E, tip_w))
    L = g.trocar(log[g.LAMP][1])                                          # modelled shadow: light at the lamp trocar
    d = tip_w - L
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    b = d @ L
    gt_sh_px = g.project(P, g.apply(E, L + (-b + np.sqrt(np.maximum(b ** 2 - (L @ L - r_eye ** 2), 0)))[:, None] * d))
    nrm = lambda u: (u - [x0, y0]) * g.WORK / side
    visible = sel & (np.linalg.norm(nrm(gt_px) - g.WORK / 2, axis=1) < 0.94 * g.WORK / 2)
    full = visible & np.isfinite(fl).all(1) & np.isfinite(sl).all(1)
    return dict(run=run, K=K, circle=(cx, cy, r), r_eye=r_eye, gt=gt, nrm=nrm, visible=visible, full=full,
                tip=tip_px, sh=sh_px, gt_tip=gt_px, gt_sh=gt_sh_px,
                cam=np.median(np.einsum("ij,nj->ni", Rc, C - E[:, :3, 3]), 0))


def light(r_eye):
    return np.array(tp.trocar_position(CAL["light_rot_up"], CAL["light_rot_clock"])) * r_eye


def orthographic(d, tip_px, sh_px):
    L, out = light(1.0), []
    for t, s in zip(d["nrm"](tip_px), d["nrm"](sh_px)):
        try:
            out.append(np.array(tp.reconstruct_tip_from_shadow(tuple(L), tuple(t), tuple(s), (540, 540), 540)))
        except ValueError:
            out.append([np.nan] * 3)
    return np.array(out) * d["r_eye"]


def perspective(K, c, tip_px, sh_px, r_eye):
    Kinv = np.linalg.inv(K)

    def ray(u):
        v = np.c_[u, np.ones(len(u))] @ Kinv.T
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    L = light(r_eye)
    dv = ray(sh_px)
    b = dv @ c
    S = c + (-b + np.sqrt(np.maximum(b ** 2 - (c @ c - r_eye ** 2), 0)))[:, None] * dv
    a, e, w = S - L, ray(tip_px), L - c
    A, B, D, Ee = (a * a).sum(1), (a * e).sum(1), (a * w).sum(1), (e * w).sum(1)
    return L + ((B * Ee - D) / (A - B * B))[:, None] * a


def camera_from_others(runs, d):
    """Leave-one-out camera: focal length / principal point relative to the eye circle and camera centre, averaged over
    the other runs, re-expressed in this run's eye circle (what a fixed, once-measured camera would give)."""
    others = [o for o in runs if o["run"] != d["run"]]
    f = np.mean([o["K"][0, 0] / o["circle"][2] for o in others])
    pp = np.mean([(o["K"][:2, 2] - o["circle"][:2]) / o["circle"][2] for o in others], 0)
    cx, cy, r = d["circle"]
    K = np.array([[f * r, 0, cx + pp[0] * r], [0, f * r, cy + pp[1] * r], [0, 0, 1]])
    return K, np.mean([o["cam"] for o in others], 0)


def height(p, r_eye):
    return r_eye - np.linalg.norm(p, axis=1)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    runs = [load(r) for r in RUNS]
    conditions = ["Orthographic, perfect 2D points", "Orthographic, segmenter (today)",
                  "Perspective, camera of other sessions", "Perspective, camera of this session"]
    med, p95, hb = np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((4, 3))
    for j, d in enumerate(runs):
        v, f = d["visible"], d["full"]
        Kx, cx = camera_from_others(runs, d)
        recs = [(orthographic(d, d["gt_tip"][v], d["gt_sh"][v]), v),
                (orthographic(d, d["tip"][f], d["sh"][f]), f),
                (perspective(Kx, cx, d["tip"][f], d["sh"][f], d["r_eye"]), f),
                (perspective(d["K"], d["cam"], d["tip"][f], d["sh"][f], d["r_eye"]), f)]
        for i, (rec, m) in enumerate(recs):
            e = np.linalg.norm(rec - d["gt"][m], axis=1)
            med[i, j], p95[i, j] = np.nanmedian(e), np.nanpercentile(e, 95)
            hb[i, j] = np.nanmedian(height(rec, d["r_eye"]) - height(d["gt"][m], d["r_eye"]))
        if d["run"] == "11-00-43":
            fig2 = (d, recs[1][0], recs[3][0])

    print("median / p95 3D error (mm), median height bias (mm)")
    for i, c in enumerate(conditions):
        print(f"  {c:40s} " + "  ".join(f"{r}: {med[i, j]:.2f} / {p95[i, j]:.2f} ({hb[i, j]:+.2f})" for j, r in enumerate(RUNS)))

    # Fig. 1 - median 3D error per run and model; thin whisker = p95
    plt.rcParams.update({"font.family": "sans-serif", "text.color": INK, "axes.labelcolor": INK2})
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=200, facecolor=SURFACE)
    style(ax)
    x, w = np.arange(len(RUNS)), 0.19
    for i, c in enumerate(conditions):
        xs = x + (i - 1.5) * (w + 0.02)
        ax.bar(xs, med[i], w, color=SERIES[i], label=c, zorder=2)
        ax.vlines(xs, med[i], p95[i], color=SERIES[i], linewidth=1.2, zorder=2)
        for xx, m in zip(xs, med[i]):
            ax.text(xx, -0.08, f"{m:.2f}", ha="center", va="top", fontsize=7, color=INK2)
    ax.set_xticks(x, [f"Run {r}" for r in RUNS])
    ax.tick_params(axis="x", length=0, pad=16)
    ax.set_ylim(0, 6)
    ax.set_ylabel("3D tip error (mm)")
    fig.suptitle("Median 3D tip error per run (bar), p95 (line; clipped at 6 mm)", fontsize=10, color=INK, x=0.01, ha="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig1_error_by_model.png"), facecolor=SURFACE)

    # Fig. 2 - height above the retina: ground truth vs reconstruction (orthographic | perspective), run 11-00-43
    d, rec_o, rec_p = fig2
    f = d["full"]
    h_gt = height(d["gt"][f], d["r_eye"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=200, sharex=True, sharey=True, facecolor=SURFACE)
    for ax, rec, name in ((axes[0], rec_o, "Orthographic model (today)"),
                          (axes[1], rec_p, "Perspective model (prototype)")):
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        h = height(rec, d["r_eye"])
        ax.plot([0, 8], [0, 8], color=INK2, linewidth=1, linestyle="--", zorder=1)
        ax.scatter(h_gt, h, s=3, color=SERIES[0], alpha=0.35, linewidths=0, zorder=2)
        bias, mae = np.nanmedian(h - h_gt), np.nanmedian(np.abs(h - h_gt))
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.text(0.03, 0.97, f"median |error| {mae:.2f} mm\nmedian bias {bias:+.2f} mm", transform=ax.transAxes,
                va="top", fontsize=8, color=INK2)
        ax.set_xlabel("True height above retina (mm)")
    axes[0].set_ylabel("Reconstructed height (mm)")
    axes[0].set_xlim(0, 8)
    axes[0].set_ylim(0, 8)
    fig.suptitle(f"Height above the retina, run 11-00-43, segmenter points (n = {f.sum()} frames); dashed = ideal",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig2_height_bias.png"), facecolor=SURFACE)


if __name__ == "__main__":
    main()
