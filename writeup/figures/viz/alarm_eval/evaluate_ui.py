"""Cached two-tip replay, physical midpoint truth proxy, and small LORO sweep."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT / "services/stream"))
import tool_positions as tp

RUNS = ("10-35-27", "10-48-23", "11-00-43")
CURRENT = (0.5, 0.08, 250)
PARAMETERS = list(itertools.product((0.3, 0.5), (0.05, 0.08, 0.15), (0, 250, 500)))
SOURCES = ("services/viz/src/feed/levels.js", "services/viz/src/feed/pipeline.js",
           "services/viz/src/config.js", "services/viz/src/geometry.js",
           "services/viz/src/ui.js", "services/stream/tool_positions.py",
           "config/default_calibration.json")


def fingerprints():
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run(run, calibration):
    frames = np.genfromtxt(ROOT / f"artifacts/sim_gt_{run}_ml_frames.csv", delimiter=",", names=True)
    path = ROOT / f"artifacts/sim_gt_{run}_ml_detections.csv"
    with path.open() as handle:
        fps = float(handle.readline().split(",")[1])
    detections = np.loadtxt(path, delimiter=",", skiprows=2)
    lookup = {int(row[0]): row[1:] for row in detections}
    tips = np.full((len(frames), 2, 3), np.nan)
    keys = ("light_rot_up", "light_rot_clock", "light_depth_mm", "forceps_rot_up",
            "forceps_rot_clock", "eye_radius_mm", "jaw_length_mm")
    arguments = {key: calibration[key] for key in keys}
    arguments.update(light_aim_tilt=calibration.get("light_aim_tilt", 0.0),
                     light_aim_clock=calibration.get("light_aim_clock", 0.0),
                     eye_center_px=(540, 540), eye_radius_px=540)
    for index, frame in enumerate(frames):
        points = lookup[int(frame["frame"])].reshape(4, 2)
        if not frame["visible"] or not np.isfinite(points).all():
            continue
        try:
            positions = tp.compute_tool_positions(**arguments, left_tip_px=tuple(points[0]),
                right_tip_px=tuple(points[1]), left_shadow_px=tuple(points[2]),
                right_shadow_px=tuple(points[3]))
            tips[index] = np.array([positions["left_tip_forceps"], positions["right_tip_forceps"]]) * calibration["eye_radius_mm"]
        except ValueError:
            pass
    valid = np.isfinite(tips).all(axis=(1, 2))
    assert np.array_equal(valid, np.isfinite(frames["height_rec_mm"]))
    midpoint = tips.mean(axis=1)
    logged_midpoint = np.column_stack([frames[key] for key in ("rec_x", "rec_y", "rec_z")])
    midpoint_roundtrip_error = float(np.max(np.linalg.norm(midpoint[valid] - logged_midpoint[valid], axis=1)))
    clearance = np.maximum(0, 12 - np.linalg.norm(tips, axis=2))
    midpoint_clearance = np.maximum(0, 12 - np.linalg.norm(midpoint, axis=1))
    frame_ids = frames["frame"].astype(int)
    assert np.all(np.diff(frame_ids) == 1)
    return dict(run=run, frames=frames, tips=tips, clearance=clearance, valid=valid,
                gt=frames["height_gt_mm"], fps=fps, times=np.arange(len(frames)) * 1000 / fps,
                delta=midpoint_clearance - np.min(clearance, axis=1),
                midpoint_roundtrip_error_mm=midpoint_roundtrip_error)


def replay(clearance, times, parameters=CURRENT):
    threshold, margin, dwell = parameters
    previous = None
    pending = [None, None]
    last_valid = None
    result = np.full(len(times), -1, dtype=int)
    for index, (distances, now) in enumerate(zip(clearance, times)):
        if not np.isfinite(distances).all():
            pending = [None, None]
            if previous is not None and now - last_valid < 400:
                result[index] = max(previous)
            else:
                previous = None
            continue
        last_valid = now
        next_levels = []
        for tip_index, distance in enumerate(distances):
            raw = 2 if distance < threshold else 1 if distance < 2 else 0
            old = None if previous is None else previous[tip_index]
            candidate = raw
            if old is not None and raw < old:
                held_distance = max(0, distance - margin)
                held = 2 if held_distance < threshold else 1 if held_distance < 2 else 0
                candidate = min(old, held)
            if old is None or candidate >= old:
                pending[tip_index] = None
                next_levels.append(candidate)
            else:
                if pending[tip_index] is None or pending[tip_index][0] != candidate:
                    pending[tip_index] = (candidate, now)
                next_levels.append(candidate if now - pending[tip_index][1] >= dwell else old)
        previous = next_levels
        result[index] = max(previous)
    return result


def onsets(mask):
    return np.flatnonzero(mask & ~np.r_[False, mask[:-1]])


def metrics(data, levels, gt_shift=0):
    gt = data["gt"] + gt_shift
    seconds = 1 / data["fps"]
    result = dict(run=data["run"], frames=len(gt), duration_s=len(gt) * seconds,
                  changes=int(np.count_nonzero(np.diff(levels))),
                  critical_episodes=len(onsets(levels == 2)),
                  true_critical_episodes=len(onsets(gt < 0.5)))
    for label, prediction, truth in (("critical", levels == 2, gt < 0.5),
                                      ("near", levels >= 1, gt < 2)):
        result.update({f"{label}_{key}_s": np.count_nonzero(mask) * seconds for key, mask in
                       (("tp", prediction & truth), ("display", prediction), ("truth", truth),
                        ("false", prediction & ~truth), ("missed", ~prediction & truth))})
    ending = (levels[:-1] == 2) & (levels[1:] != 2)
    result["critical_end_loss"] = int(np.count_nonzero(ending & (levels[1:] == -1)))
    result["critical_end_rise"] = int(np.count_nonzero(ending & (levels[1:] >= 0)))
    result["critical_right_censored"] = int(levels[-1] == 2)
    return derived(result)


def derived(result):
    result["changes_per_min"] = result["changes"] * 60 / result["duration_s"]
    for label in ("critical", "near"):
        for name, denominator in (("ppv", "display"), ("sensitivity", "truth")):
            total = result[f"{label}_{denominator}_s"]
            result[f"{label}_{name}"] = result[f"{label}_tp_s"] / total if total else float("nan")
    return result


def pool(rows):
    return derived({"run": "pooled", **{key: sum(row[key] for row in rows)
                   for key in rows[0] if key != "run" and not key.endswith(("ppv", "sensitivity", "per_min"))}})


def pareto(rows):
    return [not any(other["critical_false_s"] <= row["critical_false_s"] + 1e-9 and
                    other["critical_missed_s"] <= row["critical_missed_s"] + 1e-9 and
                    (other["critical_false_s"] < row["critical_false_s"] - 1e-9 or
                     other["critical_missed_s"] < row["critical_missed_s"] - 1e-9)
                    for other in rows) for row in rows]


def check_js(data, expected):
    runner = OUT / "verify_ui.mjs"
    payload = dict(tips=[tip.tolist() if valid else None for tip, valid in zip(data["tips"], data["valid"])],
                   times=data["times"].tolist())
    actual = subprocess.run(["node", str(runner)], input=json.dumps(payload), text=True,
                            capture_output=True, check=True, cwd=ROOT)
    np.testing.assert_array_equal(json.loads(actual.stdout), expected)


def checks():
    def sequence(distances, times, parameters=CURRENT):
        values = np.array([[distance, distance] for distance in distances], dtype=float)
        return replay(values, np.array(times, dtype=float), parameters).tolist()
    assert sequence([0.49, np.nan, np.nan], [0, 399, 400]) == [2, 2, -1]
    assert sequence([0.5, 2, 2], [0, 1, 251]) == [1, 1, 1]
    assert sequence([0.49, 0.59, 0.59], [0, 1, 251]) == [2, 2, 1]
    assert sequence([3, 0.49], [0, 1]) == [0, 2]
    assert sequence([0.49, 0.59, np.nan, 0.59, 0.59], [0, 1, 100, 251, 501]) == [2, 2, 2, 2, 1]
    assert sequence([0.49, 0.59], [0, 1], (0.5, 0.08, 0)) == [2, 1]
    assert sequence([0.49, np.nan, 3], [0, 400, 401]) == [2, -1, 0]


def main():
    before = fingerprints()
    checks()
    calibration = json.loads((ROOT / "config/default_calibration.json").read_text())
    datasets = [load_run(run, calibration) for run in RUNS]
    current_levels = [replay(data["clearance"], data["times"]) for data in datasets]
    for data, levels in zip(datasets, current_levels):
        check_js(data, levels)
    current = [metrics(data, levels) for data, levels in zip(datasets, current_levels)]
    current.append(pool(current))
    write_csv("ui_metrics.csv", current)
    shifted = [metrics(data, levels, -0.05) for data, levels in zip(datasets, current_levels)]
    write_csv("ui_truth_radius_sensitivity.csv", shifted + [pool(shifted)])
    frame_rows, onset_rows, delta_rows, dropout_rows, episode_rows = [], [], [], [], []
    bins = (("<0.5", -np.inf, 0.5), ("0.5–1", 0.5, 1), ("1–2", 1, 2),
            ("2–5", 2, 5), ("≥5", 5, np.inf))
    for data, levels in zip(datasets, current_levels):
        for index in range(len(levels)):
            frame_rows.append(dict(run=data["run"], frame=int(data["frames"]["frame"][index]),
                time_ms=data["times"][index], visible=int(data["frames"]["visible"][index]),
                gt_midpoint_mm=data["gt"][index], left_mm=data["clearance"][index, 0],
                right_mm=data["clearance"][index, 1], min_mm=np.min(data["clearance"][index]),
                midpoint_minus_min_mm=data["delta"][index], level=int(levels[index])))
        for index in onsets(levels == 2):
            onset_rows.append(dict(run=data["run"], frame=int(data["frames"]["frame"][index]),
                time_s=data["times"][index] / 1000, gt_height_mm=data["gt"][index],
                reconstructed_min_mm=np.min(data["clearance"][index])))
            ends = np.flatnonzero(levels[index:] != 2)
            end = index + int(ends[0]) if len(ends) else len(levels)
            reason = "right_censored" if end == len(levels) else "grace_expired" if levels[end] == -1 else "clearance_rose"
            episode_rows.append(dict(run=data["run"], start_frame=int(data["frames"]["frame"][index]),
                start_s=data["times"][index] / 1000, end_s=end / data["fps"],
                duration_s=(end - index) / data["fps"], end_reason=reason))
        for label, lower, upper in bins:
            selected = (data["frames"]["visible"] == 1) & (data["gt"] >= lower) & (data["gt"] < upper)
            total = int(selected.sum())
            missing = int(np.count_nonzero(selected & ~data["valid"]))
            dropout_rows.append(dict(run=data["run"], bin_mm=label, frames=total, missing=missing,
                                     missing_fraction=missing / total if total else float("nan")))
    for label, _, _ in bins:
        selected = [row for row in dropout_rows if row["bin_mm"] == label]
        total, missing = sum(row["frames"] for row in selected), sum(row["missing"] for row in selected)
        dropout_rows.append(dict(run="pooled", bin_mm=label, frames=total, missing=missing,
                                 missing_fraction=missing / total if total else float("nan")))
    onset_summary = []
    for run in (*RUNS, "pooled"):
        delta = np.concatenate([data["delta"][data["valid"]] for data in datasets if run in (data["run"], "pooled")])
        delta_rows.append(dict(run=run, valid_frames=len(delta), midpoint_minus_min_median_mm=float(np.median(delta)),
                               midpoint_minus_min_p95_mm=float(np.percentile(delta, 95))))
        heights = np.array([row["gt_height_mm"] for row in onset_rows if run in (row["run"], "pooled")])
        onset_summary.append(dict(run=run, onsets=len(heights), minimum_mm=float(heights.min()),
            p05_mm=float(np.percentile(heights, 5)), median_mm=float(np.median(heights)),
            p95_mm=float(np.percentile(heights, 95)), maximum_mm=float(heights.max())))
    for name, rows in (("ui_replay.csv", frame_rows), ("ui_critical_onsets.csv", onset_rows),
                       ("ui_onset_summary.csv", onset_summary), ("ui_tip_difference.csv", delta_rows),
                       ("ui_dropouts.csv", dropout_rows), ("ui_critical_episodes.csv", episode_rows)):
        write_csv(name, rows)
    sweep = {}
    sweep_rows = []
    for parameters in PARAMETERS:
        rows = [metrics(data, replay(data["clearance"], data["times"], parameters)) for data in datasets]
        rows.append(pool(rows))
        sweep[parameters] = rows
        for row in rows:
            sweep_rows.append(dict(critical_threshold_mm=parameters[0], hysteresis_mm=parameters[1],
                                  dwell_ms=parameters[2], **row))
    write_csv("ui_sweep.csv", sweep_rows)
    folds = []
    for held_out, run in enumerate(RUNS):
        def objective(parameters):
            train = [row for index, row in enumerate(sweep[parameters][:3]) if index != held_out]
            cost = sum(row["critical_false_s"] + row["critical_missed_s"] for row in train)
            changes = sum(row["changes"] for row in train)
            return (round(cost, 9), changes, parameters != CURRENT, PARAMETERS.index(parameters))
        selected = min(PARAMETERS, key=objective)
        train_rows = [row for index, row in enumerate(sweep[selected][:3]) if index != held_out]
        for variant, parameters in (("selected", selected), ("current", CURRENT)):
            folds.append(dict(held_out=run, variant=variant, critical_threshold_mm=parameters[0],
                hysteresis_mm=parameters[1], dwell_ms=parameters[2],
                train_true_critical_episodes=sum(row["true_critical_episodes"] for row in train_rows),
                train_cost_s=objective(parameters)[0], **sweep[parameters][held_out]))
    write_csv("ui_loro.csv", folds)
    plot_tradeoff(sweep, folds)
    plot_dropouts(dropout_rows)
    assert fingerprints() == before, "Sources changed during evaluation; rerun against a coherent version"
    (OUT / "ui_provenance.json").write_text(json.dumps(dict(source_sha256=before,
        inputs_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for run in RUNS for path in (ROOT / f"artifacts/sim_gt_{run}_ml_frames.csv",
                                        ROOT / f"artifacts/sim_gt_{run}_ml_detections.csv")},
        current_parameters=CURRENT, checks=7,
        midpoint_roundtrip_max_error_mm={data["run"]: data["midpoint_roundtrip_error_mm"] for data in datasets},
        javascript_matched_frames=sum(len(data["gt"]) for data in datasets)), indent=2) + "\n")
    print(json.dumps(dict(metrics=current, onset=onset_summary, difference=delta_rows, loro=folds), indent=2))


def plot_tradeoff(sweep, folds):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        row_index = 3 if index == 0 else index - 1
        rows = [sweep[parameters][row_index] for parameters in PARAMETERS]
        front = pareto(rows)
        for threshold, marker in ((0.3, "o"), (0.5, "s")):
            chosen = [row for parameters, row in zip(PARAMETERS, rows) if parameters[0] == threshold]
            axis.scatter([row["critical_false_s"] for row in chosen], [row["critical_missed_s"] for row in chosen],
                         marker=marker, s=55, alpha=0.65, label=f"Schwelle {threshold:.1f} mm")
        frontier = sorted(set((row["critical_false_s"], row["critical_missed_s"])
                              for row, efficient in zip(rows, front) if efficient))
        axis.plot(*zip(*frontier), "k.-", label="Paretofront (18 Kombinationen)")
        current = sweep[CURRENT][row_index]
        axis.scatter(current["critical_false_s"], current["critical_missed_s"], marker="*", s=230,
                     c="#c23b28", edgecolors="black", zorder=5, label="Aktuell: 0.5 / 0.08 / 250")
        if row_index < 3:
            selected = next(row for row in folds if row["held_out"] == RUNS[row_index] and row["variant"] == "selected")
            axis.scatter(selected["critical_false_s"], selected["critical_missed_s"], marker="D", s=95,
                         facecolors="none", edgecolors="black", linewidths=2, zorder=6, label="LORO: nur auf anderen Läufen gewählt")
        axis.set(title="Pooled (deskriptiv)" if row_index == 3 else f"Testlauf {RUNS[row_index]}",
                 xlabel="Falsch-kritische Anzeige (s)", ylabel="Verpasste kritische Zeit (s)")
        axis.set_ylim(-0.08, max(0.5, max(row["critical_missed_s"] for row in rows) + 0.15))
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    figure.suptitle("Zwei-Spitzen-Replay · Grace 400 ms · Wahrheit: GT-Mittelpunkt < 0.5 mm\n"
                    "Nur eine wahre kritische Episode; keine belastbare Parameteroptimierung", fontsize=12)
    figure.savefig(OUT / "sweep_tradeoff.png", dpi=180)
    plt.close(figure)


def plot_dropouts(rows):
    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    labels = [row["bin_mm"] for row in rows if row["run"] == "pooled"]
    for offset, run in enumerate((*RUNS, "pooled")):
        chosen = [row for row in rows if row["run"] == run]
        axis.bar(np.arange(5) + (offset - 1.5) * 0.19,
                 [100 * row["missing_fraction"] for row in chosen], width=0.18, label=run)
    axis.set(xticks=np.arange(5), xticklabels=labels, xlabel="GT-Mittelpunkt-Clearance (mm)",
             ylabel="Frames ohne 3D-Position (%)", ylim=(0, 100),
             title="Dropouts nach wahrer Höhe · nur sichtbare Spitze\nPooled: fehlend / sichtbare Frames je Bin")
    for index, row in enumerate(row for row in rows if row["run"] == "pooled"):
        axis.text(index, 96, f"{row['missing']}/{row['frames']}", ha="center", va="top", fontsize=9)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(OUT / "dropouts_by_height.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
