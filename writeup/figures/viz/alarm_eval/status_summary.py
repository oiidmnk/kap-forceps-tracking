"""Summarise ui_replay.csv at 30 Hz; never bridge run boundaries.
Run from any directory: python3 writeup/figures/viz/alarm_eval/status_summary.py
Critical durations count occupied frames; report completed and censored separately.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

runs = defaultdict(list)
with Path(__file__).with_name("ui_replay.csv").open() as f:
    for row in csv.DictReader(f):
        runs[row["run"]].append(row)

counts = dict(level=0, to_none=0, from_none=0)
completed, censored = [], []
for run, rows in runs.items():
    levels = [int(r["level"]) for r in rows]
    for a, b in zip(levels, levels[1:]):
        if a == b:
            continue
        counts["to_none" if b == -1 else "from_none" if a == -1 else "level"] += 1
    start = None
    for i, level in enumerate(levels + [-1]):
        if level == 2 and start is None:
            start = i
        elif level != 2 and start is not None:
            (censored if i == len(levels) else completed).append((i - start) / 30)
            start = None
    valid = sum(math.isfinite(float(r["min_mm"])) for r in rows)
    print(f"{run}: reconstruction mask {valid}/{len(rows)} = {100*valid/len(rows):.2f}%")
minutes = sum(map(len, runs.values())) / 30 / 60
for kind, count in counts.items():
    print(f"{kind}: {count} transitions, {count/minutes:.3f}/min")
print(f"total: {sum(counts.values())} transitions, {sum(counts.values())/minutes:.3f}/min")
print(f"Critical: {len(completed)} completed, median {median(completed):.3f} s; "
      f"{len(censored)} right-censored, observed durations {censored}; "
      f"all observed median {median(completed+censored):.3f} s")
for clearance in (4, 5.25):
    excess = max(0, clearance - 1.5)
    t = min(1, excess)
    framing = excess * t*t*(3-2*t)
    distance, lift = 8+1.6*framing, 0.5*framing
    foot_distance = math.sqrt(distance**2 + lift**2 + 2*distance*lift*math.sin(math.radians(58)))
    print(f"Scope {clearance} mm: camera–aim {distance:.3f} mm; "
          f"camera–foot {foot_distance:.3f} mm; local scale ratio {8/foot_distance:.3%}")
