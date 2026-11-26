"""Approach speed and height error from the simulator replay (writeup/viz_design.md §2, §5).

Reads alarm_eval/replay.csv (30 Hz, ground-truth and reconstructed tip-midpoint height).
Speed = central difference of the ground-truth height over ±5 frames (0.33 s), only
inside contiguous frames. Prints the numbers quoted in the design document.
"""
import csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE.parent / 'alarm_eval/replay.csv')))
speeds, err, gt_all = [], [], []
by_run = {}
for r in rows:
    by_run.setdefault(r['run'], []).append(r)
K = 5
for run, rs in by_run.items():
    f = np.array([int(r['frame']) for r in rs])
    h = np.array([float(r['height_gt_mm']) for r in rs])
    rec = np.array([float(r['height_rec_mm']) for r in rs])
    ok = np.isfinite(rec)
    err.append(rec[ok] - h[ok]); gt_all.append(h[ok])
    for i in range(K, len(rs) - K):
        if f[i + K] - f[i - K] != 2 * K:
            continue
        v = (h[i + K] - h[i - K]) / (2 * K / 30)
        if h[i] < 3 and v < -0.15:
            speeds.append(-v)
s = np.array(speeds); e = np.concatenate(err); g = np.concatenate(gt_all)
print(f'approach frames (h_gt<3 mm, v<-0.15 mm/s): {len(s)}')
print('approach speed mm/s  median %.2f  p75 %.2f  p95 %.2f' % tuple(np.percentile(s, [50, 75, 95])))
print(f'height error rec-gt (all reconstructed frames, n={len(e)}): median {np.median(e):+.2f} mm, '
      f'|err| median {np.median(abs(e)):.2f}, p95 {np.percentile(abs(e), 95):.2f}, '
      f'share rec<gt {np.mean(e < 0):.0%}')
near = g < 2
en = e[near]
print(f'near retina (h_gt<2 mm, n={len(en)}): median {np.median(en):+.2f} mm, '
      f'|err| median {np.median(abs(en)):.2f}, p95 {np.percentile(abs(en), 95):.2f}, share rec<gt {np.mean(en < 0):.0%}')
print(f'near retina: p5 / p95 of signed error {np.percentile(en, 5):+.2f} / {np.percentile(en, 95):+.2f} mm')
crit = g < 0.5
print(f'truly critical (h_gt<0.5 mm, n={crit.sum()}): reconstructed >= 0.5 mm in {np.mean(g[crit] + e[crit] >= 0.5):.0%}')
