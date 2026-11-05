"""Plot alert_trace.csv (from alert_trace.mjs) -> alert_trace.png. See viz_design.md §3."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / 'alert_trace.csv')))
t = [float(r['t_s']) for r in rows]
d = [float(r['clearance_mm']) if r['clearance_mm'] else None for r in rows]
shown = [r['displayed'] for r in rows]
held = [r['held'] == '1' for r in rows]
COL = {'safe': '#1ea6ac', 'warn': '#ffd24a', 'danger': '#ef5255', 'none': '#d5dbe0'}
raw = ['none' if x is None else 'danger' if x < 0.5 else 'warn' if x < 2 else 'safe' for x in d]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 3.9), dpi=250, sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.25], 'hspace': 0.08})
for th, lab in ((0.5, 'critical 0.5'), (2.0, 'warning 2.0')):
    ax.axhspan(th, th + 0.08, color='#9aa7b1', alpha=0.35, lw=0)
    ax.axhline(th, color='#56636e', lw=0.8, ls='--')
    ax.text(8.05, th, lab, va='center', fontsize=7.5, color='#56636e')
ax.text(8.05, 0.5 + 0.2, '+0.08 hysteresis', fontsize=6.5, color='#7a8792')
seg_t, seg_d = [], []
for ti, di in zip(t, d):
    if di is None:
        if seg_t: ax.plot(seg_t, seg_d, color='#202d37', lw=1.1)
        seg_t, seg_d = [], []
    else:
        seg_t.append(ti); seg_d.append(di)
ax.plot(seg_t, seg_d, color='#202d37', lw=1.1)
ax.set_ylabel('clearance (mm)')
ax.set_ylim(0, 2.8); ax.set_xlim(0, 8)
for s in ('top', 'right'): ax.spines[s].set_visible(False)

def note(x, y, s, xt, yt):
    ax.annotate(s, (x, y), (xt, yt), fontsize=7, color='#202d37',
                arrowprops=dict(arrowstyle='-', color='#7a8792', lw=0.6))
note(1.0, 1.97, 'A  escalation\n    immediate', 0.15, 1.15)
note(1.52, 2.25, 'B  150 ms outlier: dwell\n    (250 ms) keeps "Near"', 1.85, 2.55)
note(3.75, 0.55, 'C  300 ms dropout:\n    held + dimmed', 3.1, 0.82)
note(5.4, 0.86, 'D  de-escalates after\n    clearing +0.08 for 250 ms', 4.3, 1.55)
note(5.97, 0.75, 'E  700 ms dropout: grace\n    (400 ms) ends, then\n    "No measurement"', 6.1, 1.0)

def band(y, levels, label, hatch=None):
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or levels[i] != levels[start] or (hatch and held[i] != held[start]):
            x0, x1 = t[start], t[i] if i < len(t) else 8
            h = '////' if hatch and held[start] else None
            bx.barh(y, x1 - x0, left=x0, height=0.7, color=COL[levels[start]], hatch=h,
                    edgecolor='white' if h else 'none', lw=0)
            start = i
    bx.text(-0.08, y, label, ha='right', va='center', fontsize=7.5)
band(1, raw, 'raw threshold\n(no logic)')
band(0, shown, 'displayed\n(current UI)', hatch=True)
bx.set_yticks([]); bx.set_ylim(-0.55, 1.55)
for s in ('top', 'right', 'left'): bx.spines[s].set_visible(False)
bx.set_xlabel('time (s) - synthetic 30 Hz feed through services/viz/src/feed/pipeline.js')
bx.legend(handles=[Patch(color=COL['safe'], label='Clear'), Patch(color=COL['warn'], label='Near retina'),
                   Patch(color=COL['danger'], label='Critical'), Patch(color=COL['none'], label='no measurement'),
                   Patch(facecolor='#9aa7b1', hatch='////', edgecolor='white', label='held (grace)')],
          ncol=5, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.62), frameon=False)
fig.savefig(HERE / 'alert_trace.png', bbox_inches='tight', facecolor='white')
