"""Diagrams for writeup/viz_design.md (English, current logic of services/viz/src/feed/*.js).

Run: .venv-ml/bin/python writeup/figures/viz/diagrams/generate_diagrams.py
Writes tracking_states_alert_logic.{svg,png} and websocket_to_pixels.{svg,png}.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
INK = '#202d37'; MUT = '#50606c'; ACC = '#356b8b'; LINE = '#b9c4cb'
SAFE = '#1ea6ac'; WARN = '#e0b000'; CRIT = '#ef5255'  # warn darkened for white paper


def canvas(title, subtitle, height):
    fig, ax = plt.subplots(figsize=(12, height / 100), dpi=200)
    fig.patch.set_facecolor('white'); ax.set_xlim(0, 1200); ax.set_ylim(height, 0); ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    text(ax, 45, 30, title, 24, True)
    text(ax, 45, 63, subtitle, 14, color=MUT)
    return fig, ax


def text(ax, x, y, s, size=16, bold=False, color=INK, ha='left'):
    ax.text(x, y, s, fontsize=size * .72, color=color, weight='bold' if bold else 'normal',
            ha=ha, va='top', linespacing=1.45)


def box(ax, x, y, w, h, title, body='', status=None, fill='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=9', fc=fill,
                                ec=status or LINE, lw=1.3))
    if status: ax.plot([x + 12, x + w - 12], [y + 3, y + 3], c=status, lw=4, solid_capstyle='round')
    text(ax, x + 16, y + 14, title, 17, True)
    if body: text(ax, x + 16, y + 44, body, 14.5)


def arrow(ax, x, y, xx, yy, label=None, lx=None, ly=None, rad=0.0, color=ACC):
    ax.add_patch(FancyArrowPatch((x, y), (xx, yy), arrowstyle='-|>', mutation_scale=13, lw=1.2,
                                 color=color, connectionstyle=f'arc3,rad={rad}'))
    if label:
        text(ax, lx if lx is not None else (x + xx) / 2, ly if ly is not None else (y + yy) / 2,
             label, 13, color=color, ha='center')


def save(fig, name):
    fig.savefig(OUT / (name + '.svg'), facecolor='white')
    fig.savefig(OUT / (name + '.png'), dpi=200, facecolor='white')
    plt.close(fig)


# --- 1  Tracking states and alert logic ------------------------------------------------------
fig, ax = canvas('Tracking availability and alert logic',
                 'Two independent state axes: is there a current measurement (A), and how close is it (B)', 1130)
text(ax, 45, 100, 'A  Availability (live source; pipeline.js, useTrackingFeed.js)', 18, True, color=ACC)
box(ax, 45, 135, 200, 92, 'Connecting', 'source = live\nno socket yet')
box(ax, 300, 135, 200, 92, 'Waiting', 'socket open\nno valid frame yet')
box(ax, 560, 135, 300, 92, 'Tracking', 'valid frame, source time advanced\nnumbers + levels from raw frame')
box(ax, 915, 135, 240, 92, 'Disconnected', 'socket error / close\nno grace, no auto-reconnect')
arrow(ax, 245, 181, 300, 181, 'open', 272, 160)
arrow(ax, 500, 181, 560, 181, 'valid', 530, 160)
arrow(ax, 860, 181, 915, 181)
box(ax, 45, 290, 340, 150, 'Held  (< 400 ms grace)',
    'incomplete / invalid frame, or frozen\nsource timestamp; values dimmed,\nlevel kept, never lowered;\nnext valid frame → Tracking')
box(ax, 430, 290, 330, 150, 'Unavailable', 'gap ≥ 400 ms\nnumbers "—", "No measurement"\nlevels cleared, alarm ends\ntrend gap, motion word "—"')
box(ax, 805, 290, 350, 150, 'Stale', 'source time not advanced for ≥ 1.5 s\n(or no frame for ≥ 1.5 s)\nheader "Stale data"; checked every\nanimation frame')
arrow(ax, 600, 227, 300, 290, 'dropout / frozen t', 390, 238)
arrow(ax, 385, 365, 430, 365)
arrow(ax, 780, 227, 930, 290, '≥ 1.5 s without new source time', 1045, 236)
text(ax, 45, 452, 'After a previous live frame, Unavailable / Stale / Disconnected show the last measured pose as a faded ghost with',
     14, color=MUT)
text(ax, 45, 475, 'HUD: "No measurement · last: <level> <clearance> mm, n s ago". Current levels stay null; recovery removes the ghost.',
     14, color=MUT)

text(ax, 45, 505, 'B  Distance level per tip (levels.js); displayed level = more severe tip', 18, True, color=ACC)
box(ax, 45, 545, 320, 120, 'Clear  ●   d ≥ 2 mm', 'no scene alarm', SAFE)
box(ax, 440, 545, 320, 120, 'Near retina  ▲   0.5 ≤ d < 2', 'scene: 4 px frame + 15 px banner', WARN)
box(ax, 835, 545, 320, 120, 'Critical proximity  ■   d < 0.5', '6 px frame; 15 px banner\nlive: steady; simulation: 2 Hz glow\nno pulse with reduced motion', CRIT)
arrow(ax, 365, 585, 440, 585, 'd < 2', 402, 563)
arrow(ax, 760, 585, 835, 585, 'd < 0.5', 797, 563)
arrow(ax, 440, 640, 365, 640, '≥ 2.08', 402, 645, color=MUT)
arrow(ax, 835, 640, 760, 640, '≥ 0.58', 797, 645, color=MUT)
box(ax, 45, 700, 1110, 150, 'Rules',
    '1  Escalation is immediate on the raw threshold (one frame, no dwell, no smoothing).\n'
    '2  De-escalation needs d ≥ threshold + 0.08 mm (≈ 2 × inferred sample noise, assuming independence) continuously for 250 ms (dwell);\n'
    '    robust σ(Δh) = 0.052 mm → σ(sample) ≈ 0.037 mm; gaps / new escalations restart dwell; Critical → Clear can be direct.\n'
    '3  Missing data never lowers the level: within the 400 ms grace the level is held, afterwards it is cleared (not "Clear").',
    fill='#f6f8f9')
box(ax, 45, 875, 540, 128, 'Proximity scope (separate switch)',
    'active when min(dL, dR) < 5 mm; off at ≥ 5.25 mm\nrequires current levels and no ghost\nlayout space is permanent (no auto-promotion)')
box(ax, 615, 875, 540, 128, 'Where the level is used',
    'clearance number + status word + symbol, L/R markers,\ntrend line, scope readout, scene frame / banner,\ndrop line, reticle, distance label, tip markers')
text(ax, 45, 1030, 'Sources: services/viz/src/feed/pipeline.js · levels.js · trackingState.js · hooks/useTrackingFeed.js · config.js · ui.js ·',
     11, color=MUT)
text(ax, 45, 1050, 'components/SceneAlarm.jsx · workstation.css (alarm-pulse 0.5 s period). Values: DIST_WARN/SAFE 0.5/2, '
     'LEVEL_HYSTERESIS 0.08, DWELL 250, GRACE 400, STALE 1500 ms.', 11, color=MUT)
save(fig, 'tracking_states_alert_logic')

# --- 2  Data path and timing -----------------------------------------------------------------
fig, ax = canvas('From the microscope image to the pixels',
                 'Numbers and alarm use raw measurements; only the 3D pose is interpolated (and therefore later)', 1120)
box(ax, 45, 100, 250, 118, 'Tracker (upstream)', 'microscope video →\nsegmenter → 3D tips\n(tool_positions.py)')
box(ax, 345, 100, 250, 118, 'Stream server', 'JSON per frame on /ws\ntimestamp = send time\n(ws_server.py)')
box(ax, 645, 100, 250, 118, 'Parse + validate', 'both tips + trocar, 3 finite\nnumbers each; else null\n(trackingState.js)')
box(ax, 945, 100, 210, 118, 'Pipeline', 'freshness, grace,\nlevels, trend\n(pipeline.js)')
arrow(ax, 295, 159, 345, 159); arrow(ax, 595, 159, 645, 159); arrow(ax, 895, 159, 945, 159)
text(ax, 45, 228, 'upstream latency (capture, detection, transport) is not measured in this document', 13, color=MUT)

ax.plot([1050, 1050], [218, 246], c=ACC, lw=1.2); ax.plot([555, 1135], [246, 246], c=ACC, lw=1.2)
arrow(ax, 555, 246, 555, 295); arrow(ax, 1135, 246, 1135, 295)
text(ax, 45, 262, 'A  Readout path (numbers, status, alarm)', 18, True, color=ACC)
text(ax, 625, 262, 'B  Scene path (3D views, scope)', 18, True, color=ACC)
box(ax, 45, 295, 530, 150, 'Raw frame → clearance',
    'dL/R = max(0, 12 mm − ‖tip‖), d = min(dL, dR)\nno spatial smoothing, no bias correction\nlevels with hysteresis / dwell / grace (Fig. A)')
box(ax, 625, 295, 530, 150, 'Display interpolator',
    'drops repeated poses; up to 4 samples\ninterval I ← 0.85·I + 0.15·gap (start 66 ms)\nrender time = now − clamp(1.2·I, 30, 150) ms\nlinear blend between two measured poses')
arrow(ax, 310, 445, 310, 480); arrow(ax, 890, 445, 890, 480)
box(ax, 45, 490, 530, 160, 'Readout store (nominal 10 Hz)',
    'published at most every 100 ms;\nlevel / availability changes publish at once\nnumbers 1 decimal; trend 10 s, slope over 600 ms\nmotion word at |v| > 0.15 mm/s')
box(ax, 625, 490, 530, 160, 'Scene store (every animation frame)',
    '≈ 80 ms behind at 15 Hz input\nholds the newest pose, never extrapolates\nscope camera follows with τ = 0.08 s\ncolours from the same levels as A')
text(ax, 45, 680, 'Time constants as distance travelled at simulator approach speeds', 18, True, color=ACC)
text(ax, 45, 708, 'speed of the ground-truth tip height while approaching below 3 mm (3 runs, 1940 frames): '
     'median 0.53, p75 1.0, p95 2.2 mm/s', 13, color=MUT)
rows = [('', 'time', '0.53 mm/s', '1.0 mm/s', '2.2 mm/s', 'meaning'),
        ('readout throttle', '≤ 100 ms', '≤ 0.05', '≤ 0.10', '≤ 0.22', 'number can lag; level changes do not'),
        ('scene interpolation', '≈ 80 ms', '0.04', '0.08', '0.18', 'pose / drop line behind the number'),
        ('de-escalation dwell', '250 ms', '0.13', '0.25', '0.55', 'alarm stays on while retracting'),
        ('dropout grace', '400 ms', '0.21', '0.40', '0.88', 'held value may be this far off'),
        ('stale timeout', '1.5 s', '0.80', '1.50', '3.3', 'frozen source before "Data stale"')]
cols = [45, 250, 370, 500, 630, 760]
for i, r in enumerate(rows):
    y = 740 + i * 34
    if i == 0:
        ax.plot([45, 1155], [y + 28, y + 28], c=LINE, lw=1)
    for x, c in zip(cols, r):
        text(ax, x, y, c, 14.5, bold=(i == 0), color=INK if i else MUT)
text(ax, 45, 960, 'Distances in mm (time × speed). For comparison: median height error of the tracker ≈ 0.6 mm, '
     'hysteresis 0.08 mm, critical threshold 0.5 mm.', 13, color=MUT)
text(ax, 45, 990, 'The interpolation and throttle together are small against the measurement error; grace and stale '
     'are not, which is why held values are dimmed', 13, color=MUT)
text(ax, 45, 1012, 'and the held level may never be lowered.', 13, color=MUT)
text(ax, 45, 1060, 'Sources: services/viz/src/feed/displayInterpolation.js · pipeline.js · trend.js · config.js · '
     'scene/ScopeCamera.jsx · services/stream/ws_server.py; speeds: writeup/figures/viz/design/approach_speed.py',
     11, color=MUT)
save(fig, 'websocket_to_pixels')
