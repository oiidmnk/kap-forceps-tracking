#!/bin/sh
# Rebuild every figure used by writeup/viz_design.md (run from the repo root).
# Needs node (for the alert trace) and .venv-ml with matplotlib + Pillow.
set -e
P=.venv-ml/bin/python
V=writeup/figures/viz
A=$V/annotations
$P $V/design/approach_speed.py
$P $V/alarm_eval/status_summary.py
node $V/design/alert_trace.mjs && $P $V/design/alert_trace.py
$P $V/diagrams/generate_diagrams.py
for s in fig_live_critical fig_live_ghost fig_asymmetric_q2 fig_overview_map fig_panel_before fig_panel_after fig_cues_before_scene fig_cues_before_scope \
         fig_cues_after_scene fig_cues_after_scope; do $P $V/annotate.py $A/$s.json >/dev/null; done
for v in 03_warning_overview 05_warning_surgeon 06_warning_sagittal 07_warning_fundus; do
  $P $V/annotate.py $A/parts/view_$v.json --out $A/parts/view_$v.png >/dev/null
done
C="$P $V/design/compose.py"
$C $A/fig_panel_before_after.png "Before" $A/fig_panel_before.png "After" $A/fig_panel_after.png
$C $A/parts/cues_before.png "Before · main view" $A/fig_cues_before_scene.png "Before · scope" $A/fig_cues_before_scope.png
$C $A/parts/cues_after.png "After · main view" $A/fig_cues_after_scene.png "After · scope" $A/fig_cues_after_scope.png
$C -v $A/fig_cues_before_after.png "" $A/parts/cues_before.png "" $A/parts/cues_after.png
$C $A/parts/views_1.png "Overview (1)" $A/parts/view_03_warning_overview.png "Surgeon (2)" $A/parts/view_05_warning_surgeon.png
$C $A/parts/views_2.png "Sagittal (3)" $A/parts/view_06_warning_sagittal.png "Fundus (4)" $A/parts/view_07_warning_fundus.png
$C -v $A/fig_view_presets.png "" $A/parts/views_1.png "" $A/parts/views_2.png

$C $A/fig_live_states.png "Live Critical · steady frame" $A/fig_live_critical.png "After loss · last reading" $A/fig_live_ghost.png
