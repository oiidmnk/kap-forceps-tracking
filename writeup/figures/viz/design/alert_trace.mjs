// Drives the real viz pipeline (services/viz/src/feed/pipeline.js) with a
// synthetic live feed and writes the displayed level per frame to alert_trace.csv.
// Run: node writeup/figures/viz/design/alert_trace.mjs   (plot: alert_trace.py)
import { writeFileSync } from 'node:fs'
import { createTrackingPipeline } from '../../../../services/viz/src/feed/pipeline.js'
import { EYE_RADIUS_MM } from '../../../../services/viz/src/config.js'

// Clearance script (s -> mm); null = incomplete frame (dropout).
const noise = (t) => 0.035 * Math.sin(t * 61) + 0.025 * Math.sin(t * 23 + 1)
function clearance(t) {
  if (t >= 3.6 && t < 3.9) return null // 300 ms dropout while critical: held
  if (t >= 5.6 && t < 6.3) return null // 700 ms dropout: grace expires
  let d
  if (t < 1.0) d = 2.6 - 0.6 * t // approach to the 2 mm boundary
  else if (t >= 1.45 && t < 1.6) d = 2.25 // 150 ms outlier above the boundary: dwell absorbs it
  else if (t < 2.0) d = 2.0 // hover at the boundary (noise only)
  else if (t < 3.0) d = 2.0 - 1.5 * (t - 2) // approach into critical
  else if (t < 5.0) d = 0.5 // hover at the critical boundary
  else if (t < 7.0) d = 0.5 + 0.9 * (t - 5) // retract
  else d = 2.3
  return Math.max(0, d + noise(t))
}

const p = createTrackingPipeline()
p.reset('live', 0)
const rows = ['t_s,clearance_mm,displayed,held']
let lastPublish = -1
for (let ms = 0; ms <= 8000; ms += 1000 / 60) {
  // 30 Hz frames, 60 Hz animation ticks
  if (ms - lastPublish >= 1000 / 30 - 1e-6) {
    lastPublish = ms
    const t = ms / 1000
    const d = clearance(t)
    if (d == null) p.publish(null, 'degraded', ms)
    else {
      const r = EYE_RADIUS_MM - d
      const tip = [0, -r, 0]
      p.publish({ t: ms, tip_left: tip, tip_right: tip, trocar: [0, 11, 3], light_tip: null, light_trocar: null }, 'live', ms)
    }
    const s = p.readout.get()
    const lv = s.levels // the readout publishes level changes immediately
    rows.push([t.toFixed(4), d == null ? '' : d.toFixed(4), lv ? lv.min : 'none', s.held ? 1 : 0].join(','))
  }
  p.tick(ms)
}
writeFileSync(new URL('./alert_trace.csv', import.meta.url), rows.join('\n') + '\n')
console.log('wrote', rows.length - 1, 'rows')
