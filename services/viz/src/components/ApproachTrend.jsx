import { useStore } from '../feed/store.js'
import { TREND_WINDOW_MS, STEADY_MM_S } from '../feed/trend.js'
import { STATUS, depthScale } from '../ui.js'
import { DIST_SAFE_MM, DIST_WARN_MM } from '../config.js'

const W = 280
const H = 64
const y = (mm) => H - depthScale(mm) * H

// Minimum distance over the last 10 s on the same square-root scale as the
// depth bars, with the threshold zones behind it, plus approach speed and an
// estimated time to contact (only while approaching, at most 10 s ahead).
export default function ApproachTrend({ readout }) {
  const { trend, levels, frame } = useStore(readout)
  const color = levels ? STATUS[levels.min].color : 'var(--muted)'

  let path = ''
  let last = null
  if (trend) {
    let pen = false
    for (const [t, d] of trend.points) {
      if (d == null) {
        pen = false
        continue
      }
      const px = W - ((trend.now - t) / TREND_WINDOW_MS) * W
      path += `${pen ? 'L' : 'M'}${px.toFixed(1)} ${y(d).toFixed(1)}`
      pen = true
      last = [px, y(d)]
    }
  }

  const v = frame ? trend?.velocity : null
  const ttc = frame ? trend?.ttc : null
  const ttcTone = ttc == null ? null : ttc < 1 ? 'var(--danger)' : ttc < 3 ? 'var(--warn)' : undefined
  const motion =
    v == null
      ? '—'
      : v < -STEADY_MM_S
        ? `Approaching · ${(-v).toFixed(2)} mm/s`
        : v > STEADY_MM_S
          ? `Retracting · ${v.toFixed(2)} mm/s`
          : 'Holding steady'

  return (
    <section className="approach-panel" aria-labelledby="approach-title">
      <div className="section-heading">
        <h2 id="approach-title">Approach</h2>
        <span className="eyebrow">Last 10 s</span>
      </div>
      <svg
        className="approach-chart"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Minimum distance to the retina over the last 10 seconds"
      >
        <rect x="0" y={y(DIST_WARN_MM)} width={W} height={H - y(DIST_WARN_MM)} className="zone-danger" />
        <rect x="0" y={y(DIST_SAFE_MM)} width={W} height={y(DIST_WARN_MM) - y(DIST_SAFE_MM)} className="zone-warn" />
        {path && (
          <path d={path} fill="none" stroke={color} strokeWidth="1.75" vectorEffect="non-scaling-stroke" />
        )}
        {last && frame && <circle cx={last[0]} cy={last[1]} r="2.5" fill={color} />}
      </svg>
      <div className="approach-readout">
        <span>{motion}</span>
        {ttc != null && (
          <strong style={{ color: ttcTone }}>Contact in ≈ {ttc.toFixed(1)} s</strong>
        )}
      </div>
    </section>
  )
}
