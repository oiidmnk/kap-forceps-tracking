import { useStore } from '../feed/store.js'
import { TREND_WINDOW_MS, STEADY_MM_S } from '../feed/trend.js'
import { STATUS, depthScale } from '../ui.js'
import { DIST_SAFE_MM, DIST_WARN_MM } from '../config.js'

const W = 280
const H = 64
const y = (mm) => H - depthScale(mm) * H

// Minimum distance over the last 10 s on the same square-root scale as the
// depth bars, with labelled threshold boundaries and approach speed.
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
  const motion =
    v == null
      ? '—'
      : v < -STEADY_MM_S
        ? `Approaching · ${(-v).toFixed(1)} mm/s`
        : v > STEADY_MM_S
          ? `Retracting · ${v.toFixed(1)} mm/s`
          : `Holding steady · ${Math.abs(v).toFixed(1)} mm/s`

  return (
    <section className="approach-panel" aria-labelledby="approach-title">
      <div className="section-heading">
        <h2 id="approach-title">Approach</h2>
        <span className="eyebrow">Last 10 s</span>
      </div>
      <svg
        className="approach-chart"
        viewBox={`0 0 ${W + 32} ${H}`}
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
        {[DIST_WARN_MM, DIST_SAFE_MM].map((mm) => (
          <g key={mm} className="approach-boundary">
            <line x1="0" x2={W} y1={y(mm)} y2={y(mm)} />
            <text x={W + 8} y={y(mm)} dominantBaseline="middle">{mm}</text>
          </g>
        ))}
      </svg>
      <div className="approach-axis" aria-hidden><span>−10 s</span><span>now</span></div>
      <div className="approach-readout">
        <span>{motion}</span>
      </div>
    </section>
  )
}
