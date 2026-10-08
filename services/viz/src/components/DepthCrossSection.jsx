import { useState } from 'react'
import { distanceToRetina } from '../geometry.js'
import { distanceStatus } from '../ui.js'
import { DIST_SAFE_MM, DIST_WARN_MM } from '../config.js'
export default function DepthCrossSection({ frame }) {
  const [near, setNear] = useState(false)
  const max = near ? 3 : 10
  const y = (mm) => 172 - (Math.min(Math.max(mm, 0), max) / max) * 142
  const tips = frame
    ? [frame.tip_left, frame.tip_right].map((tip, i) => ({
        label: i ? 'R' : 'L',
        x: i ? 207 : 122,
        dist: distanceToRetina(tip),
      }))
    : []
  return (
    <section className="depth-panel" aria-labelledby="depth-title">
      <div className="section-heading">
        <h2 id="depth-title">Depth profile</h2>
        <button
          className="scale-button"
          aria-pressed={near}
          onClick={() => setNear(!near)}
        >
          {near ? '0–3 mm' : '0–10 mm'} <span aria-hidden>⌕</span>
        </button>
      </div>
      <svg
        viewBox="0 0 292 203"
        role="img"
        aria-label={`Radial distance to retina, zero to ${max} millimeters. Warning threshold ${DIST_SAFE_MM} mm; danger threshold ${DIST_WARN_MM} mm.`}
      >
        <rect
          x="42"
          y={y(DIST_SAFE_MM)}
          width="228"
          height={y(DIST_WARN_MM) - y(DIST_SAFE_MM)}
          fill="var(--warn)"
          opacity="0.06"
        />
        <rect
          x="42"
          y={y(DIST_WARN_MM)}
          width="228"
          height={172 - y(DIST_WARN_MM)}
          fill="var(--danger)"
          opacity="0.12"
        />
        {(near ? [1, 2, 3] : [2, 4, 6, 8, 10]).map((mm) => (
          <g key={mm}>
            <line
              x1="42"
              x2="270"
              y1={y(mm)}
              y2={y(mm)}
              stroke="var(--border)"
            />
            <text x="25" y={y(mm) + 4} textAnchor="end">
              {mm}
            </text>
          </g>
        ))}
        {[
          { mm: DIST_SAFE_MM, tone: 'var(--warn)' },
          { mm: DIST_WARN_MM, tone: 'var(--danger)' },
        ].map((t) => (
          <g key={t.mm}>
            <line
              x1="42"
              x2="270"
              y1={y(t.mm)}
              y2={y(t.mm)}
              stroke={t.tone}
              strokeOpacity="0.6"
              strokeDasharray="3 4"
            />
            {t.mm === DIST_WARN_MM && (
              <text
                x={near ? 25 : 283}
                y={y(t.mm) + (near ? 4 : -6)}
                textAnchor="end"
                style={{ fill: t.tone }}
              >
                {t.mm}
              </text>
            )}
          </g>
        ))}
        <line
          x1="42"
          x2="270"
          y1="172"
          y2="172"
          stroke="#aab9c9"
          strokeWidth="2"
        />
        <text x="25" y="176" textAnchor="end">
          0
        </text>
        <text x="270" y="196" textAnchor="end">
          Retinal surface
        </text>
        <text x="42" y="16">
          mm
        </text>
        {tips.map((t) => (
          <g key={t.label} style={{ color: distanceStatus(t.dist).color }}>
            <line
              x1={t.x}
              x2={t.x}
              y1={y(t.dist)}
              y2="172"
              stroke="currentColor"
              strokeDasharray="3 4"
              opacity="0.7"
            />
            <circle cx={t.x} cy={y(t.dist)} r="5" fill="currentColor" />
            <text
              x={t.x}
              y={y(t.dist) - 12}
              textAnchor="middle"
              style={{ fill: 'currentColor', fontWeight: 600 }}
            >
              {t.label} {t.dist > max ? '↑ ' : ''}
              {t.dist.toFixed(2)}
            </text>
          </g>
        ))}
      </svg>
      <p className="panel-note">
        {near
          ? 'Magnified scale · values above 3 mm marked ↑'
          : 'Radial clearance · select scale to magnify'}
      </p>
    </section>
  )
}
