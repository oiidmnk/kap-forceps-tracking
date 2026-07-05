import { distanceToRetina } from '../geometry.js'
import { distanceStatus } from '../ui.js'

// 2D side-profile of the forceps tips approaching the retina wall, isolating the
// depth (Z) axis the operator can't perceive from the 2D microscope feed.
// Vertical position = radial distance from the retina; the wall is at the bottom.
const W = 240
const H = 200
const MAX_MM = 10 // full-scale depth shown
const PAD_TOP = 26
const PAD_BOTTOM = 34
const wallY = H - PAD_BOTTOM
const scale = (wallY - PAD_TOP) / MAX_MM
const yFor = (mm) => wallY - Math.min(Math.max(mm, 0), MAX_MM) * scale

export default function DepthCrossSection({ frame }) {
  const tips = frame
    ? [
        { name: 'L', x: W * 0.38, dist: distanceToRetina(frame.tip_left) },
        { name: 'R', x: W * 0.62, dist: distanceToRetina(frame.tip_right) },
      ]
    : []

  return (
    <div style={panel}>
      <div style={title}>DEPTH TO RETINA — SIDE VIEW</div>
      <svg width={W} height={H}>
        {/* depth gridlines */}
        {[2, 4, 6, 8].map((mm) => (
          <g key={mm}>
            <line x1={30} y1={yFor(mm)} x2={W - 10} y2={yFor(mm)} stroke="rgba(255,255,255,0.08)" />
            <text x={8} y={yFor(mm) + 3} fill="#5f7085" fontSize="9">{mm}</text>
          </g>
        ))}

        {/* retina wall (curved to suggest the sphere interior) */}
        <path
          d={`M 20 ${wallY} Q ${W / 2} ${wallY + 16} ${W - 20} ${wallY}`}
          fill="none"
          stroke="#7fd1ff"
          strokeWidth="3"
        />
        <text x={W - 20} y={wallY + 28} fill="#7fd1ff" fontSize="10" textAnchor="end">RETINA</text>

        {/* tips + gap lines */}
        {tips.map((t) => {
          const s = distanceStatus(t.dist)
          const y = yFor(t.dist)
          return (
            <g key={t.name}>
              <line x1={t.x} y1={y} x2={t.x} y2={wallY} stroke={s.color} strokeWidth="1" strokeDasharray="3 2" opacity="0.7" />
              <circle cx={t.x} cy={y} r="5" fill={s.color} />
              <text x={t.x} y={y - 9} fill={s.color} fontSize="11" textAnchor="middle" fontWeight="600">
                {t.dist.toFixed(2)}
              </text>
              <text x={t.x} y={y + 4} fill="#0b1120" fontSize="8" textAnchor="middle" fontWeight="700">{t.name}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

const panel = {
  position: 'absolute',
  left: 20,
  top: 280,
  background: 'rgba(10,14,22,0.72)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 12,
  padding: '10px 12px',
  backdropFilter: 'blur(8px)',
}
const title = { font: '600 10px sans-serif', letterSpacing: '0.12em', color: '#8fa3ba', marginBottom: 6 }
