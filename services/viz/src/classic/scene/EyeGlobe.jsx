import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import { DoubleSide, BackSide } from 'three'
import { EYE_RADIUS_MM, LIMBUS_RADIUS_MM } from '../config.js'

// The eye globe (+Y-up convention): a clean latitude/longitude reference grid
// (not a triangulated wireframe — no moiré diagonals) over a faint glassy
// shell, with the posterior (lower, -Y) hemisphere shaded as the retina, a
// limbus ring at the top, and dim principal axes. The grid's curvature is the
// primary depth cue; colors stay on the neutral steel / muted-anatomical
// palette so it reads clinical.

const PARALLEL_STEP_DEG = 15
const MERIDIAN_STEP_DEG = 20
const SEGMENTS = 96

// theta = polar angle from +Y (0 at top pole). Grid spans limbus → near -Y pole.
function gridLines(R, limbusTheta) {
  const parallels = []
  for (let deg = 45; deg <= 165; deg += PARALLEL_STEP_DEG) {
    const theta = (deg * Math.PI) / 180
    if (theta <= limbusTheta) continue
    const y = R * Math.cos(theta)
    const r = R * Math.sin(theta)
    const ring = []
    for (let s = 0; s <= SEGMENTS; s++) {
      const phi = (s / SEGMENTS) * Math.PI * 2
      ring.push([r * Math.cos(phi), y, r * Math.sin(phi)])
    }
    parallels.push(ring)
  }

  const meridians = []
  const thetaEnd = (172 * Math.PI) / 180 // stop short of the pole to avoid clutter
  for (let deg = 0; deg < 360; deg += MERIDIAN_STEP_DEG) {
    const phi = (deg * Math.PI) / 180
    const arc = []
    for (let s = 0; s <= 48; s++) {
      const theta = limbusTheta + (s / 48) * (thetaEnd - limbusTheta)
      arc.push([
        R * Math.sin(theta) * Math.cos(phi),
        R * Math.cos(theta),
        R * Math.sin(theta) * Math.sin(phi),
      ])
    }
    meridians.push(arc)
  }
  return { parallels, meridians }
}

export default function EyeGlobe({ showRetina = true }) {
  const R = EYE_RADIUS_MM
  const limbusY = Math.sqrt(R * R - LIMBUS_RADIUS_MM * LIMBUS_RADIUS_MM)
  const limbusTheta = Math.asin(LIMBUS_RADIUS_MM / R)
  const { parallels, meridians } = useMemo(() => gridLines(R, limbusTheta), [R, limbusTheta])

  return (
    <group>
      {/* Glassy shell — interior backface slightly stronger for volume depth */}
      <mesh>
        <sphereGeometry args={[R, 64, 32, 0, Math.PI * 2, limbusTheta, Math.PI - limbusTheta]} />
        <meshPhysicalMaterial color="#9db4cc" transparent opacity={0.05} roughness={0.15} side={BackSide} depthWrite={false} />
      </mesh>
      <mesh>
        <sphereGeometry args={[R, 64, 32, 0, Math.PI * 2, limbusTheta, Math.PI - limbusTheta]} />
        <meshPhysicalMaterial color="#9db4cc" transparent opacity={0.03} roughness={0.15} depthWrite={false} />
      </mesh>

      {/* Reference grid — parallels + meridians as crisp screen-space lines */}
      {parallels.map((pts, i) => (
        <Line key={`par-${i}`} points={pts} color="#64778c" lineWidth={0.75} transparent opacity={0.45} />
      ))}
      {meridians.map((pts, i) => (
        <Line key={`mer-${i}`} points={pts} color="#64778c" lineWidth={0.75} transparent opacity={0.3} />
      ))}

      {/* Retina — lower (-Y) hemisphere: smooth, muted anatomical rose */}
      {showRetina && (
        <mesh>
          <sphereGeometry args={[R - 0.08, 64, 28, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2]} />
          <meshStandardMaterial
            color="#8b3a4a"
            emissive="#1a060c"
            transparent
            opacity={0.22}
            roughness={0.85}
            depthWrite={false}
            side={DoubleSide}
          />
        </mesh>
      )}

      {/* Limbus ring at the top (+Y) — neutral steel */}
      <mesh position={[0, limbusY, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[LIMBUS_RADIUS_MM, 0.06, 12, 96]} />
        <meshStandardMaterial color="#8aa0b6" emissive="#1c2733" />
      </mesh>

      {/* Principal axes — dim reference only */}
      <Line color="#3c4a63" lineWidth={0.75} points={[[-R, 0, 0], [R, 0, 0]]} />
      <Line color="#3c4a63" lineWidth={0.75} points={[[0, -R, 0], [0, R, 0]]} />
      <Line color="#3c4a63" lineWidth={0.75} points={[[0, 0, -R], [0, 0, R]]} />
    </group>
  )
}
