import { Line } from '@react-three/drei'
import { DoubleSide } from 'three'
import { EYE_RADIUS_MM, LIMBUS_RADIUS_MM } from '../config.js'

// The eye globe (+Y-up convention): a glassy wireframe grid sphere for depth
// feel, with the posterior (lower, -Y) hemisphere shaded as the retina, a
// limbus ring at the top, and faint principal axes. The dense wireframe is the
// primary depth cue (its curvature reads the sphere as a 3D volume); colors are
// kept on the neutral steel / muted-anatomical palette so it stays clinical.
export default function EyeGlobe({ showRetina = true }) {
  const R = EYE_RADIUS_MM
  const limbusY = Math.sqrt(R * R - LIMBUS_RADIUS_MM * LIMBUS_RADIUS_MM)
  const limbusTheta = Math.asin(LIMBUS_RADIUS_MM / R)

  return (
    <group>
      {/* Glassy wireframe grid sphere — neutral steel instead of neon cyan, with opening at the limbus */}
      <mesh>
        <sphereGeometry args={[R, 64, 32, 0, Math.PI * 2, limbusTheta, Math.PI - limbusTheta]} />
        <meshPhysicalMaterial
          color="#89a3bd"
          wireframe
          transparent
          opacity={0.15}
          roughness={0.2}
          transmission={0.72}
          side={DoubleSide}
        />
      </mesh>

      {/* Retina — lower (-Y) hemisphere: muted anatomical rose */}
      {showRetina && (
        <mesh>
          <sphereGeometry args={[R - 0.08, 64, 28, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2]} />
          <meshStandardMaterial
            color="#a5455d"
            emissive="#2a0810"
            transparent
            opacity={0.34}
            roughness={0.78}
            side={DoubleSide}
          />
        </mesh>
      )}

      {/* Limbus ring at the top (+Y) — neutral steel */}
      <mesh position={[0, limbusY, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[LIMBUS_RADIUS_MM, 0.06, 12, 96]} />
        <meshStandardMaterial color="#8aa0b6" emissive="#1c2733" />
      </mesh>

      {/* Principal axes */}
      <Line color="#4a5876" lineWidth={1} points={[[-R, 0, 0], [R, 0, 0]]} />
      <Line color="#4a5876" lineWidth={1} points={[[0, -R, 0], [0, R, 0]]} />
      <Line color="#4a5876" lineWidth={1} points={[[0, 0, -R], [0, 0, R]]} />
    </group>
  )
}
