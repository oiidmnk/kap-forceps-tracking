import { Line } from '@react-three/drei'
import { DoubleSide } from 'three'
import { EYE_RADIUS_MM, LIMBUS_RADIUS_MM } from '../config.js'

// The eye globe (+Y-up convention): a glassy wireframe grid sphere for depth
// feel, with the posterior (lower, -Y) hemisphere shaded as the retina, a
// limbus ring at the top, and faint principal axes. Adapted from the prototype.
export default function EyeGlobe({ showRetina = true }) {
  const R = EYE_RADIUS_MM
  const limbusY = Math.sqrt(R * R - LIMBUS_RADIUS_MM * LIMBUS_RADIUS_MM)

  return (
    <group>
      {/* Glassy wireframe grid sphere */}
      <mesh>
        <sphereGeometry args={[R, 64, 32]} />
        <meshPhysicalMaterial
          color="#9dd8ff"
          wireframe
          transparent
          opacity={0.16}
          roughness={0.2}
          transmission={0.72}
          side={DoubleSide}
        />
      </mesh>

      {/* Retina — lower (-Y) hemisphere */}
      {showRetina && (
        <mesh>
          <sphereGeometry args={[R - 0.08, 64, 28, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2]} />
          <meshStandardMaterial
            color="#b5425f"
            emissive="#2f0711"
            transparent
            opacity={0.34}
            roughness={0.75}
            side={DoubleSide}
          />
        </mesh>
      )}

      {/* Limbus ring at the top (+Y) */}
      <mesh position={[0, limbusY, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[LIMBUS_RADIUS_MM, 0.06, 12, 96]} />
        <meshStandardMaterial color="#78e7ff" emissive="#174e5c" />
      </mesh>

      {/* Principal axes */}
      <Line color="#5a6b88" lineWidth={1} points={[[-R, 0, 0], [R, 0, 0]]} />
      <Line color="#5a6b88" lineWidth={1} points={[[0, -R, 0], [0, R, 0]]} />
      <Line color="#5a6b88" lineWidth={1} points={[[0, 0, -R], [0, 0, R]]} />
    </group>
  )
}
