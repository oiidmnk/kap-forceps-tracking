import { Line } from '@react-three/drei'
import { shadowOf } from '../geometry.js'

// Estimated shadows of the forceps tips on the retina, cast by the light pipe:
// a ray from the light tip through each forceps tip onto the sphere interior.
// Shows the rays, the two shadow points, and the line joining them.
export default function Shadow({ frame, showRays = true }) {
  if (!frame?.light_tip) return null
  const { tip_left, tip_right, light_tip } = frame
  const ls = shadowOf(tip_left, light_tip)
  const rs = shadowOf(tip_right, light_tip)

  return (
    <group>
      {showRays && ls && (
        <Line points={[light_tip, tip_left, ls]} color="#fb7185" lineWidth={1.5} dashed dashSize={0.15} gapSize={0.08} />
      )}
      {showRays && rs && (
        <Line points={[light_tip, tip_right, rs]} color="#f43f5e" lineWidth={1.5} dashed dashSize={0.15} gapSize={0.08} />
      )}
      {ls && rs && <Line points={[ls, rs]} color="#fb7185" lineWidth={3} />}

      {ls && (
        <mesh position={ls}>
          <sphereGeometry args={[0.12, 16, 12]} />
          <meshStandardMaterial color="#fb7185" emissive="#7f1d1d" />
        </mesh>
      )}
      {rs && (
        <mesh position={rs}>
          <sphereGeometry args={[0.12, 16, 12]} />
          <meshStandardMaterial color="#f43f5e" emissive="#7f1d1d" />
        </mesh>
      )}
    </group>
  )
}
