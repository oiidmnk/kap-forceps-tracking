import { Line } from '@react-three/drei'
import { shadowOf, forcepsRenderPose } from '../geometry.js'

// Estimated shadows of the forceps tips on the retina, cast by the light pipe:
// a ray from the light tip through each forceps tip onto the sphere interior.
// Shows the rays, the two shadow points, and the line joining them.
// Rays pass through the *rendered* tip positions (jaw spread amplified by
// VIEW_JAW_SPREAD), so the shadow line visibly runs through the drawn tips
// rather than the raw, un-amplified tracking coordinates.
export default function Shadow({ frame, showRays = true }) {
  if (!frame?.light_tip) return null
  const { light_tip } = frame
  const { tipLeftRender, tipRightRender } = forcepsRenderPose(frame)
  const ls = shadowOf(tipLeftRender, light_tip)
  const rs = shadowOf(tipRightRender, light_tip)

  return (
    <group>
      {showRays && ls && (
        <Line points={[light_tip, tipLeftRender, ls]} color="#fb7185" lineWidth={1} dashed dashSize={0.15} gapSize={0.08} transparent opacity={0.55} />
      )}
      {showRays && rs && (
        <Line points={[light_tip, tipRightRender, rs]} color="#f43f5e" lineWidth={1} dashed dashSize={0.15} gapSize={0.08} transparent opacity={0.55} />
      )}
      {ls && rs && <Line points={[ls, rs]} color="#fb7185" lineWidth={2.5} />}

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
