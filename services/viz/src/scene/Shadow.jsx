import DynamicLine from './DynamicLine.jsx'
import { shadowOf, forcepsRenderPose } from '../geometry.js'

// Estimated shadows of the forceps tips on the retina, cast by the light pipe:
// a ray from the light tip through each forceps tip onto the sphere interior.
// Shows the rays, the two shadow points, and the line joining them.
// Rendered and tracked tip positions coincide, so rays and distance guides
// meet the same points on the instrument.
export default function Shadow({ frame, showRays = true }) {
  if (!frame?.light_tip) return null
  const { light_tip } = frame
  const { tipLeftRender, tipRightRender } = forcepsRenderPose(frame)
  const ls = shadowOf(tipLeftRender, light_tip)
  const rs = shadowOf(tipRightRender, light_tip)

  return (
    <group>
      {showRays && ls && (
        <DynamicLine points={[light_tip, tipLeftRender, ls]} color="#fb7185" lineWidth={1} dashed dashSize={0.15} gapSize={0.08} transparent opacity={0.55} />
      )}
      {showRays && rs && (
        <DynamicLine points={[light_tip, tipRightRender, rs]} color="#f43f5e" lineWidth={1} dashed dashSize={0.15} gapSize={0.08} transparent opacity={0.55} />
      )}
      {ls && rs && <DynamicLine points={[ls, rs]} color="#fb7185" lineWidth={2.5} />}

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
