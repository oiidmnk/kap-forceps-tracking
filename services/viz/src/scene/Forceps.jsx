import { forcepsRenderPose, distanceToRetina, sub } from '../geometry.js'
import { distanceStatus } from '../ui.js'
import { Rod, SteelMaterial } from './primitives.jsx'
import Jaw from './Jaw.jsx'
import {
  SHAFT_RADIUS_MM,
  JAW_BASE_RADIUS_MM,
  JAW_TIP_RADIUS_MM,
  TIP_INDICATOR_MM,
  JAW_FLATTEN,
} from '../config.js'

// Solid microforceps: a brushed-steel shaft (trocar → hinge) and two
// tapered jaw prongs meeting in a proper V at the hinge. The hinge is placed
// back along the shaft from the tips, so the jaws are NOT drawn from the tips'
// midpoint (which would make them point straight out sideways at 180°).
// Thin steel blades follow the tracked endpoints. Small colored tip markers
// carry the distance state without enlarging the jaw opening.
export default function Forceps({ frame }) {
  if (!frame) return null
  const { tip_left, tip_right, trocar } = frame
  const { hinge, tipLeftRender, tipRightRender } = forcepsRenderPose(frame)
  const openingDir = sub(tip_right, tip_left)

  return (
    <group>
      {/* Shaft — slight taper toward the hinge */}
      <Rod a={trocar} b={hinge} radius={SHAFT_RADIUS_MM} radiusTop={SHAFT_RADIUS_MM * 0.8}>
        <SteelMaterial />
      </Rod>

      {/* Hinge collar where the prongs emerge */}
      <mesh position={hinge}>
        <sphereGeometry args={[SHAFT_RADIUS_MM * 0.85, 20, 16]} />
        <SteelMaterial roughness={0.45} />
      </mesh>

      {/* Thin blades with distance-colored tip markers */}
      {[[tip_left, tipLeftRender], [tip_right, tipRightRender]].map(([tip, renderTip], i) => {
        const { color } = distanceStatus(distanceToRetina(tip))
        return (
          <group key={i}>
            <Jaw
              from={hinge}
              to={renderTip}
              openingDir={openingDir}
              radiusFrom={JAW_BASE_RADIUS_MM}
              radiusTo={JAW_TIP_RADIUS_MM}
              flatten={JAW_FLATTEN}
            />
            <mesh position={renderTip}>
              <sphereGeometry args={[TIP_INDICATOR_MM, 12, 10]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.2} />
            </mesh>
          </group>
        )
      })}
    </group>
  )
}
