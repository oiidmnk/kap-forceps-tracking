import { forcepsRenderPose, distanceToRetina } from '../geometry.js'
import { distanceStatus } from '../ui.js'
import { Rod, SteelMaterial } from './primitives.jsx'
import {
  SHAFT_RADIUS_MM,
  JAW_BASE_RADIUS_MM,
  JAW_TIP_RADIUS_MM,
  TIP_INDICATOR_MM,
} from '../config.js'

// Solid microforceps: a brushed-steel 23G shaft (trocar → hinge) and two
// tapered jaw prongs meeting in a proper V at the hinge. The hinge is placed
// back along the shaft from the tips, so the jaws are NOT drawn from the tips'
// midpoint (which would make them point straight out sideways at 180°).
// Jaws stay colored by each tip's Distance to Retina — the safety cue lives on
// the part of the instrument that can touch the wall — with a small emissive
// bead at each very tip.
export default function Forceps({ frame }) {
  if (!frame) return null
  const { tip_left, tip_right, trocar } = frame
  const { hinge, tipLeftRender, tipRightRender } = forcepsRenderPose(frame)

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

      {/* Jaws — tapered prongs, colored by Distance to Retina */}
      {[[tip_left, tipLeftRender], [tip_right, tipRightRender]].map(([tip, renderTip], i) => {
        const { color } = distanceStatus(distanceToRetina(tip))
        return (
          <group key={i}>
            <Rod a={hinge} b={renderTip} radius={JAW_BASE_RADIUS_MM} radiusTop={JAW_TIP_RADIUS_MM}>
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.3}
                metalness={0.35}
                roughness={0.4}
              />
            </Rod>
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
