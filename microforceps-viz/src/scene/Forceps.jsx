import { Line } from '@react-three/drei'
import { forcepsRenderPose, distanceToRetina } from '../geometry.js'
import { distanceStatus } from '../ui.js'

// Minimal forceps: a thin shaft line (trocar -> hinge) and two delicate jaw
// lines (hinge -> tips) that meet in a proper V. The hinge is placed back along
// the shaft from the tips, so the jaws are NOT drawn from the tips' midpoint
// (which would make them point straight out sideways at 180°). Jaw lines are
// colored by each tip's Distance to Retina.
export default function Forceps({ frame }) {
  if (!frame) return null
  const { tip_left, tip_right, trocar } = frame
  const { hinge, tipLeftRender, tipRightRender } = forcepsRenderPose(frame)

  return (
    <group>
      {/* Shaft */}
      <Line points={[trocar, hinge]} color="#aab4c4" lineWidth={3.5} />

      {/* Jaws — delicate lines meeting at the hinge, colored by Distance to Retina */}
      {[[tip_left, tipLeftRender], [tip_right, tipRightRender]].map(([tip, renderTip], i) => {
        const { color } = distanceStatus(distanceToRetina(tip))
        return <Line key={i} points={[hinge, renderTip]} color={color} lineWidth={2.5} />
      })}
    </group>
  )
}
