import { Html } from '@react-three/drei'
import { nearestTip, midpoint } from '../geometry.js'
import { STATUS } from '../ui.js'
import { useStore } from '../feed/store.js'

// Distance readout next to the drop line in the main view, so the clearance
// can be read where the eyes already are. Position follows the display-rate
// scene frame; the number and color come from the readout store (raw
// measurement, refreshed at the readout rate, hysteresis level).
export default function DistanceLabel({ frame, readout, portal }) {
  const { frame: measured, levels } = useStore(readout)
  if (!frame || !measured || !levels) return null
  const { tip, wall } = nearestTip(frame)
  const { color } = STATUS[levels.min]
  return (
    <Html portal={portal} position={midpoint(tip, wall)} zIndexRange={[2, 2]}>
      <div className="scene-distance" style={{ color, borderColor: color }}>
        {nearestTip(measured).dist.toFixed(2)} mm
      </div>
    </Html>
  )
}
