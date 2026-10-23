import { nearestTip } from '../geometry.js'
import { STATUS } from '../ui.js'
import { useStore } from '../feed/store.js'

const selectLevel = (s) => s.levels?.min ?? null
const selectDistance = (s) => (s.frame ? nearestTip(s.frame).dist : null)

// View-wide warning: a colored frame around the 3D view plus a banner while
// the nearest tip is in the warning or critical zone, so the state is visible
// without looking at the side panel. Critical proximity pulses unless the
// user prefers reduced motion (see CSS).
export default function SceneAlarm({ readout }) {
  const level = useStore(readout, selectLevel)
  const dist = useStore(readout, selectDistance)
  if (!level || level === 'safe' || dist == null) return null
  const { symbol, label } = STATUS[level]
  return (
    <div className={`scene-alarm scene-alarm-${level}`}>
      <div className="scene-alarm-banner" role="status">
        <span aria-hidden>{symbol}</span> {label} · {dist.toFixed(2)} mm
      </div>
    </div>
  )
}
