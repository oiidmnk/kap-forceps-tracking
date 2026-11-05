import { nearestTip } from '../geometry.js'
import { STATUS, formatClearance } from '../ui.js'
import { useStore } from '../feed/store.js'
import { CRITICAL_PULSE_LIVE } from '../config.js'

const selectLevel = (s) => s.levels?.min ?? null
const selectDistance = (s) => (s.frame ? nearestTip(s.frame).dist : null)

// View-wide warning: a colored frame around the 3D view plus a banner while
// the nearest tip is in the warning or critical zone, so the state is visible
// without looking at the side panel. Critical proximity pulses unless the
// user prefers reduced motion (see CSS) or the data is live tracking with a
// tracker whose Critical is not yet reliable (CRITICAL_PULSE_LIVE).
export default function SceneAlarm({ readout, source }) {
  const level = useStore(readout, selectLevel)
  const dist = useStore(readout, selectDistance)
  const held = useStore(readout, (s) => s.held)
  const retained = useStore(readout, (s) => s.levels?.retained?.min)
  if (!level || level === 'safe' || dist == null) return null
  const { symbol, label } = STATUS[level]
  return (
    <div className={`scene-alarm scene-alarm-${level}${source === 'live' && !CRITICAL_PULSE_LIVE ? ' scene-alarm-steady' : ''}`}>
      <div className="scene-alarm-banner" role="status">
        <span aria-hidden>{symbol}</span> {label}{retained ? ' · clearing' : ''} · <span style={{ opacity: held ? 0.55 : undefined }}>{formatClearance(dist)} mm</span>
      </div>
    </div>
  )
}
