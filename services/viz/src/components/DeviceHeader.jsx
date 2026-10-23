import { useStore } from '../feed/store.js'
const labels = {
  live: 'Receiving data',
  mock: 'Simulation active',
  debug: 'Manual simulation',
  connecting: 'Connecting',
  waiting: 'Awaiting data',
  error: 'Disconnected',
  stale: 'Stale data',
  degraded: 'Incomplete data',
  idle: 'Idle',
}
// Selectors return primitives, so the header re-renders only when its text
// actually changes.
const selectStatus = (s) => s.status
const selectConfidence = (s) => s.frame?.confidence ?? null
const selectCalibrated = (s) => s.frame?.calibrated ?? null

export default function DeviceHeader({ readout, source, onSourceChange }) {
  const status = useStore(readout, selectStatus)
  const rawConfidence = useStore(readout, selectConfidence)
  const calibrated = useStore(readout, selectCalibrated)
  const confidence = rawConfidence != null ? `${Math.round(rawConfidence * 100)}%` : '—'
  const calibration =
    calibrated === true ? 'Calibrated' : calibrated === false ? 'Not calibrated' : 'Unknown'
  // Calibration and confidence only exist for live tracking; simulated data
  // has neither, so the header leaves them out.
  return (
    <header className="device-header">
      <div className="brand">
        <img className="brand-logo" src="/maps-logo.png" alt="MAPS" />
        <h1>
          Microforceps<span className="brand-divider"> / </span>
          <span className="brand-subtitle">Depth guidance</span>
        </h1>
      </div>
      {source === 'live' && (
        <div className="telemetry">
          <div>
            <span className="eyebrow">Calibration</span>
            <strong>{calibration}</strong>
          </div>
          <div>
            <span className="eyebrow">Confidence</span>
            <strong title="Reported by the tracking source">{confidence}</strong>
          </div>
        </div>
      )}
      <div className="source-control">
        <label className="eyebrow" htmlFor="data-source">
          Data source
        </label>
        <select
          id="data-source"
          value={source}
          onChange={(e) => onSourceChange(e.target.value)}
        >
          <option value="mock">Simulation</option>
          <option value="live">Live tracking</option>
          <option value="debug">Manual simulation</option>
        </select>
      </div>
      <div className={`connection connection-${status}`} role="status">
        <i />
        <span>{labels[status] || status}</span>
      </div>
    </header>
  )
}
