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
export default function DeviceHeader({
  status,
  source,
  onSourceChange,
  frame,
}) {
  const confidence =
    source === 'live' && frame?.confidence != null
      ? `${Math.round(frame.confidence * 100)}%`
      : '—'
  const calibration =
    source !== 'live'
      ? 'Not applicable'
      : frame?.calibrated === true
        ? 'Calibrated'
        : frame?.calibrated === false
          ? 'Not calibrated'
          : 'Unknown'
  return (
    <header className="device-header">
      <div className="brand">
        <img className="brand-logo" src="/maps-logo.png" alt="MAPS" />
        <h1>
          Microforceps<span className="brand-divider"> / </span>
          <span className="brand-subtitle">Depth guidance</span>
        </h1>
      </div>
      <div className="telemetry">
        <div>
          <span className="eyebrow">Calibration</span>
          <strong>{calibration}</strong>
        </div>
        <div>
          <span className="eyebrow">Confidence</span>
          <strong
            title={
              source !== 'live'
                ? 'Confidence is not measured for simulated data'
                : 'Reported by the tracking source'
            }
          >
            {confidence}
          </strong>
        </div>
      </div>
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
