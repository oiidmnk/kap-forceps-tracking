import { useEffect, useRef, useState } from 'react'
import { distanceToRetina } from '../geometry.js'
import { measurementQualifier, lastMeasurementStatus, STATUS, formatClearance, depthScale, DEPTH_SCALE_MAX_MM } from '../ui.js'
import { useStore } from '../feed/store.js'
import { DIST_SAFE_MM, DIST_WARN_MM } from '../config.js'
export function ToggleBar({ toggles, onToggle }) {
  const [open, setOpen] = useState(false)
  const root = useRef(null)
  const button = useRef(null)
  useEffect(() => {
    if (!open) return
    const outside = (e) => {
      if (!root.current?.contains(e.target)) setOpen(false)
    }
    const escape = (e) => {
      if (e.key === 'Escape') {
        setOpen(false)
        button.current?.focus()
      }
    }
    document.addEventListener('pointerdown', outside)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('keydown', escape)
    }
  }, [open])
  return (
    <div className="display-control" ref={root}>
      <button
        ref={button}
        className="display-button"
        aria-expanded={open}
        aria-controls="display-options"
        onClick={() => setOpen(!open)}
      >
        <span aria-hidden>☷</span> Display <span aria-hidden>⌄</span>
      </button>
      {open && (
        <div id="display-options" className="display-popover">
          <h3>Scene layers</h3>
          {[
            ['showRetina', 'Retinal surface'],
            ['showReticle', 'Distance reticle'],
            ['showInstrumentShadow', 'Instrument shadow'],
            ['showShadow', 'Shadow rays'],
            ['cutaway', 'Globe cutaway'],
            ['showBeam', 'Light beam'],
            ['showProximityScope', 'Proximity scope'],
          ].map(([key, label]) => (
            <label key={key}>
              <span>{label}</span>
              <input
                type="checkbox"
                checked={toggles[key]}
                onChange={() => onToggle(key)}
              />
            </label>
          ))}
        </div>
      )}
    </div>
  )
}
const pct = (mm) => `${(depthScale(mm) * 100).toFixed(2)}%`
const TICKS = [0, DIST_WARN_MM, DIST_SAFE_MM, 5, DEPTH_SCALE_MAX_MM]

// Distance panel: the minimum distance as the primary readout, and both tips
// on compact depth bars (square-root scale, threshold zones shaded). Numbers
// come from the readout store (raw measurements, refreshed at
// READOUT_INTERVAL_MS); colors and labels use the hysteresis levels.
export default function HUD({ readout, source }) {
  const { frame, levels, held, nearestTip, lastLevel, lastClearance, lostAtMs } = useStore(readout)
  const [now, setNow] = useState(() => performance.now())
  useEffect(() => {
    if (lostAtMs == null) return
    setNow(performance.now())
    const timer = setInterval(() => setNow(performance.now()), 500)
    return () => clearInterval(timer)
  }, [lostAtMs])
  const tips = frame
    ? [
        { key: 'L', name: 'Left tip', dist: distanceToRetina(frame.tip_left), level: levels?.left },
        { key: 'R', name: 'Right tip', dist: distanceToRetina(frame.tip_right), level: levels?.right },
      ]
    : []
  const minDist = tips.length ? Math.min(...tips.map((t) => t.dist)) : null
  const state = minDist == null || !levels ? null : STATUS[levels.min]
  const nearest = nearestTip
  const unreliable = frame?.calibrated === false
    ? 'Uncalibrated - distances unreliable'
    : frame?.confidence != null && frame.confidence < 0.5 ? 'Low tracking confidence' : null
  const qualifier = measurementQualifier({ source, level: levels?.min, held, frame, hasMeasurement: minDist != null })
  return (
    <section className="distance-panel" aria-labelledby="distance-title">
      <div className="section-heading">
        <h2 id="distance-title">Estimated clearance</h2>
        <span className="eyebrow">Minimum</span>
      </div>
      <div className={`distance-value${minDist == null ? ' distance-empty' : ''}${held ? ' measurement-held' : ''}`} style={{ color: state?.color }}>
        <span>{minDist == null ? '—' : formatClearance(minDist)}</span>
        <span className="distance-unit">mm</span>
      </div>
      <div className={`distance-status${held ? ' measurement-held' : ''}`} style={{ color: state?.color }}>
        <span aria-hidden>{state?.symbol || '○'}</span>
        {state?.label || lastMeasurementStatus(lastLevel, lastClearance, lostAtMs, now)}
        {levels?.retained?.min ? ' · clearing' : ''}
      </div>
      <p className={`distance-qualifier${!held && unreliable ? ' distance-qualifier-warning' : ''}`}>{qualifier}</p>
      <div className={`distance-nearest${held ? ' measurement-held' : ''}`}>{nearest == null ? '—' : nearest === 'tie' ? 'L ≈ R' : nearest === 'left' ? 'L' : 'R'}</div>
      <div
        className={`depth-bars${held ? ' measurement-held' : ''}`}
        style={{ '--warn-at': pct(DIST_WARN_MM), '--safe-at': pct(DIST_SAFE_MM) }}
      >
        {['L', 'R'].map((key, i) => {
          const tip = tips[i]
          const color = tip?.level ? STATUS[tip.level].color : undefined
          return (
            <div className="depth-bar" key={key}>
              <span title={i ? 'Right tip' : 'Left tip'}>{key}</span>
              <div className="depth-track">
                {tip && (
                  <i
                    style={{ left: pct(tip.dist), background: color }}
                    aria-hidden
                  />
                )}
              </div>
              <strong style={{ color }}>
                {tip ? formatClearance(tip.dist) : '—'} <small>mm</small>
              </strong>
            </div>
          )
        })}
        <div className="depth-ticks" aria-hidden>
          {TICKS.map((mm) => (
            <span key={mm} style={{ left: pct(mm) }}>
              {mm}
            </span>
          ))}
        </div>
      </div>
      <div className="threshold-key">
        <span>
          <i style={{ background: 'var(--danger)' }} /> &lt; {DIST_WARN_MM} mm
        </span>
        <span>
          <i style={{ background: 'var(--warn)' }} /> {DIST_WARN_MM}–
          {DIST_SAFE_MM} mm
        </span>
        <span>
          <i style={{ background: 'var(--safe)' }} /> ≥ {DIST_SAFE_MM} mm
        </span>
      </div>
    </section>
  )
}
