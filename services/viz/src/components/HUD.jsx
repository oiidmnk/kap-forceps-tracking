import { useEffect, useRef, useState } from 'react'
import { distanceToRetina } from '../geometry.js'
import { distanceStatus } from '../ui.js'
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
            ['showShadow', 'Projected shadows'],
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
export default function HUD({ frame, status }) {
  const tips = frame
    ? [
        { name: 'Left tip', dist: distanceToRetina(frame.tip_left) },
        { name: 'Right tip', dist: distanceToRetina(frame.tip_right) },
      ]
    : []
  const minDist = tips.length ? Math.min(...tips.map((t) => t.dist)) : null
  const state = minDist == null ? null : distanceStatus(minDist)
  return (
    <section className="distance-panel" aria-labelledby="distance-title">
      <div className="section-heading">
        <h2 id="distance-title">Distance to retina</h2>
        <span className="eyebrow">Minimum</span>
      </div>
      <div className="distance-value" style={{ color: state?.color }}>
        <span>{minDist == null ? '—' : minDist.toFixed(2)}</span>
        <span className="distance-unit">mm</span>
      </div>
      <div className="distance-status" style={{ color: state?.color }}>
        <span aria-hidden>{state?.symbol || '○'}</span>
        {state?.label || (status === 'stale' ? 'Data stale' : 'No measurement')}
      </div>
      <div className="tip-readings">
        {['Left tip', 'Right tip'].map((name, i) => (
          <div key={name}>
            <span>{name}</span>
            <strong>
              {tips[i] ? tips[i].dist.toFixed(2) : '—'} <small>mm</small>
            </strong>
          </div>
        ))}
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
