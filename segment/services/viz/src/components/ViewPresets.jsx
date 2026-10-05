import { useEffect } from 'react'
import { color, font, radius } from '../theme.js'

// Bottom-center segmented control for the main-camera view presets. Free orbit
// stays available; these just snap the camera to a standardized clinical frame.
// Keys 1/2/3 mirror the buttons (free — debug mode uses WASD/QE/arrows).
const PRESETS = [
  { key: 'overview', label: 'Overview' },
  { key: 'surgeon', label: 'Surgeon' },
  { key: 'sagittal', label: 'Sagittal' },
]

export default function ViewPresets({ view, onSelect }) {
  useEffect(() => {
    const onKey = (e) => {
      const idx = { '1': 0, '2': 1, '3': 2 }[e.key]
      if (idx != null) onSelect(PRESETS[idx].key)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onSelect])

  return (
    <div style={bar}>
      {PRESETS.map(({ key, label }) => {
        const on = view === key
        return (
          <button
            key={key}
            onClick={() => onSelect(key)}
            style={{
              ...btn,
              background: on ? color.accentSoft : 'transparent',
              borderColor: on ? color.accentBorder : 'transparent',
              color: on ? color.text : color.textDim,
            }}
          >
            {label}
          </button>
        )
      })}
      <span style={hint}>drag to orbit · scroll to zoom</span>
    </div>
  )
}

const bar = {
  position: 'absolute',
  bottom: 20,
  left: '50%',
  transform: 'translateX(-50%)',
  display: 'flex',
  alignItems: 'center',
  gap: 4,
  padding: 5,
  borderRadius: radius.md,
  background: color.surface,
  border: `1px solid ${color.border}`,
  pointerEvents: 'auto',
}
const btn = {
  padding: '6px 14px',
  borderRadius: radius.sm,
  border: '1px solid',
  font: `600 12px ${font.sans}`,
  letterSpacing: '0.04em',
  cursor: 'pointer',
}
const hint = {
  padding: '0 10px 0 6px',
  font: `400 10px ${font.sans}`,
  color: color.textFaint,
  whiteSpace: 'nowrap',
}
