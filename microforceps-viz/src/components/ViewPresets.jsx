import { useEffect } from 'react'

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
              background: on ? 'rgba(127,209,255,0.18)' : 'transparent',
              borderColor: on ? '#7fd1ff' : 'transparent',
              color: on ? '#dff1ff' : '#8fa3ba',
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
  borderRadius: 12,
  background: 'rgba(10,14,22,0.72)',
  border: '1px solid rgba(255,255,255,0.1)',
  backdropFilter: 'blur(8px)',
  pointerEvents: 'auto',
}
const btn = {
  padding: '6px 14px',
  borderRadius: 8,
  border: '1px solid',
  font: '600 12px sans-serif',
  letterSpacing: '0.04em',
  cursor: 'pointer',
}
const hint = {
  padding: '0 10px 0 6px',
  font: '400 10px sans-serif',
  color: '#5a6b80',
  whiteSpace: 'nowrap',
}
