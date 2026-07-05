// Compact toggle bar for scene layers (shadows, light beam, retina shading).
export default function Controls({ toggles, onToggle }) {
  const items = [
    { key: 'showShadow', label: 'Shadows' },
    { key: 'showBeam', label: 'Light beam' },
    { key: 'showRetina', label: 'Retina' },
  ]
  return (
    <div style={bar}>
      {items.map(({ key, label }) => {
        const on = toggles[key]
        return (
          <button
            key={key}
            onClick={() => onToggle(key)}
            style={{
              ...btn,
              background: on ? 'rgba(127,209,255,0.18)' : 'rgba(255,255,255,0.05)',
              borderColor: on ? '#7fd1ff' : 'rgba(255,255,255,0.15)',
              color: on ? '#dff1ff' : '#8fa3ba',
            }}
          >
            <span style={{ ...dot, background: on ? '#7fd1ff' : '#4a5a70' }} />
            {label}
          </button>
        )
      })}
    </div>
  )
}

const bar = {
  position: 'absolute',
  bottom: 20,
  left: '50%',
  transform: 'translateX(-50%)',
  display: 'flex',
  gap: 8,
  padding: 8,
  borderRadius: 12,
  background: 'rgba(10,14,22,0.72)',
  border: '1px solid rgba(255,255,255,0.1)',
  backdropFilter: 'blur(8px)',
  pointerEvents: 'auto',
}
const btn = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  padding: '6px 12px',
  borderRadius: 8,
  border: '1px solid',
  font: '500 12px sans-serif',
  cursor: 'pointer',
}
const dot = { width: 8, height: 8, borderRadius: '50%' }
