import { useEffect } from 'react'
const presets = [
  { key: 'overview', label: 'Overview' },
  { key: 'surgeon', label: 'Surgeon' },
  { key: 'sagittal', label: 'Sagittal' },
]
export default function ViewPresets({ view, onSelect }) {
  useEffect(() => {
    const onKey = (e) => {
      if (
        e.target.closest('input, select, textarea, [contenteditable="true"]') ||
        e.altKey ||
        e.ctrlKey ||
        e.metaKey
      )
        return
      const preset = presets[Number(e.key) - 1]
      if (preset) onSelect(preset.key)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onSelect])
  return (
    <div className="view-presets" role="group" aria-label="Camera view">
      {presets.map((p, i) => (
        <button
          key={p.key}
          aria-pressed={view === p.key}
          onClick={() => onSelect(p.key)}
          title={`${p.label} view (${i + 1})`}
        >
          {p.label}
        </button>
      ))}
    </div>
  )
}
