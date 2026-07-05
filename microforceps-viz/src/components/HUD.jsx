import { distanceToRetina } from '../geometry.js'
import { distanceStatus } from '../ui.js'

// 2D overlay dashboard: the critical safety readout + feed status/controls.
export default function HUD({ frame, status, source, onToggleSource, toggles, onToggle }) {
  const toggleItems = [
    { key: 'showShadow', label: 'Shadows' },
    { key: 'showBeam', label: 'Light beam' },
    { key: 'showRetina', label: 'Retina' },
  ]
  const tips = frame
    ? [
        { name: 'L', tip: frame.tip_left },
        { name: 'R', tip: frame.tip_right },
      ].map((t) => ({ ...t, dist: distanceToRetina(t.tip) }))
    : []

  const minDist = tips.length ? Math.min(...tips.map((t) => t.dist)) : null
  const minStatus = minDist != null ? distanceStatus(minDist) : null

  return (
    <div style={styles.root}>
      {/* Feed status + source toggle + scene controls */}
      <div style={styles.topBar}>
        <span style={{ ...styles.dot, background: statusColor(status) }} />
        <span style={styles.status}>{statusLabel(status)}</span>
        <button style={styles.toggle} onClick={onToggleSource}>
          {nextSourceLabel(source)}
        </button>
        <div style={styles.divider} />
        {toggleItems.map(({ key, label }) => {
          const on = toggles[key]
          return (
            <button
              key={key}
              onClick={() => onToggle(key)}
              style={{
                ...styles.toggleBtn,
                background: on ? 'rgba(127,209,255,0.18)' : 'rgba(255,255,255,0.05)',
                borderColor: on ? '#7fd1ff' : 'rgba(255,255,255,0.15)',
                color: on ? '#dff1ff' : '#8fa3ba',
              }}
            >
              <span style={{ ...styles.toggleDot, background: on ? '#7fd1ff' : '#4a5a70' }} />
              {label}
            </button>
          )
        })}
        {source === 'debug' && (
          <>
            <div style={styles.divider} />
            <div style={styles.debugHint}>WASD sway · Q/E depth · ↑/↓ jaw · ←/→ roll</div>
          </>
        )}
      </div>

      {/* Critical safety metric */}
      <div style={styles.panel}>
        <div style={styles.panelTitle}>DISTANCE TO RETINA</div>
        {minStatus ? (
          <div style={{ ...styles.big, color: minStatus.color }}>
            {minDist.toFixed(2)}
            <span style={styles.unit}> mm</span>
          </div>
        ) : (
          <div style={styles.big}>—</div>
        )}
        <div style={styles.tips}>
          {tips.map((t) => {
            const s = distanceStatus(t.dist)
            return (
              <div key={t.name} style={styles.tipRow}>
                <span style={styles.tipName}>Tip {t.name}</span>
                <span style={{ ...styles.tipVal, color: s.color }}>
                  {t.dist.toFixed(2)} mm
                </span>
              </div>
            )
          })}
        </div>
        {frame && (
          <div style={styles.coords}>
            <div>tipL {fmt(frame.tip_left)}</div>
            <div>tipR {fmt(frame.tip_right)}</div>
            <div>troc {fmt(frame.trocar)}</div>
          </div>
        )}
      </div>

      <div style={styles.hint}>drag to orbit · scroll to zoom</div>
    </div>
  )
}

const fmt = (v) => `[${v.map((n) => n.toFixed(1)).join(', ')}]`

function statusLabel(s) {
  return (
    { mock: 'MOCK FEED', live: 'LIVE', debug: 'DEBUG MODE', connecting: 'CONNECTING…', error: 'NO SIGNAL', idle: 'IDLE' }[
      s
    ] || s
  )
}
function statusColor(s) {
  return { live: '#52c41a', mock: '#7fd1ff', debug: '#c084fc', connecting: '#faad14', error: '#ff4d4f' }[s] || '#888'
}
function nextSourceLabel(source) {
  return { mock: 'Use live feed', live: 'Use debug mode', debug: 'Use mock feed' }[source] || 'Use mock feed'
}

const styles = {
  root: {
    position: 'absolute',
    inset: 0,
    pointerEvents: 'none',
    padding: 20,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
  },
  topBar: { display: 'flex', alignItems: 'center', gap: 10, pointerEvents: 'auto' },
  debugHint: {
    font: '600 11px sans-serif',
    letterSpacing: '0.06em',
    color: '#c084fc',
    background: 'rgba(192,132,252,0.12)',
    border: '1px solid rgba(192,132,252,0.35)',
    borderRadius: 6,
    padding: '4px 10px',
    whiteSpace: 'nowrap',
  },
  dot: { width: 10, height: 10, borderRadius: '50%', boxShadow: '0 0 8px currentColor' },
  status: { font: '600 12px sans-serif', letterSpacing: '0.1em', color: '#cbd5e1' },
  toggle: {
    pointerEvents: 'auto',
    background: 'rgba(255,255,255,0.08)',
    color: '#e6edf3',
    border: '1px solid rgba(255,255,255,0.15)',
    borderRadius: 6,
    padding: '6px 12px',
    font: '500 12px sans-serif',
    cursor: 'pointer',
  },
  divider: {
    width: 1,
    height: 20,
    background: 'rgba(255,255,255,0.15)',
    marginLeft: 4,
    marginRight: 4,
  },
  toggleBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 10px',
    borderRadius: 8,
    border: '1px solid',
    font: '500 11px sans-serif',
    cursor: 'pointer',
    pointerEvents: 'auto',
  },
  toggleDot: { width: 7, height: 7, borderRadius: '50%' },
  panel: {
    position: 'absolute',
    top: 60,
    left: 20,
    width: 240,
    background: 'rgba(10,14,22,0.72)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: 12,
    padding: 16,
    backdropFilter: 'blur(8px)',
  },
  panelTitle: { font: '600 11px sans-serif', letterSpacing: '0.14em', color: '#8fa3ba' },
  big: { font: '700 44px sans-serif', lineHeight: 1.1, marginTop: 4 },
  unit: { fontSize: 18, fontWeight: 500, opacity: 0.7 },
  tips: { marginTop: 12, display: 'flex', flexDirection: 'column', gap: 4 },
  tipRow: { display: 'flex', justifyContent: 'space-between', font: '500 13px sans-serif' },
  tipName: { color: '#8fa3ba' },
  tipVal: { fontVariantNumeric: 'tabular-nums' },
  coords: {
    marginTop: 12,
    paddingTop: 10,
    borderTop: '1px solid rgba(255,255,255,0.08)',
    font: '400 10px ui-monospace, monospace',
    color: '#6b7c91',
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  },
  hint: { font: '400 11px sans-serif', color: '#5a6b80', alignSelf: 'center' },
}
