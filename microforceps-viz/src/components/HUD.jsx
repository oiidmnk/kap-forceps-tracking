import { distanceToRetina } from '../geometry.js'
import { distanceStatus } from '../ui.js'
import { color, font, space, radius, tnum, HEADER_H } from '../theme.js'
import { SIDE_PANEL_W } from './SidePanel.jsx'

// Scene layer toggles — rendered above the 3D canvas, right of the side panel.
export function ToggleBar({ toggles, onToggle, source }) {
  const toggleItems = [
    { key: 'showShadow', label: 'Shadows' },
    { key: 'showBeam', label: 'Light beam' },
    { key: 'showRetina', label: 'Retina' },
    { key: 'showReticle', label: 'Reticle' },
  ]

  return (
    <div style={styles.toggleBarWrap}>
      <div style={styles.topBar}>
        {toggleItems.map(({ key, label }) => {
          const on = toggles[key]
          return (
            <button
              key={key}
              onClick={() => onToggle(key)}
              style={{
                ...styles.toggleBtn,
                background: on ? color.accentSoft : color.surfaceInset,
                borderColor: on ? color.accentBorder : color.border,
                color: on ? color.text : color.textDim,
              }}
            >
              <span style={{ ...styles.toggleDot, background: on ? color.accent : color.textFaint }} />
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
    </div>
  )
}

// Critical safety readout: Distance to Retina panel. Rendered inside the
// SidePanel flex column so it inherits the panel width automatically.
export default function HUD({ frame }) {
  const tips = frame
    ? [
        { name: 'L', tip: frame.tip_left },
        { name: 'R', tip: frame.tip_right },
      ].map((t) => ({ ...t, dist: distanceToRetina(t.tip) }))
    : []

  const minDist = tips.length ? Math.min(...tips.map((t) => t.dist)) : null
  const minStatus = minDist != null ? distanceStatus(minDist) : null

  return (
    <div style={styles.panel}>
      <div style={styles.panelTitle}>DISTANCE TO RETINA</div>
      {minStatus ? (
        <>
          <div style={{ ...styles.big, ...tnum, color: minStatus.color }}>
            {minDist.toFixed(2)}
            <span style={styles.unit}> mm</span>
          </div>
          {/* Redundant, non-hue safety cue (symbol + word) for CVD legibility */}
          <div style={{ ...styles.badge, color: minStatus.color, borderColor: minStatus.color }}>
            <span aria-hidden>{minStatus.symbol}</span>
            {minStatus.label}
          </div>
        </>
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
                <span aria-hidden style={{ marginRight: 5 }}>{s.symbol}</span>
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
  )
}

const fmt = (v) => `[${v.map((n) => n.toFixed(1)).join(', ')}]`

const styles = {
  // Toggle bar lives above the canvas, offset past the side panel
  toggleBarWrap: {
    position: 'absolute',
    top: HEADER_H + space.md,
    left: SIDE_PANEL_W + 24 + space.lg,
    pointerEvents: 'none',
    zIndex: 5,
  },
  topBar: { display: 'flex', alignItems: 'center', gap: space.sm, pointerEvents: 'auto' },
  debugHint: {
    font: `600 11px ${font.sans}`,
    letterSpacing: '0.06em',
    color: color.textDim,
    background: color.surfaceInset,
    border: `1px solid ${color.border}`,
    borderRadius: radius.sm,
    padding: '4px 10px',
    whiteSpace: 'nowrap',
  },
  divider: {
    width: 1,
    height: 18,
    background: color.border,
    marginLeft: space.xs,
    marginRight: space.xs,
  },
  toggleBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: space.xs + 2,
    padding: '6px 10px',
    borderRadius: radius.sm,
    border: '1px solid',
    font: `500 11px ${font.sans}`,
    cursor: 'pointer',
    pointerEvents: 'auto',
  },
  toggleDot: { width: 6, height: 6, borderRadius: '50%' },

  // Distance-to-Retina panel — flow layout inside SidePanel
  panel: {
    background: color.surface,
    border: `1px solid ${color.border}`,
    borderRadius: radius.md,
    padding: space.lg,
    pointerEvents: 'auto',
  },
  panelTitle: { font: `600 11px ${font.sans}`, letterSpacing: '0.14em', color: color.textDim },
  big: { font: `700 44px ${font.sans}`, lineHeight: 1.1, marginTop: space.xs },
  unit: { fontSize: 18, fontWeight: 500, color: color.textDim },
  badge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: space.xs + 2,
    marginTop: space.sm,
    padding: '3px 10px',
    borderRadius: radius.sm,
    border: '1px solid',
    font: `700 12px ${font.sans}`,
    letterSpacing: '0.12em',
    background: color.surfaceInset,
  },
  tips: { marginTop: space.md, display: 'flex', flexDirection: 'column', gap: space.xs },
  tipRow: { display: 'flex', justifyContent: 'space-between', font: `500 13px ${font.sans}` },
  tipName: { color: color.textDim },
  tipVal: { ...tnum },
  coords: {
    marginTop: space.md,
    paddingTop: space.sm + 2,
    borderTop: `1px solid ${color.border}`,
    font: `400 10px ${font.mono}`,
    color: color.textFaint,
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  },
}
