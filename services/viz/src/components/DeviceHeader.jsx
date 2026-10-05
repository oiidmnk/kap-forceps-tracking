import { useEffect, useState } from 'react'
import { color, font, space, radius, tnum, HEADER_H } from '../theme.js'

// Slim device chrome across the top: product identity on the left, live
// telemetry (calibration, tracking confidence, clock) and feed status on the
// right. This framing is what makes the dashboard read as a certified
// instrument rather than a web page.
const STATUS_LABEL = {
  live: 'LIVE',
  mock: 'MOCK FEED',
  debug: 'DEBUG MODE',
  connecting: 'CONNECTING…',
  error: 'NO SIGNAL',
  idle: 'IDLE',
}
const statusColor = (s) => ({ live: color.safe, connecting: color.warn, error: color.danger }[s] || color.accent)
const nextSourceLabel = (s) => ({ mock: 'Use live feed', live: 'Use debug mode', debug: 'Use mock feed' }[s] || 'Use mock feed')

export default function DeviceHeader({ status, source, onToggleSource, frame }) {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  const conf = frame?.confidence != null ? Math.round(frame.confidence * 100) : null
  const confTone = conf == null ? color.textDim : conf < 80 ? color.warn : color.text

  return (
    <header style={bar}>
      <div style={identity}>
        <span style={mark} />
        <span style={title}>MICROFORCEPS · DIGITAL TWIN</span>
        <span style={subtitle}>Vitreoretinal Depth Guidance</span>
      </div>

      <div style={right}>
        <Metric label="CALIBRATION" value="CALIBRATED" tone={color.safe} />
        <Metric label="CONFIDENCE" value={conf != null ? `${conf}%` : '—'} tone={confTone} />
        <Metric label="TIME" value={now.toLocaleTimeString('en-GB')} mono />
        <span style={divider} />
        <div style={statusWrap}>
          <span style={{ ...dot, background: statusColor(status) }} />
          <span style={statusLabel}>{STATUS_LABEL[status] || status}</span>
        </div>
        <button style={sourceBtn} onClick={onToggleSource}>
          {nextSourceLabel(source)}
        </button>
      </div>
    </header>
  )
}

function Metric({ label, value, tone = color.text, mono }) {
  return (
    <div style={metric}>
      <span style={metricLabel}>{label}</span>
      <span style={{ ...metricVal, ...tnum, color: tone, fontFamily: mono ? font.mono : font.sans }}>{value}</span>
    </div>
  )
}

const bar = {
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  height: HEADER_H,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: `0 ${space.xl}px`,
  background: 'rgba(9,13,19,0.92)',
  borderBottom: `1px solid ${color.border}`,
  pointerEvents: 'auto',
  zIndex: 10,
}
const identity = { display: 'flex', alignItems: 'baseline', gap: space.md }
const mark = {
  alignSelf: 'center',
  width: 10,
  height: 10,
  borderRadius: 2,
  border: `2px solid ${color.accent}`,
  transform: 'rotate(45deg)',
}
const title = { font: `700 13px ${font.sans}`, letterSpacing: '0.16em', color: color.text }
const subtitle = { font: `500 11px ${font.sans}`, letterSpacing: '0.08em', color: color.textFaint }

const right = { display: 'flex', alignItems: 'center', gap: space.lg }
const metric = { display: 'flex', flexDirection: 'column', alignItems: 'flex-end', lineHeight: 1.25 }
const metricLabel = { font: `600 8px ${font.sans}`, letterSpacing: '0.16em', color: color.textFaint }
const metricVal = { font: `600 12px ${font.sans}` }
const divider = { width: 1, height: 22, background: color.border }
const statusWrap = { display: 'flex', alignItems: 'center', gap: space.sm }
const dot = { width: 8, height: 8, borderRadius: '50%' }
const statusLabel = { font: `600 12px ${font.sans}`, letterSpacing: '0.1em', color: color.text }
const sourceBtn = {
  background: color.surfaceInset,
  color: color.text,
  border: `1px solid ${color.border}`,
  borderRadius: radius.sm,
  padding: '6px 12px',
  font: `500 12px ${font.sans}`,
  cursor: 'pointer',
}
