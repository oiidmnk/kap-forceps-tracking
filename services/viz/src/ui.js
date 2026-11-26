import { DIST_SAFE_MM, DIST_WARN_MM } from './config.js'
import { color } from './theme.js'

// Safety coding for a Distance-to-Retina value (mm). This drives a
// SAFETY-CRITICAL readout, so status is never carried by hue alone — every
// level also gets a distinct text `label` and a distinct `symbol` shape, so it
// stays legible under red-green color vision deficiency (the most common kind).
// Hues come from the shared theme — these are the only saturated colors in the UI.
// Clearance with one decimal, rounded DOWN: the shown number is never larger
// than the estimate, so it agrees with the instantaneous level (0.46 mm shows
// "0.4" next to "Critical proximity", 1.97 mm shows "1.9" next to "Near retina").
// A level held by hysteresis/dwell is marked " · clearing" instead.
export const formatClearance = (mm) => (Math.floor(mm * 10 + 1e-9) / 10).toFixed(1)

// Display-conditioned simulator replay, finite measurements, P10–P90.
export const LIVE_ACCURACY_NOTES = {
  danger: 'Sim: Critical shown → tip 1–5.6 mm away',
  warn: 'Sim: Near shown → tip 1.6–3.4 mm away',
  safe: 'Sim runs: typically reads ≈ 0.6 mm too close',
}
export const SIMULATION_NOTE = 'Simulation: exact geometry, no tracker error'

export function measurementQualifier({ source, level, held, frame, hasMeasurement }) {
  if (held) return 'Signal gap · holding last value'
  if (frame?.calibrated === false) return 'Uncalibrated - distances unreliable'
  if (frame?.confidence != null && frame.confidence < 0.5) return 'Low tracking confidence'
  if (!hasMeasurement) return 'Distances unavailable until a complete frame arrives'
  return source === 'mock' || source === 'debug' ? SIMULATION_NOTE : LIVE_ACCURACY_NOTES[level]
}

export const STATUS = {
  danger: {
    level: 'danger',
    color: color.danger,
    label: 'Critical proximity',
    symbol: '■',
  },
  warn: {
    level: 'warn',
    color: color.warn,
    label: 'Near retina',
    symbol: '▲',
  },
  safe: {
    level: 'safe',
    color: color.safe,
    label: 'Clear',
    symbol: '●',
  },
}

export function distanceLevel(mm) {
  if (mm < DIST_WARN_MM) return 'danger'
  if (mm < DIST_SAFE_MM) return 'warn'
  return 'safe'
}

export function distanceStatus(mm) {
  return STATUS[distanceLevel(mm)]
}

// Position (0–1) of a distance on the panel's depth bars. Square-root scale
// over 0–DEPTH_SCALE_MAX_MM so the critical first millimetres get most of the
// room: 0.5 mm sits at ~22 %, 2 mm at ~45 %.
export const DEPTH_SCALE_MAX_MM = 10
export function depthScale(mm) {
  return Math.sqrt(Math.min(Math.max(mm, 0), DEPTH_SCALE_MAX_MM) / DEPTH_SCALE_MAX_MM)
}

// Historical context only: callers keep current levels null during a ghost.
export function lastMeasurementStatus(level, clearance, lostAtMs, now) {
  if (lostAtMs == null || !STATUS[level] || clearance == null) return 'No measurement'
  const seconds = Math.max(0, Math.floor((now - lostAtMs) / 1000))
  return `No measurement · last: ${STATUS[level].label} ${formatClearance(clearance)} mm, ${seconds} s ago`
}
