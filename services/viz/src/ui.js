import { DIST_SAFE_MM, DIST_WARN_MM } from './config.js'
import { color } from './theme.js'

// Safety coding for a Distance-to-Retina value (mm). This drives a
// SAFETY-CRITICAL readout, so status is never carried by hue alone — every
// level also gets a distinct text `label` and a distinct `symbol` shape, so it
// stays legible under red-green color vision deficiency (the most common kind).
// Hues come from the shared theme — these are the only saturated colors in the UI.
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
    label: 'Approaching retina',
    symbol: '▲',
  },
  safe: {
    level: 'safe',
    color: color.safe,
    label: 'Above warning threshold',
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
