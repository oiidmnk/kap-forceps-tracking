import { DIST_SAFE_MM, DIST_WARN_MM } from './config.js'
import { color } from './theme.js'

// Safety coding for a Distance-to-Retina value (mm). This drives a
// SAFETY-CRITICAL readout, so status is never carried by hue alone — every
// level also gets a distinct text `label` and a distinct `symbol` shape, so it
// stays legible under red-green color vision deficiency (the most common kind).
// Hues come from the shared theme — these are the only saturated colors in the UI.
export function distanceStatus(mm) {
  if (mm < DIST_WARN_MM) return { level: 'danger', color: color.danger, label: 'DANGER', symbol: '■' }
  if (mm < DIST_SAFE_MM) return { level: 'warn', color: color.warn, label: 'WARN', symbol: '▲' }
  return { level: 'safe', color: color.safe, label: 'SAFE', symbol: '●' }
}
