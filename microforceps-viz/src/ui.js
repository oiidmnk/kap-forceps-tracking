import { DIST_SAFE_MM, DIST_WARN_MM } from './config.js'

// Safety coding for a Distance-to-Retina value (mm). This drives a
// SAFETY-CRITICAL readout, so status is never carried by hue alone — every
// level also gets a distinct text `label` and a distinct `symbol` shape, so it
// stays legible under red-green color vision deficiency (the most common kind).
export function distanceStatus(mm) {
  if (mm < DIST_WARN_MM) return { level: 'danger', color: '#ff4d4f', label: 'DANGER', symbol: '■' }
  if (mm < DIST_SAFE_MM) return { level: 'warn', color: '#faad14', label: 'WARN', symbol: '▲' }
  return { level: 'safe', color: '#52c41a', label: 'SAFE', symbol: '●' }
}
