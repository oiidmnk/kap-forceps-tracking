import { DIST_SAFE_MM, DIST_WARN_MM } from './config.js'

// Safety color coding for a Distance-to-Retina value (mm).
export function distanceStatus(mm) {
  if (mm < DIST_WARN_MM) return { level: 'danger', color: '#ff4d4f' }
  if (mm < DIST_SAFE_MM) return { level: 'warn', color: '#faad14' }
  return { level: 'safe', color: '#52c41a' }
}
