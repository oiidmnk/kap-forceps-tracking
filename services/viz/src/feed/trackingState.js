import { EYE_RADIUS_MM } from '../config.js'
export const STALE_AFTER_MS = 1500
const point = (p) =>
  Array.isArray(p) && p.length === 3 && p.every(Number.isFinite)

// Accept native millimeter frames and the upstream normalized reconstruction.
// Never substitute old tips into a new frame: that would imply current tracking.
export function parseTrackingFrame(raw) {
  if (!raw || typeof raw !== 'object') return null
  let frame
  if (raw.positions) {
    const p = raw.positions
    const map = (v) =>
      point(v)
        ? [v[0] * EYE_RADIUS_MM, -v[2] * EYE_RADIUS_MM, v[1] * EYE_RADIUS_MM]
        : null
    frame = {
      t: raw.timestamp,
      tip_left: map(p.left_tip_forceps),
      tip_right: map(p.right_tip_forceps),
      trocar: map(p.trocar_forceps),
      light_tip: map(p.tip_light),
      light_trocar: map(p.trocar_light),
    }
  } else {
    frame = {
      t: raw.t,
      tip_left: raw.tip_left,
      tip_right: raw.tip_right,
      trocar: raw.trocar,
      light_tip: point(raw.light_tip) ? raw.light_tip : null,
      light_trocar: point(raw.light_trocar) ? raw.light_trocar : null,
    }
  }
  if (![frame.tip_left, frame.tip_right, frame.trocar].every(point)) return null
  return {
    ...frame,
    confidence:
      Number.isFinite(raw.confidence) &&
      raw.confidence >= 0 &&
      raw.confidence <= 1
        ? raw.confidence
        : null,
    calibrated: typeof raw.calibrated === 'boolean' ? raw.calibrated : null,
  }
}

export function trackingIsStale(lastReceived, now) {
  return lastReceived != null && now - lastReceived >= STALE_AFTER_MS
}
