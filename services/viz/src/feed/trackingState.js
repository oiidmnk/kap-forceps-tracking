import { EYE_RADIUS_MM, ROD_LENGTH_MM } from '../config.js'
export const STALE_AFTER_MS = 1500
const point = (p) =>
  Array.isArray(p) && p.length === 3 && p.every(Number.isFinite)

// The rod is rigid: keep the tracked midpoint and direction, fix the length.
function fixRodLength(a, b) {
  const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
  const len = Math.hypot(...d)
  if (!(len > 1e-9)) return [a, b]
  const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]
  const h = ROD_LENGTH_MM / 2 / len
  return [m.map((v, i) => v - d[i] * h), m.map((v, i) => v + d[i] * h)]
}

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
      rod_a: map(p.rod_a),
      rod_b: map(p.rod_b),
      rod_contact: p.rod_contact === true,
    }
  } else {
    frame = {
      t: raw.t,
      tip_left: raw.tip_left,
      tip_right: raw.tip_right,
      trocar: raw.trocar,
      light_tip: point(raw.light_tip) ? raw.light_tip : null,
      light_trocar: point(raw.light_trocar) ? raw.light_trocar : null,
      rod_a: point(raw.rod_a) ? raw.rod_a : null,
      rod_b: point(raw.rod_b) ? raw.rod_b : null,
      rod_contact: raw.rod_contact === true,
    }
  }
  // The rod (a free-floating capsule) is optional and only drawn with both ends.
  // rod_contact: the forceps touches it in the video (the stream has already
  // moved the rod onto the forceps tips).
  if (!point(frame.rod_a) || !point(frame.rod_b)) {
    frame.rod_a = frame.rod_b = null
    frame.rod_contact = false
  } else {
    ;[frame.rod_a, frame.rod_b] = fixRodLength(frame.rod_a, frame.rod_b)
  }
  // A frame needs the forceps or the rod; without the forceps its points are
  // all null (see hasForceps), never partly filled.
  if (![frame.tip_left, frame.tip_right, frame.trocar].every(point)) {
    if (!frame.rod_a) return null
    frame.tip_left = frame.tip_right = frame.trocar = null
  }
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

// Forceps in this frame (a frame may carry only the rod).
export const hasForceps = (frame) => !!frame?.tip_left

export function trackingIsStale(lastReceived, now) {
  return lastReceived != null && now - lastReceived >= STALE_AFTER_MS
}
