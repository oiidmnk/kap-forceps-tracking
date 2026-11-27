import { EYE_RADIUS_MM, ROD_LENGTH_MM, ROD_RADIUS_MM, STALE_MS } from '../config.js'
export const STALE_AFTER_MS = STALE_MS
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

const ROD_CLEARANCE_MM = 0.05 // gap kept between a tip and the rod surface without contact

// Closest point to p on segment a-b.
function closestOnSegment(p, a, b) {
  const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
  const dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
  const t = dd > 0 ? Math.min(1, Math.max(0, ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2]) / dd)) : 0
  return [a[0] + t * d[0], a[1] + t * d[1], a[2] + t * d[2]]
}

// Forceps tips (and their midpoint) to the rod surface, mm; 0 when inside.
export function rodGap(frame) {
  if (!frame?.rod_a || !frame?.tip_left) return null
  const tips = [frame.tip_left, frame.tip_right, frame.tip_left.map((v, i) => (v + frame.tip_right[i]) / 2)]
  const d = Math.min(...tips.map((p) => Math.hypot(...p.map((v, i) => v - closestOnSegment(p, frame.rod_a, frame.rod_b)[i]))))
  return Math.max(0, d - ROD_RADIUS_MM)
}

// Without contact in the video the forceps must not pierce the rod: move the
// rod (as a whole) away from each tip that is inside it. In contact the stream
// has already put the rod onto the tips.
function clearRod(frame) {
  let [a, b] = [frame.rod_a, frame.rod_b]
  const keep = ROD_RADIUS_MM + ROD_CLEARANCE_MM
  for (let pass = 0; pass < 3; pass++) {
    for (const p of [frame.tip_left, frame.tip_right, frame.tip_left.map((v, i) => (v + frame.tip_right[i]) / 2)]) {
      const c = closestOnSegment(p, a, b)
      let away = c.map((v, i) => v - p[i])
      let dist = Math.hypot(...away)
      if (dist >= keep) continue
      if (dist < 1e-9) {
        // tip on the axis: push perpendicular to the rod
        const d = b.map((v, i) => v - a[i])
        away = Math.abs(d[1]) < 0.9 * Math.hypot(...d) ? [-d[2], 0, d[0]] : [0, d[2], -d[1]]
        dist = 0
      }
      const n = Math.hypot(...away)
      const shift = away.map((v) => (v / n) * (keep - dist))
      a = a.map((v, i) => v + shift[i])
      b = b.map((v, i) => v + shift[i])
    }
  }
  return [a, b]
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
      t: raw.timestamp ?? raw.t,
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
      t: raw.t ?? raw.timestamp,
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
  if (frame.rod_a && frame.tip_left && !frame.rod_contact) {
    ;[frame.rod_a, frame.rod_b] = clearRod(frame)
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
