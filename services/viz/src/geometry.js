// Pure geometry helpers. Vectors are plain [x, y, z] arrays in mm,
// origin at the eye-globe center.

import {
  EYE_RADIUS_MM,
  JAW_LENGTH_MM,
  JAW_EXIT_OFFSET_MM,
  JAW_CLOSED_OFFSET_MM,
  JAW_FULL_OPEN_MM,
} from './config.js'

export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s]
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
export const length = (a) => Math.hypot(a[0], a[1], a[2])
export const midpoint = (a, b) => scale(add(a, b), 0.5)

export function normalize(a) {
  const l = length(a)
  return l > 1e-9 ? scale(a, 1 / l) : [0, 0, 0]
}

// Absolute distance from a forceps tip to the interior sphere wall (the retina),
// measured radially. Tip is expected inside the globe (|tip| <= EYE_RADIUS_MM).
// Returns >= 0; clamps to 0 if the tip is at/through the wall.
export function distanceToRetina(tip, radius = EYE_RADIUS_MM) {
  return Math.max(0, radius - length(tip))
}

// First forward intersection of a ray (origin + t·dir, t>0) with the eye sphere.
// Returns the [x,y,z] hit point, or null if the ray misses.
export function raySphereIntersection(origin, dir, radius = EYE_RADIUS_MM) {
  const d = normalize(dir) // a = 1
  const b = 2 * dot(origin, d)
  const c = dot(origin, origin) - radius * radius
  const disc = b * b - 4 * c
  if (disc < 0) return null
  const s = Math.sqrt(disc)
  const roots = [(-b - s) / 2, (-b + s) / 2].filter((t) => t > 1e-4)
  if (roots.length === 0) return null
  return add(origin, scale(d, Math.min(...roots)))
}

// Shadow of a forceps tip on the retina: cast a ray from the light tip through
// the forceps tip and intersect the sphere interior.
export function shadowOf(tip, lightTip, radius = EYE_RADIUS_MM) {
  return raySphereIntersection(lightTip, sub(tip, lightTip), radius)
}

// The tip closest to the retina, with its distance and nearest wall point.
export function nearestTip(frame, radius = EYE_RADIUS_MM) {
  const dl = distanceToRetina(frame.tip_left, radius)
  const dr = distanceToRetina(frame.tip_right, radius)
  const tip = dl <= dr ? frame.tip_left : frame.tip_right
  const dist = Math.min(dl, dr)
  const wall = scale(normalize(tip), radius) // nearest point on the retina
  return { tip, dist, wall }
}

// Full instrument pose derived from the two tips + the (static) trocar.
// - jawCenter: midpoint of the tips, lies on the shaft axis
// - shaftDir: unit vector from trocar toward the jaw center (pivot constraint)
// - openingMm: tip-to-tip separation (proxy for jaw open/close)
export function forcepsPose(tipLeft, tipRight, trocar) {
  const jawCenter = midpoint(tipLeft, tipRight)
  const shaftDir = normalize(sub(jawCenter, trocar))
  const openingMm = length(sub(tipLeft, tipRight))
  return { jawCenter, shaftDir, openingMm }
}

// Unit vector perpendicular to `axis`, used when the jaws are closed and the
// tip separation no longer defines an opening direction.
function anyPerpendicular(axis) {
  const reference = Math.abs(axis[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]
  return normalize(cross(cross(axis, reference), axis))
}

// Render the tracked tips at their measured positions. The instrument,
// distance reticle, shadows, and scope must all use the same coordinates.
// Only the untracked tube end is inferred, from the nominal jaw length.
//
// Model (Haag-Streit simulator style microforceps): a straight tube whose cut
// end releases two spring jaws. Each jaw leaves the tube at the rim, offset
// sideways from the axis, and runs a constant length to its tracked tip (the
// simulator keeps the exposed length constant while closing). As the jaws
// close the tube squeezes their bases toward the axis, so a shut jaw pair
// reads as one blade tapering to the tips.
// - tubeEnd: centre of the tube's distal rim
// - openingDir: unit vector from the left toward the right jaw, ⟂ shaft
// - widthDir: blade width direction, ⟂ shaft and opening (normal of jaw plane)
// - jawBaseLeft/Right: where each jaw emerges from the tube
export function forcepsRenderPose(frame) {
  const { tip_left, tip_right, trocar } = frame
  const { jawCenter, shaftDir, openingMm } = forcepsPose(tip_left, tip_right, trocar)

  const separation = sub(tip_right, tip_left)
  const lateral = sub(separation, scale(shaftDir, dot(separation, shaftDir)))
  const halfOpen = length(lateral) / 2
  const openingDir = halfOpen > 1e-6 ? normalize(lateral) : anyPerpendicular(shaftDir)
  const widthDir = normalize(cross(shaftDir, openingDir))

  const released = Math.min(1, (2 * halfOpen) / JAW_FULL_OPEN_MM)
  const exitOffset = JAW_CLOSED_OFFSET_MM + (JAW_EXIT_OFFSET_MM - JAW_CLOSED_OFFSET_MM) * released
  const spread = halfOpen - exitOffset
  const axial = Math.sqrt(Math.max(JAW_LENGTH_MM ** 2 - spread ** 2, (0.5 * JAW_LENGTH_MM) ** 2))
  const tubeEnd = sub(jawCenter, scale(shaftDir, axial))

  return {
    jawCenter,
    shaftDir,
    openingMm,
    openingDir,
    widthDir,
    tubeEnd,
    jawBaseLeft: sub(tubeEnd, scale(openingDir, exitOffset)),
    jawBaseRight: add(tubeEnd, scale(openingDir, exitOffset)),
    tipLeftRender: [...tip_left],
    tipRightRender: [...tip_right],
  }
}

// Retinal wall point picked by a view ray (double-click focus). From outside
// the globe the near shell is see-through (or cut away), so the far wall is the
// surface actually seen; from inside it is the first wall ahead.
export function pickWall(origin, dir, radius = EYE_RADIUS_MM) {
  const d = normalize(dir)
  const b = dot(origin, d)
  const c = dot(origin, origin) - radius * radius
  const disc = b * b - c
  if (disc < 0) return null
  const t = -b + Math.sqrt(disc) // far root; also the forward hit from inside
  return t > 0 ? add(origin, scale(d, t)) : null
}

// Instrument state for the panel: jaw opening (tip separation), insertion
// depth from the trocar to the jaws, and the shaft's angle to the retinal
// surface where its axis would land (90° = perpendicular, 0° = tangential).
export function instrumentMetrics(frame, radius = EYE_RADIUS_MM) {
  const { jawCenter, shaftDir, openingMm } = forcepsPose(frame.tip_left, frame.tip_right, frame.trocar)
  const landing = raySphereIntersection(jawCenter, shaftDir, radius)
  const angle = landing
    ? (Math.asin(Math.min(1, Math.abs(dot(shaftDir, normalize(landing))))) * 180) / Math.PI
    : null
  return {
    openingMm,
    insertionMm: length(sub(jawCenter, frame.trocar)),
    retinaAngleDeg: angle,
  }
}
