// Pure geometry helpers. Vectors are plain [x, y, z] arrays in mm,
// origin at the eye-globe center.

import { EYE_RADIUS_MM, JAW_LENGTH_MM, VIEW_JAW_SPREAD } from './config.js'

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

// Full render-space pose: the hinge and the two tip positions as they're
// actually drawn (jaw opening amplified by VIEW_JAW_SPREAD for legibility).
// Anything that needs to frame or measure against what's on screen — not just
// the raw tracking data — should read from this, so the camera and the
// geometry it's framing never disagree.
export function forcepsRenderPose(frame) {
  const { tip_left, tip_right, trocar } = frame
  const { jawCenter, shaftDir, openingMm } = forcepsPose(tip_left, tip_right, trocar)
  const hinge = sub(jawCenter, scale(shaftDir, JAW_LENGTH_MM))

  const renderTip = (tip) => {
    const v = sub(tip, jawCenter)
    const along = scale(shaftDir, dot(v, shaftDir))
    const perp = sub(v, along)
    return add(jawCenter, add(along, scale(perp, VIEW_JAW_SPREAD)))
  }

  return {
    jawCenter,
    shaftDir,
    openingMm,
    hinge,
    tipLeftRender: renderTip(tip_left),
    tipRightRender: renderTip(tip_right),
  }
}
