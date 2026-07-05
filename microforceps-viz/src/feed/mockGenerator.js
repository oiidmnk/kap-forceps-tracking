// Browser-side synthetic generator, mirroring feed/synthetic_feed.py so the
// dashboard runs standalone (no Python) with realistic pivoting motion.
// Produces the same message shape the live feed emits.

import { EYE_RADIUS_MM } from '../config.js'
import { add, sub, scale, normalize, cross } from '../geometry.js'

// +Y-up convention: instruments enter near the top (+Y); retina is the lower
// (-Y) hemisphere. Forceps trocar is fixed on the surface near the top.
const TROCAR = scale(normalize([0.35, 0.9, 0.45]), EYE_RADIUS_MM)

// Light pipe (illumination): a second instrument, held roughly static. Enters
// on the opposite side near the top and aims down at the posterior pole.
const LIGHT_TROCAR = scale(normalize([-0.5, 0.82, -0.28]), EYE_RADIUS_MM)
const LIGHT_AXIS = normalize(sub([1.5, -EYE_RADIUS_MM, 1.0], LIGHT_TROCAR))
const LIGHT_TIP = add(LIGHT_TROCAR, scale(LIGHT_AXIS, 9))

// Build an orthonormal basis {u, v} perpendicular to a direction d.
function perpBasis(d) {
  const ref = Math.abs(d[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]
  const u = normalize([
    d[1] * ref[2] - d[2] * ref[1],
    d[2] * ref[0] - d[0] * ref[2],
    d[0] * ref[1] - d[1] * ref[0],
  ])
  const v = normalize([
    d[1] * u[2] - d[2] * u[1],
    d[2] * u[0] - d[0] * u[2],
    d[0] * u[1] - d[1] * u[0],
  ])
  return { u, v }
}

// Builds a frame from an explicit forceps pose: `depth` is insertion distance
// from the trocar (mm), `wobbleU`/`wobbleV` tilt the shaft off the straight-in
// direction along the two axes perpendicular to it, `half` is the jaw
// half-angle (radians). Shared by the time-driven mock motion below and by the
// keyboard-driven debug pose (useDebugPose), so both stay geometrically
// consistent with each other and with the fixed trocar/light-pipe placement.
export function poseFrame({ depth, wobbleU, wobbleV, half, roll = 0 }) {
  const inward = normalize(scale(TROCAR, -1))
  const { u, v } = perpBasis(inward)
  const shaftDir = normalize(add(inward, add(scale(u, wobbleU), scale(v, wobbleV))))

  const jawCenter = add(TROCAR, scale(shaftDir, depth))

  const jawLen = 1.2
  const { u: ju0 } = perpBasis(shaftDir)
  // Rotate the jaw spread axis around the shaft by the roll angle
  const cosR = Math.cos(roll)
  const sinR = Math.sin(roll)
  const ju = add(scale(ju0, cosR), scale(cross(shaftDir, ju0), sinR))
  const along = scale(shaftDir, jawLen * Math.cos(half))
  const spread = scale(ju, jawLen * Math.sin(half))

  const tipLeft = add(jawCenter, add(along, spread))
  const tipRight = add(jawCenter, add(along, scale(spread, -1)))

  return {
    tip_left: tipLeft,
    tip_right: tipRight,
    trocar: TROCAR,
    light_tip: LIGHT_TIP,
    light_trocar: LIGHT_TROCAR,
  }
}

// Returns a tracking frame for a given time t (seconds).
// Overall pace of the synthetic motion. Lower = slower.
const SPEED = 0.35

export function sampleFrame(t) {
  const s = t * SPEED
  const wobbleU = 0.35 * Math.sin(s * 0.7)
  const wobbleV = 0.35 * Math.cos(s * 0.5)
  // Insertion depth sweeps from mid-vitreous to near the retina, exercising the
  // Distance-to-Retina safety states while keeping the tips inside the globe.
  const depth = 12.5 + 8.0 * Math.sin(s * 0.4)
  // Jaws open/close within a realistic microforceps range (~7°–18° full angle).
  const half = 0.06 + 0.1 * (0.5 + 0.5 * Math.sin(s * 1.3)) // radians (half-angle)

  return { t, confidence: 0.99, ...poseFrame({ depth, wobbleU, wobbleV, half }) }
}
