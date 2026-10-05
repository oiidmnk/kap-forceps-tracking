// Browser-side synthetic generator, mirroring feed/synthetic_feed.py so the
// dashboard runs standalone (no Python) with realistic pivoting motion.
// Produces the same message shape the live feed emits.

import { EYE_RADIUS_MM, VIEW_PRESETS } from '../config.js'
import { add, sub, scale, normalize, cross, dot } from '../geometry.js'

// +Y-up convention: instruments enter near the top (+Y); retina is the lower
// (-Y) hemisphere. Forceps trocar is fixed on the surface near the top.
const TROCAR = scale(normalize([0.35, 0.9, 0.45]), EYE_RADIUS_MM)
const INWARD = normalize(scale(TROCAR, -1)) // straight-in shaft direction

// Light pipe (illumination): a second instrument, held roughly static. Enters
// on the opposite side near the top and aims down at the posterior pole.
const LIGHT_TROCAR = scale(normalize([-0.5, 0.82, -0.28]), EYE_RADIUS_MM)
const LIGHT_AXIS = normalize(sub([1.5, -EYE_RADIUS_MM, 1.0], LIGHT_TROCAR))
const LIGHT_TIP = add(LIGHT_TROCAR, scale(LIGHT_AXIS, 9))

// Screen-space basis (right, up) of a camera at `camPos` looking at the origin
// with world +Y up — the same framing drei's CameraControls produces.
function screenBasis(camPos) {
  const forward = normalize(scale(camPos, -1)) // camera -> origin
  const right = normalize(cross(forward, [0, 1, 0]))
  const up = cross(right, forward) // already unit (right ⟂ forward)
  return { right, up }
}

// Debug-control wobble axes, aligned to the SURGEON view so the keyboard maps
// to what the operator sees in that framing: pressing D moves the tip purely
// right on screen, W purely up, etc. `u` is chosen ⟂ to both the shaft and the
// screen-up axis (so it projects to pure horizontal), `v` ⟂ to shaft and
// screen-right (pure vertical). Signs make u -> screen-right, v -> screen-up.
const SURGEON = screenBasis(VIEW_PRESETS.surgeon.pos)
const WOBBLE_U = normalize(cross(INWARD, SURGEON.up)) // D/A: screen horizontal
const WOBBLE_V = normalize(cross(SURGEON.right, INWARD)) // W/S: screen vertical

// Builds a frame from an explicit forceps pose: `depth` is insertion distance
// from the trocar (mm), `wobbleU`/`wobbleV` tilt the shaft off the straight-in
// direction along the two axes perpendicular to it, `half` is the jaw
// half-angle (radians). Shared by the time-driven mock motion below and by the
// keyboard-driven debug pose (useDebugPose), so both stay geometrically
// consistent with each other and with the fixed trocar/light-pipe placement.
export function poseFrame({ depth, wobbleU, wobbleV, half, roll = 0 }) {
  const inward = INWARD
  const u = WOBBLE_U
  const shaftDir = normalize(add(inward, add(scale(WOBBLE_U, wobbleU), scale(WOBBLE_V, wobbleV))))

  const jawCenter = add(TROCAR, scale(shaftDir, depth))

  const jawLen = 1.2
  // Jaw spread axis. Derive it from the *fixed* in-plane reference (u,
  // perpendicular to the constant inward direction), projected perpendicular to
  // the CURRENT shaft. This keeps the jaw plane continuous as the shaft wobbles.
  // Calling perpBasis(shaftDir) here instead would flip the basis ~125° the
  // moment the near-vertical shaft crosses |dir.y| = 0.9 — a hard discontinuity
  // that makes the jaws visibly teleport/rotate on a tiny nudge.
  const ju0 = normalize(sub(u, scale(shaftDir, dot(u, shaftDir))))
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
  const wobbleU = 0.25 * Math.sin(s * 0.7)
  const wobbleV = 0.25 * Math.cos(s * 0.5)
  // Insertion depth keeps the tips in the lower (-Y) hemisphere (retina side).
  // ~16 mm puts tips just below the equator; ~22 mm brings them near the retina
  // wall, exercising the Distance-to-Retina safety states.
  const depth = 19.0 + 3.0 * Math.sin(s * 0.4)
  // Jaws open/close within a realistic microforceps range (~7°–18° full angle).
  const half = 0.06 + 0.1 * (0.5 + 0.5 * Math.sin(s * 1.3)) // radians (half-angle)

  return { t, confidence: 0.99, ...poseFrame({ depth, wobbleU, wobbleV, half }) }
}
