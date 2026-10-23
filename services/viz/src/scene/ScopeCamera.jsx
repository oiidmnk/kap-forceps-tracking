import { useEffect, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { midpoint, normalize, scale } from '../geometry.js'
import { EYE_RADIUS_MM } from '../config.js'

const VIEW_DIST_MM = 8 // fixed camera distance from the wall — no zoom-to-fit, tune here if the framing feels off

const FOCUS_TAU_S = 0.08 // follow time constant; frame-rate independent

// Retinal point under the jaw midpoint. Unlike the nearest tip's wall point it
// cannot jump when the left/right tips swap which one is closer.
export function scopeFocusTarget(frame) {
  return scale(normalize(midpoint(frame.tip_left, frame.tip_right)), EYE_RADIUS_MM)
}

const DEG = THREE.MathUtils.degToRad
export const ELEV_DEFAULT = DEG(58) // mostly-from-the-top, but tilted enough to still read the gap
export const ELEV_MIN = DEG(10) // never let the camera reach the tangent plane — this is what kept it off the wall's underside
export const ELEV_MAX = DEG(82) // stop just short of straight-down, where roll becomes undefined

// Scratch vectors for smooth world-reference blending
const _wUp = new THREE.Vector3(0, 1, 0)
const _wRight = new THREE.Vector3(1, 0, 0)
const _rA = new THREE.Vector3()
const _rB = new THREE.Vector3()

// Drives the proximity-scope camera and owns the shared scope focus.
// - focus: the retinal point under the jaw midpoint, followed with a short
//   time constant and published in `focusRef` so ProximityWall centres its
//   patch on exactly the point the camera looks at. Runs at priority -1, before
//   the wall reads it.
// - orientation: an instrument-relative basis (outward/rightAxis/forward) from
//   the focus, so it never flips on world axes, plus the user-adjustable
//   (azimuth, elevation) from `dragState`. Elevation is clamped away from the
//   tangent plane, so the camera never ends up under the retina.
// The camera is rigidly attached to the focus at a fixed distance: all motion
// smoothing happens in one place, so the wall, reticle and instrument never
// swim against each other.
export default function ScopeCamera({ frame, dragState, focusRef }) {
  const { camera } = useThree()
  const v = useMemo(
    () => ({
      target: new THREE.Vector3(),
      desiredPos: new THREE.Vector3(),
      up: new THREE.Vector3(),
      dir: new THREE.Vector3(),
      outward: new THREE.Vector3(),
      rightAxis: new THREE.Vector3(),
      forward: new THREE.Vector3(),
    }),
    [],
  )
  // Start fresh (no glide from an old position) whenever the scope reactivates.
  useEffect(() => () => {
    focusRef.current.valid = false
  }, [focusRef])

  useFrame((_, dt) => {
    if (!frame) return
    const focus = focusRef.current
    v.target.set(...scopeFocusTarget(frame))
    if (!focus.valid) {
      focus.point.copy(v.target)
      focus.valid = true
    } else {
      focus.point.lerp(v.target, 1 - Math.exp(-Math.min(dt, 0.1) / FOCUS_TAU_S))
    }
    v.outward.copy(focus.point).normalize()

    // Use a world-stable reference axis so the scope doesn't roll with the
    // forceps. Smoothly blend between two world references to avoid a
    // discontinuous jump when outward.y crosses a threshold.
    const absY = Math.abs(v.outward.y)
    const t = absY <= 0.85 ? 0 : absY >= 0.99 ? 1
      : ((x) => x * x * (3 - 2 * x))((absY - 0.85) / (0.99 - 0.85))

    _rA.crossVectors(_wUp, v.outward)
    _rB.crossVectors(_wRight, v.outward)
    const lA = _rA.length(), lB = _rB.length()
    if (lA > 1e-6) _rA.multiplyScalar(1 / lA)
    if (lB > 1e-6) _rB.multiplyScalar(1 / lB)
    v.rightAxis.lerpVectors(_rA, _rB, t).normalize()

    v.forward.crossVectors(v.outward, v.rightAxis).normalize()

    const azimuth = dragState?.current?.azimuth ?? 0
    const elevation = THREE.MathUtils.clamp(dragState?.current?.elevation ?? ELEV_DEFAULT, ELEV_MIN, ELEV_MAX)

    // Direction from the wall to the camera: azimuth spins around `outward`
    // within the tangent plane, elevation tilts that toward -outward — i.e.
    // away from the wall, back into the vitreous cavity, since `outward`
    // points from the eye center OUT toward the retina, not toward where the
    // instrument sits.
    v.dir
      .copy(v.rightAxis)
      .multiplyScalar(Math.cos(azimuth))
      .addScaledVector(v.forward, Math.sin(azimuth))
      .multiplyScalar(Math.cos(elevation))
      .addScaledVector(v.outward, -Math.sin(elevation))
      .normalize()

    v.desiredPos.copy(focus.point).addScaledVector(v.dir, VIEW_DIST_MM)

    // Up = the part of "away from the wall" that's still perpendicular to the
    // view direction — stays well-defined (never collinear with the view dir)
    // at any elevation inside the clamp range.
    v.up.copy(v.dir).multiplyScalar(v.outward.dot(v.dir)).sub(v.outward).normalize()

    camera.position.copy(v.desiredPos)
    camera.up.copy(v.up)
    camera.lookAt(focus.point)
  }, -1)

  return null
}
