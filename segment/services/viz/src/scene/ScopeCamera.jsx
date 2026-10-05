import { useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { forcepsRenderPose, nearestTip } from '../geometry.js'

const VIEW_DIST_MM = 8 // fixed camera distance from the wall — no zoom-to-fit, tune here if the framing feels off

const DEG = THREE.MathUtils.degToRad
export const ELEV_DEFAULT = DEG(58) // mostly-from-the-top, but tilted enough to still read the gap
export const ELEV_MIN = DEG(10) // never let the camera reach the tangent plane — this is what kept it off the wall's underside
export const ELEV_MAX = DEG(82) // stop just short of straight-down, where roll becomes undefined

// Scratch vectors for smooth world-reference blending
const _wUp = new THREE.Vector3(0, 1, 0)
const _wRight = new THREE.Vector3(1, 0, 0)
const _rA = new THREE.Vector3()
const _rB = new THREE.Vector3()

// Any vector perpendicular to v — used only as a last-resort reference when the
// jaws are exactly closed and give no natural "sideways" direction.
function anyPerpendicular(v) {
  const ref = Math.abs(v.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0)
  return ref.addScaledVector(v, -ref.dot(v)).normalize()
}

// Drives the proximity-scope camera. Orientation splits into two parts:
//  - an instrument-relative basis (outward/rightAxis/forward), recomputed every
//    frame from the tracking data so it can never flip based on world axes.
//  - a user-adjustable (azimuth, elevation) pair, read from `dragState` (a
//    plain ref the panel updates on drag) that orbits within that basis.
//    Elevation is clamped away from the tangent plane, so neither the default
//    view nor a user drag can ever put the camera under the retina looking up
//    through it.
// The camera always looks at the wall's contact point — the same point
// ProximityWall's patch is centered on — at a fixed distance. Earlier this
// re-fit distance and look-at target to the tip+wall bounding box every frame,
// which made the view zoom and drift around as the jaws opened/closed; now
// only the tracked position moves, angle and zoom stay put unless dragged.
export default function ScopeCamera({ frame, dragState }) {
  const { camera } = useThree()
  const v = useMemo(
    () => ({
      desiredPos: new THREE.Vector3(),
      up: new THREE.Vector3(),
      dir: new THREE.Vector3(),
      outward: new THREE.Vector3(),
      rightAxis: new THREE.Vector3(),
      forward: new THREE.Vector3(),
      tipL: new THREE.Vector3(),
      tipR: new THREE.Vector3(),
      wall: new THREE.Vector3(),
    }),
    [],
  )

  useFrame(() => {
    if (!frame) return
    const { wall } = nearestTip(frame)
    const { tipLeftRender, tipRightRender, jawCenter } = forcepsRenderPose(frame)

    v.tipL.set(...tipLeftRender)
    v.tipR.set(...tipRightRender)
    v.wall.set(...wall)
    v.outward.set(...jawCenter).normalize()

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

    v.desiredPos.copy(v.wall).addScaledVector(v.dir, VIEW_DIST_MM)

    // Up = the part of "away from the wall" that's still perpendicular to the
    // view direction — reduces to the old fixed -outward at elevation 0, and
    // stays well-defined (never collinear with the view dir) at any elevation
    // inside the clamp range.
    v.up.copy(v.dir).multiplyScalar(v.outward.dot(v.dir)).sub(v.outward).normalize()

    camera.position.lerp(v.desiredPos, 0.2)
    camera.up.lerp(v.up, 0.2).normalize()
    camera.lookAt(v.wall)
  })

  return null
}
