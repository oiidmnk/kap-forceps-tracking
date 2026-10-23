import { useLayoutEffect, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { EYE_RADIUS_MM } from '../config.js'

// Cutaway of the globe: one shared clipping plane through the globe center,
// facing the camera, removes the near half of the glass shell, grid and axes
// so the interior is unobstructed. Through the center, the cut edge always
// coincides with the globe's outline from any direction (a plane anchored
// elsewhere leaves a visible circle inside the eye). The retina, its shadow
// and the limbus ring are never cut (userData.keepInCutaway). From inside the
// globe nothing is cut; the inside/outside switch has a small hysteresis so a
// camera right at the wall cannot make the cut flicker.
export const CUT_PLANE = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0)
const NO_CUT = 1e6
const WALL_HYSTERESIS_MM = 0.5

// Keeps CUT_PLANE facing the main camera. Mount once in the main view.
export function GlobeCutaway() {
  const inside = useRef(false)
  useFrame(({ camera }) => {
    const d = camera.position.length()
    inside.current = inside.current
      ? d < EYE_RADIUS_MM + WALL_HYSTERESIS_MM
      : d < EYE_RADIUS_MM - WALL_HYSTERESIS_MM
    if (inside.current) {
      CUT_PLANE.constant = NO_CUT
      return
    }
    // Keep points with dot(p, camDir) <= 0, i.e. the half facing away.
    CUT_PLANE.normal.copy(camera.position).multiplyScalar(-1 / d)
    CUT_PLANE.constant = 0
  }, -1)
  return null
}

// Applies (or removes) the cut on every material under `ref`. Objects marked
// `userData.keepInCutaway` (landmarks like the limbus ring) are never cut.
// `deps` should include anything that mounts or unmounts meshes in that subtree.
export function useCutaway(ref, enabled, deps = []) {
  useLayoutEffect(() => {
    ref.current?.traverse((obj) => {
      if (obj.userData.keepInCutaway) return
      const materials = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : []
      for (const m of materials) {
        m.clippingPlanes = enabled ? [CUT_PLANE] : null
        m.needsUpdate = true
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, ...deps])
}
