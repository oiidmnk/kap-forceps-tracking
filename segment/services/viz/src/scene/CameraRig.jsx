import { useEffect, useRef } from 'react'
import { CameraControls } from '@react-three/drei'
import { VIEW_PRESETS, EYE_RADIUS_MM } from '../config.js'

// Wraps drei's CameraControls and snaps to a named view preset whenever `view`
// changes, with a smooth built-in transition. Free orbit/zoom stay available
// between snaps; panning (truck) is disabled to keep the globe centered.
//
// A single world up (+Y) is kept for every preset. camera-controls interpolates
// position/target smoothly but applies any `camera.up` change instantly, so
// varying up per preset would snap-roll the whole scene (the forceps appears to
// "teleport" to a new rotation) before the glide. None of the presets look
// exactly down the Y axis, so a constant up needs no gimbal handling.
export default function CameraRig({ view, snapNonce }) {
  const ref = useRef()

  // Disable right-drag pan once, matching the previous OrbitControls behavior.
  useEffect(() => {
    const c = ref.current
    if (c) c.mouseButtons.right = 0 // camera-controls ACTION.NONE
  }, [])

  useEffect(() => {
    const c = ref.current
    const preset = VIEW_PRESETS[view]
    if (!c || !preset) return
    c.setLookAt(preset.pos[0], preset.pos[1], preset.pos[2], 0, 0, 0, true)
    // snapNonce is intentionally a dependency: re-selecting the same preset
    // (nonce bumps, view doesn't) re-snaps the framing after free orbit/zoom.
  }, [view, snapNonce])

  return (
    <CameraControls
      ref={ref}
      makeDefault
      minDistance={EYE_RADIUS_MM * 1.5}
      maxDistance={EYE_RADIUS_MM * 8}
    />
  )
}
