import { useEffect, useLayoutEffect, useState } from 'react'
import { createPortal, useFrame } from '@react-three/fiber'
import * as THREE from 'three'

// One WebGL context for the whole workstation: a single fixed, transparent
// canvas sits over the page and each 3D view renders into the screen rectangle
// of a DOM placeholder (`track`). Views share GPU resources (shaders,
// environment map) and one render loop.
//
// Compared with drei's <View>, this keeps a separate scene + camera per view
// in an R3F portal, but also clips the drawn area to an optional scrolling
// container (`clip`) and leaves event targets to the caller (controls and HTML
// labels are given their DOM element explicitly).

// Clears the shared canvas once per frame, before any view renders.
export function ViewClear() {
  useFrame(({ gl }) => {
    gl.autoClear = false
    gl.localClippingEnabled = true // per-material planes (globe cutaway)
    gl.setScissorTest(false)
    gl.clear(true, true, true)
  }, 1)
  return null
}

export default function SceneView({
  track,
  clip,
  index = 2,
  active = true,
  camera: cameraProps,
  cameraRef,
  children,
}) {
  const [scene] = useState(() => new THREE.Scene())
  const [camera] = useState(() => {
    const { fov = 45, near = 0.1, far = 1000, position } = cameraProps ?? {}
    const c = new THREE.PerspectiveCamera(fov, 1, near, far)
    if (position) c.position.set(...position)
    return c
  })
  const [size, setSize] = useState(null)

  useEffect(() => {
    if (cameraRef) cameraRef.current = camera
  }, [camera, cameraRef])

  // Portal `size` feeds drei helpers (Html labels, line resolution, framing).
  useLayoutEffect(() => {
    const el = track.current
    if (!el) return
    const measure = () => {
      const r = el.getBoundingClientRect()
      setSize({ width: r.width, height: r.height, top: r.top, left: r.left })
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    window.addEventListener('resize', measure)
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [track])

  if (!size) return null
  return createPortal(
    <>
      <ViewRenderer track={track} clip={clip} index={index} active={active} />
      {children}
    </>,
    scene,
    { camera, size }
  )
}

function ViewRenderer({ track, clip, index, active }) {
  useFrame(({ gl, scene, camera }) => {
    const el = track.current
    if (!active || !el) return
    const r = el.getBoundingClientRect()
    if (r.width < 1 || r.height < 1) return

    let left = r.left
    let top = r.top
    let right = r.right
    let bottom = r.bottom
    const c = clip?.current?.getBoundingClientRect()
    if (c) {
      left = Math.max(left, c.left)
      top = Math.max(top, c.top)
      right = Math.min(right, c.right)
      bottom = Math.min(bottom, c.bottom)
    }
    if (right - left < 1 || bottom - top < 1) return

    const canvas = gl.domElement.getBoundingClientRect()
    const aspect = r.width / r.height
    if (camera.aspect !== aspect) {
      camera.aspect = aspect
      camera.updateProjectionMatrix()
    }
    // WebGL's origin is bottom-left; rectangles are in CSS pixels.
    gl.setViewport(r.left - canvas.left, canvas.bottom - r.bottom, r.width, r.height)
    gl.setScissor(left - canvas.left, canvas.bottom - bottom, right - left, bottom - top)
    gl.setScissorTest(true)
    gl.clearDepth()
    gl.render(scene, camera)
    gl.setScissorTest(false)
  }, index)
  return null
}
