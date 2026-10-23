import { useEffect, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { CameraControls } from '@react-three/drei'
import * as THREE from 'three'
import {
  VIEW_PRESETS,
  EYE_RADIUS_MM,
  NAV_MIN_DISTANCE_MM,
  NAV_MAX_DISTANCE_MM,
  NAV_FOLLOW_DISTANCE_MM,
  NAV_FOCUS_DISTANCE_MM,
} from '../config.js'
import { add, sub, scale, normalize, midpoint, pickWall } from '../geometry.js'

const ACTION_TRUCK = 2 // camera-controls ACTION.TRUCK
const ORIGIN = [0, 0, 0]
// The orbit pivot may move anywhere inside the globe, never outside it.
const PIVOT_BOUNDS = new THREE.Box3(
  new THREE.Vector3(-EYE_RADIUS_MM, -EYE_RADIUS_MM, -EYE_RADIUS_MM),
  new THREE.Vector3(EYE_RADIUS_MM, EYE_RADIUS_MM, EYE_RADIUS_MM)
)

const jawCenter = (frame) => midpoint(frame.tip_left, frame.tip_right)

// Main-view navigation on drei's CameraControls:
// - left-drag orbits, right-drag pans (pivot bounded to the globe), scroll
//   zooms toward the cursor, down to NAV_MIN_DISTANCE_MM (also inside the eye).
// - pivot 'globe' orbits the globe center; 'instrument' keeps the pivot on the
//   jaws and follows them smoothly; 'point' is a pivot picked on the
//   instrument or the retinal wall — by a click while `picking` (Point
//   button), or by double-click at any time.
// - presets snap the framing; in instrument mode they only set the direction.
// A single world up (+Y) is kept for every preset: camera-controls applies
// `camera.up` changes instantly, which would snap-roll the scene mid-glide.
// `domElementRef` is the view's DOM placeholder (the shared canvas is
// pointer-transparent); it is resolved here, after SceneView has mounted.
export default function CameraRig({
  view,
  snapNonce,
  pivot,
  onPivot,
  domElementRef,
  sceneStore,
  pickRef,
  picking = false,
  onPickDone,
  navApi,
}) {
  const ref = useRef()
  const { width, height } = useThree((state) => state.size)
  const camera = useThree((state) => state.camera)

  useEffect(() => {
    const c = ref.current
    if (!c) return
    c.mouseButtons.right = ACTION_TRUCK
    c.dollyToCursor = true
    c.setBoundary(PIVOT_BOUNDS)
  }, [])

  // Buttons in the navigation cluster (outside the canvas) drive the controls.
  useEffect(() => {
    if (!navApi) return
    navApi.current = {
      zoom(direction) {
        const c = ref.current
        if (c) c.dolly(direction > 0 ? c.distance * 0.3 : -c.distance * 0.45, true)
      },
    }
    return () => {
      navApi.current = null
    }
  }, [navApi])

  // Snap to the selected preset (also on Reset: snapNonce bumps).
  useEffect(() => {
    const c = ref.current
    const preset = VIEW_PRESETS[view]
    if (!c || !preset || pivot === 'point') return
    const animate = !window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const target = preset.target ?? ORIGIN
    if (pivot === 'instrument') {
      const frame = sceneStore.get().frame
      const center = frame ? jawCenter(frame) : target
      const dir = normalize(sub(preset.pos, target))
      c.setLookAt(...add(center, scale(dir, NAV_FOLLOW_DISTANCE_MM)), ...center, animate)
      return
    }
    const fit = preset.fit === false ? 1 : 0.88 * Math.max(1, height / width)
    c.setLookAt(...add(target, scale(sub(preset.pos, target), fit)), ...target, animate)
  }, [view, snapNonce, pivot, width, height, sceneStore])

  // Instrument pivot: keep the jaws at the orbit center. moveTo shifts camera
  // and pivot together, so the user's angle and zoom are kept while following.
  useFrame(() => {
    const c = ref.current
    if (!c || pivot !== 'instrument') return
    const frame = sceneStore.get().frame
    if (frame) c.moveTo(...jawCenter(frame), true)
  })

  // Pivot picking: a plain click while `picking`, or a double-click anytime,
  // orbits around the clicked instrument part or retinal point.
  const pickingRef = useRef(picking)
  pickingRef.current = picking
  useEffect(() => {
    const el = domElementRef.current
    if (!el) return
    const raycaster = new THREE.Raycaster()
    const ndc = new THREE.Vector2()
    const pickAt = (clientX, clientY) => {
      const c = ref.current
      if (!c) return false
      const r = el.getBoundingClientRect()
      ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1)
      raycaster.setFromCamera(ndc, camera)
      const hit = pickRef.current ? raycaster.intersectObject(pickRef.current, true)[0] : null
      const point = hit
        ? hit.point.toArray()
        : pickWall(raycaster.ray.origin.toArray(), raycaster.ray.direction.toArray())
      if (!point) return false
      onPivot('point')
      c.moveTo(...point, true)
      if (c.distance > NAV_FOCUS_DISTANCE_MM) c.dollyTo(NAV_FOCUS_DISTANCE_MM, true)
      return true
    }

    // Overlay controls inside the view (navigation cluster, legend) are
    // marked data-scene-ui and never pick.
    const onOverlay = (e) => e.target !== el && !!e.target.closest?.('[data-scene-ui]')
    let down = null
    const onPointerDown = (e) => {
      down = e.button === 0 && !onOverlay(e) ? { x: e.clientX, y: e.clientY } : null
    }
    const onPointerUp = (e) => {
      if (!pickingRef.current || !down) return
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y)
      down = null
      if (moved < 4 && pickAt(e.clientX, e.clientY)) onPickDone?.()
    }
    const onDoubleClick = (e) => {
      if (!onOverlay(e) && pickAt(e.clientX, e.clientY)) onPickDone?.()
    }
    // Capture phase: camera-controls captures the pointer on pointerdown.
    el.addEventListener('pointerdown', onPointerDown, true)
    window.addEventListener('pointerup', onPointerUp, true)
    el.addEventListener('dblclick', onDoubleClick)
    return () => {
      el.removeEventListener('pointerdown', onPointerDown, true)
      window.removeEventListener('pointerup', onPointerUp, true)
      el.removeEventListener('dblclick', onDoubleClick)
    }
  }, [domElementRef, camera, pickRef, onPivot, onPickDone])

  return (
    <CameraControls
      ref={ref}
      makeDefault
      domElement={domElementRef.current}
      minDistance={NAV_MIN_DISTANCE_MM}
      maxDistance={NAV_MAX_DISTANCE_MM}
    />
  )
}
