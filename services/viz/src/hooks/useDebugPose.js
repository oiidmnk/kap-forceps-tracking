import { useEffect, useRef } from 'react'
import { navigationHelpOpen } from './navigationHelp.js'
import { poseFrame } from '../feed/mockGenerator.js'

// Rates the pose changes per second of a key being held.
const DEPTH_RATE_MM_S = 1.5
const WOBBLE_RATE_S = 0.15
const JAW_RATE_RAD_S = 0.1
const ROLL_RATE_RAD_S = 1.5

const DEPTH_MIN_MM = 2
const DEPTH_MAX_MM = 22
const WOBBLE_LIMIT = 1.0
const HALF_MIN_RAD = 0.02
const HALF_MAX_RAD = 0.4

const INITIAL_POSE = { depth: 14, wobbleU: 0, wobbleV: 0, half: 0.15, roll: 0 }

// WASD sways the shaft laterally (the two axes perpendicular to straight-in),
// Q/E moves insertion depth, Up/Down opens/closes the jaws.
const KEY_MAP = {
  KeyW: 'v+',
  KeyS: 'v-',
  KeyA: 'u-',
  KeyD: 'u+',
  KeyQ: 'depth-',
  KeyE: 'depth+',
  ArrowUp: 'jaw+',
  ArrowDown: 'jaw-',
  ArrowLeft: 'roll-',
  ArrowRight: 'roll+',
}

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x))

const LIMITS = {
  depth: [DEPTH_MIN_MM, DEPTH_MAX_MM],
  wobbleU: [-WOBBLE_LIMIT, WOBBLE_LIMIT],
  wobbleV: [-WOBBLE_LIMIT, WOBBLE_LIMIT],
  half: [HALF_MIN_RAD, HALF_MAX_RAD],
  roll: [-Infinity, Infinity],
}
const ALIASES = { u: 'wobbleU', v: 'wobbleV' }

// Applies a partial pose ({ depth, wobbleU|u, wobbleV|v, half, roll }), clamped
// to the keyboard limits; unknown keys and non-numbers are ignored.
function applyPose(pose, partial) {
  for (const [k, v] of Object.entries(partial || {})) {
    const key = ALIASES[k] || k
    if (LIMITS[key] && Number.isFinite(+v)) pose[key] = clamp(+v, ...LIMITS[key])
  }
  return { ...pose }
}

// `?pose=depth:20.5,u:0.1,v:0,half:0.15,roll:0` sets the starting pose, so
// screenshots of a given configuration are reproducible.
function poseFromUrl() {
  const raw = new URLSearchParams(window.location.search).get('pose')
  if (!raw) return {}
  return Object.fromEntries(raw.split(',').map((kv) => kv.split(':')).filter((kv) => kv.length === 2))
}

// Keyboard-controlled forceps pose for debugging the visualization without
// needing a real feed. Only active while `active` is true (i.e. the 'debug'
// source is selected) — otherwise no listeners are attached and the held keys
// don't do anything, so they're free to use normally the rest of the time.
export function useDebugPose(active, onFrame) {
  const pose = useRef(applyPose({ ...INITIAL_POSE }, poseFromUrl()))
  const pressed = useRef(new Set())

  useEffect(() => {
    if (!active) return

    // Scripted control (screenshots, tests): window.__vizDebug.setPose({ depth: 20.5 }).
    window.__vizDebug = {
      setPose: (partial) => applyPose(pose.current, partial),
      getPose: () => ({ ...pose.current }),
      resetPose: () => applyPose(pose.current, INITIAL_POSE),
    }

    const onKeyDown = (e) => {
      if (navigationHelpOpen() || e.target.closest('input, select, textarea, [contenteditable="true"]') || !KEY_MAP[e.code]) return
      pressed.current.add(e.code)
      e.preventDefault()
    }
    const onKeyUp = (e) => {
      pressed.current.delete(e.code)
    }
    const onBlur = () => pressed.current.clear()
    window.addEventListener('blur', onBlur)
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)

    let raf
    let last = performance.now()
    const tick = () => {
      const now = performance.now()
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now

      if (navigationHelpOpen()) pressed.current.clear()
      const p = pose.current
      for (const code of pressed.current) {
        switch (KEY_MAP[code]) {
          case 'v+':
            p.wobbleV = clamp(p.wobbleV + WOBBLE_RATE_S * dt, -WOBBLE_LIMIT, WOBBLE_LIMIT)
            break
          case 'v-':
            p.wobbleV = clamp(p.wobbleV - WOBBLE_RATE_S * dt, -WOBBLE_LIMIT, WOBBLE_LIMIT)
            break
          case 'u+':
            p.wobbleU = clamp(p.wobbleU + WOBBLE_RATE_S * dt, -WOBBLE_LIMIT, WOBBLE_LIMIT)
            break
          case 'u-':
            p.wobbleU = clamp(p.wobbleU - WOBBLE_RATE_S * dt, -WOBBLE_LIMIT, WOBBLE_LIMIT)
            break
          case 'depth+':
            p.depth = clamp(p.depth + DEPTH_RATE_MM_S * dt, DEPTH_MIN_MM, DEPTH_MAX_MM)
            break
          case 'depth-':
            p.depth = clamp(p.depth - DEPTH_RATE_MM_S * dt, DEPTH_MIN_MM, DEPTH_MAX_MM)
            break
          case 'jaw+':
            p.half = clamp(p.half + JAW_RATE_RAD_S * dt, HALF_MIN_RAD, HALF_MAX_RAD)
            break
          case 'jaw-':
            p.half = clamp(p.half - JAW_RATE_RAD_S * dt, HALF_MIN_RAD, HALF_MAX_RAD)
            break
          case 'roll+':
            p.roll += ROLL_RATE_RAD_S * dt
            break
          case 'roll-':
            p.roll -= ROLL_RATE_RAD_S * dt
            break
        }
      }
      onFrame({ t: now / 1000, confidence: 1, ...poseFrame(p) }, 'debug', now)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)

    return () => {
      pressed.current.clear()
      delete window.__vizDebug
      window.removeEventListener('blur', onBlur)
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      cancelAnimationFrame(raf)
    }
  }, [active, onFrame])
}
