import { distanceLevel } from '../ui.js'
import {
  LEVEL_HYSTERESIS_MM,
  LEVEL_DEESCALATE_DWELL_MS,
  DIST_SCOPE_SHOW_MM,
  DIST_SCOPE_HIDE_MM,
} from '../config.js'
import { distanceToRetina } from '../geometry.js'

const SEVERITY = { safe: 0, warn: 1, danger: 2 }

// Safety level with one-sided hysteresis: escalation is immediate, while
// de-escalation only happens once the distance clears the threshold by
// `margin`. Never shows a safer level than the raw measurement implies.
export function nextLevel(prev, mm, margin = LEVEL_HYSTERESIS_MM) {
  const raw = distanceLevel(mm)
  if (!prev || SEVERITY[raw] >= SEVERITY[prev]) return raw
  const held = distanceLevel(Math.max(0, mm - margin))
  return SEVERITY[held] > SEVERITY[prev] ? prev : held
}

const moreSevere = (a, b) => (SEVERITY[a] >= SEVERITY[b] ? a : b)

// Tracks the displayed levels across frames. `update(frame)` returns
// { left, right, min, scope, retained } or null when there is no measurement.
export function createLevelTracker({ margin = LEVEL_HYSTERESIS_MM, dwellMs = LEVEL_DEESCALATE_DWELL_MS } = {}) {
  let state = null
  const pending = { left: null, right: null }
  function level(tip, mm, now) {
    const prev = state?.[tip]
    const candidate = nextLevel(prev, mm, margin)
    if (!prev || SEVERITY[candidate] >= SEVERITY[prev]) {
      pending[tip] = null
      return candidate
    }
    if (pending[tip]?.level !== candidate) pending[tip] = { level: candidate, since: now }
    return now - pending[tip].since >= dwellMs ? candidate : prev
  }
  return {
    update(frame, now = 0) {
      if (!frame) {
        state = null
        pending.left = pending.right = null
        return null
      }
      const dl = distanceToRetina(frame.tip_left)
      const dr = distanceToRetina(frame.tip_right)
      const left = level('left', dl, now)
      const right = level('right', dr, now)
      const nearest = Math.min(dl, dr)
      const scope = state?.scope
        ? nearest < DIST_SCOPE_HIDE_MM
        : nearest < DIST_SCOPE_SHOW_MM
      const min = moreSevere(left, right)
      const retained = {
        left: SEVERITY[left] > SEVERITY[distanceLevel(dl)],
        right: SEVERITY[right] > SEVERITY[distanceLevel(dr)],
        min: SEVERITY[min] > SEVERITY[distanceLevel(nearest)],
      }
      const next = { left, right, min, scope, retained }
      if (
        state &&
        state.left === next.left &&
        state.right === next.right &&
        state.scope === next.scope &&
        state.retained.left === retained.left &&
        state.retained.right === retained.right &&
        state.retained.min === retained.min
      )
        return state
      state = next
      return state
    },
    // Missing measurements interrupt evidence for a continuous safer interval.
    interrupt() {
      pending.left = pending.right = null
    },
    reset() {
      pending.left = pending.right = null
      state = null
    },
  }
}
