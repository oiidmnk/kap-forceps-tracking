import { distanceLevel } from '../ui.js'
import {
  DIST_HYSTERESIS_MM,
  DIST_SCOPE_SHOW_MM,
  DIST_SCOPE_HIDE_MM,
} from '../config.js'
import { distanceToRetina } from '../geometry.js'

const SEVERITY = { safe: 0, warn: 1, danger: 2 }

// Safety level with one-sided hysteresis: escalation is immediate, while
// de-escalation only happens once the distance clears the threshold by
// `margin`. Never shows a safer level than the raw measurement implies.
export function nextLevel(prev, mm, margin = DIST_HYSTERESIS_MM) {
  const raw = distanceLevel(mm)
  if (!prev || SEVERITY[raw] >= SEVERITY[prev]) return raw
  const held = distanceLevel(Math.max(0, mm - margin))
  return SEVERITY[held] > SEVERITY[prev] ? prev : held
}

const moreSevere = (a, b) => (SEVERITY[a] >= SEVERITY[b] ? a : b)

// Tracks the displayed levels across frames. `update(frame)` returns
// { left, right, min, scope } or null when there is no measurement.
export function createLevelTracker() {
  let state = null
  return {
    update(frame) {
      if (!frame) {
        state = null
        return null
      }
      const dl = distanceToRetina(frame.tip_left)
      const dr = distanceToRetina(frame.tip_right)
      const left = nextLevel(state?.left, dl)
      const right = nextLevel(state?.right, dr)
      const nearest = Math.min(dl, dr)
      const scope = state?.scope
        ? nearest < DIST_SCOPE_HIDE_MM
        : nearest < DIST_SCOPE_SHOW_MM
      const next = { left, right, min: moreSevere(left, right), scope }
      if (
        state &&
        state.left === next.left &&
        state.right === next.right &&
        state.scope === next.scope
      )
        return state
      state = next
      return state
    },
    reset() {
      state = null
    },
  }
}
