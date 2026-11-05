import { createStore } from './store.js'
import { createInterpolator } from './displayInterpolation.js'
import { createLevelTracker } from './levels.js'
import { createTrend } from './trend.js'
import { distanceToRetina } from '../geometry.js'
import { READOUT_INTERVAL_MS, DROPOUT_GRACE_MS, MISSED_UPDATE_MS, STALE_MS, NEAREST_TIE_ENTER_MM, NEAREST_TIE_EXIT_MM, NEAREST_SIDE_LEAD_MM, NEAREST_DWELL_MS } from '../config.js'

/* Readout additions (all times use the local monotonic receipt clock):
 * held: boolean — frame/levels retained without a new measurement.
 * heldSinceMs: number|null — local time holding began; null when not held.
 * measurementAgeMs: number|null — elapsed since source time last advanced,
 *   or since the last valid receipt when the source has no finite time.
 * nearestTip: 'left'|'right'|'tie'|null — nearer tip; enter tie below 0.10 mm,
 *   leave above 0.15 mm; side switches require a lead above 0.05 mm; changes dwell for 500 ms;
 *   null without a displayed measurement.
 * lastLevel / lastClearance: last published minimum level / raw clearance (mm).
 *   Historical context only; current levels remain null during a ghost.
 * lostAtMs: ghost onset on the receipt clock; null without a ghost.
 */

// Statuses in which live tracking was lost after it had been running. The
// scene then keeps the last measured pose as a faded "ghost"; readouts stay
// empty because there is no current measurement.
const LOST = new Set(['stale', 'degraded', 'error'])

const EMPTY_SCENE = { frame: null, ghost: false, levels: null, lostAt: null }

export function initialStatus(source) {
  return source === 'live' ? 'connecting' : source
}

export function nextNearestTip(previous, left, right) {
  const difference = left - right
  const gap = Math.abs(difference)
  if (gap < NEAREST_TIE_ENTER_MM || (previous === 'tie' && gap <= NEAREST_TIE_EXIT_MM)) return 'tie'
  if (previous === 'left' && difference <= NEAREST_SIDE_LEAD_MM) return 'left'
  if (previous === 'right' && difference >= -NEAREST_SIDE_LEAD_MM) return 'right'
  return difference < 0 ? 'left' : 'right'
}

// Dwell runs on fresh measurements only. Gaps interrupt candidate evidence;
// the current label survives grace, then resets with measurement availability.
export function createNearestTipTracker(dwellMs = NEAREST_DWELL_MS) {
  let current = null, pending = null, since = null
  return {
    interrupt() { pending = since = null },
    reset() { current = pending = since = null },
    update(left, right, now) {
      const candidate = nextNearestTip(current, left, right)
      if (current == null || candidate === current) {
        current = candidate
        pending = since = null
      } else {
        if (pending !== candidate) { pending = candidate; since = now }
        if (now - since >= dwellMs) { current = candidate; pending = since = null }
      }
      return current
    },
  }
}

// Routes tracking frames to two stores with different update policies:
// - scene:   every display frame (live poses are blended between measurements,
//            see displayInterpolation.js) plus a ghost of the last pose when
//            live tracking is lost.
// - readout: raw measurements for numbers and status, refreshed at most every
//            READOUT_INTERVAL_MS — except that availability, status and safety
//            level changes publish immediately. Includes the approach trend
//            (history and speed; see trend.js).
// Safety levels come from one hysteresis tracker, so every panel and the 3D
// views always agree on the color.
export function createTrackingPipeline({ nearestDwellMs = NEAREST_DWELL_MS } = {}) {
  const readout = createStore({ frame: null, status: 'idle', levels: null, trend: null,
    held: false, heldSinceMs: null, measurementAgeMs: null, nearestTip: null,
    lastLevel: null, lastClearance: null, lostAtMs: null,
  })
  const scene = createStore(EMPTY_SCENE)
  const interpolator = createInterpolator()
  const tracker = createLevelTracker()
  const trend = createTrend()
  const nearestTracker = createNearestTipTracker(nearestDwellMs)

  let source = null
  let raw = { frame: null, status: 'idle' }
  let lastMeasured = null
  let levels = null
  let readoutDirty = false
  let lastReadoutAt = -Infinity
  let measurementAt = null
  let sourceTime = null
  let heldSinceMs = null
  let missing = false
  let nearest = null
  let lastLevel = null
  let lastClearance = null

  const age = (now) => measurementAt == null ? null : Math.max(0, now - measurementAt)

  function publishReadout(now) {
    if (raw.frame && levels) {
      lastLevel = levels.min
      lastClearance = Math.min(distanceToRetina(raw.frame.tip_left), distanceToRetina(raw.frame.tip_right))
    }
    readout.set({ frame: raw.frame, status: raw.status, levels, trend: trend.snapshot(now),
      held: heldSinceMs != null, heldSinceMs, measurementAgeMs: age(now),
      nearestTip: nearest, lastLevel, lastClearance,
      lostAtMs: scene.get().ghost ? scene.get().lostAt : null,
    })
    lastReadoutAt = now
    readoutDirty = false
  }

  function showFrame(frame) {
    const s = scene.get()
    if (s.frame === frame && s.levels === levels && !s.ghost) return
    scene.set({ frame, ghost: false, levels, lostAt: null })
  }

  function showUnavailable(now) {
    const s = scene.get()
    if (source === 'live' && lastMeasured && LOST.has(raw.status)) {
      if (!s.ghost) scene.set({ frame: lastMeasured, ghost: true, levels: null, lostAt: now })
    } else if (s.frame) {
      scene.set(EMPTY_SCENE)
    }
  }

  return {
    readout,
    scene,

    reset(nextSource, now = 0) {
      source = nextSource
      raw = { frame: null, status: initialStatus(nextSource) }
      lastMeasured = null
      levels = null
      measurementAt = sourceTime = heldSinceMs = null
      nearest = null
      lastLevel = lastClearance = null
      missing = false
      interpolator.push(null)
      tracker.reset()
      trend.reset()
      nearestTracker.reset()
      scene.set(EMPTY_SCENE)
      publishReadout(now)
    },

    publish(frame, status, now) {
      const previousHeld = heldSinceMs != null
      let newMeasurement = !!frame
      if (frame) {
        const t = frame.t ?? frame.timestamp
        if (Number.isFinite(t)) {
          newMeasurement = sourceTime == null || t > sourceTime
          if (newMeasurement) sourceTime = t
        } else {
          sourceTime = null
        }
        if (newMeasurement) measurementAt = now
        if (source === 'live' && age(now) >= STALE_MS) {
          frame = null
          status = 'stale'
        } else if (source === 'live' && !newMeasurement && age(now) >= DROPOUT_GRACE_MS) {
          frame = null
          status = 'degraded'
        }
      }
      missing = !frame && status === 'degraded'
      const withinGrace = missing && lastMeasured && measurementAt != null && age(now) < DROPOUT_GRACE_MS
      if (withinGrace || (frame && !newMeasurement)) {
        heldSinceMs ??= now
        frame = lastMeasured
        tracker.interrupt()
        nearestTracker.interrupt()
        trend.push(now, null)
      } else {
        heldSinceMs = null
      }
      const changed = !!frame !== !!raw.frame || status !== raw.status || previousHeld !== (heldSinceMs != null)
      raw = { frame, status }
      if (frame && newMeasurement && !withinGrace) lastMeasured = frame
      const previousLevels = levels
      if (heldSinceMs == null) {
        levels = tracker.update(frame, now)
        trend.push(now, frame ? Math.min(distanceToRetina(frame.tip_left), distanceToRetina(frame.tip_right)) : null)
        nearest = frame ? nearestTracker.update(distanceToRetina(frame.tip_left), distanceToRetina(frame.tip_right), now) : null
        if (!frame) nearestTracker.reset()
      }

      if (heldSinceMs != null) {
        interpolator.push(null)
        showFrame(frame)
      } else if (!frame) {
        interpolator.push(null)
        showUnavailable(now)
      } else if (source === 'live') {
        interpolator.push(frame, now)
        if (scene.get().ghost) showFrame(frame) // recovered: drop the ghost now
      } else {
        showFrame(frame)
      }

      if (changed || levels !== previousLevels) publishReadout(now)
      else readoutDirty = true
    },

    // Called once per animation frame.
    tick(now) {
      if (source === 'live' && measurementAt != null && ['live', 'degraded'].includes(raw.status) && age(now) >= STALE_MS) {
        this.publish(null, 'stale', now)
      } else if (source === 'live' && raw.frame && age(now) >= MISSED_UPDATE_MS) {
        this.publish(null, 'degraded', now)
      } else if (missing && raw.frame && age(now) >= DROPOUT_GRACE_MS) {
        this.publish(null, 'degraded', now)
      }
      if (source === 'live' && raw.frame && heldSinceMs == null) {
        const frame = interpolator.sample(now)
        if (frame) showFrame(frame)
      }
      if ((readoutDirty || measurementAt != null) && now - lastReadoutAt >= READOUT_INTERVAL_MS) publishReadout(now)
    },
  }
}
