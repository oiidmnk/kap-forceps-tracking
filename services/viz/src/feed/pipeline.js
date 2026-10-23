import { createStore } from './store.js'
import { createInterpolator } from './displayInterpolation.js'
import { createLevelTracker } from './levels.js'
import { createTrend } from './trend.js'
import { distanceToRetina } from '../geometry.js'
import { READOUT_INTERVAL_MS } from '../config.js'

// Statuses in which live tracking was lost after it had been running. The
// scene then keeps the last measured pose as a faded "ghost"; readouts stay
// empty because there is no current measurement.
const LOST = new Set(['stale', 'degraded', 'error'])

const EMPTY_SCENE = { frame: null, ghost: false, levels: null, lostAt: null }

export function initialStatus(source) {
  return source === 'live' ? 'connecting' : source
}

// Routes tracking frames to two stores with different update policies:
// - scene:   every display frame (live poses are blended between measurements,
//            see displayInterpolation.js) plus a ghost of the last pose when
//            live tracking is lost.
// - readout: raw measurements for numbers and status, refreshed at most every
//            READOUT_INTERVAL_MS — except that availability, status and safety
//            level changes publish immediately. Includes the approach trend
//            (history, speed, time to contact; see trend.js).
// Safety levels come from one hysteresis tracker, so every panel and the 3D
// views always agree on the color.
export function createTrackingPipeline() {
  const readout = createStore({ frame: null, status: 'idle', levels: null, trend: null })
  const scene = createStore(EMPTY_SCENE)
  const interpolator = createInterpolator()
  const tracker = createLevelTracker()
  const trend = createTrend()

  let source = null
  let raw = { frame: null, status: 'idle' }
  let lastMeasured = null
  let levels = null
  let readoutDirty = false
  let lastReadoutAt = -Infinity

  function publishReadout(now) {
    readout.set({ frame: raw.frame, status: raw.status, levels, trend: trend.snapshot(now) })
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
      interpolator.push(null)
      tracker.reset()
      trend.reset()
      scene.set(EMPTY_SCENE)
      publishReadout(now)
    },

    publish(frame, status, now) {
      const changed = !!frame !== !!raw.frame || status !== raw.status
      raw = { frame, status }
      if (frame) lastMeasured = frame
      const previousLevels = levels
      levels = tracker.update(frame)
      trend.push(
        now,
        frame ? Math.min(distanceToRetina(frame.tip_left), distanceToRetina(frame.tip_right)) : null
      )

      if (!frame) {
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
      if (source === 'live' && raw.frame) {
        const frame = interpolator.sample(now)
        if (frame) showFrame(frame)
      }
      if (readoutDirty && now - lastReadoutAt >= READOUT_INTERVAL_MS) publishReadout(now)
    },
  }
}
