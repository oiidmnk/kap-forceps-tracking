// Display-only smoothing for live tracking. The detector publishes ~15 Hz and
// the stream re-sends the latest result every ~33 ms, so the dashboard sees a
// jittery mix of repeated and new poses. Rendering those directly makes the
// instrument step, which the magnified proximity scope shows as stutter.
//
// The interpolator drops repeats, assigns each new pose a steady timestamp,
// and renders slightly in the past (about one sample interval) so it can
// blend between two real measurements. It never extrapolates: past the newest
// sample it holds that sample. Numeric readouts should keep using raw frames.

const POINT_KEYS = ['tip_left', 'tip_right', 'trocar', 'light_tip', 'light_trocar']
const MAX_SAMPLES = 4
const MIN_DELAY_MS = 30
const MAX_DELAY_MS = 150
const RESYNC_INTERVALS = 4 // gaps longer than this many intervals snap instead of blend

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x))

function samePoint(a, b) {
  if (a === b) return true
  if (!a || !b) return false
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2]
}

export function samePose(a, b) {
  return POINT_KEYS.every((k) => samePoint(a[k], b[k]))
}

function lerpPoint(a, b, u) {
  if (!a || !b) return b ?? null
  return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u]
}

export function lerpFrame(a, b, u) {
  const out = { ...b }
  for (const k of POINT_KEYS) out[k] = lerpPoint(a[k], b[k], u)
  return out
}

export function createInterpolator() {
  let samples = [] // { t, frame } with steady (regularised) timestamps
  let interval = 66 // estimated ms between new poses

  return {
    push(frame, now) {
      if (!frame) {
        samples = []
        return
      }
      const last = samples[samples.length - 1]
      if (!last) {
        samples.push({ t: now, frame })
        return
      }
      if (samePose(last.frame, frame)) {
        last.frame = frame // repeated pose: keep its timestamp, refresh metadata
        return
      }
      const gap = now - last.t
      let t
      if (gap > RESYNC_INTERVALS * interval) {
        samples = [] // tracking resumed after a pause: start over, no long blend
        t = now
      } else {
        interval = 0.85 * interval + 0.15 * clamp(gap, 5, 500)
        const predicted = last.t + interval
        t = Math.min(now, predicted + 0.3 * (now - predicted))
      }
      samples.push({ t, frame })
      if (samples.length > MAX_SAMPLES) samples.shift()
    },

    sample(now) {
      if (!samples.length) return null
      const t = now - clamp(1.2 * interval, MIN_DELAY_MS, MAX_DELAY_MS)
      for (let i = samples.length - 1; i > 0; i--) {
        const a = samples[i - 1]
        const b = samples[i]
        if (t >= b.t) return b.frame
        if (t >= a.t) return lerpFrame(a.frame, b.frame, (t - a.t) / (b.t - a.t))
      }
      return samples[0].frame
    },
  }
}
