// Approach trend of the minimum tip–retina distance, for the panel's sparkline
// and the approach-speed / time-to-contact readout.
//
// Samples are kept for TREND_WINDOW_MS (thinned to one per SAMPLE_MS); a null
// distance marks a tracking gap. Velocity is the least-squares slope over the
// last SLOPE_MS of contiguous samples, lightly smoothed, so a single noisy
// frame cannot swing it. Time to contact is only estimated while approaching.

export const TREND_WINDOW_MS = 10000
const SAMPLE_MS = 50
const SLOPE_MS = 600
const MIN_SLOPE_SAMPLES = 4
const MIN_SLOPE_SPAN_MS = 250
export const STEADY_MM_S = 0.15 // below this speed the instrument counts as holding
export const MAX_TTC_S = 10

export function createTrend() {
  let samples = []
  let velocity = null

  function slope(now) {
    // Contiguous run of measured samples ending at the newest one.
    const run = []
    for (let i = samples.length - 1; i >= 0; i--) {
      const s = samples[i]
      if (s.d == null || now - s.t > SLOPE_MS) break
      run.push(s)
    }
    if (run.length < MIN_SLOPE_SAMPLES) return null
    if (run[0].t - run[run.length - 1].t < MIN_SLOPE_SPAN_MS) return null
    let st = 0
    let sd = 0
    for (const s of run) {
      st += s.t
      sd += s.d
    }
    const mt = st / run.length
    const md = sd / run.length
    let num = 0
    let den = 0
    for (const s of run) {
      num += (s.t - mt) * (s.d - md)
      den += (s.t - mt) ** 2
    }
    return den > 0 ? (num / den) * 1000 : null // mm/s
  }

  return {
    push(t, d) {
      const last = samples[samples.length - 1]
      const gap = d == null
      if (last && (last.d == null) === gap && t - last.t < SAMPLE_MS) return
      if (gap && last?.d == null) return // one marker per gap is enough
      samples.push({ t, d })
      while (samples.length && t - samples[0].t > TREND_WINDOW_MS) samples.shift()
    },

    snapshot(now) {
      const raw = slope(now)
      velocity = raw == null ? null : velocity == null ? raw : velocity + (raw - velocity) * 0.5
      const latest = samples[samples.length - 1]
      const current = latest?.d ?? null
      const ttc =
        velocity != null && current != null && velocity < -STEADY_MM_S
          ? current / -velocity
          : null
      return {
        now,
        points: samples.map((s) => [s.t, s.d]),
        velocity,
        ttc: ttc != null && ttc <= MAX_TTC_S ? ttc : null,
      }
    },

    reset() {
      samples = []
      velocity = null
    },
  }
}
