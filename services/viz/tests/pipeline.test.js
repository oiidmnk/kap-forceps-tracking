import test from 'node:test'
import assert from 'node:assert/strict'
import { nextLevel, createLevelTracker } from '../src/feed/levels.js'
import { createTrackingPipeline, createNearestTipTracker, nextNearestTip } from '../src/feed/pipeline.js'
import { EYE_RADIUS_MM, READOUT_INTERVAL_MS, MISSED_UPDATE_MS, DROPOUT_GRACE_MS, STALE_MS } from '../src/config.js'

// Both tips at the given distance from the retina (straight down).
const at = (mm, x = 0) => ({
  tip_left: [x, -(EYE_RADIUS_MM - mm), 0],
  tip_right: [x + 0.2, -(EYE_RADIUS_MM - mm), 0],
  trocar: [0, 10, 0],
  light_tip: null,
  light_trocar: null,
})

test('escalation is immediate, de-escalation needs the margin', () => {
  assert.equal(nextLevel('safe', 1.99), 'warn')
  assert.equal(nextLevel('warn', 0.49), 'danger')
  assert.equal(nextLevel('danger', 0.52), 'danger')
  assert.equal(nextLevel('danger', 0.66), 'warn')
  assert.equal(nextLevel('warn', 2.03), 'warn')
  assert.equal(nextLevel('warn', 2.16), 'safe')
  assert.equal(nextLevel('danger', 3), 'safe', 'a large jump goes straight to the true level')
})

test('noise around a threshold does not flicker the level', () => {
  const tracker = createLevelTracker()
  const seen = new Set()
  for (const mm of [0.49, 0.51, 0.48, 0.53, 0.5, 0.52, 0.49]) seen.add(tracker.update(at(mm)).min)
  assert.deepEqual([...seen], ['danger'])
})

test('readout numbers are throttled, level changes publish at once', () => {
  const p = createTrackingPipeline()
  p.reset('mock', 0)
  let readouts = 0
  p.readout.subscribe(() => readouts++)
  for (let t = 0; t < 1000; t += 16) {
    p.publish(at(5 + t / 1000), 'mock', t)
    p.tick(t)
  }
  // 1 availability change + ~10 per second
  assert.ok(readouts <= 1 + 1000 / READOUT_INTERVAL_MS + 1, `throttled (${readouts})`)
  assert.ok(readouts >= 1000 / READOUT_INTERVAL_MS - 1, `still refreshing (${readouts})`)

  const before = readouts
  p.publish(at(1.5), 'mock', 1001) // safe → warn
  assert.equal(readouts, before + 1)
  assert.equal(p.readout.get().levels.min, 'warn')
})

test('mock frames reach the scene store every frame', () => {
  const p = createTrackingPipeline()
  p.reset('mock', 0)
  let scenes = 0
  p.scene.subscribe(() => scenes++)
  for (let t = 0; t < 160; t += 16) p.publish(at(4, t / 1000), 'mock', t)
  assert.equal(scenes, 10)
})

test('lost live tracking keeps the last pose as a ghost without measurements', () => {
  const p = createTrackingPipeline()
  p.reset('live', 0)
  p.publish(at(3, 0), 'live', 0)
  p.publish(at(3, 1), 'live', 66)
  p.tick(200)
  p.publish(null, 'stale', 1800)

  const scene = p.scene.get()
  assert.equal(scene.ghost, true)
  assert.deepEqual(scene.frame.tip_left, at(3, 1).tip_left, 'ghost is the last measured pose')
  assert.equal(scene.levels, null)
  assert.equal(p.readout.get().frame, null, 'no measurement is invented')
  assert.equal(p.readout.get().status, 'stale')

  p.publish(at(2.5, 2), 'live', 2500)
  assert.equal(p.scene.get().ghost, false, 'recovers on the next complete frame')
})

test('switching source never carries a ghost or frame over', () => {
  const p = createTrackingPipeline()
  p.reset('live', 0)
  p.publish(at(3), 'live', 0)
  p.publish(null, 'error', 100)
  assert.equal(p.scene.get().ghost, true)
  p.reset('mock', 200)
  assert.equal(p.scene.get().frame, null)
  assert.equal(p.readout.get().frame, null)
})

test('waiting for a first live frame shows no ghost', () => {
  const p = createTrackingPipeline()
  p.reset('live', 0)
  p.publish(null, 'waiting', 10)
  assert.equal(p.scene.get().frame, null)
  assert.equal(p.scene.get().ghost, false)
})

test('dropout grace holds danger with one trend gap, then expires without messages', () => {
  const p = createTrackingPipeline()
  p.reset('live', 0)
  const frame = at(0.3)
  p.publish(frame, 'live', 0)
  p.publish(null, 'degraded', 50)
  assert.equal(p.readout.get().frame, frame)
  assert.equal(p.readout.get().levels.min, 'danger')
  assert.equal(p.readout.get().held, true)
  assert.equal(p.readout.get().heldSinceMs, 50)
  assert.equal(p.readout.get().measurementAgeMs, 50)
  assert.equal(p.scene.get().ghost, false)
  p.publish(null, 'degraded', 150)
  p.tick(399)
  assert.equal(p.readout.get().levels.min, 'danger')
  assert.equal(p.readout.get().trend.points.filter(([, d]) => d == null).length, 1)
  assert.equal(p.readout.get().trend.points.filter(([, d]) => d != null).length, 1)
  p.tick(400)
  assert.equal(p.readout.get().frame, null)
  assert.equal(p.readout.get().levels, null)
  assert.equal(p.readout.get().held, false)
  assert.equal(p.scene.get().ghost, true)
  p.publish(at(0.2), 'live', 450)
  assert.equal(p.scene.get().ghost, false)
  assert.equal(p.readout.get().held, false)
  assert.equal(p.readout.get().measurementAgeMs, 0)
})

test('frozen or regressing source time becomes stale despite ongoing receipt', () => {
  const p = createTrackingPipeline()
  p.reset('live', 0)
  for (let now = 0; now <= 1500; now += 100) p.publish({ ...at(1), t: 90000 }, 'live', now)
  assert.equal(p.readout.get().status, 'stale')
  assert.equal(p.readout.get().measurementAgeMs, 1500)
  assert.equal(p.scene.get().ghost, true)
  p.publish({ ...at(1), t: 89999 }, 'live', 1600)
  assert.equal(p.readout.get().status, 'stale')
  p.publish({ ...at(0.2), t: 90001 }, 'live', 1700)
  assert.equal(p.readout.get().status, 'live')
  assert.equal(p.readout.get().measurementAgeMs, 0)
  assert.equal(p.readout.get().levels.min, 'danger')
})

test('source-less frames use receipt freshness; silence becomes stale on tick', () => {
  const p = createTrackingPipeline()
  p.reset('live', 0)
  for (let now = 0; now <= 2000; now += 100) p.publish(at(1), 'live', now)
  p.tick(2100)
  assert.equal(p.readout.get().status, 'live')
  assert.equal(p.readout.get().measurementAgeMs, 100)
  p.tick(3500)
  assert.equal(p.readout.get().status, 'stale')
})

test('dwell requires continuous evidence; escalation immediate; dwell configurable', () => {
  const tracker = createLevelTracker({ margin: 0.15, dwellMs: 250 })
  assert.equal(tracker.update(at(0.3), 0).min, 'danger')
  assert.equal(tracker.update(at(0.6), 100).min, 'danger')
  assert.equal(tracker.update(at(0.8), 200).min, 'danger')
  assert.equal(tracker.update(at(0.8), 449).min, 'danger')
  assert.equal(tracker.update(at(0.8), 450).min, 'warn')
  assert.equal(tracker.update(at(0.3), 451).min, 'danger')
  tracker.update(at(3), 500)
  tracker.interrupt()
  assert.equal(tracker.update(at(3), 800).min, 'danger')
  assert.equal(tracker.update(at(3), 1050).min, 'safe')
  const immediate = createLevelTracker({ margin: 0.05, dwellMs: 0 })
  immediate.update(at(0.3), 0)
  assert.equal(immediate.update(at(0.6), 1).min, 'warn')
})

test('readout identifies nearer tip or tie and omits numeric TTC', () => {
  const p = createTrackingPipeline({ nearestDwellMs: 0 })
  p.reset('mock', 0)
  const frame = (left, right) => ({ ...at(1), tip_left: [0, left - 12, 0], tip_right: [0, right - 12, 0] })
  for (const [left, right, expected, now] of [[1, 2, 'left', 0], [2, 1, 'right', 100], [1, 1.09, 'tie', 200], [1, 1.11, 'tie', 300], [1, 1.16, 'left', 400]]) {
    p.publish(frame(left, right), 'mock', now)
    p.tick(now)
    assert.equal(p.readout.get().nearestTip, expected)
    assert.equal('ttc' in p.readout.get().trend, false)
  }
})

test('silence holds after missed updates, ghosts after grace, then becomes stale', () => {
  const pipeline = createTrackingPipeline()
  pipeline.reset('live')
  const frame = at(0.3)
  pipeline.publish(frame, 'live', 100)
  const levels = pipeline.readout.get().levels
  pipeline.tick(100 + MISSED_UPDATE_MS - 1)
  assert.equal(pipeline.readout.get().held, false)
  pipeline.tick(100 + MISSED_UPDATE_MS)
  assert.equal(pipeline.readout.get().held, true)
  assert.equal(pipeline.readout.get().heldSinceMs, 100 + MISSED_UPDATE_MS)
  assert.equal(pipeline.readout.get().frame, frame)
  assert.equal(pipeline.readout.get().levels, levels)
  assert.equal(pipeline.scene.get().frame, frame)
  pipeline.tick(100 + DROPOUT_GRACE_MS - 1)
  assert.equal(pipeline.scene.get().ghost, false)
  pipeline.tick(100 + DROPOUT_GRACE_MS)
  assert.equal(pipeline.scene.get().ghost, true)
  assert.equal(pipeline.readout.get().frame, null)
  assert.equal(pipeline.readout.get().nearestTip, null)
  pipeline.tick(100 + STALE_MS)
  assert.equal(pipeline.readout.get().status, 'stale')
  pipeline.publish(at(1), 'live', 1700)
  assert.equal(pipeline.readout.get().held, false)
  assert.equal(pipeline.scene.get().ghost, false)
})

test('repeated source timestamps cannot extend the dropout grace', () => {
  const pipeline = createTrackingPipeline()
  pipeline.reset('live')
  for (let now = 0; now <= DROPOUT_GRACE_MS; now += 50) {
    pipeline.publish({ ...at(0.3), t: 10 }, 'live', now)
  }
  assert.equal(pipeline.readout.get().frame, null)
  assert.equal(pipeline.scene.get().ghost, true)
})

test('animation-rate simulation frames never enter held', () => {
  for (const source of ['mock', 'debug', 'live']) {
    const pipeline = createTrackingPipeline()
    pipeline.reset(source)
    for (let now = 0; now < 2000; now += 16) {
      pipeline.tick(now)
      pipeline.publish(at(1), source, now)
      assert.equal(pipeline.readout.get().held, false)
    }
  }
})

test('retained flags track raw classification through hysteresis and dwell', () => {
  const tracker = createLevelTracker()
  assert.deepEqual(tracker.update(at(0.3), 0).retained, { left: false, right: false, min: false })
  assert.deepEqual(tracker.update(at(0.54), 100).retained, { left: true, right: true, min: true })
  assert.equal(tracker.update(at(0.8), 200).retained.min, true)
  assert.equal(tracker.update(at(0.8), 449).retained.min, true)
  assert.equal(tracker.update(at(0.8), 450).retained.min, false)
  assert.equal(tracker.update(at(2.04), 500).retained.min, true)
  assert.equal(tracker.update(at(2.2), 600).retained.min, true)
  assert.equal(tracker.update(at(2.2), 850).retained.min, false)
  assert.equal(tracker.update(at(0.3), 851).retained.min, false)
  assert.equal(tracker.update(null), null)
})

test('retained changes publish immediately even when the alarm level stays unchanged', () => {
  const pipeline = createTrackingPipeline()
  pipeline.reset('mock')
  pipeline.publish(at(0.3), 'mock', 0)
  pipeline.publish(at(0.54), 'mock', 1)
  assert.equal(pipeline.readout.get().levels.min, 'danger')
  assert.equal(pipeline.readout.get().levels.retained.min, true)
  pipeline.publish(at(0.3), 'mock', 2)
  assert.equal(pipeline.readout.get().levels.retained.min, false)
})

test('per-tip retention does not imply a retained minimum when the other tip is critical', () => {
  const tracker = createLevelTracker()
  tracker.update(at(0.3), 0)
  const levels = tracker.update({ ...at(0.3), tip_left: [0, 0.54 - EYE_RADIUS_MM, 0] }, 100)
  assert.deepEqual(levels.retained, { left: true, right: false, min: false })
})

test('nearest-tip tie band has distinct entry and exit boundaries', () => {
  assert.equal(nextNearestTip('left', 0, 0.10), 'left')
  assert.equal(nextNearestTip('left', 0, 0.099), 'tie')
  assert.equal(nextNearestTip('tie', 0, 0.15), 'tie')
  assert.equal(nextNearestTip('tie', 0, 0.151), 'left')
  assert.equal(nextNearestTip('tie', 0.15, 0), 'tie')
  assert.equal(nextNearestTip('tie', 0.151, 0), 'right')
  assert.equal(nextNearestTip('left', 0.049, 0), 'tie')
  assert.equal(nextNearestTip('left', 0.16, 0), 'right')
  assert.equal(nextNearestTip('right', 0, 0.049), 'tie')
  assert.equal(nextNearestTip('right', 0, 0.16), 'left')
})


test('ghost retains the last displayed level and clearance without a current alarm', () => {
  const p = createTrackingPipeline()
  p.reset('live')
  p.publish(at(0.3), 'live', 0)
  // Hysteresis keeps Critical even as the numeric clearance increases.
  p.publish(at(0.54), 'live', 100)
  const displayed = p.readout.get()
  assert.equal(displayed.lastLevel, 'danger')
  assert.ok(Math.abs(displayed.lastClearance - (EYE_RADIUS_MM - Math.hypot(0.2, EYE_RADIUS_MM - 0.54))) < 1e-9)
  p.tick(100 + DROPOUT_GRACE_MS)
  const ghost = p.readout.get()
  assert.equal(ghost.levels, null)
  assert.equal(ghost.frame, null)
  assert.equal(ghost.lastLevel, displayed.lastLevel)
  assert.equal(ghost.lastClearance, displayed.lastClearance)
  assert.equal(ghost.lostAtMs, p.scene.get().lostAt)
  p.tick(100 + STALE_MS)
  assert.equal(p.readout.get().lostAtMs, ghost.lostAtMs)
  assert.equal(p.readout.get().lastClearance, displayed.lastClearance)
  p.publish(at(3), 'live', 2000)
  assert.equal(p.readout.get().lostAtMs, null)
  assert.equal(p.readout.get().lastLevel, 'safe')
  p.reset('debug', 2100)
  assert.equal(p.readout.get().lastLevel, null)
  assert.equal(p.readout.get().lastClearance, null)
  assert.equal(p.readout.get().lostAtMs, null)
})

test('ghost history uses the last published readout, not an unpublished raw frame', () => {
  const p = createTrackingPipeline()
  p.reset('live')
  p.publish(at(3), 'live', 0)
  const displayed = p.readout.get().lastClearance
  p.publish(at(4), 'live', 1)
  p.publish(null, 'error', 2)
  assert.equal(p.readout.get().lastClearance, displayed)
  assert.equal(p.readout.get().lastLevel, 'safe')
  assert.equal(p.readout.get().levels, null)
})

test('nearest label requires continuous dwell, interrupts on gaps and resets', () => {
  const t = createNearestTipTracker()
  assert.equal(t.update(1, 2, 0), 'left')
  assert.equal(t.update(2, 1, 10), 'left')
  assert.equal(t.update(2, 1, 509), 'left')
  assert.equal(t.update(2, 1, 510), 'right')
  assert.equal(t.update(1, 1.01, 600), 'right')
  assert.equal(t.update(1, 1.01, 1100), 'tie')
  assert.equal(t.update(1, 2, 1200), 'tie')
  assert.equal(t.update(1, 2, 1700), 'left')
  t.update(2, 1, 1800)
  t.update(1, 2, 1900)
  assert.equal(t.update(2, 1, 2000), 'left')
  assert.equal(t.update(2, 1, 2499), 'left', 'reversal cancels pending evidence')
  t.interrupt()
  assert.equal(t.update(2, 1, 2600), 'left', 'gap cannot complete dwell')
  assert.equal(t.update(2, 1, 3100), 'right')
  t.reset()
  assert.equal(t.update(1, 2, 3200), 'left')
})

test('pipeline holds nearest label through grace without accumulating evidence', () => {
  const p = createTrackingPipeline()
  p.reset('replay')
  const f = (l,r,t) => ({ ...at(l), tip_right: [0, r-12, 0], t })
  p.publish(f(1,2,0), 'replay', 0)
  p.publish(f(2,1,100), 'replay', 100)
  p.publish(null, 'degraded', 200)
  p.publish(f(2,1,300), 'replay', 300)
  p.tick(600)
  assert.equal(p.readout.get().nearestTip, 'left')
  p.publish(f(2,1,800), 'replay', 800)
  p.tick(800)
  assert.equal(p.readout.get().nearestTip, 'right')
  p.publish(null, 'degraded', 1200)
  assert.equal(p.readout.get().nearestTip, null)
  p.publish(f(1,2,1300), 'replay', 1300)
  assert.equal(p.readout.get().nearestTip, 'left')
})
