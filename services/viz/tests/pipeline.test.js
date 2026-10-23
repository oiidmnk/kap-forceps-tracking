import test from 'node:test'
import assert from 'node:assert/strict'
import { nextLevel, createLevelTracker } from '../src/feed/levels.js'
import { createTrackingPipeline } from '../src/feed/pipeline.js'
import { EYE_RADIUS_MM, READOUT_INTERVAL_MS } from '../src/config.js'

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
  assert.equal(nextLevel('danger', 0.56), 'warn')
  assert.equal(nextLevel('warn', 2.03), 'warn')
  assert.equal(nextLevel('warn', 2.06), 'safe')
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
