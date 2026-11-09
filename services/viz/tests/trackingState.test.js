import test from 'node:test'
import assert from 'node:assert/strict'
import { ROD_LENGTH_MM } from '../src/config.js'
import {
  parseTrackingFrame,
  trackingIsStale,
  STALE_AFTER_MS,
} from '../src/feed/trackingState.js'
import { distanceStatus } from '../src/ui.js'
const native = {
  t: 1,
  tip_left: [0, -11, 0],
  tip_right: [0, -10, 0],
  trocar: [0, 10, 0],
}

test('native millimeter frames preserve coordinates and do not invent telemetry', () => {
  const frame = parseTrackingFrame(native)
  assert.deepEqual(frame.tip_left, native.tip_left)
  assert.equal(frame.confidence, null)
  assert.equal(frame.calibrated, null)
})
test('normalized reconstruction maps upstream axes and millimeter scale', () => {
  const frame = parseTrackingFrame({
    timestamp: 2,
    positions: {
      left_tip_forceps: [0.1, 0.2, 0.3],
      right_tip_forceps: [0, 0, 0.5],
      trocar_forceps: [0, 0, -0.8],
    },
  })
  frame.tip_left.forEach((n, i) =>
    assert.ok(Math.abs(n - [1.2, -3.6, 2.4][i]) < 1e-10)
  )
  assert.equal(frame.confidence, null)
})
test('missing tips invalidate the entire frame instead of carrying forward measurements', () => {
  assert.equal(parseTrackingFrame({ ...native, tip_right: null }), null)
  assert.equal(
    parseTrackingFrame({
      positions: {
        left_tip_forceps: [0, 0, 0],
        right_tip_forceps: null,
        trocar_forceps: [1, 0, 0],
      },
    }),
    null
  )
})
test('malformed coordinates and non-finite values are rejected', () => {
  for (const tip_left of [
    [1, 2],
    [1, NaN, 3],
    [1, Infinity, 3],
    ['1', 2, 3],
  ])
    assert.equal(parseTrackingFrame({ ...native, tip_left }), null)
  for (const value of [null, 'text', {}, []])
    assert.equal(parseTrackingFrame(value), null)
})
test('only explicit valid confidence and calibration are shown', () => {
  const frame = parseTrackingFrame({
    ...native,
    confidence: 0.81,
    calibrated: false,
  })
  assert.equal(frame.confidence, 0.81)
  assert.equal(frame.calibrated, false)
  for (const confidence of [-1, 1.5, '0.9', NaN])
    assert.equal(parseTrackingFrame({ ...native, confidence }).confidence, null)
  assert.equal(
    parseTrackingFrame({ ...native, calibrated: 'true' }).calibrated,
    null
  )
})
test('invalid optional light geometry does not invalidate tracked forceps', () => {
  assert.equal(
    parseTrackingFrame({ ...native, light_tip: [NaN, 1, 2] }).light_tip,
    null
  )
})
test('stale timeout starts at the last complete received frame', () => {
  assert.equal(trackingIsStale(null, 10000), false)
  assert.equal(trackingIsStale(0, STALE_AFTER_MS - 1), false)
  assert.equal(trackingIsStale(0, STALE_AFTER_MS), true)
  assert.equal(trackingIsStale(1600, 1700), false)
})
test('threshold boundaries preserve distinct text and shape cues', () => {
  assert.equal(distanceStatus(0.49).level, 'danger')
  assert.equal(distanceStatus(0.5).level, 'warn')
  assert.equal(distanceStatus(1.99).level, 'warn')
  assert.equal(distanceStatus(2).level, 'safe')
  assert.equal(new Set([0, 1, 3].map((n) => distanceStatus(n).symbol)).size, 3)
})

test('depth bars give the critical first millimetres most of the room', async () => {
  const { depthScale } = await import('../src/ui.js')
  assert.equal(depthScale(0), 0)
  assert.equal(depthScale(10), 1)
  assert.equal(depthScale(25), 1, 'clamped above the scale')
  assert.ok(depthScale(0.5) > 0.2 && depthScale(2) > 0.4, 'thresholds sit well inside the bar')
})
test('the rod is optional and maps like the tips', () => {
  const positions = {
    left_tip_forceps: [0.1, 0.2, 0.3],
    right_tip_forceps: [0, 0, 0.5],
    trocar_forceps: [0, 0, -0.8],
  }
  assert.equal(parseTrackingFrame({ positions }).rod_a, null)
  const frame = parseTrackingFrame({ positions: { ...positions, rod_a: [0.1, 0.2, 0.3], rod_b: null } })
  assert.equal(frame.rod_a, null) // never half a rod
  const both = parseTrackingFrame({ positions: { ...positions, rod_a: [0.1, 0.2, 0.3], rod_b: [0, 0, 0.5] } })
  // same midpoint and direction as the tips it was given, but the rod's fixed length
  const mid = (p, q) => p.map((v, i) => (v + q[i]) / 2)
  const dir = (p, q) => {
    const d = q.map((v, i) => v - p[i])
    const n = Math.hypot(...d)
    return d.map((v) => v / n)
  }
  const close = (p, q) => p.every((v, i) => Math.abs(v - q[i]) < 1e-9)
  assert.ok(close(mid(both.rod_a, both.rod_b), mid(both.tip_left, both.tip_right)))
  assert.ok(close(dir(both.rod_a, both.rod_b), dir(both.tip_left, both.tip_right)))
  assert.ok(Math.abs(Math.hypot(...both.rod_b.map((v, i) => v - both.rod_a[i])) - ROD_LENGTH_MM) < 1e-9)
})
test('a frame with only the rod is valid and has no forceps', async () => {
  const { hasForceps } = await import('../src/feed/trackingState.js')
  const frame = parseTrackingFrame({
    positions: { left_tip_forceps: null, right_tip_forceps: null, trocar_forceps: [0, 0, -1], rod_a: [0.1, 0.2, 0.3], rod_b: [0, 0, 0.5] },
  })
  assert.ok(frame && frame.rod_a)
  assert.equal(hasForceps(frame), false)
  assert.equal(frame.trocar, null)
})
test('rod contact is reported only with a whole rod', () => {
  const positions = {
    left_tip_forceps: [0.1, 0.2, 0.3],
    right_tip_forceps: [0, 0, 0.5],
    trocar_forceps: [0, 0, -0.8],
    rod_a: [0.1, 0.2, 0.3],
    rod_b: [0, 0, 0.5],
  }
  assert.equal(parseTrackingFrame({ positions: { ...positions, rod_contact: true } }).rod_contact, true)
  assert.equal(parseTrackingFrame({ positions }).rod_contact, false)
  assert.equal(parseTrackingFrame({ positions: { ...positions, rod_b: null, rod_contact: true } }).rod_contact, false)
})
