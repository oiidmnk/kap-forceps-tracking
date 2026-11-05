import test from 'node:test'
import assert from 'node:assert/strict'
import { createTrend, TREND_WINDOW_MS } from '../src/feed/trend.js'
import { instrumentMetrics } from '../src/geometry.js'

test('steady approach gives negative velocity', () => {
  const trend = createTrend()
  // 3 mm, approaching at 1 mm/s, sampled at 60 Hz
  for (let t = 0; t <= 1000; t += 16) trend.push(t, 3 - t / 1000)
  const s = trend.snapshot(1000)
  assert.ok(Math.abs(s.velocity + 1) < 0.05, `velocity ${s.velocity}`)
  assert.equal('ttc' in s, false)
})

test('holding is near zero velocity; retracting has positive velocity', () => {
  const hold = createTrend()
  for (let t = 0; t <= 1000; t += 16) hold.push(t, 2 + 0.01 * Math.sin(t))
  assert.ok(Math.abs(hold.snapshot(1000).velocity) < 0.15)

  const back = createTrend()
  for (let t = 0; t <= 1000; t += 16) back.push(t, 1 + t / 1000)
  const s = back.snapshot(1000)
  assert.ok(s.velocity > 0.9)
})

test('a tracking gap breaks the slope and stays visible in the history', () => {
  const trend = createTrend()
  for (let t = 0; t <= 500; t += 16) trend.push(t, 3 - t / 1000)
  trend.push(520, null)
  trend.push(540, null)
  const s = trend.snapshot(600)
  assert.equal(s.velocity, null)
  assert.equal(s.points.filter(([, d]) => d == null).length, 1)
})

test('history is limited to the trend window', () => {
  const trend = createTrend()
  for (let t = 0; t <= 30000; t += 16) trend.push(t, 5)
  const { points } = trend.snapshot(30000)
  assert.ok(points[points.length - 1][0] - points[0][0] <= TREND_WINDOW_MS)
  assert.ok(points.length <= TREND_WINDOW_MS / 50 + 2)
})

test('instrument metrics: opening, insertion depth and angle to the retina', () => {
  // Shaft straight down from the top: perpendicular to the posterior pole.
  const m = instrumentMetrics({ tip_left: [-0.2, -5, 0], tip_right: [0.2, -5, 0], trocar: [0, 10, 0] })
  assert.ok(Math.abs(m.openingMm - 0.4) < 1e-9)
  assert.ok(Math.abs(m.insertionMm - 15) < 1e-9)
  assert.ok(Math.abs(m.retinaAngleDeg - 90) < 1e-6)
})
