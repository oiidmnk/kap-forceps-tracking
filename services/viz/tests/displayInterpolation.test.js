import test from 'node:test'
import assert from 'node:assert/strict'
import { createInterpolator } from '../src/feed/displayInterpolation.js'

const pose = (x) => ({ tip_left: [x, 0, 0], tip_right: [x + 0.5, 0, 0], trocar: [0, 10, 0], light_tip: null, light_trocar: null })

test('repeated poses are ignored and motion is blended between measurements', () => {
  const it = createInterpolator()
  // 15 Hz poses, each re-sent by the stream every ~33 ms; rendered at 60 Hz.
  const xs = []
  let k = 0
  for (let t = 0; t <= 1000; t += 1) {
    if (t % 33 === 0) {
      k = Math.floor(t / 66)
      it.push(pose(k), t)
    }
    if (t % 16 === 0 && t >= 400) xs.push(it.sample(t).tip_left[0])
  }
  for (let i = 1; i < xs.length; i++) {
    const step = xs[i] - xs[i - 1]
    assert.ok(step > 0.1 && step < 0.4, `steady forward motion (step ${step.toFixed(3)})`)
  }
})

test('never extrapolates past the newest measurement', () => {
  const it = createInterpolator()
  it.push(pose(0), 0)
  it.push(pose(1), 66)
  assert.deepEqual(it.sample(10_000).tip_left, [1, 0, 0])
})

test('a pause resynchronises instead of sweeping across the gap', () => {
  const it = createInterpolator()
  it.push(pose(0), 0)
  it.push(pose(1), 66)
  it.push(pose(5), 2000)
  assert.deepEqual(it.sample(2001).tip_left, [5, 0, 0])
})

test('losing tracking clears the buffer', () => {
  const it = createInterpolator()
  it.push(pose(0), 0)
  it.push(null, 10)
  assert.equal(it.sample(20), null)
})
