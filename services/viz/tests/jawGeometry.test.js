import test from 'node:test'
import assert from 'node:assert/strict'
import { jawCenterline, sweepBlade } from '../src/scene/jawGeometry.js'
import { length, sub, dot, cross } from '../src/geometry.js'

const jaw = {
  base: [0.17, 0, 0],
  tip: [0.3, 1.65, 0],
  axis: [0, 1, 0],
  inward: [-1, 0, 0],
  widthBase: 0.17,
  widthTip: 0.09,
  hook: 0.09,
}

test('jaw reaches the tracked tip and hooks toward the other jaw', () => {
  const { points } = jawCenterline(jaw)
  const nearest = Math.min(...points.map((p) => length(sub(p, jaw.tip))))
  assert.ok(nearest < 0.03, `centerline passes the tracked tip (${nearest})`)
  const lip = points[points.length - 1]
  assert.ok(lip[0] < jaw.tip[0] - 0.05, 'hook lip points inward')
  assert.ok(lip[1] <= jaw.tip[1] + 0.02, 'hook does not overshoot the tip')
  assert.ok(points[0][1] < 0, 'stub starts inside the tube')
})

test('blade faces wind outward', () => {
  const g = sweepBlade(jawCenterline(jaw), [0, 0, 1], 0.06)
  const pos = g.getAttribute('position').array
  const nrm = g.getAttribute('normal').array
  const idx = g.getIndex().array
  const v = (i) => [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]
  for (let k = 0; k < idx.length; k += 3) {
    const [a, b, c] = [idx[k], idx[k + 1], idx[k + 2]]
    const face = cross(sub(v(b), v(a)), sub(v(c), v(a)))
    if (length(face) < 1e-12) continue
    const n = [nrm[3 * a], nrm[3 * a + 1], nrm[3 * a + 2]]
    assert.ok(dot(face, n) > 0, `triangle ${k / 3} faces its normal`)
  }
  assert.ok(Array.from(pos).every(Number.isFinite))
})
