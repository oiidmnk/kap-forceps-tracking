import test from 'node:test'
import assert from 'node:assert/strict'
import {
  forcepsRenderPose,
  nearestTip,
  distanceToRetina,
  length,
  sub,
} from '../src/geometry.js'

for (const [name, left, right] of [
  ['open jaws near the surface', [-0.3, -11.7, 0.2], [0.4, -11.5, 0.1]],
  ['rotated asymmetric jaws', [7, -8, 1], [7.2, -7.8, 1.4]],
  ['closed jaws', [1, -11, 0], [1, -11, 0]],
]) {
  test(`rendered tips agree with measurements and reticle: ${name}`, () => {
    const frame = { tip_left: left, tip_right: right, trocar: [0, 10, 2] }
    const pose = forcepsRenderPose(frame)
    assert.deepEqual(pose.tipLeftRender, left)
    assert.deepEqual(pose.tipRightRender, right)
    assert.equal(length(sub(pose.tipLeftRender, pose.tipRightRender)), pose.openingMm)

    const nearest = nearestTip(frame)
    const displayedNearest = nearest.tip === left ? pose.tipLeftRender : pose.tipRightRender
    assert.deepEqual(displayedNearest, nearest.tip)
    assert.equal(distanceToRetina(displayedNearest), nearest.dist)
    assert.ok(Math.abs(length(sub(nearest.wall, displayedNearest)) - nearest.dist) < 1e-10)
  })
}
