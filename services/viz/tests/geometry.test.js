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

test('tube end sits one jaw length behind the tips, jaws leave at the rim', async () => {
  const { JAW_LENGTH_MM, JAW_EXIT_OFFSET_MM, SHAFT_LUMEN_RADIUS_MM } = await import('../src/config.js')
  const frame = { tip_left: [-0.3, -10, 0], tip_right: [0.3, -10, 0], trocar: [0, 10, 0] }
  const pose = forcepsRenderPose(frame)
  for (const [base, tip] of [[pose.jawBaseLeft, frame.tip_left], [pose.jawBaseRight, frame.tip_right]]) {
    assert.ok(Math.abs(length(sub(tip, base)) - JAW_LENGTH_MM) < 1e-9)
    const offset = length(sub(base, pose.tubeEnd))
    assert.ok(offset <= JAW_EXIT_OFFSET_MM + 1e-12 && offset < SHAFT_LUMEN_RADIUS_MM)
  }
  // Opening direction points from the left to the right jaw, ⟂ to the shaft.
  assert.ok(Math.abs(pose.openingDir[0] - 1) < 1e-9)
  assert.ok(Math.abs(pose.widthDir[1]) < 1e-9)
})

test('closed jaws still get an orthonormal jaw frame', () => {
  const pose = forcepsRenderPose({ tip_left: [1, -11, 0], tip_right: [1, -11, 0], trocar: [0, 10, 2] })
  for (const v of [pose.openingDir, pose.widthDir, pose.tubeEnd]) assert.ok(v.every(Number.isFinite))
  assert.ok(Math.abs(length(pose.openingDir) - 1) < 1e-9)
  assert.ok(Math.abs(length(pose.widthDir) - 1) < 1e-9)
})

test('double-click focus picks the visible retinal wall', async () => {
  const { pickWall } = await import('../src/geometry.js')
  // From outside, looking through the near shell: the far wall is picked.
  assert.deepEqual(pickWall([0, 30, 0], [0, -1, 0]), [0, -12, 0])
  // From inside the globe: the first wall ahead.
  assert.deepEqual(pickWall([0, 5, 0], [0, -1, 0]), [0, -12, 0])
  // Missing the globe picks nothing.
  assert.equal(pickWall([0, 30, 0], [1, 0, 0]), null)
})
