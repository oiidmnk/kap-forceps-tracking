import test from 'node:test'
import assert from 'node:assert/strict'
import { lastMeasurementStatus } from '../src/ui.js'

test('ghost status includes last level, downward-rounded clearance and whole seconds', () => {
  assert.equal(lastMeasurementStatus('danger', 0.39, 500, 1999),
    'No measurement · last: Critical proximity 0.3 mm, 1 s ago')
  assert.equal(lastMeasurementStatus('warn', 1.99, 500, 2499),
    'No measurement · last: Near retina 1.9 mm, 1 s ago')
  assert.match(lastMeasurementStatus('safe', 3, 500, 2500), /Clear 3.0 mm, 2 s ago$/)
  assert.match(lastMeasurementStatus('danger', 0.39, 500, 400), /0 s ago$/)
})

test('waiting and non-ghost states do not present historical measurements', () => {
  assert.equal(lastMeasurementStatus(null, null, null, 0), 'No measurement')
  assert.equal(lastMeasurementStatus('danger', 0.3, null, 2000), 'No measurement')
})

import { measurementQualifier, LIVE_ACCURACY_NOTES, SIMULATION_NOTE } from '../src/ui.js'
import { readoutTip, distanceToRetina } from '../src/geometry.js'

test('live accuracy notes follow displayed level and preserve qualifier precedence', () => {
  const base = { source: 'live', hasMeasurement: true, frame: {}, held: false }
  for (const level of ['safe', 'warn', 'danger']) {
    assert.equal(measurementQualifier({ ...base, level }), LIVE_ACCURACY_NOTES[level])
    assert.ok(LIVE_ACCURACY_NOTES[level].length <= 60)
    for (const source of ['mock','debug']) assert.equal(measurementQualifier({ ...base, level, source }), SIMULATION_NOTE)
  }
  assert.match(measurementQualifier({ ...base, hasMeasurement: false }), /unavailable/)
  assert.match(measurementQualifier({ ...base, frame: { confidence: .2 } }), /Low tracking/)
  assert.match(measurementQualifier({ ...base, frame: { calibrated: false, confidence: .2 } }), /Uncalibrated/)
  assert.match(measurementQualifier({ ...base, held: true, frame: { calibrated: false } }), /Signal gap/)
})

test('spatial cue follows published side or midpoint while number stays minimum', () => {
  const f = { tip_left: [0,-10,0], tip_right: [0,-11,0] }
  assert.deepEqual(readoutTip(f,'left').tip, f.tip_left)
  assert.deepEqual(readoutTip(f,'right').tip, f.tip_right)
  assert.deepEqual(readoutTip(f,'tie').tip, [0,-10.5,0])
  for(const side of ['left','right','tie']) assert.equal(readoutTip(f,side).dist, distanceToRetina(f.tip_right))
})
