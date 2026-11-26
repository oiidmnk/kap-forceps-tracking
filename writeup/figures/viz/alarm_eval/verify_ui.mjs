import { createTrackingPipeline } from '../../../../services/viz/src/feed/pipeline.js'

let input = ''
for await (const chunk of process.stdin) input += chunk
const { tips, times } = JSON.parse(input)
const pipeline = createTrackingPipeline()
pipeline.reset('replay', 0)
const severity = { safe: 0, warn: 1, danger: 2 }
const levels = tips.map((pair, index) => {
  const frame = pair ? { tip_left: pair[0], tip_right: pair[1], t: times[index] } : null
  pipeline.publish(frame, frame ? 'replay' : 'degraded', times[index])
  pipeline.tick(times[index])
  const current = pipeline.readout.get().levels
  return current ? severity[current.min] : -1
})
process.stdout.write(JSON.stringify(levels))
