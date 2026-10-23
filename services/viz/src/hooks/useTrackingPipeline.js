import { useEffect, useLayoutEffect, useState } from 'react'
import { createTrackingPipeline } from '../feed/pipeline.js'
import { useTrackingFeed } from './useTrackingFeed.js'
import { useDebugPose } from './useDebugPose.js'

// Owns the tracking pipeline for the selected source. Returns a stable object
// whose `scene` and `readout` stores components subscribe to with useStore —
// tracking updates never re-render the app shell.
export function useTrackingPipeline(source) {
  const [pipeline] = useState(createTrackingPipeline)

  // Before paint and before the new source connects: never show the previous
  // source's frame under the new source's label.
  useLayoutEffect(() => {
    pipeline.reset(source, performance.now())
  }, [pipeline, source])

  useTrackingFeed(source === 'debug' ? null : source, pipeline.publish)
  useDebugPose(source === 'debug', pipeline.publish)

  useEffect(() => {
    let raf
    const loop = () => {
      pipeline.tick(performance.now())
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [pipeline])

  return pipeline
}
