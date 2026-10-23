import { useEffect } from 'react'
import { WS_URL } from '../config.js'
import { sampleFrame } from '../feed/mockGenerator.js'
import { parseTrackingFrame, trackingIsStale } from '../feed/trackingState.js'

// Connects the selected data source (mock generator or live WebSocket) and
// forwards every frame to `onFrame(frame, status, now)`. No React state: the
// tracking pipeline decides which components update and how often.
export function useTrackingFeed(source, onFrame) {
  useEffect(() => {
    const publish = (frame, status) => onFrame(frame, status, performance.now())
    if (source === 'mock') {
      const start = performance.now()
      let raf
      const tick = () => {
        publish(sampleFrame((performance.now() - start) / 1000), 'mock')
        raf = requestAnimationFrame(tick)
      }
      raf = requestAnimationFrame(tick)
      return () => cancelAnimationFrame(raf)
    }
    if (source !== 'live') return

    let ws,
      closed = false,
      lastReceived = null
    const staleTimer = setInterval(() => {
      if (!closed && trackingIsStale(lastReceived, performance.now())) {
        publish(null, 'stale')
        lastReceived = null
      }
    }, 250)
    try {
      ws = new WebSocket(WS_URL)
      ws.onopen = () => {
        if (!closed) publish(null, 'waiting')
      }
      ws.onmessage = (event) => {
        if (closed) return
        let frame = null
        try {
          frame = parseTrackingFrame(JSON.parse(event.data))
        } catch {
          /* malformed data is explicitly unavailable */
        }
        lastReceived = frame ? performance.now() : null
        publish(frame, frame ? 'live' : 'degraded')
      }
      const disconnect = () => {
        if (!closed) {
          lastReceived = null
          publish(null, 'error')
        }
      }
      ws.onerror = disconnect
      ws.onclose = disconnect
    } catch {
      publish(null, 'error')
    }
    return () => {
      closed = true
      clearInterval(staleTimer)
      ws?.close()
    }
  }, [source, onFrame])
}
