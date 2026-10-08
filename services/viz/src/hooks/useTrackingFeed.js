import { useEffect, useState } from 'react'
import { WS_URL } from '../config.js'
import { sampleFrame } from '../feed/mockGenerator.js'
import { parseTrackingFrame, trackingIsStale } from '../feed/trackingState.js'

export function useTrackingFeed(source = 'mock') {
  const [state, setState] = useState({ source, frame: null, status: 'idle' })
  useEffect(() => {
    const publish = (frame, status) => setState({ source, frame, status })
    publish(null, source === 'live' ? 'connecting' : source)
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
    if (source === 'debug') return

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
  }, [source])
  // Suppress the prior source synchronously, before effect cleanup runs.
  return state.source === source
    ? state
    : { frame: null, status: source === 'live' ? 'connecting' : source }
}
