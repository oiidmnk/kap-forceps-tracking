import { useEffect, useRef, useState } from 'react'
import { WS_URL, EYE_RADIUS_MM } from '../config.js'
import { sampleFrame } from '../feed/mockGenerator.js'

// Subscribes to the tracking data source and returns the latest frame plus
// connection status. `source` is 'live' (WebSocket), 'mock' (in-browser), or
// 'debug' (keyboard-driven — see useDebugPose, which supplies the frame
// directly; this hook just stays idle so it doesn't also try to connect).
//
// Frame shape: { t, tip_left:[x,y,z], tip_right:[x,y,z], trocar:[x,y,z], confidence }
export function useTrackingFeed(source = 'mock') {
  const [frame, setFrame] = useState(null)
  const [status, setStatus] = useState('idle') // idle | connecting | live | mock | debug | error
  const frameRef = useRef(null)

  useEffect(() => {
    frameRef.current = null

    if (source === 'mock') {
      setStatus('mock')
      const start = performance.now()
      let raf
      const tick = () => {
        const t = (performance.now() - start) / 1000
        setFrame(sampleFrame(t))
        raf = requestAnimationFrame(tick)
      }
      raf = requestAnimationFrame(tick)
      return () => cancelAnimationFrame(raf)
    }

    if (source === 'debug') {
      setStatus('debug')
      return
    }

    // Live WebSocket feed.
    setStatus('connecting')
    let ws
    let closed = false
    try {
      ws = new WebSocket(WS_URL)
      ws.onopen = () => !closed && setStatus('live')
      ws.onmessage = (ev) => {
        try {
          const raw = JSON.parse(ev.data)
          if (raw.positions) {
            const mapPoint = (p) => [
              p[0] * EYE_RADIUS_MM,
              -p[2] * EYE_RADIUS_MM,
              -p[1] * EYE_RADIUS_MM
            ]
            setFrame({
              t: raw.timestamp,
              tip_left: mapPoint(raw.positions.left_tip_forceps),
              tip_right: mapPoint(raw.positions.right_tip_forceps),
              trocar: mapPoint(raw.positions.trocar_forceps),
              light_tip: mapPoint(raw.positions.tip_light),
              light_trocar: mapPoint(raw.positions.trocar_light),
              confidence: 1.0,
            })
          }
        } catch {
          /* ignore malformed frame */
        }
      }
      ws.onerror = () => !closed && setStatus('error')
      ws.onclose = () => !closed && setStatus('error')
    } catch {
      setStatus('error')
    }

    return () => {
      closed = true
      ws && ws.close()
    }
  }, [source])

  return { frame, status }
}
