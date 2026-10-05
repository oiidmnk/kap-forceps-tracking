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
            // A point may be null when the reconstruction degrades to just the
            // anchors (trocars/light) — e.g. calibrating before a matching frame
            // is processed. Carry the last known value forward so the forceps
            // holds its pose while the trocars still update, instead of crashing.
            // Reconstruction space is the microscope image frame: x-right,
            // y-DOWN, z-depth (into the eye). Map to the viz's +Y-up frame:
            // depth -> +Y, and flip the image y so "lower in the image" reads as
            // "lower in the viz" instead of being vertically mirrored.
            const prev = frameRef.current
            const mapPoint = (p, fallback) =>
              Array.isArray(p)
                ? [p[0] * EYE_RADIUS_MM, -p[2] * EYE_RADIUS_MM, p[1] * EYE_RADIUS_MM]
                : fallback ?? null
            const next = {
              t: raw.timestamp,
              tip_left: mapPoint(raw.positions.left_tip_forceps, prev?.tip_left),
              tip_right: mapPoint(raw.positions.right_tip_forceps, prev?.tip_right),
              trocar: mapPoint(raw.positions.trocar_forceps, prev?.trocar),
              light_tip: mapPoint(raw.positions.tip_light, prev?.light_tip),
              light_trocar: mapPoint(raw.positions.trocar_light, prev?.light_trocar),
              confidence: 1.0,
            }
            // Only publish once the forceps pose is known, so downstream
            // components never receive null tips.
            if (next.tip_left && next.tip_right && next.trocar) {
              frameRef.current = next
              setFrame(next)
            }
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
