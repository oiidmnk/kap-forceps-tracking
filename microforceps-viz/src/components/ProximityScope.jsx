import { useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import Forceps from '../scene/Forceps.jsx'
import ProximityWall from '../scene/ProximityWall.jsx'
import ScopeCamera, { ELEV_DEFAULT, ELEV_MIN, ELEV_MAX } from '../scene/ScopeCamera.jsx'
import { nearestTip } from '../geometry.js'
import { distanceStatus } from '../ui.js'
import { DIST_SCOPE_SHOW_MM } from '../config.js'
import { color, font, radius, tnum, HEADER_H } from '../theme.js'

const DRAG_SENSITIVITY = 0.006
const SCOPE_SIZE_PX = 300
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

// Picture-in-picture "proximity scope": a magnified close-up of the gap
// between the nearest tip and the retina. Unlike the main overview, this is
// its own purpose-built scene (curved retina segment + dimension line) rather
// than the full eye. Only mounts once a tip is actually closing in on the
// retina (DIST_SCOPE_SHOW_MM) — otherwise it's a redundant close-up of empty
// space. Size stays fixed (no zoom-in-on-approach) so the panel itself isn't
// moving around at the same time as the camera; only the border glow/pulse
// still ramps up as the tip gets closer.
//
// The default view angle is mostly-from-the-top (ScopeCamera's ELEV_DEFAULT);
// dragging orbits within that, azimuth around the contact point and elevation
// tilting toward/away from it. Elevation is clamped by ScopeCamera itself, so
// no drag can rotate the camera down past the tangent plane.
export default function ProximityScope({ frame }) {
  const dragState = useRef({ azimuth: 0, elevation: ELEV_DEFAULT })
  const pointer = useRef(null)

  const onPointerDown = (e) => {
    pointer.current = { x: e.clientX, y: e.clientY }
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const onPointerMove = (e) => {
    if (!pointer.current) return
    const dx = e.clientX - pointer.current.x
    const dy = e.clientY - pointer.current.y
    pointer.current = { x: e.clientX, y: e.clientY }
    dragState.current.azimuth -= dx * DRAG_SENSITIVITY
    dragState.current.elevation = clamp(dragState.current.elevation + dy * DRAG_SENSITIVITY, ELEV_MIN, ELEV_MAX)
  }
  const onPointerUp = (e) => {
    pointer.current = null
    e.currentTarget.releasePointerCapture(e.pointerId)
  }

  const info = frame ? nearestTip(frame) : null
  if (!info || info.dist >= DIST_SCOPE_SHOW_MM) return null

  const status = distanceStatus(info.dist)
  // Ambient look stays matte; a glow is reserved as a DANGER alarm only.
  const danger = status?.level === 'danger'

  return (
    <div
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      style={{
        position: 'relative',
        width: '100%',
        flex: 1,
        minHeight: 180,
        borderRadius: radius.md,
        overflow: 'hidden',
        border: `2px solid ${status ? status.color : color.border}`,
        boxShadow: '0 6px 20px rgba(0,0,0,0.55)',
        background: color.surface,
        animation: danger ? 'scopePulse 0.8s ease-in-out infinite' : 'none',
        cursor: 'grab',
        touchAction: 'none',
        pointerEvents: 'auto',
      }}
    >
      <style>{`@keyframes scopePulse { 0%,100%{box-shadow:0 6px 20px rgba(0,0,0,0.55)} 50%{box-shadow:0 0 24px ${status?.color}} }`}</style>

      <div style={header}>
        <span>PROXIMITY SCOPE</span>
        {info && (
          <span style={{ color: status.color, ...tnum }}>
            {info.dist.toFixed(2)} mm
          </span>
        )}
      </div>

      <Canvas gl={{ alpha: true }} camera={{ fov: 42, near: 0.01, far: 100 }}>
        <ambientLight intensity={0.8} />
        <directionalLight position={[4, 6, 4]} intensity={1.1} />
        <directionalLight position={[-4, -2, -3]} intensity={0.4} />

        <ProximityWall frame={frame} />
        <Forceps frame={frame} />
        <ScopeCamera frame={frame} dragState={dragState} />
      </Canvas>

      <div style={hint}>drag to rotate</div>
    </div>
  )
}

const header = {
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  zIndex: 1,
  display: 'flex',
  justifyContent: 'space-between',
  padding: '6px 10px',
  font: `600 10px ${font.sans}`,
  letterSpacing: '0.1em',
  color: color.textDim,
  background: 'linear-gradient(rgba(10,14,22,0.92), rgba(10,14,22,0))',
  pointerEvents: 'none',
}

const hint = {
  position: 'absolute',
  bottom: 6,
  left: 0,
  right: 0,
  textAlign: 'center',
  font: `500 9px ${font.sans}`,
  letterSpacing: '0.06em',
  color: color.textFaint,
  pointerEvents: 'none',
}
