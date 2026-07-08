import { useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import Forceps from '../scene/Forceps.jsx'
import ProximityWall from '../scene/ProximityWall.jsx'
import ScopeCamera, { ELEV_DEFAULT, ELEV_MIN, ELEV_MAX } from '../scene/ScopeCamera.jsx'
import { nearestTip } from '../geometry.js'
import { distanceStatus } from '../ui.js'
import { DIST_SAFE_MM, DIST_SCOPE_SHOW_MM } from '../config.js'

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
  const active = info.dist < DIST_SAFE_MM // glow/pulse harder when getting close (size no longer changes)

  return (
    <div
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      style={{
        position: 'absolute',
        right: 20,
        top: 60,
        width: SCOPE_SIZE_PX,
        height: SCOPE_SIZE_PX,
        borderRadius: 12,
        overflow: 'hidden',
        border: `2px solid ${status ? status.color : 'rgba(255,255,255,0.15)'}`,
        boxShadow: active ? `0 0 18px ${status.color}` : '0 4px 18px rgba(0,0,0,0.5)',
        background: 'rgba(6,9,16,0.85)',
        transition: 'box-shadow 0.25s',
        animation: status?.level === 'danger' ? 'scopePulse 0.8s ease-in-out infinite' : 'none',
        cursor: 'grab',
        touchAction: 'none',
      }}
    >
      <style>{`@keyframes scopePulse { 0%,100%{box-shadow:0 0 8px ${status?.color}} 50%{box-shadow:0 0 26px ${status?.color}} }`}</style>

      <div style={header}>
        <span>PROXIMITY SCOPE</span>
        {info && (
          <span style={{ color: status.color, fontVariantNumeric: 'tabular-nums' }}>
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
  font: '600 10px sans-serif',
  letterSpacing: '0.1em',
  color: '#9fb0c4',
  background: 'linear-gradient(rgba(6,9,16,0.9), rgba(6,9,16,0))',
  pointerEvents: 'none',
}

const hint = {
  position: 'absolute',
  bottom: 6,
  left: 0,
  right: 0,
  textAlign: 'center',
  font: '500 9px sans-serif',
  letterSpacing: '0.06em',
  color: 'rgba(159,176,196,0.6)',
  pointerEvents: 'none',
}
