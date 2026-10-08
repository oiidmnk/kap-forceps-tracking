import { useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import Forceps from '../scene/Forceps.jsx'
import LandingReticle from '../scene/LandingReticle.jsx'
import ProximityWall from '../scene/ProximityWall.jsx'
import StudioEnvironment from '../scene/StudioEnvironment.jsx'
import ScopeCamera, {
  ELEV_DEFAULT,
  ELEV_MIN,
  ELEV_MAX,
} from '../scene/ScopeCamera.jsx'
import { nearestTip } from '../geometry.js'
import { distanceStatus } from '../ui.js'
import { DIST_SCOPE_SHOW_MM } from '../config.js'

const DRAG_SENSITIVITY = 0.006
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

// Keep the scope canvas and its space stable across the proximity threshold.
// Only its scene is active within range; drag adjusts the close-up angle.
export default function ProximityScope({ frame, enabled = true }) {
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
    dragState.current.elevation = clamp(
      dragState.current.elevation + dy * DRAG_SENSITIVITY,
      ELEV_MIN,
      ELEV_MAX
    )
  }
  const onPointerUp = (e) => {
    pointer.current = null
    e.currentTarget.releasePointerCapture(e.pointerId)
  }

  const info = frame ? nearestTip(frame) : null
  const inRange = !!info && info.dist < DIST_SCOPE_SHOW_MM
  const active = enabled && inRange
  const status = info ? distanceStatus(info.dist) : null
  return (
    <section className="scope-panel" aria-labelledby="scope-title">
      <div className="section-heading">
        <h2 id="scope-title">Proximity scope</h2>
        <span
          className="scope-readout"
          style={{ color: active ? status.color : undefined }}
        >
          {active ? `${info.dist.toFixed(2)} mm` : 'Standby'}
        </span>
      </div>
      <div
        className="scope-stage"
        onPointerDown={active ? onPointerDown : undefined}
        onPointerMove={active ? onPointerMove : undefined}
        onPointerUp={onPointerUp}
        onLostPointerCapture={() => {
          pointer.current = null
        }}
        onPointerCancel={() => {
          pointer.current = null
        }}
      >
        <Canvas
          gl={{ alpha: true }}
          camera={{ fov: 42, near: 0.01, far: 100 }}
          frameloop={active ? 'always' : 'demand'}
        >
          <StudioEnvironment intensity={0.35} />
          <ambientLight intensity={0.4} />
          <directionalLight position={[4, 6, 4]} intensity={1.1} />
          <directionalLight position={[-4, -2, -3]} intensity={0.4} />
          {active && (
            <>
              <ProximityWall frame={frame} />
              <Forceps frame={frame} />
              <LandingReticle frame={frame} />
              <ScopeCamera frame={frame} dragState={dragState} />
            </>
          )}
        </Canvas>
        {!active && (
          <div className="scope-empty">
            <span className="scope-symbol" aria-hidden>
              ◎
            </span>
            <strong>
              {!enabled
                ? 'Proximity scope hidden'
                : !frame
                  ? 'Waiting for tracking'
                  : 'Outside proximity range'}
            </strong>
            <span>
              {!enabled
                ? 'Enable in Display settings'
                : `Close-up available within ${DIST_SCOPE_SHOW_MM} mm`}
            </span>
          </div>
        )}
      </div>
      <p className="panel-note">
        {active
          ? 'Nearest tip to retinal surface · drag to rotate'
          : 'Fixed close-up · retinal clearance'}
      </p>
    </section>
  )
}
