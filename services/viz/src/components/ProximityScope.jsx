import { useRef } from 'react'
import ProximityWall from '../scene/ProximityWall.jsx'
import Forceps from '../scene/Forceps.jsx'
import LandingReticle from '../scene/LandingReticle.jsx'
import StudioEnvironment from '../scene/StudioEnvironment.jsx'
import SceneView from '../scene/SceneView.jsx'
import { ShadowLight } from '../scene/InstrumentShadow.jsx'
import ScopeCamera, { ELEV_MIN, ELEV_MAX } from '../scene/ScopeCamera.jsx'
import { nearestTip } from '../geometry.js'
import { STATUS } from '../ui.js'
import { useStore } from '../feed/store.js'
import { DIST_SCOPE_SHOW_MM } from '../config.js'

const DRAG_SENSITIVITY = 0.006
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

// The close-up is active while tracking is current and the nearest tip is in
// proximity range (with hysteresis, see levels.js). Panel and 3D view derive
// this from the same tracker, so they switch together.
const scopeActive = (enabled) => (s) => enabled && !!s.levels?.scope && !s.ghost

// Panel: readout (raw measurement, refreshed at the readout rate), the stage
// placeholder the 3D view renders into, and drag-to-rotate handling.
export default function ProximityScope({ readout, stageRef, dragState, enabled = true }) {
  const pointer = useRef(null)
  const { frame, status, levels } = useStore(readout)
  const active = scopeActive(enabled)({ levels })

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
    if (e.currentTarget.hasPointerCapture(e.pointerId))
      e.currentTarget.releasePointerCapture(e.pointerId)
  }

  const info = frame ? nearestTip(frame) : null
  const lost = !frame && ['stale', 'degraded', 'error'].includes(status)
  const color = active && levels ? STATUS[levels.min].color : undefined
  return (
    <section className="scope-panel" aria-labelledby="scope-title">
      <div className="section-heading">
        <h2 id="scope-title">Proximity scope</h2>
        <span className="scope-readout" style={{ color }}>
          {active ? `${info.dist.toFixed(2)} mm` : 'Standby'}
        </span>
      </div>
      <div
        ref={stageRef}
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
        {!active && (
          <div className="scope-empty">
            <span className="scope-symbol" aria-hidden>
              ◎
            </span>
            <strong>
              {!enabled
                ? 'Proximity scope hidden'
                : lost
                  ? 'Tracking lost'
                  : !frame
                    ? 'Waiting for tracking'
                    : 'Outside proximity range'}
            </strong>
            <span>
              {!enabled
                ? 'Enable in Display settings'
                : lost
                  ? 'Close-up resumes with the next complete frame'
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

// 3D close-up, rendered into the panel's stage inside the shared canvas.
// `clip` keeps it inside the scrolling side panel.
export function ProximityScopeView({ sceneStore, stageRef, clip, dragState, focusRef, enabled, showShadow, index }) {
  const active = useStore(sceneStore, scopeActive(enabled))
  return (
    <SceneView
      track={stageRef}
      clip={clip}
      index={index}
      active={active}
      camera={{ fov: 42, near: 0.01, far: 100 }}
    >
      <StudioEnvironment intensity={0.35} />
      <ambientLight intensity={0.4} />
      <directionalLight position={[4, 6, 4]} intensity={1.1} />
      <directionalLight position={[-4, -2, -3]} intensity={0.4} />
      {active && (
        <ScopeContents
          sceneStore={sceneStore}
          dragState={dragState}
          focusRef={focusRef}
          showShadow={showShadow}
        />
      )}
    </SceneView>
  )
}

function ScopeContents({ sceneStore, dragState, focusRef, showShadow }) {
  const { frame, levels } = useStore(sceneStore)
  return (
    <>
      {showShadow && <ShadowLight frame={frame} angle={0.45} />}
      <ProximityWall frame={frame} focusRef={focusRef} showShadow={showShadow && !!frame?.light_tip} />
      <Forceps frame={frame} levels={levels} />
      <LandingReticle frame={frame} level={levels?.min} />
      <ScopeCamera frame={frame} dragState={dragState} focusRef={focusRef} />
    </>
  )
}
