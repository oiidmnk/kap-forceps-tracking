import { useCallback, useEffect, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import * as THREE from 'three'
import { useTrackingPipeline } from './hooks/useTrackingPipeline.js'
import { useStore } from './feed/store.js'
import { useSceneShield } from './hooks/useSceneShield.js'
import SurgicalScene from './scene/SurgicalScene.jsx'
import CameraRig from './scene/CameraRig.jsx'
import SceneView, { ViewClear } from './scene/SceneView.jsx'
import AxisGizmo from './scene/AxisGizmo.jsx'
import { ELEV_DEFAULT } from './scene/ScopeCamera.jsx'
import HUD, { ToggleBar } from './components/HUD.jsx'
import DeviceHeader from './components/DeviceHeader.jsx'
import ProximityScope, { ProximityScopeView } from './components/ProximityScope.jsx'
import NavigationControls from './components/NavigationControls.jsx'
import SceneAlarm from './components/SceneAlarm.jsx'
import ApproachTrend from './components/ApproachTrend.jsx'
import InstrumentStatus from './components/InstrumentStatus.jsx'
import ViewPresets from './components/ViewPresets.jsx'
import SidePanel from './components/SidePanel.jsx'
import { EYE_RADIUS_MM } from './config.js'
import './workstation.css'

// Layout shell. Tracking data flows through the pipeline's stores, so this
// component only re-renders on user interaction, never per tracking frame.
// All 3D views share one WebGL context: a fixed transparent canvas renders
// each view into its DOM placeholder (see SceneView).
export default function App() {
  const [source, setSource] = useState(
    import.meta.env.VITE_DEFAULT_SOURCE || 'mock'
  )
  const pipeline = useTrackingPipeline(source)
  const [toggles, setToggles] = useState({
    showShadow: false,
    showInstrumentShadow: true,
    cutaway: true,
    showBeam: false,
    showRetina: true,
    showReticle: true,
    showProximityScope: true,
  })
  const onToggle = (key) => setToggles((t) => ({ ...t, [key]: !t[key] }))
  const [view, setView] = useState('overview')
  const [snapNonce, setSnapNonce] = useState(0)
  // Orbit pivot: 'globe' center, followed 'instrument', or a double-clicked 'point'.
  const [pivot, setPivot] = useState('globe')
  // Point picking: the next click in the main view sets the pivot.
  const [picking, setPicking] = useState(false)
  const endPicking = useCallback(() => setPicking(false), [])
  const selectView = useCallback((v) => {
    setView(v)
    setPivot((p) => (p === 'point' ? 'globe' : p))
    setSnapNonce((n) => n + 1)
  }, [])
  const resetView = useCallback(() => {
    setPivot((p) => (p === 'point' ? 'globe' : p))
    setSnapNonce((n) => n + 1)
  }, [])

  const mainStage = useRef(null)
  const gizmoStage = useRef(null)
  const scopeStage = useRef(null)
  const sidePanel = useRef(null)
  const mainCamera = useRef(null)
  const instrumentPick = useRef(null)
  const navApi = useRef(null)
  const scopeDrag = useRef({ azimuth: 0, elevation: ELEV_DEFAULT })
  const scopeFocus = useRef({ point: new THREE.Vector3(), valid: false })

  return (
    <div className="workstation">
      <DeviceHeader
        readout={pipeline.readout}
        source={source}
        onSourceChange={setSource}
      />
      <main className="workspace">
        <SidePanel ref={sidePanel}>
          <HUD readout={pipeline.readout} />
          <ApproachTrend readout={pipeline.readout} />
          <InstrumentStatus readout={pipeline.readout} />
          <ProximityScope
            readout={pipeline.readout}
            stageRef={scopeStage}
            dragState={scopeDrag}
            enabled={toggles.showProximityScope}
          />
        </SidePanel>
        <section
          className="viewport"
          aria-label="Three-dimensional instrument view"
        >
          <div className="viewport-toolbar">
            <ViewPresets view={view} onSelect={selectView} />
            <ToggleBar toggles={toggles} onToggle={onToggle} />
          </div>
          <div className={picking ? 'scene picking' : 'scene'} ref={mainStage}>
            <div className="scene-caption">
              <span className="scene-tag">
                {source === 'live'
                  ? 'Live reconstruction'
                  : source === 'debug'
                    ? 'Manual simulation'
                    : 'Simulation'}
              </span>
              <span>Spherical eye model · R {EYE_RADIUS_MM} mm</span>
            </div>
            <SceneAlarm readout={pipeline.readout} />
            <SceneStatus pipeline={pipeline} source={source} />
            {picking && (
              <div className="scene-hint" role="status">
                Click the retina or the instrument to orbit around that spot · Esc cancels
              </div>
            )}
            <div className="gizmo-stage" ref={gizmoStage} aria-hidden />
            <NavigationControls
              pivot={pivot}
              onPivot={setPivot}
              picking={picking}
              onPick={setPicking}
              onReset={resetView}
              navApi={navApi}
              debug={source === 'debug'}
            />
            <SceneLegend shadow={toggles.showInstrumentShadow} />
          </div>
        </section>
      </main>

      <Canvas
        className="gl-layer"
        gl={{ alpha: true, antialias: true }}
        shadows="percentage"
        style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 1 }}
      >
        <ViewClear />
        <SceneView
          track={mainStage}
          index={2}
          cameraRef={mainCamera}
          camera={{
            fov: 45,
            near: 0.1,
            far: 1000,
            position: [EYE_RADIUS_MM * 1.4, EYE_RADIUS_MM * 0.9, EYE_RADIUS_MM * 3.4],
          }}
        >
          <SurgicalScene
            sceneStore={pipeline.scene}
            readout={pipeline.readout}
            labelPortal={mainStage}
            pickRef={instrumentPick}
            {...toggles}
          />
          <CameraRig
            view={view}
            snapNonce={snapNonce}
            pivot={pivot}
            onPivot={setPivot}
            domElementRef={mainStage}
            sceneStore={pipeline.scene}
            pickRef={instrumentPick}
            picking={picking}
            onPickDone={endPicking}
            navApi={navApi}
          />
        </SceneView>
        <SceneView
          track={gizmoStage}
          index={3}
          camera={{ fov: 22.6, near: 1, far: 1000, position: [0, 0, 300] }}
        >
          <AxisGizmo mainCamera={mainCamera} />
        </SceneView>
        <ProximityScopeView
          sceneStore={pipeline.scene}
          stageRef={scopeStage}
          clip={sidePanel}
          dragState={scopeDrag}
          focusRef={scopeFocus}
          enabled={toggles.showProximityScope}
          showShadow={toggles.showInstrumentShadow}
          index={4}
        />
      </Canvas>
    </div>
  )
}

// Collapsible key for the 3D view's visual encodings.
function SceneLegend({ shadow }) {
  const [open, setOpen] = useState(true)
  const root = useRef(null)
  useSceneShield(root)
  return (
    <div
      ref={root}
      className={open ? 'scene-legend' : 'scene-legend collapsed'}
      data-scene-ui
    >
      <button
        className="legend-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        Legend <span aria-hidden>{open ? '⌄' : '›'}</span>
      </button>
      {open && (
        <>
          <span>
            <i className="legend-line" />
            Instrument
          </span>
          {shadow && (
            <span>
              <i className="legend-shadow" />
              Instrument shadow
            </span>
          )}
          <span>
            <i className="legend-status" />
            Distance · color = status
          </span>
          <span>
            <i className="legend-ghost" />
            Last known pose
          </span>
        </>
      )}
    </div>
  )
}

const UNAVAILABLE_TEXT = {
  stale: 'Tracking data is stale',
  degraded: 'Incomplete tracking data',
  error: 'Tracking disconnected',
}

const selectSceneState = (s) => (s.ghost ? 'ghost' : s.frame ? 'tracking' : 'empty')
const selectLostAt = (s) => s.lostAt
const selectStatus = (s) => s.status

// Overlay for the main view: a full notice while live tracking has not
// produced a frame, or a compact banner while a lost pose is shown as a ghost.
function SceneStatus({ pipeline, source }) {
  const state = useStore(pipeline.scene, selectSceneState)
  const lostAt = useStore(pipeline.scene, selectLostAt)
  const status = useStore(pipeline.readout, selectStatus)

  if (state === 'ghost')
    return (
      <div className="scene-lost" role="status">
        <strong>{UNAVAILABLE_TEXT[status] || 'Tracking lost'}</strong>
        <span>
          Showing last known pose · <LostFor since={lostAt} />
        </span>
      </div>
    )
  if (source !== 'live' || state !== 'empty') return null
  return (
    <div className="scene-empty" role="status">
      <span className="empty-cross">+</span>
      <strong>{UNAVAILABLE_TEXT[status] || 'Waiting for tracking data'}</strong>
      <span>Measurements resume when a complete frame arrives.</span>
    </div>
  )
}

function LostFor({ since }) {
  const [now, setNow] = useState(() => performance.now())
  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 500)
    return () => clearInterval(id)
  }, [])
  const seconds = Math.max(0, (now - since) / 1000)
  return <>{seconds < 1 ? 'just now' : `${Math.floor(seconds)} s ago`}</>
}
