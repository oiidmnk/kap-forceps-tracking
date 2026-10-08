import { useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { GizmoHelper, GizmoViewport } from '@react-three/drei'
import { useTrackingFeed } from './hooks/useTrackingFeed.js'
import { useDebugPose } from './hooks/useDebugPose.js'
import SurgicalScene from './scene/SurgicalScene.jsx'
import CameraRig from './scene/CameraRig.jsx'
import HUD, { ToggleBar } from './components/HUD.jsx'
import DeviceHeader from './components/DeviceHeader.jsx'
import ProximityScope from './components/ProximityScope.jsx'
import DepthCrossSection from './components/DepthCrossSection.jsx'
import ViewPresets from './components/ViewPresets.jsx'
import SidePanel from './components/SidePanel.jsx'
import { EYE_RADIUS_MM } from './config.js'
import './workstation.css'

export default function App() {
  const [source, setSource] = useState(
    import.meta.env.VITE_DEFAULT_SOURCE || 'mock'
  )
  const { frame: feedFrame, status } = useTrackingFeed(source)
  const debugFrame = useDebugPose(source === 'debug')
  const frame = source === 'debug' ? debugFrame : feedFrame
  const [toggles, setToggles] = useState({
    showShadow: false,
    showBeam: false,
    showRetina: true,
    showReticle: true,
    showProximityScope: true,
  })
  const onToggle = (key) => setToggles((t) => ({ ...t, [key]: !t[key] }))
  const [view, setView] = useState('overview')
  const [snapNonce, setSnapNonce] = useState(0)
  const selectView = (v) => {
    setView(v)
    setSnapNonce((n) => n + 1)
  }
  const unavailable = source === 'live' && !frame

  return (
    <div className="workstation">
      <DeviceHeader
        frame={frame}
        status={status}
        source={source}
        onSourceChange={setSource}
      />
      <main className="workspace">
        <SidePanel>
          <HUD frame={frame} status={status} />
          <DepthCrossSection frame={frame} />
          <ProximityScope frame={frame} enabled={toggles.showProximityScope} />
        </SidePanel>
        <section
          className="viewport"
          aria-label="Three-dimensional instrument view"
        >
          <div className="viewport-toolbar">
            <div className="view-identity">
              <span className="eyebrow">Digital twin</span>
              <span>Instrument overview</span>
            </div>
            <ViewPresets view={view} onSelect={selectView} />
            <ToggleBar toggles={toggles} onToggle={onToggle} />
          </div>
          <div className="scene">
            <div className="scene-caption">
              <span className="scene-tag">
                {source === 'live'
                  ? 'Live reconstruction'
                  : source === 'debug'
                    ? 'Manual simulation'
                    : 'Simulation'}
              </span>
              <span>Vitreoretinal anatomy · mm</span>
            </div>
            {unavailable && (
              <div className="scene-empty" role="status">
                <span className="empty-cross">+</span>
                <strong>
                  {status === 'stale'
                    ? 'Tracking data is stale'
                    : status === 'degraded'
                      ? 'Incomplete tracking data'
                      : status === 'error'
                        ? 'Tracking disconnected'
                        : 'Waiting for tracking data'}
                </strong>
                <span>Measurements resume when a complete frame arrives.</span>
              </div>
            )}
            <Canvas
              gl={{ alpha: true }}
              camera={{
                position: [
                  EYE_RADIUS_MM * 1.4,
                  EYE_RADIUS_MM * 0.9,
                  EYE_RADIUS_MM * 3.4,
                ],
                fov: 45,
                near: 0.1,
                far: 1000,
              }}
            >
              <SurgicalScene frame={frame} {...toggles} />
              <CameraRig view={view} snapNonce={snapNonce} />
              <GizmoHelper alignment="bottom-right" margin={[48, 48]}>
                <GizmoViewport
                  axisColors={['#9c797e', '#7d9c8c', '#7c92af']}
                  labelColor="#e6edf4"
                  axisHeadScale={0.7}
                />
              </GizmoHelper>
            </Canvas>
            <div className="scene-legend">
              <span>
                <i className="legend-line" />
                Forceps
              </span>
              <span>
                <i className="legend-dot" />
                Nearest retinal point
              </span>
            </div>
          </div>
          <footer className="viewport-footer">
            <span>
              {source === 'debug'
                ? 'WASD move · Q/E depth · ↑/↓ jaws · ←/→ roll'
                : 'Drag to orbit · Scroll to zoom'}
            </span>
            <button className="text-button" onClick={() => selectView(view)}>
              Reset view <span aria-hidden>↺</span>
            </button>
          </footer>
        </section>
      </main>
      <footer className="system-footer">
        <span>
          Microforceps tracking <span className="footer-separator">/</span>{' '}
          Research visualization
        </span>
        <span>
          {source === 'live' ? 'Live source' : 'Simulated data'}{' '}
          <span className="footer-separator">·</span> Spherical eye model · R{' '}
          {EYE_RADIUS_MM} mm
        </span>
      </footer>
    </div>
  )
}
