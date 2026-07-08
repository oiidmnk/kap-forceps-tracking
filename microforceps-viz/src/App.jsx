import { useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { GizmoHelper, GizmoViewport } from '@react-three/drei'
import { useTrackingFeed } from './hooks/useTrackingFeed.js'
import { useDebugPose } from './hooks/useDebugPose.js'
import SurgicalScene from './scene/SurgicalScene.jsx'
import CameraRig from './scene/CameraRig.jsx'
import HUD from './components/HUD.jsx'
import ProximityScope from './components/ProximityScope.jsx'
import DepthCrossSection from './components/DepthCrossSection.jsx'
import ViewPresets from './components/ViewPresets.jsx'
import { EYE_RADIUS_MM } from './config.js'

const SOURCES = ['mock', 'live', 'debug']

export default function App() {
  const [source, setSource] = useState(import.meta.env.VITE_DEFAULT_SOURCE || 'mock')
  const { frame: feedFrame, status } = useTrackingFeed(source)
  const debugFrame = useDebugPose(source === 'debug')
  const frame = source === 'debug' ? debugFrame : feedFrame
  const [toggles, setToggles] = useState({
    showShadow: true,
    showBeam: false,
    showRetina: true,
    showReticle: true,
  })
  const onToggle = (key) => setToggles((t) => ({ ...t, [key]: !t[key] }))
  const cycleSource = () => setSource((s) => SOURCES[(SOURCES.indexOf(s) + 1) % SOURCES.length])
  const [view, setView] = useState('overview')
  // Bumped on every preset selection — even re-selecting the current view — so
  // CameraRig re-snaps and acts as a "reset framing" after free orbit/zoom.
  const [snapNonce, setSnapNonce] = useState(0)
  const selectView = (v) => {
    setView(v)
    setSnapNonce((n) => n + 1)
  }

  return (
    <>
      {/* Main overview (+Y up: instruments enter at top, retina at bottom) */}
      <Canvas
        gl={{ alpha: true }}
        camera={{ position: [EYE_RADIUS_MM * 1.4, EYE_RADIUS_MM * 0.9, EYE_RADIUS_MM * 3.4], fov: 45, near: 0.1, far: 1000 }}
      >
        <SurgicalScene frame={frame} {...toggles} />
        <CameraRig view={view} snapNonce={snapNonce} />
        <GizmoHelper alignment="bottom-right" margin={[70, 70]}>
          <GizmoViewport axisColors={['#e05555', '#55e055', '#5577e0']} labelColor="#fff" />
        </GizmoHelper>
      </Canvas>

      <HUD
        frame={frame}
        status={status}
        source={source}
        onToggleSource={cycleSource}
        toggles={toggles}
        onToggle={onToggle}
      />
      <DepthCrossSection frame={frame} />
      <ProximityScope frame={frame} />
      <ViewPresets view={view} onSelect={selectView} />
    </>
  )
}
