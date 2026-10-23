import EyeGlobe from './EyeGlobe.jsx'
import Trocar from './Trocar.jsx'
import Forceps from './Forceps.jsx'
import LightPipe from './LightPipe.jsx'
import Shadow from './Shadow.jsx'
import LandingReticle from './LandingReticle.jsx'
import StudioEnvironment from './StudioEnvironment.jsx'
import { useStore } from '../feed/store.js'
import { ShadowLight, RetinaShadow } from './InstrumentShadow.jsx'
import { GlobeCutaway } from './Cutaway.jsx'
import DistanceLabel from './DistanceLabel.jsx'

// The main overview's 3D content (lights + eye + forceps + light pipe +
// shadows). Subscribes to the pipeline's scene store itself, so tracking
// updates re-render only this subtree. After tracking is lost the last pose is
// shown as a ghost, without the measurement guides (shadows, reticle).
// Ghost changes remount the instruments so their materials switch cleanly
// between opaque and transparent shader variants. `pickRef` marks the
// instrument for double-click focus.
export default function SurgicalScene({
  sceneStore,
  readout,
  labelPortal,
  pickRef,
  showRetina = true,
  showShadow = true,
  showInstrumentShadow = true,
  showBeam = false,
  showReticle = true,
  cutaway = true,
}) {
  const { frame, ghost, levels } = useStore(sceneStore)
  const variant = ghost ? 'ghost' : 'live'
  return (
    <>
      {/* IBL carries the metallic sheen; directionals just shape the forms.
          The warm interior fill comes from the light pipe's own point light. */}
      <StudioEnvironment intensity={0.45} />
      <ambientLight intensity={0.25} />
      <directionalLight position={[20, 30, 20]} intensity={1.0} />
      <directionalLight position={[-20, -10, -20]} intensity={0.3} />

      <GlobeCutaway />
      <EyeGlobe showRetina={showRetina} cutaway={cutaway} />
      <Trocar position={frame?.trocar} labelPortal={labelPortal} />
      <group ref={pickRef}>
        <Forceps key={`forceps-${variant}`} frame={frame} levels={levels} ghost={ghost} />
      </group>
      {showInstrumentShadow && !ghost && (
        <>
          <ShadowLight frame={frame} angle={0.75} />
          <RetinaShadow />
        </>
      )}
      <LightPipe
        key={`light-${variant}`}
        lightTrocar={frame?.light_trocar}
        lightTip={frame?.light_tip}
        showBeam={showBeam}
        ghost={ghost}
      />
      {showShadow && !ghost && <Shadow frame={frame} />}
      {showReticle && !ghost && <LandingReticle frame={frame} level={levels?.min} />}
      {showReticle && !ghost && (
        <DistanceLabel frame={frame} readout={readout} portal={labelPortal} />
      )}
    </>
  )
}
