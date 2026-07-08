import EyeGlobe from './EyeGlobe.jsx'
import Trocar from './Trocar.jsx'
import Forceps from './Forceps.jsx'
import LightPipe from './LightPipe.jsx'
import Shadow from './Shadow.jsx'
import LandingReticle from './LandingReticle.jsx'
import { EYE_RADIUS_MM } from '../config.js'

// The shared 3D content (lights + eye + forceps + light pipe + shadows), reused
// by both the main overview camera and the proximity-scope close-up camera. No
// camera or controls here — those belong to whichever view renders this.
export default function SurgicalScene({
  frame,
  showRetina = true,
  showShadow = true,
  showBeam = false,
  showReticle = true,
}) {
  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[20, 30, 20]} intensity={1.2} />
      <directionalLight position={[-20, -10, -20]} intensity={0.4} />
      <pointLight position={[0, EYE_RADIUS_MM * 2, EYE_RADIUS_MM * 3]} intensity={0.6} />

      <EyeGlobe showRetina={showRetina} />
      <Trocar position={frame?.trocar} />
      <Forceps frame={frame} />
      <LightPipe lightTrocar={frame?.light_trocar} lightTip={frame?.light_tip} showBeam={showBeam} />
      {showShadow && <Shadow frame={frame} />}
      {showReticle && <LandingReticle frame={frame} />}
    </>
  )
}
