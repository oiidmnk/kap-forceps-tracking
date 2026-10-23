import { useMemo } from 'react'
import * as THREE from 'three'
import { EYE_RADIUS_MM } from '../config.js'
import { midpoint, shadowOf } from '../geometry.js'

// Real shadow of the instrument on the retina, cast from the light-pipe tip —
// the depth cue the simulator footage relies on: the gap between the tips and
// their shadow closes as the instrument approaches the retina.
//
// The light has zero intensity and only drives the shadow map, so enabling the
// shadow never changes the scene's lighting. The canvas uses PCF shadow
// filtering so `shadow-radius` softens the edges. Receivers use ShadowMaterial,
// which draws nothing but the shadow. Casters opt in with `castShadow`
// (Forceps does; the light pipe and globe do not).

const SHADOW_COLOR = '#14070a'
const SHADOW_OPACITY = 0.6
const RECEIVER_RADIUS_MM = EYE_RADIUS_MM - 0.1 // just inside the retina mesh
const RETINA_FROM = (70 * Math.PI) / 180 // polar angle from +Y: retina region

// Shadow-only spot light at the light-pipe tip, aimed at where the jaws'
// shadow falls (a real endoilluminator is a cone light too). One 2D shadow map
// focused on the instrument is much sharper than a point light's cube map,
// which spreads its resolution over every direction. `angle` is the cone
// half-angle: wide in the overview so the shaft's shadow fits, narrower in
// the close-up.
export function ShadowLight({ frame, angle = 0.7 }) {
  const target = useMemo(() => new THREE.Object3D(), [])
  const lightTip = frame?.light_tip
  if (!lightTip) return null
  const jaws = midpoint(frame.tip_left, frame.tip_right)
  const aim = shadowOf(jaws, lightTip) ?? jaws
  return (
    <>
      <primitive object={target} position={aim} />
      <spotLight
        position={lightTip}
        target={target}
        intensity={0}
        angle={angle}
        // `distance` also sets the shadow camera's far plane (and makes three
        // update its projection, which a plain shadow-camera-far would not).
        distance={EYE_RADIUS_MM * 2 + 2}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-near={0.05}
        shadow-bias={-0.0002}
        shadow-radius={3}
      />
    </>
  )
}

// Shadow-only surface over the posterior retina for the main view. Its inner
// side faces the light and the camera. Like the retina it is never cut away.
export function RetinaShadow() {
  return (
    <mesh receiveShadow renderOrder={1}>
      <sphereGeometry
        args={[RECEIVER_RADIUS_MM, 96, 48, 0, Math.PI * 2, RETINA_FROM, Math.PI - RETINA_FROM]}
      />
      <shadowMaterial
        color={SHADOW_COLOR}
        opacity={SHADOW_OPACITY}
        transparent
        depthWrite={false}
        side={THREE.BackSide}
      />
    </mesh>
  )
}

// Shadow-only overlay for a spherical cap (the proximity scope's wall patch).
export function CapShadow({ radius, capAngle }) {
  return (
    <mesh receiveShadow renderOrder={1}>
      <sphereGeometry args={[radius, 64, 32, 0, Math.PI * 2, 0, capAngle]} />
      <shadowMaterial
        color={SHADOW_COLOR}
        opacity={SHADOW_OPACITY}
        transparent
        depthWrite={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}
