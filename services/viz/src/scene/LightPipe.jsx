import { useMemo } from 'react'
import DynamicLine from './DynamicLine.jsx'
import * as THREE from 'three'
import { raySphereIntersection, sub, scale, normalize } from '../geometry.js'
import { Rod, SteelMaterial, Cannula, ghostMaterial, OutsideShaft } from './primitives.jsx'

const PIPE_RADIUS_MM = 0.25 // 25G endoilluminator body
const FIBER_RADIUS_MM = 0.19 // exposed optical fiber at the distal end
const FIBER_LENGTH_MM = 0.3
const LIGHT_TINT = '#d9b06a' // light-pipe trocar

// The illumination light pipe (endoilluminator): a 25G steel probe from its
// trocar cannula, ending in a slightly narrower glowing fiber that actually
// emits light into the scene (a real point light — the retina picks up its
// warm falloff), plus (optionally) the beam
// cone toward the illuminated spot. The probe continues a little outside the
// eye. Fed by light_trocar + light_tip.
// `ghost` (tracking lost) fades the pipe and hides the aim guides; the point
// light stays so the light count, and therefore every shader, is unchanged.
export default function LightPipe({ lightTrocar, lightTip, showBeam = false, ghost = false }) {
  if (!lightTrocar || !lightTip) return null
  const axis = normalize(sub(lightTip, lightTrocar))
  const lightedCenter = raySphereIntersection(lightTip, axis)
  const fiberStart = sub(lightTip, scale(axis, FIBER_LENGTH_MM))

  return (
    <group>
      <Cannula position={lightTrocar} tint={LIGHT_TINT} />
      <OutsideShaft entry={lightTrocar} axis={axis} radius={PIPE_RADIUS_MM} ghost={ghost} />

      {/* Probe body */}
      <Rod a={lightTrocar} b={fiberStart} radius={PIPE_RADIUS_MM}>
        <SteelMaterial {...ghostMaterial(ghost)} />
      </Rod>

      {/* Fiber end — emissive, plus a real light so the interior reads lit */}
      <Rod a={fiberStart} b={lightTip} radius={FIBER_RADIUS_MM}>
        <meshStandardMaterial
          color="#fff3d0"
          emissive="#ffcf6e"
          emissiveIntensity={ghost ? 0.3 : 2.2}
          {...ghostMaterial(ghost)}
        />
      </Rod>
      <pointLight position={lightTip} color="#ffd27a" intensity={22} distance={30} decay={2} />

      {/* Aim line to the illuminated center */}
      {!ghost && lightedCenter && (
        <DynamicLine
          points={[lightTip, lightedCenter]}
          color="#ffe9a6"
          lineWidth={1}
          dashed
          dashSize={0.15}
          gapSize={0.08}
          transparent
          opacity={0.6}
        />
      )}

      {!ghost && showBeam && lightedCenter && <BeamCone apex={lightTip} base={lightedCenter} />}

      {!ghost && lightedCenter && (
        <mesh position={lightedCenter}>
          <sphereGeometry args={[0.12, 16, 12]} />
          <meshStandardMaterial color="#fff7cc" emissive="#c68a00" />
        </mesh>
      )}
    </group>
  )
}

// Translucent cone widening from the light tip (apex) to the illuminated spot.
function BeamCone({ apex, base }) {
  const { position, quaternion, height, radius } = useMemo(() => {
    const a = new THREE.Vector3(...apex)
    const b = new THREE.Vector3(...base)
    const h = a.distanceTo(b) || 1e-6
    const dir = a.clone().sub(b).normalize() // tip (radius 0) toward apex
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir)
    const mid = a.clone().add(b).multiplyScalar(0.5)
    return { position: mid, quaternion: q, height: h, radius: Math.max(h * 0.26, 0.2) }
  }, [apex, base])

  return (
    <mesh position={position} quaternion={quaternion}>
      <coneGeometry args={[radius, height, 40, 1, true]} />
      <meshBasicMaterial color="#ffd666" transparent opacity={0.14} depthWrite={false} side={THREE.DoubleSide} />
    </mesh>
  )
}
