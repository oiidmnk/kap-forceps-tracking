import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import { raySphereIntersection, sub, normalize } from '../geometry.js'
import { Rod, SteelMaterial, Cannula } from './primitives.jsx'

const PIPE_RADIUS_MM = 0.25 // 25G endoilluminator body

// The illumination light pipe: a steel shaft from its trocar cannula to a
// glowing fiber tip that actually emits light into the scene (a real point
// light — the retina picks up its warm falloff), plus (optionally) the beam
// cone toward the illuminated spot. Fed by light_trocar + light_tip.
export default function LightPipe({ lightTrocar, lightTip, showBeam = false }) {
  if (!lightTrocar || !lightTip) return null
  const axis = normalize(sub(lightTip, lightTrocar))
  const lightedCenter = raySphereIntersection(lightTip, axis)

  return (
    <group>
      <Cannula position={lightTrocar} tint="#d9b06a" />

      {/* Pipe body */}
      <Rod a={lightTrocar} b={lightTip} radius={PIPE_RADIUS_MM}>
        <SteelMaterial />
      </Rod>

      {/* Fiber tip — emissive, plus a real light so the interior reads lit */}
      <mesh position={lightTip}>
        <sphereGeometry args={[PIPE_RADIUS_MM * 1.1, 16, 12]} />
        <meshStandardMaterial color="#fff3d0" emissive="#ffcf6e" emissiveIntensity={2.2} />
      </mesh>
      <pointLight position={lightTip} color="#ffd27a" intensity={22} distance={30} decay={2} />

      {/* Aim line to the illuminated center */}
      {lightedCenter && (
        <Line
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

      {showBeam && lightedCenter && <BeamCone apex={lightTip} base={lightedCenter} />}

      {lightedCenter && (
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
