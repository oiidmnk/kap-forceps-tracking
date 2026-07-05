import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import { raySphereIntersection, sub, normalize } from '../geometry.js'

// The illumination light pipe: a line from its trocar to its tip, markers at
// each, and (optionally) the light beam cone toward the point it illuminates on
// the retina. Fed by light_trocar + light_tip from the input.
export default function LightPipe({ lightTrocar, lightTip, showBeam = false }) {
  if (!lightTrocar || !lightTip) return null
  const axis = normalize(sub(lightTip, lightTrocar))
  const lightedCenter = raySphereIntersection(lightTip, axis)

  return (
    <group>
      {/* Light pipe shaft */}
      <Line points={[lightTrocar, lightTip]} color="#f7cc5b" lineWidth={4} />
      {/* Aim line to the illuminated center */}
      {lightedCenter && (
        <Line points={[lightTip, lightedCenter]} color="#ffe9a6" lineWidth={1.5} dashed dashSize={0.15} gapSize={0.08} />
      )}

      {showBeam && lightedCenter && <BeamCone apex={lightTip} base={lightedCenter} />}

      <mesh position={lightTrocar}>
        <sphereGeometry args={[0.35, 20, 14]} />
        <meshStandardMaterial color="#f59e0b" emissive="#5a3300" />
      </mesh>
      <mesh position={lightTip}>
        <sphereGeometry args={[0.42, 20, 14]} />
        <meshStandardMaterial color="#ffe08a" emissive="#8a5e00" />
      </mesh>
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
