import { useMemo } from 'react'
import * as THREE from 'three'

const UP = new THREE.Vector3(0, 1, 0)

// A cylinder/cone rendered between two world-space points [x,y,z] (mm).
// Handles orientation so callers only supply endpoints. Pass radiusTo=0 to
// taper to a sharp point at `to` (e.g. a forceps jaw tip).
export default function Bone({
  from,
  to,
  radius = 0.25,
  radiusFrom,
  radiusTo,
  color = '#cbd5e1',
  metalness = 0.6,
  roughness = 0.35,
}) {
  const rTop = radiusTo ?? radius // `to` end (+Y after orientation)
  const rBottom = radiusFrom ?? radius // `from` end
  const { position, quaternion, height } = useMemo(() => {
    const a = new THREE.Vector3(...from)
    const b = new THREE.Vector3(...to)
    const dir = new THREE.Vector3().subVectors(b, a)
    const h = dir.length()
    const q = new THREE.Quaternion().setFromUnitVectors(
      UP,
      dir.clone().normalize(),
    )
    const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5)
    return { position: mid, quaternion: q, height: h || 1e-6 }
  }, [from, to])

  return (
    <mesh position={position} quaternion={quaternion}>
      <cylinderGeometry args={[rTop, rBottom, height, 16]} />
      <meshStandardMaterial color={color} metalness={metalness} roughness={roughness} />
    </mesh>
  )
}
