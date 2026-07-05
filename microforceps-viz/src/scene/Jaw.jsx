import { useMemo } from 'react'
import * as THREE from 'three'

// A single forceps jaw: a tapered blade from `from` (hinge) to `to` (tip).
// Rendered flat — thin perpendicular to the opening plane — like a real
// microforceps jaw, rather than a round cone. `openingDir` is the sideways
// direction in which the jaws part; the blade lies in the plane spanned by the
// jaw axis and that direction.
export default function Jaw({
  from,
  to,
  openingDir,
  radiusFrom = 0.2,
  radiusTo = 0.05,
  flatten = 0.35,
  color = '#e2e7ef',
}) {
  const { position, quaternion, height } = useMemo(() => {
    const a = new THREE.Vector3(...from)
    const b = new THREE.Vector3(...to)
    const yAxis = new THREE.Vector3().subVectors(b, a)
    const h = yAxis.length() || 1e-6
    yAxis.normalize()

    const op = new THREE.Vector3(...openingDir)
    // Thin axis = normal to the opening plane (jaw axis × opening direction).
    let xAxis = new THREE.Vector3().crossVectors(yAxis, op)
    if (xAxis.lengthSq() < 1e-9) xAxis.set(1, 0, 0) // jaws perfectly closed
    xAxis.normalize()
    const zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis).normalize()

    const m = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis)
    const q = new THREE.Quaternion().setFromRotationMatrix(m)
    const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5)
    return { position: mid, quaternion: q, height: h }
  }, [from, to, openingDir])

  return (
    <mesh position={position} quaternion={quaternion} scale={[flatten, 1, 1]}>
      <cylinderGeometry args={[radiusTo, radiusFrom, height, 16]} />
      <meshStandardMaterial color={color} metalness={0.25} roughness={0.45} />
    </mesh>
  )
}
