import { useMemo } from 'react'
import * as THREE from 'three'

// Solid instrument primitives shared by the forceps, light pipe and trocar
// cannulas. Endpoints are plain [x, y, z] mm arrays from geometry.js.

const UP = new THREE.Vector3(0, 1, 0)

// Brushed-steel instrument body. The sheen comes from the procedural
// RoomEnvironment IBL (see StudioEnvironment) — without it metals go flat.
export function SteelMaterial(props) {
  return <meshStandardMaterial color="#b9c4cf" metalness={0.85} roughness={0.32} {...props} />
}

// Cylinder from a → b. The +Y (top) end of the cylinder maps onto b, so
// radiusTop tapers the b end (e.g. jaw tips). Material goes in as children.
export function Rod({ a, b, radius = 0.1, radiusTop, children, ...meshProps }) {
  const { position, quaternion, length } = useMemo(() => {
    const va = new THREE.Vector3(...a)
    const vb = new THREE.Vector3(...b)
    const dir = vb.clone().sub(va)
    const len = Math.max(dir.length(), 1e-6)
    const q = new THREE.Quaternion().setFromUnitVectors(UP, dir.normalize())
    return { position: va.add(vb).multiplyScalar(0.5), quaternion: q, length: len }
  }, [a[0], a[1], a[2], b[0], b[1], b[2]])

  return (
    <mesh position={position} quaternion={quaternion} {...meshProps}>
      <cylinderGeometry args={[radiusTop ?? radius, radius, length, 24]} />
      {children}
    </mesh>
  )
}

// Trocar marker sitting flush on the sphere surface: a single flange ring lying
// tangent to the surface (no protruding tube), marking the instrument entry
// port. `tint` keeps the per-instrument identity color (steel-blue forceps
// port, amber light port).
const FLANGE_R = 0.62

export function Cannula({ position, tint = '#8fb3d9' }) {
  const quaternion = useMemo(() => {
    const outward = new THREE.Vector3(...position).normalize()
    return new THREE.Quaternion().setFromUnitVectors(UP, outward)
  }, [position[0], position[1], position[2]])

  return (
    <group position={position} quaternion={quaternion}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[FLANGE_R, 0.09, 12, 40]} />
        <meshStandardMaterial color={tint} metalness={0.8} roughness={0.4} />
      </mesh>
    </group>
  )
}
