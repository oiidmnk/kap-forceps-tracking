import { useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import { EYE_RADIUS_MM } from '../config.js'
import { CapShadow } from './InstrumentShadow.jsx'

// Reusable scratch objects to avoid per-frame allocations
const _worldUp = new THREE.Vector3(0, 1, 0)
const _worldRight = new THREE.Vector3(1, 0, 0)
const _tangentX = new THREE.Vector3()
const _tangentZ = new THREE.Vector3()
const _outward = new THREE.Vector3()
const _basisMat = new THREE.Matrix4()
const _txA = new THREE.Vector3() // tangent from worldUp ref
const _txB = new THREE.Vector3() // tangent from worldRight ref

// Hermite smoothstep — continuous first derivative at edges
function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

const PATCH_HALF_MM = 6 // approx half-width of the rendered patch, measured along the sphere surface
const GRID_STEP_MM = 2
const WALL_RADIUS_MM = EYE_RADIUS_MM - 0.08 // matches EyeGlobe's retina inset, avoids z-fighting with the main sphere
const CAP_ANGLE = PATCH_HALF_MM / WALL_RADIUS_MM // small-angle arc-length approximation

// Point on the sphere in a "pole = local +Y" frame: theta is the polar angle
// from the pole (0 = pole itself), phi sweeps around it.
function sphericalPoint(theta, phi, radius) {
  return [
    radius * Math.sin(theta) * Math.cos(phi),
    radius * Math.cos(theta),
    radius * Math.sin(theta) * Math.sin(phi),
  ]
}

// Precomputed once: the lat/long ruler lines over the cap, in the unrotated
// pole-aligned frame. Reused every frame — only the group's orientation moves.
const GRID_RADIUS_MM = WALL_RADIUS_MM + 0.01 // lift just off the mesh surface
const LAT_LINES = (() => {
  const lines = []
  const dTheta = GRID_STEP_MM / WALL_RADIUS_MM
  const segments = 48
  for (let theta = dTheta; theta <= CAP_ANGLE + 1e-6; theta += dTheta) {
    const ring = []
    for (let s = 0; s <= segments; s++) {
      ring.push(
        sphericalPoint(theta, (s / segments) * Math.PI * 2, GRID_RADIUS_MM)
      )
    }
    lines.push(ring)
  }
  return lines
})()
const LON_LINES = (() => {
  const lines = []
  const edgeRadius = Math.sin(CAP_ANGLE) * WALL_RADIUS_MM
  const dPhi = GRID_STEP_MM / edgeRadius
  const steps = 16
  for (let phi = 0; phi < Math.PI * 2 - 1e-6; phi += dPhi) {
    const line = []
    for (let s = 0; s <= steps; s++) {
      line.push(sphericalPoint((s / steps) * CAP_ANGLE, phi, GRID_RADIUS_MM))
    }
    lines.push(line)
  }
  return lines
})()

// At proximity-scope zoom the eye's curvature is subtle but real — render an
// actual segment of the 12mm sphere rather than a flat stand-in, so the
// geometry stays anatomically honest. A small polar cap is built once in a
// "pole = local +Y" frame; each frame we just re-orient the whole group so the
// pole lands on the scope focus (`focusRef`, smoothed by ScopeCamera, which
// runs first). Rotation about the sphere's own center preserves every vertex's
// radius, so this is always a true piece of the real sphere — just re-aimed.
// The exact predicted contact point is marked by the LandingReticle.
export default function ProximityWall({ frame, focusRef, showShadow = false }) {
  const groupRef = useRef()

  useFrame(() => {
    const focus = focusRef.current
    if (!focus.valid || !groupRef.current) return
    _outward.copy(focus.point).normalize()

    // Build a stable basis from the focus direction. Blend between two world
    // references to avoid the discontinuous jump that a hard threshold causes
    // when outward.y crosses a cutoff.
    // Smoothstep blend: 0 when |y|<0.85 (use worldUp), 1 when |y|>0.99 (use worldRight)
    const absY = Math.abs(_outward.y)
    const t = absY <= 0.85 ? 0 : absY >= 0.99 ? 1 : smoothstep(0.85, 0.99, absY)

    _txA.crossVectors(_outward, _worldUp)
    const lenA = _txA.length()
    _txB.crossVectors(_outward, _worldRight)
    const lenB = _txB.length()
    if (lenA > 1e-6) _txA.multiplyScalar(1 / lenA)
    if (lenB > 1e-6) _txB.multiplyScalar(1 / lenB)
    _tangentX.lerpVectors(_txA, _txB, t).normalize()

    _tangentZ.crossVectors(_tangentX, _outward).normalize()
    _basisMat.makeBasis(_tangentX, _outward, _tangentZ)
    groupRef.current.quaternion.setFromRotationMatrix(_basisMat)
  })

  if (!frame) return null

  return (
    <group ref={groupRef}>
      <mesh>
        <sphereGeometry
          args={[WALL_RADIUS_MM, 48, 24, 0, Math.PI * 2, 0, CAP_ANGLE]}
        />
        <meshStandardMaterial
          color="#654451"
          emissive="#20151b"
          emissiveIntensity={0.6}
          roughness={0.8}
          side={THREE.DoubleSide}
        />
      </mesh>
      {showShadow && <CapShadow radius={WALL_RADIUS_MM - 0.01} capAngle={CAP_ANGLE} />}
      {LAT_LINES.map((pts, i) => (
        <Line
          key={`lat-${i}`}
          points={pts}
          color="#f0c3cf"
          transparent
          opacity={0.18}
          lineWidth={0.7}
          depthTest={false}
          renderOrder={1}
        />
      ))}
      {LON_LINES.map((pts, i) => (
        <Line
          key={`lon-${i}`}
          points={pts}
          color="#f0c3cf"
          transparent
          opacity={0.18}
          lineWidth={0.7}
          depthTest={false}
          renderOrder={1}
        />
      ))}
    </group>
  )
}
