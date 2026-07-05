import { Html } from '@react-three/drei'

// Fixed spatial anchor marking the trocar (instrument entry / pivot point)
// on the sphere surface.
export default function Trocar({ position }) {
  if (!position) return null
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[0.4, 20, 20]} />
        <meshStandardMaterial
          color="#60a5fa"
          emissive="#1e3a8a"
          emissiveIntensity={0.5}
        />
      </mesh>
      <Html distanceFactor={40} position={[0, 0.9, 0]} center>
        <div style={labelStyle}>FORCEPS TROCAR</div>
      </Html>
    </group>
  )
}

const labelStyle = {
  color: '#93c5fd',
  font: '600 11px -apple-system, sans-serif',
  letterSpacing: '0.08em',
  whiteSpace: 'nowrap',
  pointerEvents: 'none',
  textShadow: '0 0 4px #000',
}
