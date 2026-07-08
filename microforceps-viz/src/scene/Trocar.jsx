import { Html } from '@react-three/drei'

// Fixed spatial anchor marking the trocar (instrument entry / pivot point)
// on the sphere surface. Label is deliberately subtle — a small tag, not a
// billboard — so it reads as a quiet reference, not a call-to-action.
export default function Trocar({ position }) {
  if (!position) return null
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[0.35, 20, 20]} />
        <meshStandardMaterial
          color="#60a5fa"
          emissive="#1e3a8a"
          emissiveIntensity={0.4}
        />
      </mesh>
      <Html distanceFactor={50} position={[0, 0.6, 0]} center>
        <div style={labelStyle}>TCR</div>
      </Html>
    </group>
  )
}

const labelStyle = {
  color: 'rgba(147,197,253,0.55)',
  font: '500 8px -apple-system, sans-serif',
  letterSpacing: '0.12em',
  whiteSpace: 'nowrap',
  pointerEvents: 'none',
  textShadow: '0 0 3px rgba(0,0,0,0.8)',
}
