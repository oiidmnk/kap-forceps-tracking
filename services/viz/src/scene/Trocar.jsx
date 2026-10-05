import { Html } from '@react-three/drei'
import { Cannula } from './primitives.jsx'

// Fixed spatial anchor marking the forceps trocar (instrument entry / pivot
// point) on the sphere surface — rendered as an actual entry cannula, steel
// with the blue identity tint. Label is deliberately subtle — a small tag, not
// a billboard — so it reads as a quiet reference, not a call-to-action.
export default function Trocar({ position }) {
  if (!position) return null
  return (
    <group>
      <Cannula position={position} tint="#8fb3d9" />
      <Html distanceFactor={50} position={[position[0], position[1] + 1.6, position[2]]} center>
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
