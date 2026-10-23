import { Html } from '@react-three/drei'
import { Cannula } from './primitives.jsx'

// Fixed spatial anchor marking the forceps trocar (instrument entry / pivot
// point) on the sphere surface — rendered as an actual entry cannula, steel
// with the blue identity tint. Label is deliberately subtle — a small tag, not
// a billboard — so it reads as a quiet reference, not a call-to-action.
export default function Trocar({ position, labelPortal }) {
  if (!position) return null
  return (
    <group>
      <Cannula position={position} tint="#8fb3d9" />
      <Html
        portal={labelPortal}
        zIndexRange={[2, 2]}
        position={[position[0], position[1] + 1.6, position[2]]}
        center
      >
        <div style={labelStyle}>Trocar</div>
      </Html>
    </group>
  )
}

const labelStyle = {
  color: '#b3c6d8',
  font: '500 12px Inter, sans-serif',
  letterSpacing: '0.02em',
  whiteSpace: 'nowrap',
  pointerEvents: 'none',
  textShadow: '0 0 3px rgba(0,0,0,0.8)',
}
