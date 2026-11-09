import { ghostMaterial, Rod } from './primitives.jsx'
import { length, sub } from '../geometry.js'
import { ROD_COLOR, ROD_CONTACT_GLOW, ROD_RADIUS_MM } from '../config.js'

// The free-floating rod of the simulator's rod task: a glossy red capsule
// between its two tracked end points (the cap apexes, so the cylinder is
// shortened by one radius at each end). It casts the instrument shadow like
// the forceps does. Glows while the forceps touches it. Not drawn while the
// stream reports no rod.
export default function RodObject({ frame, ghost = false }) {
  const a = frame?.rod_a
  const b = frame?.rod_b
  if (!a || !b) return null
  const L = length(sub(b, a))
  const r = ROD_RADIUS_MM
  const inset = L > 2 * r ? r / L : 0.5
  const lerp = (u) => [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u]
  const ca = lerp(inset)
  const cb = lerp(1 - inset)
  const material = (
    <meshStandardMaterial
      color={ROD_COLOR}
      emissive={frame.rod_contact ? ROD_CONTACT_GLOW : '#000000'}
      emissiveIntensity={frame.rod_contact ? 0.6 : 0}
      metalness={0.1}
      roughness={0.28}
      {...ghostMaterial(ghost)}
    />
  )
  return (
    <group>
      <Rod a={ca} b={cb} radius={r} castShadow>
        {material}
      </Rod>
      {[ca, cb].map((p, i) => (
        <mesh key={i} position={p} castShadow>
          <sphereGeometry args={[r, 24, 16]} />
          {material}
        </mesh>
      ))}
    </group>
  )
}
