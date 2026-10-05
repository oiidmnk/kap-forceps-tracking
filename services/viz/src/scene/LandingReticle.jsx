import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import { DoubleSide, Quaternion, Vector3 } from 'three'
import { nearestTip } from '../geometry.js'
import { distanceStatus } from '../ui.js'

// Landing reticle: projects the nearest forceps tip radially onto the retina
// and marks the predicted contact point with a target ring plus a dashed drop
// line from the tip. This turns the abstract Distance-to-Retina number into a
// spatial "where will it land" cue — the core depth-awareness aid. Colored by
// the nearest tip's safety status (hue is redundant here: the drop line already
// encodes the gap by length).
export default function LandingReticle({ frame }) {
  const info = frame ? nearestTip(frame) : null

  // Orient the flat ring so its face is tangent to the sphere at the contact
  // point (ring normal = outward radial direction at that wall point).
  const quaternion = useMemo(() => {
    if (!info) return null
    const normal = new Vector3(info.wall[0], info.wall[1], info.wall[2]).normalize()
    return new Quaternion().setFromUnitVectors(new Vector3(0, 0, 1), normal)
  }, [info?.wall[0], info?.wall[1], info?.wall[2]])

  if (!info) return null
  const { color } = distanceStatus(info.dist)

  return (
    <group>
      {/* Drop line: tip -> predicted contact point on the retina */}
      <Line
        points={[info.tip, info.wall]}
        color={color}
        lineWidth={1.5}
        dashed
        dashSize={0.3}
        gapSize={0.2}
        transparent
        opacity={0.85}
      />

      {/* Target rings on the retina at the contact point */}
      <group position={info.wall} quaternion={quaternion}>
        <mesh>
          <ringGeometry args={[0.32, 0.46, 40]} />
          <meshBasicMaterial color={color} transparent opacity={0.95} side={DoubleSide} depthWrite={false} />
        </mesh>
        <mesh>
          <ringGeometry args={[0.66, 0.74, 40]} />
          <meshBasicMaterial color={color} transparent opacity={0.4} side={DoubleSide} depthWrite={false} />
        </mesh>
      </group>
    </group>
  )
}
