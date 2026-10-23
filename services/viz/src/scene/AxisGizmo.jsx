import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { GizmoViewport } from '@react-three/drei'

// Orientation gizmo for the main view, rendered as its own small SceneView
// (drei's GizmoHelper assumes it owns the whole canvas). It counter-rotates
// with the main camera so the axes show the world orientation.
export default function AxisGizmo({ mainCamera }) {
  const group = useRef()
  useFrame(() => {
    const cam = mainCamera.current
    if (cam && group.current) group.current.quaternion.copy(cam.quaternion).invert()
  })
  return (
    <group ref={group}>
      <GizmoViewport
        disabled
        axisColors={['#9c797e', '#7d9c8c', '#7c92af']}
        labelColor="#e6edf4"
        axisHeadScale={0.7}
      />
    </group>
  )
}
