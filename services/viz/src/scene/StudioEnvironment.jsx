import { useEffect } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'

// Procedural image-based lighting (three's RoomEnvironment) so the metallic
// instrument bodies pick up studio-like reflections. Generated on the GPU —
// no HDR fetch, so the demo stays fully offline. The map is generated once per
// renderer and shared by every view's scene.
const shared = new WeakMap()

function acquire(gl) {
  let entry = shared.get(gl)
  if (!entry) {
    const pmrem = new THREE.PMREMGenerator(gl)
    const room = new RoomEnvironment()
    entry = { texture: pmrem.fromScene(room, 0.04).texture, users: 0 }
    room.dispose?.()
    pmrem.dispose()
    shared.set(gl, entry)
  }
  entry.users++
  return entry.texture
}

function release(gl) {
  const entry = shared.get(gl)
  if (entry && --entry.users === 0) {
    entry.texture.dispose()
    shared.delete(gl)
  }
}

export default function StudioEnvironment({ intensity = 0.45 }) {
  const gl = useThree((s) => s.gl)
  const scene = useThree((s) => s.scene)

  useEffect(() => {
    scene.environment = acquire(gl)
    return () => {
      scene.environment = null
      release(gl)
    }
  }, [gl, scene])

  useEffect(() => {
    scene.environmentIntensity = intensity
  }, [scene, intensity])

  return null
}
