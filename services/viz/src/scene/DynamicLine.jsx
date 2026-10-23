import { useEffect, useLayoutEffect, useMemo } from 'react'
import { useThree } from '@react-three/fiber'
import { DynamicDrawUsage, InstancedInterleavedBuffer, InterleavedBufferAttribute } from 'three'
import { Line2 } from 'three/examples/jsm/lines/Line2.js'
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'

// Screen-space polyline for geometry that moves every frame (tracking guides).
// drei's <Line> rebuilds its geometry whenever `points` changes and its cleanup
// disposes the shared material as well, so a per-frame update recompiles the
// line shader each frame. Here the geometry, buffers and material are created
// once per point count and the vertex data is rewritten in place.
export default function DynamicLine({
  points,
  color = '#ffffff',
  lineWidth = 1,
  dashed = false,
  dashSize = 1,
  gapSize = 1,
  opacity = 1,
  transparent = false,
  depthTest = true,
  renderOrder = 0,
}) {
  const segments = Math.max(points.length - 1, 1)
  const size = useThree((s) => s.size)

  const line = useMemo(() => {
    const geometry = new LineGeometry()
    const positions = new InstancedInterleavedBuffer(new Float32Array(6 * segments), 6, 1)
    const distances = new InstancedInterleavedBuffer(new Float32Array(2 * segments), 2, 1)
    positions.setUsage(DynamicDrawUsage)
    distances.setUsage(DynamicDrawUsage)
    geometry.setAttribute('instanceStart', new InterleavedBufferAttribute(positions, 3, 0))
    geometry.setAttribute('instanceEnd', new InterleavedBufferAttribute(positions, 3, 3))
    geometry.setAttribute('instanceDistanceStart', new InterleavedBufferAttribute(distances, 1, 0))
    geometry.setAttribute('instanceDistanceEnd', new InterleavedBufferAttribute(distances, 1, 1))
    const object = new Line2(geometry, new LineMaterial())
    // Bounds would need recomputing every frame; these guides are always small
    // and in view, so skip culling instead.
    object.frustumCulled = false
    return { object, positions, distances }
  }, [segments])

  useEffect(
    () => () => {
      line.object.geometry.dispose()
      line.object.material.dispose()
    },
    [line]
  )

  useLayoutEffect(() => {
    const pos = line.positions.array
    const dist = line.distances.array
    let travelled = 0
    for (let i = 0; i < segments; i++) {
      const a = points[i]
      const b = points[Math.min(i + 1, points.length - 1)]
      pos[6 * i] = a[0]
      pos[6 * i + 1] = a[1]
      pos[6 * i + 2] = a[2]
      pos[6 * i + 3] = b[0]
      pos[6 * i + 4] = b[1]
      pos[6 * i + 5] = b[2]
      dist[2 * i] = travelled
      travelled += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2])
      dist[2 * i + 1] = travelled
    }
    line.positions.needsUpdate = true
    line.distances.needsUpdate = true
  })

  useLayoutEffect(() => {
    const m = line.object.material
    m.color.set(color)
    m.linewidth = lineWidth
    m.dashed = dashed
    m.dashSize = dashSize
    m.gapSize = gapSize
    m.opacity = opacity
    m.transparent = transparent
    m.depthTest = depthTest
    m.resolution.set(size.width, size.height)
  }, [line, color, lineWidth, dashed, dashSize, gapSize, opacity, transparent, depthTest, size.width, size.height])

  return <primitive object={line.object} renderOrder={renderOrder} />
}
