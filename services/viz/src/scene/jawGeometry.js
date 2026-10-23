import * as THREE from 'three'
import { add, sub, scale, normalize, cross, dot, length } from '../geometry.js'

// Centerline of one forceps jaw, as plain [x, y, z] mm points with per-point
// blade width. The jaw lies in the plane spanned by the shaft axis and the
// opening direction:
//   hidden stub inside the tube → base at the rim → spring bend (quadratic
//   Bezier, leaving the tube parallel to the axis) → tracked tip → small hook
//   bent toward the other jaw.
// The tracked tip is the most distal point of the centerline.
export function jawCenterline({
  base,
  tip,
  axis,
  inward,
  widthBase,
  widthTip,
  hook,
  stub = 0.3,
  bendSamples = 18,
  arcSamples = 6,
}) {
  const reach = length(sub(tip, base))
  // Short spring bend right at the rim; the rest of the jaw runs straight.
  const control = add(base, scale(axis, 0.12 * reach))
  const tangent = normalize(sub(tip, control))
  // Hook = quarter arc of radius r, then a short straight inward lip. The arc
  // radius must exceed the blade half-thickness or the inner face folds.
  const r = Math.min(hook * 0.6, reach * 0.2)
  // In-plane inward direction, ⟂ to the end tangent.
  const side = normalize(sub(inward, scale(tangent, dot(tangent, inward))))
  const arcStart = sub(tip, scale(tangent, r))
  const arcCenter = add(arcStart, scale(side, r))

  const points = [sub(base, scale(axis, stub))]
  const widths = [widthBase]
  for (let i = 0; i <= bendSamples; i++) {
    const t = i / bendSamples
    const a = (1 - t) * (1 - t)
    const b = 2 * (1 - t) * t
    const c = t * t
    points.push(add(add(scale(base, a), scale(control, b)), scale(arcStart, c)))
    widths.push(widthBase + (widthTip - widthBase) * t)
  }
  for (let i = 1; i <= arcSamples; i++) {
    const phi = (i / arcSamples) * (Math.PI / 2)
    points.push(add(arcCenter, add(scale(side, -r * Math.cos(phi)), scale(tangent, r * Math.sin(phi)))))
    widths.push(widthTip)
  }
  // Lip: continues inward and tucks slightly back, like the simulator's hook.
  const lipEnd = add(add(tip, scale(side, hook)), scale(tangent, -0.15 * r))
  points.push(lipEnd)
  widths.push(widthTip * 0.85)
  return { points, widths }
}

// Sweeps a rectangular blade section along the centerline. `widthDir` is the
// jaw-plane normal (blade width); the thin dimension (`thickness`) lies in the
// jaw plane, ⟂ to the local tangent. Flat-shaded sides, capped lip end.
export function sweepBlade({ points, widths }, widthDir, thickness) {
  const n = points.length
  const positions = []
  const normals = []
  const indices = []
  const w = widthDir

  const rings = points.map((p, i) => {
    const prev = points[Math.max(i - 1, 0)]
    const next = points[Math.min(i + 1, n - 1)]
    const t = normalize(sub(next, prev))
    const nrm = normalize(cross(w, t))
    const h = thickness / 2
    const hw = widths[i] / 2
    return {
      nrm,
      c: [
        add(add(p, scale(nrm, h)), scale(w, hw)),
        add(add(p, scale(nrm, -h)), scale(w, hw)),
        add(add(p, scale(nrm, -h)), scale(w, -hw)),
        add(add(p, scale(nrm, h)), scale(w, -hw)),
      ],
    }
  })

  // Four faces: +nrm (c3,c0), +w (c0,c1), -nrm (c1,c2), -w (c2,c3).
  const faces = [
    [3, 0, (r) => r.nrm],
    [0, 1, () => w],
    [1, 2, (r) => scale(r.nrm, -1)],
    [2, 3, () => scale(w, -1)],
  ]
  for (const [ia, ib, faceNormal] of faces) {
    const start = positions.length / 3
    for (const ring of rings) {
      const fn = faceNormal(ring)
      positions.push(...ring.c[ia], ...ring.c[ib])
      normals.push(...fn, ...fn)
    }
    for (let i = 0; i < n - 1; i++) {
      const a = start + 2 * i
      indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2)
    }
  }

  // End cap on the hook lip.
  const last = rings[n - 1]
  const endT = normalize(sub(points[n - 1], points[n - 2]))
  const cap = positions.length / 3
  for (const c of last.c) {
    positions.push(...c)
    normals.push(...endT)
  }
  indices.push(cap, cap + 1, cap + 2, cap, cap + 2, cap + 3)

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
  geometry.setIndex(indices)
  return geometry
}
