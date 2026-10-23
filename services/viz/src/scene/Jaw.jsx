import { useEffect, useMemo } from 'react'
import { jawCenterline, sweepBlade } from './jawGeometry.js'
import { ghostMaterial } from './primitives.jsx'
import {
  JAW_THICKNESS_MM,
  JAW_WIDTH_BASE_MM,
  JAW_WIDTH_TIP_MM,
  JAW_HOOK_MM,
} from '../config.js'

// A single forceps jaw: a flat spring-steel blade that leaves the tube rim at
// `base`, bends outward to the tracked `tip` and ends in a small hook pointing
// `inward` (toward the other jaw), as on the simulator instrument. The blade is
// thin in the opening direction and wide along `widthDir`.
export default function Jaw({ base, tip, axis, inward, widthDir, ghost = false, color = '#cfd6de' }) {
  const geometry = useMemo(
    () =>
      sweepBlade(
        jawCenterline({
          base,
          tip,
          axis,
          inward,
          widthBase: JAW_WIDTH_BASE_MM,
          widthTip: JAW_WIDTH_TIP_MM,
          hook: JAW_HOOK_MM,
        }),
        widthDir,
        JAW_THICKNESS_MM
      ),
    [base, tip, axis, inward, widthDir]
  )
  useEffect(() => () => geometry.dispose(), [geometry])

  return (
    <mesh geometry={geometry} castShadow>
      <meshStandardMaterial color={color} metalness={0.85} roughness={0.28} {...ghostMaterial(ghost)} />
    </mesh>
  )
}
