import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { forcepsRenderPose, length, sub, scale } from '../geometry.js'
import { STATUS } from '../ui.js'
import { SteelMaterial, ghostMaterial, OutsideShaft } from './primitives.jsx'
import Jaw from './Jaw.jsx'
import {
  SHAFT_RADIUS_MM,
  SHAFT_LUMEN_RADIUS_MM,
  TIP_INDICATOR_MM,
} from '../config.js'

const UP = new THREE.Vector3(0, 1, 0)
const RIM_BEVEL_MM = 0.05 // rounded nose radius at the outer edge
const BORE_EDGE_MM = 0.015
const LUMEN_DEPTH_MM = 0.6

// Microforceps modelled on the Haag-Streit simulator instrument: a polished
// constant-diameter tube from the trocar to its cut distal end, where two flat
// jaws emerge from the bore at the rim and spread to the tracked tips. The
// tips themselves are drawn at the measured coordinates; only the tube end is
// inferred (see forcepsRenderPose). Small colored tip markers carry the
// distance state (`levels`, from the shared hysteresis tracker).
// `ghost` draws the last known pose faded and neutral after tracking is lost.
// Tube and jaws cast the instrument shadow (see InstrumentShadow). The shaft
// continues a little outside the eye (OutsideShaft).
const GHOST_TIP = '#8a97a3'

export default function Forceps({ frame, levels, ghost = false }) {
  if (!frame) return null
  const pose = forcepsRenderPose(frame)
  const { tubeEnd, shaftDir, openingDir, widthDir } = pose
  const inwardLeft = openingDir
  const inwardRight = scale(openingDir, -1)

  return (
    <group>
      <Tube end={tubeEnd} axis={shaftDir} shaftLength={length(sub(tubeEnd, frame.trocar))} ghost={ghost} />
      <OutsideShaft
        entry={frame.trocar}
        axis={shaftDir}
        radius={SHAFT_RADIUS_MM}
        ghost={ghost}
        color="#aab4be"
        metalness={0.9}
        roughness={0.22}
      />

      {[
        [pose.jawBaseLeft, pose.tipLeftRender, inwardLeft],
        [pose.jawBaseRight, pose.tipRightRender, inwardRight],
      ].map(([base, tip, inward], i) => {
        const level = levels?.[i ? 'right' : 'left']
        const color = !ghost && level ? STATUS[level].color : GHOST_TIP
        return (
          <group key={i}>
            <Jaw base={base} tip={tip} axis={shaftDir} inward={inward} widthDir={widthDir} ghost={ghost} />
            <mesh position={tip}>
              <sphereGeometry args={[TIP_INDICATOR_MM, 12, 10]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={ghost ? 0 : 1.2}
                {...ghostMaterial(ghost)}
              />
            </mesh>
          </group>
        )
      })}
    </group>
  )
}

// The instrument tube in a local frame with +Y along the shaft and the rim at
// y = 0: open outer wall back to the trocar, a rounded nose and annular rim
// (as on the simulator instrument), and a dark bore so the cut end reads as
// hollow.
function Tube({ end, axis, shaftLength, ghost }) {
  const fade = ghostMaterial(ghost)
  const quaternion = useMemo(
    () => new THREE.Quaternion().setFromUnitVectors(UP, new THREE.Vector3(...axis)),
    [axis[0], axis[1], axis[2]]
  )
  const rim = useMemo(() => {
    const R = SHAFT_RADIUS_MM
    const r = SHAFT_LUMEN_RADIUS_MM
    const b = RIM_BEVEL_MM
    const e = BORE_EDGE_MM
    const profile = [new THREE.Vector2(r, -e), new THREE.Vector2(r + e, 0)]
    for (let i = 0; i <= 8; i++) {
      const phi = (i / 8) * (Math.PI / 2)
      profile.push(new THREE.Vector2(R - b + b * Math.sin(phi), -b + b * Math.cos(phi)))
    }
    return new THREE.LatheGeometry(profile, 48)
  }, [])
  useEffect(() => () => rim.dispose(), [rim])
  const outerLength = Math.max(shaftLength - RIM_BEVEL_MM, 1e-3)

  return (
    <group position={end} quaternion={quaternion}>
      <mesh position={[0, -RIM_BEVEL_MM - outerLength / 2, 0]} castShadow>
        <cylinderGeometry args={[SHAFT_RADIUS_MM, SHAFT_RADIUS_MM, outerLength, 40, 1, true]} />
        <SteelMaterial color="#aab4be" metalness={0.9} roughness={0.22} {...fade} />
      </mesh>
      <mesh geometry={rim} castShadow>
        <SteelMaterial color="#aab4be" metalness={0.9} roughness={0.22} side={THREE.DoubleSide} {...fade} />
      </mesh>
      <mesh position={[0, -BORE_EDGE_MM - LUMEN_DEPTH_MM / 2, 0]}>
        <cylinderGeometry args={[SHAFT_LUMEN_RADIUS_MM, SHAFT_LUMEN_RADIUS_MM, LUMEN_DEPTH_MM, 32, 1, true]} />
        <meshStandardMaterial color="#25292e" metalness={0.5} roughness={0.6} side={THREE.BackSide} {...fade} />
      </mesh>
      <mesh position={[0, -BORE_EDGE_MM - LUMEN_DEPTH_MM, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[SHAFT_LUMEN_RADIUS_MM, 32]} />
        <meshStandardMaterial color="#121417" roughness={0.8} {...fade} />
      </mesh>
    </group>
  )
}
