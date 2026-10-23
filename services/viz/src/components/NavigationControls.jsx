import { useEffect, useRef, useState } from 'react'
import { typing } from './ViewPresets.jsx'
import { useSceneShield } from '../hooks/useSceneShield.js'

const pivots = [
  { key: 'globe', label: 'Globe', title: 'Orbit around the globe center (G)' },
  { key: 'instrument', label: 'Instrument', title: 'Follow and orbit around the jaws (F)' },
  { key: 'point', label: 'Point', title: 'Click, then click the retina or instrument to orbit around that spot (P)' },
]

const SHORTCUTS = [
  ['Drag', 'Orbit'],
  ['Right-drag', 'Pan'],
  ['Scroll', 'Zoom toward cursor'],
  ['Double-click', 'Orbit around that spot'],
  ['1 – 4', 'View presets'],
  ['F / G / P', 'Orbit instrument / globe / point'],
  ['R', 'Reset view'],
  ['Esc', 'Cancel point picking'],
]
const DEBUG_SHORTCUTS = [
  ['W A S D', 'Move instrument'],
  ['Q / E', 'Insertion depth'],
  ['↑ / ↓', 'Open / close jaws'],
  ['← / →', 'Roll'],
]

// Floating navigation cluster in the 3D view (map-style): orbit pivot, zoom,
// reset and a shortcut reference. `navApi` is filled in by CameraRig.
export default function NavigationControls({ pivot, onPivot, picking, onPick, onReset, navApi, debug }) {
  const [help, setHelp] = useState(false)
  const root = useRef(null)
  useSceneShield(root)

  useEffect(() => {
    const onKey = (e) => {
      if (e.code === 'Escape') {
        onPick(false)
        setHelp(false)
        return
      }
      if (typing(e)) return
      if (e.code === 'KeyF') (onPick(false), onPivot('instrument'))
      else if (e.code === 'KeyG') (onPick(false), onPivot('globe'))
      else if (e.code === 'KeyP') onPick(true)
      else if (e.code === 'KeyR') onReset()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onPivot, onPick, onReset])

  useEffect(() => {
    if (!help) return
    const outside = (e) => {
      if (!root.current?.contains(e.target)) setHelp(false)
    }
    document.addEventListener('pointerdown', outside)
    return () => document.removeEventListener('pointerdown', outside)
  }, [help])

  return (
    // data-scene-ui + shield: clicks here never orbit or pick.
    <div className="nav-cluster" ref={root} data-scene-ui>
      {help && (
        <div className="nav-help" role="dialog" aria-label="Navigation shortcuts">
          <h3>Navigation</h3>
          <dl>
            {[...SHORTCUTS, ...(debug ? DEBUG_SHORTCUTS : [])].map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
      <div className="nav-orbit" role="group" aria-labelledby="nav-orbit-label">
        <span id="nav-orbit-label">Orbit around</span>
        {pivots.map((p) => {
          const isPoint = p.key === 'point'
          const pressed = isPoint ? pivot === 'point' || picking : pivot === p.key && !picking
          return (
            <button
              key={p.key}
              aria-pressed={pressed}
              className={isPoint && picking ? 'picking' : undefined}
              onClick={() => (isPoint ? onPick(!picking) : (onPick(false), onPivot(p.key)))}
              title={p.title}
            >
              {p.label}
            </button>
          )
        })}
      </div>
      <div className="nav-buttons">
        <button onClick={() => navApi.current?.zoom(1)} title="Zoom in" aria-label="Zoom in">
          +
        </button>
        <button onClick={() => navApi.current?.zoom(-1)} title="Zoom out" aria-label="Zoom out">
          −
        </button>
        <button onClick={onReset} title="Reset view (R)" aria-label="Reset view">
          ↺
        </button>
        <button
          onClick={() => setHelp(!help)}
          aria-expanded={help}
          title="Navigation shortcuts"
          aria-label="Navigation shortcuts"
        >
          ?
        </button>
      </div>
    </div>
  )
}
