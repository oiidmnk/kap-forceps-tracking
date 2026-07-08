import { color, radius, HEADER_H } from '../theme.js'

// Fixed left sidebar that contains the three instrument panels stacked
// vertically: Distance to Retina → Depth Side View → Proximity Scope.
// Uses a flex column so sections shrink proportionally if the viewport is short,
// preventing overlap. All children stretch to the panel's 320px width.
const PANEL_W = 320

export const SIDE_PANEL_W = PANEL_W

export default function SidePanel({ children }) {
  return (
    <div style={root}>
      {children}
    </div>
  )
}

const root = {
  position: 'absolute',
  top: HEADER_H + 12,
  left: 12,
  bottom: 12,
  width: PANEL_W,
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
  pointerEvents: 'none',       // pass-through by default; children opt in
  zIndex: 5,
}
