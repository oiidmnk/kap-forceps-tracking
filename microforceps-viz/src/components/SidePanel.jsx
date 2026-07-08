import React from 'react'
import { color, radius, HEADER_H } from '../theme.js'

// Fixed left sidebar that contains the three instrument panels stacked
// vertically: Distance to Retina → Depth Side View → Proximity Scope.
// Uses a flex column so sections shrink proportionally if the viewport is short,
// preventing overlap. All children stretch to the panel's 320px width.
// Thin horizontal dividers separate each section for clean visual hierarchy.
const PANEL_W = 320

export const SIDE_PANEL_W = PANEL_W

const divider = {
  width: '85%',
  height: 1,
  alignSelf: 'center',
  background: color.border,
  flexShrink: 0,
}

export default function SidePanel({ children }) {
  // Interleave thin dividers between each child.
  const items = []
  React.Children.forEach(children, (child, i) => {
    if (!child) return
    if (items.length > 0) {
      items.push(<div key={`div-${i}`} style={divider} />)
    }
    items.push(child)
  })

  return (
    <div style={root}>
      {items}
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
  gap: 8,
  background: color.surface,
  border: `1px solid ${color.border}`,
  borderRadius: radius.md,
  padding: '10px 0',
  pointerEvents: 'none',       // pass-through by default; children opt in
  zIndex: 5,
}
