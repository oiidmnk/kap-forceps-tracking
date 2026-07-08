// Shared design tokens for a restrained, clinical instrument UI.
//
// Guiding rule: saturated color is reserved for SAFETY state only. Everything
// else — chrome, labels, accents, active toggles — uses the neutral steel/slate
// scale below, so the operator's eye is drawn to the safety readout and nowhere
// else. Surfaces are matte (opaque, hairline-bordered, no neon glow).

export const font = {
  // Inter is loaded in index.html; the stack degrades to the platform grotesk.
  sans: "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
  mono: "'IBM Plex Mono', ui-monospace, 'SF Mono', monospace",
}

export const color = {
  // Matte surfaces (no frosted blur — crisp and legible).
  surface: 'rgba(14,19,26,0.92)',
  surfaceInset: 'rgba(255,255,255,0.04)',
  border: 'rgba(150,170,195,0.14)',
  borderStrong: 'rgba(150,170,195,0.30)',

  // Text ramp.
  text: '#dfe6ee',
  textDim: '#8a99ab',
  textFaint: '#5d6b7d',

  // Neutral steel accent — replaces the old cyan/purple everywhere that isn't
  // a safety signal (active toggles, selected preset, dividers, grid).
  accent: '#8aa0b6',
  accentSoft: 'rgba(138,160,182,0.16)',
  accentBorder: 'rgba(138,160,182,0.55)',

  // The ONLY saturated colors in the UI — muted, high-contrast medical variants.
  safe: '#3fb27f',
  warn: '#e0a020',
  danger: '#e5484d',
}

export const space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 20 }
export const radius = { sm: 6, md: 8 }

// Height of the fixed device header strip; overlays below offset by this.
export const HEADER_H = 46

// Tabular figures so digits don't jitter as values update — applied to every
// numeric readout.
export const tnum = { fontVariantNumeric: 'tabular-nums' }
