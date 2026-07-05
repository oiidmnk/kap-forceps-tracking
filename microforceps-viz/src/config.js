// Central configuration for the digital-twin dashboard.
// All spatial units are millimeters (mm); origin is the eye-globe center.

export const EYE_RADIUS_MM = 12 // globe radius (perfect sphere)
export const LIMBUS_RADIUS_MM = 6 // limbus circle radius

// Distance-to-Retina safety thresholds (mm), tunable.
// >= SAFE => green, [WARN, SAFE) => amber, < WARN => red.
export const DIST_SAFE_MM = 2.0
export const DIST_WARN_MM = 0.5

// The proximity scope only appears once a tip is within this range of the
// retina — otherwise it's dead screen real estate for most of the procedure.
export const DIST_SCOPE_SHOW_MM = 5.0

// Live tracking feed (Python) — see feed/synthetic_feed.py
// Live tracking feed — proxied through nginx at /ws in Docker; override via VITE_WS_URL.
function defaultWsUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}/ws`
}
export const WS_URL = import.meta.env.VITE_WS_URL || defaultWsUrl()

// Physical size hints for rendering the instrument (mm). A real 23–25 gauge
// microforceps shaft is ~0.5–0.6 mm diameter with fine jaws that taper to a point.
export const SHAFT_RADIUS_MM = 0.28 // shaft ~0.56 mm dia
export const JAW_BASE_RADIUS_MM = 0.22 // jaw half-width at the hinge
export const JAW_TIP_RADIUS_MM = 0.05 // jaws taper to a fine tip
export const JAW_FLATTEN = 0.32 // blade thickness / width ratio (flat jaw)
export const TIP_INDICATOR_MM = 0.1 // subtle safety-color accent at the very tip

// Rendered jaw length (mm). The hinge is placed this far back along the shaft
// from the tips, so the two jaws form a proper V instead of a flat line.
// ~1.5 mm matches typical vitreoretinal microforceps jaw length.
export const JAW_LENGTH_MM = 1.5

// Render-only multiplier on the sideways jaw opening, so the two prongs are
// legible even when the real forceps is nearly closed. 1 = true opening.
// Tuned with JAW_LENGTH_MM for a realistic ~15°–37° full opening angle.
export const VIEW_JAW_SPREAD = 2.4
