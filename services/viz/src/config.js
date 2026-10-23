// Central configuration for the digital-twin dashboard.
// All spatial units are millimeters (mm); origin is the eye-globe center.

export const EYE_RADIUS_MM = 12 // globe radius (perfect sphere)
export const LIMBUS_RADIUS_MM = 6 // limbus circle radius

// Distance-to-Retina safety thresholds (mm), tunable.
// >= SAFE => green, [WARN, SAFE) => amber, < WARN => red.
export const DIST_SAFE_MM = 2.0
export const DIST_WARN_MM = 0.5
// Hysteresis: a more critical level shows immediately; returning to a safer
// level requires clearing the threshold by this margin, so noise at a boundary
// cannot make the status flicker.
export const DIST_HYSTERESIS_MM = 0.05

// The persistent proximity panel activates its close-up inside this range;
// outside it, a standby state keeps the workstation layout stable.
export const DIST_SCOPE_SHOW_MM = 5.0
export const DIST_SCOPE_HIDE_MM = 5.25 // scope stays active until this far

// Numeric readouts refresh at this rate (ms); 3D views update every frame.
export const READOUT_INTERVAL_MS = 100

// Live tracking feed (Python) — see feed/synthetic_feed.py
// Live tracking feed — proxied through nginx at /ws in Docker; override via VITE_WS_URL.
function defaultWsUrl() {
  if (typeof window === 'undefined') return ''
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}/ws`
}
export const WS_URL = import.meta.env?.VITE_WS_URL || defaultWsUrl()

// Instrument dimensions (mm), matched to the Haag-Streit (Eyesi) simulator
// forceps: a 23G tube (Ø 0.64 mm) whose cut end releases two flat jaws with
// small inward hooks. Proportions were measured on simulator footage — the
// exposed jaws are ~2.6 shaft diameters long and open to ~1.1 diameters.
export const SHAFT_RADIUS_MM = 0.32 // 23G outer diameter 0.64 mm
export const SHAFT_LUMEN_RADIUS_MM = 0.25 // inner bore the jaws slide in
export const JAW_LENGTH_MM = 1.65 // exposed jaw, tube rim → tracked tip
export const JAW_EXIT_OFFSET_MM = 0.17 // jaw base offset from axis, fully open
export const JAW_CLOSED_OFFSET_MM = 0.05 // jaw base offset when squeezed shut
export const JAW_FULL_OPEN_MM = 0.7 // tip separation of a fully opened jaw
export const JAW_THICKNESS_MM = 0.07 // blade thickness in the opening direction
export const JAW_WIDTH_BASE_MM = 0.17 // blade width where it leaves the tube
export const JAW_WIDTH_TIP_MM = 0.09 // blade width toward the tip
export const JAW_HOOK_MM = 0.07 // inward-bent grasping hook at the tip
export const TIP_INDICATOR_MM = 0.045 // small accent centered on the tracked tip

// --- Clinical view presets --------------------------------------------------
// Named camera framings for the main overview (+Y up; instruments enter +Y;
// retina = -Y hemisphere). All keep a common world up (+Y) so preset-to-preset
// transitions glide without a snap-roll (see CameraRig); none looks exactly
// along the Y axis, so no gimbal handling is needed. Positions are in mm.
// `target` defaults to the globe center; `fit: false` skips aspect fitting.
// When orbiting the instrument, presets only supply the viewing direction.
const LIMBUS_Y_MM = Math.sqrt(EYE_RADIUS_MM ** 2 - LIMBUS_RADIUS_MM ** 2)
export const VIEW_PRESETS = {
  overview: { pos: [EYE_RADIUS_MM * 1.4, EYE_RADIUS_MM * 0.9, EYE_RADIUS_MM * 3.4] },
  surgeon: { pos: [0, EYE_RADIUS_MM * 3.4, EYE_RADIUS_MM * 0.6] },
  sagittal: { pos: [EYE_RADIUS_MM * 3.8, EYE_RADIUS_MM * 0.3, 0] },
  // Inside the globe, just below the pupil, looking at the posterior pole —
  // roughly the simulator's wide-field view of the fundus.
  fundus: { pos: [0, LIMBUS_Y_MM - 0.8, 0.6], target: [0, -EYE_RADIUS_MM, 0], fit: false },
}

// --- Navigation --------------------------------------------------------------
export const NAV_MIN_DISTANCE_MM = 1 // close enough to inspect the jaws
export const NAV_MAX_DISTANCE_MM = EYE_RADIUS_MM * 8
export const NAV_FOLLOW_DISTANCE_MM = 14 // camera distance when orbiting the instrument
export const NAV_FOCUS_DISTANCE_MM = 18 // zoom-in distance after double-click focus
