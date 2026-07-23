const calibrationForm = document.querySelector('#calibration-form')
const processForm = document.querySelector('#process-form')
const result = document.querySelector('#result')
const calibrationStatus = document.querySelector('#calibration-status')
const processStatus = document.querySelector('#process-status')
const serviceStatus = document.querySelector('#service-status')

function setStatus(el, message, ok) {
  el.textContent = message
  el.className = `status ${ok ? 'ok' : 'error'}`
}

function fillCalibration(data) {
  for (const [key, value] of Object.entries(data)) {
    if (key === 'eye_center_px') {
      calibrationForm.elements.eye_center_px_0.value = value[0]
      calibrationForm.elements.eye_center_px_1.value = value[1]
    } else if (calibrationForm.elements[key]) {
      calibrationForm.elements[key].value = value
    }
  }
}

function readCalibration() {
  const f = calibrationForm.elements
  return {
    light_rot_up: Number(f.light_rot_up.value),
    light_rot_clock: Number(f.light_rot_clock.value),
    light_depth_mm: Number(f.light_depth_mm.value),
    forceps_rot_up: Number(f.forceps_rot_up.value),
    forceps_rot_clock: Number(f.forceps_rot_clock.value),
    eye_center_px: [Number(f.eye_center_px_0.value), Number(f.eye_center_px_1.value)],
    eye_radius_px: Number(f.eye_radius_px.value),
    eye_radius_mm: Number(f.eye_radius_mm.value),
    jaw_length_mm: Number(f.jaw_length_mm.value),
  }
}

async function loadCalibration() {
  const response = await fetch('/api/calibration')
  const data = await response.json()
  if (!response.ok) throw new Error(data.detail || response.statusText)
  fillCalibration(data)
  setStatus(calibrationStatus, 'Calibration loaded.', true)
}

// Push the saved calibration onto the live stream so the viz updates without
// re-uploading a frame. Best-effort: the caller reports the outcome.
async function pushCalibrationToStream() {
  const response = await fetch('/api/apply-calibration', { method: 'POST' })
  const data = await response.json()
  if (!response.ok) throw new Error(data.detail || response.statusText)
  return data
}

async function loadServiceStatus() {
  const response = await fetch('/api/status')
  const data = await response.json()
  if (!response.ok) throw new Error(data.detail || response.statusText)

  if (data.ready) {
    serviceStatus.textContent = 'Segmentation and stream services are ready.'
    serviceStatus.className = 'service-status ok'
    return
  }

  const segmentation = data.segmentation?.payload || {}
  const parts = []
  if (!data.segmentation?.available) {
    parts.push(`segmentation unavailable: ${data.segmentation?.error || data.segmentation?.status_code}`)
  } else if (segmentation.weights_available === false) {
    parts.push(`weights missing: ${segmentation.weights}`)
  }
  if (!data.stream?.available) {
    parts.push(`stream unavailable: ${data.stream?.error || data.stream?.status_code}`)
  }

  serviceStatus.textContent = data.message || parts.join('; ') || 'One or more services are not ready.'
  serviceStatus.className = 'service-status error'
}

calibrationForm.addEventListener('submit', async (event) => {
  event.preventDefault()
  try {
    const response = await fetch('/api/calibration', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(readCalibration()),
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || response.statusText)
    fillCalibration(data)
    try {
      const push = await pushCalibrationToStream()
      if (push?.stream_result?.error) {
        setStatus(calibrationStatus, 'Calibration saved and trocars applied. Tips need a processed frame to reconstruct.', true)
      } else {
        setStatus(calibrationStatus, 'Calibration saved and applied to the live stream.', true)
      }
    } catch (pushError) {
      setStatus(calibrationStatus, `Calibration saved, but live stream update failed: ${pushError.message}`, false)
    }
  } catch (error) {
    setStatus(calibrationStatus, error.message, false)
  }
})

document.querySelector('#reload-calibration').addEventListener('click', () => {
  loadCalibration().catch((error) => setStatus(calibrationStatus, error.message, false))
})

processForm.addEventListener('submit', async (event) => {
  event.preventDefault()
  const button = document.querySelector('#process-button')
  button.disabled = true
  processStatus.textContent = 'Processing...'
  processStatus.className = 'status'
  try {
    const body = new FormData(processForm)
    const response = await fetch('/api/process', { method: 'POST', body })
    const data = await response.json()
    result.textContent = JSON.stringify({
      predicted_points: data.predicted_points,
      stream_result: data.stream_result,
    }, null, 2)
    if (!response.ok) throw new Error(data.detail || response.statusText)
    setStatus(processStatus, 'Stream updated.', true)
  } catch (error) {
    setStatus(processStatus, error.message, false)
  } finally {
    button.disabled = false
  }
})

// ---- Live calibration ----------------------------------------------------
// A live camera view with a centered reference ring sized to the LIMBUS (the
// sharp, visible cornea/sclera boundary — the full eyeball silhouette usually
// isn't). The operator zooms the microscope until the limbus lands on the ring.
// Under orthographic projection the limbus (6 mm) projects at exactly half the
// eye silhouette (12 mm), so the eye radius in pixels is derived as
// limbus_px * (EYE_RADIUS_MM / LIMBUS_RADIUS_MM). Each trocar click is then
// inverted through trocar_position() to recover (rot_up, rot_clock): a surface
// point projects orthographically to (nx, ny) normalized by the EYE radius, and
// the anterior cap gives rot_up = asin(|n|), rot_clock = atan2(ny, nx).
// Everything is expressed in the camera's native pixel frame so it matches the
// frames segmentation sees.
const EYE_RADIUS_MM = 12
const LIMBUS_RADIUS_MM = 6
const LIMBUS_TO_EYE = EYE_RADIUS_MM / LIMBUS_RADIUS_MM
const calibVideo = document.querySelector('#calib-video')
const calibCanvas = document.querySelector('#calib-canvas')
const calibCtx = calibCanvas.getContext('2d')
const cameraSelect = document.querySelector('#camera-select')
const cameraStartButton = document.querySelector('#camera-start')
const cameraStopButton = document.querySelector('#camera-stop')
const ringRadiusInput = document.querySelector('#ring-radius')
const verticalStretchInput = document.querySelector('#vertical-stretch')
const stretchResetButton = document.querySelector('#stretch-reset')
const freezeButton = document.querySelector('#freeze-toggle')
const zoomResetButton = document.querySelector('#zoom-reset')
const labelingStage = document.querySelector('#labeling-stage')
const focusModeButton = document.querySelector('#focus-mode-toggle')
const manualPointsDetails = document.querySelector('#manual-points-details')
const pickForcepsButton = document.querySelector('#pick-forceps')
const pickLightButton = document.querySelector('#pick-light')
const calibReadout = document.querySelector('#calib-readout')
const calibApply = document.querySelector('#calib-apply')
const calibModeStatus = document.querySelector('#calib-mode-status')

// Manual forceps points (tips + shadows) — hand-clicked stand-ins for
// segmentation output, on the SAME canvas/frame as the trocars above, so they
// share the exact same native pixel space. Pushed straight to the stream via
// /api/manual-points, bypassing the segmentation service entirely — for
// testing the viz before the YOLO model is trained.
const pickLeftTipButton = document.querySelector('#pick-left-tip')
const pickRightTipButton = document.querySelector('#pick-right-tip')
const pickLeftShadowButton = document.querySelector('#pick-left-shadow')
const pickRightShadowButton = document.querySelector('#pick-right-shadow')
const pointsReadout = document.querySelector('#points-readout')
const pointsPushButton = document.querySelector('#points-push')
const pointsClearButton = document.querySelector('#points-clear')
const pointsStatus = document.querySelector('#points-status')

const FORCEPS_COLOR = '#3b82f6'
const LIGHT_COLOR = '#e0a44a'
const POINT_COLORS = {
  left_tip: '#22d3ee',
  right_tip: '#f472b6',
  left_shadow: '#0e7490',
  right_shadow: '#9d174d',
}
const POINT_LABELS = { left_tip: 'LT', right_tip: 'RT', left_shadow: 'LS', right_shadow: 'RS' }
const POINT_KEYS = Object.keys(POINT_COLORS)

// One marker store for every pickable target (2 trocars + 4 forceps points),
// all in native camera pixel space so they stay mutually consistent.
const markers = {
  forceps: null,
  light: null,
  left_tip: null,
  right_tip: null,
  left_shadow: null,
  right_shadow: null,
}
const PICK_BUTTONS = {
  forceps: pickForcepsButton,
  light: pickLightButton,
  left_tip: pickLeftTipButton,
  right_tip: pickRightTipButton,
  left_shadow: pickLeftShadowButton,
  right_shadow: pickRightShadowButton,
}
const PICK_LABELS = {
  forceps: 'forceps trocar',
  light: 'light trocar',
  left_tip: 'left forceps tip',
  right_tip: 'right forceps tip',
  left_shadow: 'left tip shadow',
  right_shadow: 'right tip shadow',
}
let pickTarget = null
let cameraStream = null
let frozenFrame = null // ImageBitmap snapshot, or null while showing the live feed

// Zoom/pan view, applied only as a canvas DRAW transform (canvasPx = worldPx *
// scale + translate). The backing-store pixel grid itself never changes, so
// eye_center_px/eye_radius_px/trocar/point math stays in native camera pixel
// space regardless of zoom — every click is converted from screen -> canvas
// backing-store pixel (existing rect-ratio math, unaffected) -> world pixel
// (inverting this transform) before it's ever stored in `markers`.
const view = { scale: 1, tx: 0, ty: 0 }
const MIN_ZOOM = 1
const MAX_ZOOM = 12

function clampView() {
  view.scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, view.scale))
  const minTx = calibCanvas.width - calibCanvas.width * view.scale
  const minTy = calibCanvas.height - calibCanvas.height * view.scale
  view.tx = Math.min(0, Math.max(minTx, view.tx))
  view.ty = Math.min(0, Math.max(minTy, view.ty))
}

function resetView() {
  view.scale = 1
  view.tx = 0
  view.ty = 0
}

// The element's CSS box, narrowed down to the sub-rectangle the image is
// actually drawn into. In focus mode the canvas uses `object-fit: contain`
// (so it can fill the available height rather than being width-bound), which
// letterboxes when the box aspect ratio doesn't match the camera's native
// aspect ratio — getBoundingClientRect() still reports the FULL box in that
// case, not the visible image area, so using it directly would offset every
// click by the size of the letterbox bar. Outside focus mode there's no
// object-fit and this returns the same box getBoundingClientRect() would.
function canvasContentRect() {
  const rect = calibCanvas.getBoundingClientRect()
  const boxAspect = rect.width / rect.height
  const contentAspect = calibCanvas.width / calibCanvas.height
  let { width, height } = rect
  let offsetX = 0
  let offsetY = 0
  if (boxAspect > contentAspect) {
    width = rect.height * contentAspect
    offsetX = (rect.width - width) / 2
  } else if (boxAspect < contentAspect) {
    height = rect.width / contentAspect
    offsetY = (rect.height - height) / 2
  }
  return { left: rect.left + offsetX, top: rect.top + offsetY, width, height }
}

// Screen (CSS) coordinates -> canvas backing-store pixel coordinates. Already
// accounts for the CSS display size, the vertical-stretch transform, and any
// object-fit letterboxing (via canvasContentRect).
function canvasPointFromEvent(event) {
  const content = canvasContentRect()
  return {
    x: (event.clientX - content.left) * (calibCanvas.width / content.width),
    y: (event.clientY - content.top) * (calibCanvas.height / content.height),
  }
}

// Canvas backing-store pixel -> world (native camera) pixel, inverting the
// zoom/pan draw transform.
function worldFromCanvasPoint(canvasPt) {
  return {
    x: (canvasPt.x - view.tx) / view.scale,
    y: (canvasPt.y - view.ty) / view.scale,
  }
}

function statusElFor(which) {
  return POINT_KEYS.includes(which) ? pointsStatus : calibModeStatus
}

function updateFreezeButtonState() {
  if (frozenFrame) {
    freezeButton.disabled = false
    freezeButton.textContent = 'Resume live'
  } else {
    freezeButton.disabled = !cameraStream
    freezeButton.textContent = 'Freeze frame'
  }
}

// Freezing captures a still snapshot of the current video frame and displays
// that in place of the live feed, so the operator can click precise points on
// a static image (the forceps keeps moving otherwise). The ring, crosshair,
// and markers are drawn fresh every frame regardless, so they stay fully live
// and adjustable on top of the frozen picture.
async function toggleFreeze() {
  if (frozenFrame) {
    frozenFrame.close?.()
    frozenFrame = null
    updateFreezeButtonState()
    setStatus(calibModeStatus, 'Live again.', true)
    return
  }
  if (!cameraStream || calibVideo.readyState < 2) {
    setStatus(calibModeStatus, 'Start the camera first.', false)
    return
  }
  frozenFrame = await createImageBitmap(calibVideo)
  updateFreezeButtonState()
  setStatus(calibModeStatus, 'Frame frozen — click precisely, then "Resume live" when done.', true)
}

// Correct for non-square-pixel camera/capture hardware (common with USB/HDMI
// capture dongles used to digitize a microscope feed) by visually stretching
// the CANVAS ELEMENT only, via a CSS transform, until a round object (the
// limbus) actually looks round. This never touches the canvas's backing-store
// pixel grid, so eye_center_px/eye_radius_px and trocar-click math stay in
// unmodified native pixel space. Click coordinates still resolve correctly
// because canvasContentRect() reads getBoundingClientRect(), which already
// reports the post-CSS-transform box.
function applyVerticalStretch() {
  calibCanvas.style.transform = `scaleY(${Number(verticalStretchInput.value)})`
}

function ringGeometry() {
  // The ring is the LIMBUS; the eye silhouette is twice its radius.
  const limbusRadius = Math.min(calibCanvas.width, calibCanvas.height) * Number(ringRadiusInput.value)
  return {
    cx: calibCanvas.width / 2,
    cy: calibCanvas.height / 2,
    limbusRadius,
    eyeRadius: limbusRadius * LIMBUS_TO_EYE,
  }
}

function pixelToTrocarAngles(marker, ring) {
  // Normalize by the EYE radius (surface = unit sphere), not the limbus ring.
  const nx = (marker.x - ring.cx) / ring.eyeRadius
  const ny = (marker.y - ring.cy) / ring.eyeRadius
  const rho = Math.hypot(nx, ny)
  return {
    rot_up: Math.asin(Math.min(rho, 1)),
    rot_clock: Math.atan2(ny, nx),
    outside: rho > 1,
  }
}

function drawMarker(marker, color, label) {
  if (!marker) return
  // Divide by view.scale so markers/labels stay a constant on-screen size
  // instead of visually ballooning as you zoom in.
  const s = 1 / view.scale
  const r = 9 * s
  calibCtx.strokeStyle = color
  calibCtx.fillStyle = color
  calibCtx.lineWidth = 2 * s
  calibCtx.beginPath()
  calibCtx.arc(marker.x, marker.y, r, 0, Math.PI * 2)
  calibCtx.moveTo(marker.x - r - 4 * s, marker.y)
  calibCtx.lineTo(marker.x + r + 4 * s, marker.y)
  calibCtx.moveTo(marker.x, marker.y - r - 4 * s)
  calibCtx.lineTo(marker.x, marker.y + r + 4 * s)
  calibCtx.stroke()
  calibCtx.font = `bold ${16 * s}px sans-serif`
  calibCtx.fillText(label, marker.x + r + 6 * s, marker.y - r)
}

function drawFrame() {
  const { width, height } = calibCanvas

  // Background in screen space so panned-out areas don't show stale pixels.
  calibCtx.setTransform(1, 0, 0, 1, 0, 0)
  calibCtx.fillStyle = '#0f172a'
  calibCtx.fillRect(0, 0, width, height)

  // Everything below is drawn in WORLD (native camera pixel) space; this
  // transform maps it to the current zoom/pan. Ring/marker geometry itself
  // never changes with zoom — only how much of it is visible and how big it
  // renders on screen.
  calibCtx.setTransform(view.scale, 0, 0, view.scale, view.tx, view.ty)
  const zoomStroke = 1 / view.scale

  if (frozenFrame) {
    calibCtx.drawImage(frozenFrame, 0, 0, width, height)
  } else if (cameraStream && calibVideo.readyState >= 2) {
    calibCtx.drawImage(calibVideo, 0, 0, width, height)
  }

  const ring = ringGeometry()
  // Full-canvas crosshair through the ring center — helps center the eye and
  // makes it easy to spot residual vertical stretch (the ring should look
  // exactly as tall as it is wide relative to these lines).
  calibCtx.save()
  calibCtx.strokeStyle = 'rgba(74, 222, 128, 0.45)'
  calibCtx.lineWidth = zoomStroke
  calibCtx.setLineDash([4 * zoomStroke, 6 * zoomStroke])
  calibCtx.beginPath()
  calibCtx.moveTo(0, ring.cy)
  calibCtx.lineTo(width, ring.cy)
  calibCtx.moveTo(ring.cx, 0)
  calibCtx.lineTo(ring.cx, height)
  calibCtx.stroke()
  calibCtx.restore()

  // Derived eye silhouette (2x the limbus) — faint dashed reference.
  calibCtx.strokeStyle = 'rgba(74, 222, 128, 0.28)'
  calibCtx.lineWidth = zoomStroke
  calibCtx.setLineDash([6 * zoomStroke, 6 * zoomStroke])
  calibCtx.beginPath()
  calibCtx.arc(ring.cx, ring.cy, ring.eyeRadius, 0, Math.PI * 2)
  calibCtx.stroke()
  calibCtx.setLineDash([])
  // Limbus reference ring — align the limbus to this.
  calibCtx.strokeStyle = 'rgba(74, 222, 128, 0.95)'
  calibCtx.lineWidth = 2 * zoomStroke
  calibCtx.beginPath()
  calibCtx.arc(ring.cx, ring.cy, ring.limbusRadius, 0, Math.PI * 2)
  calibCtx.stroke()

  drawMarker(markers.forceps, FORCEPS_COLOR, 'F')
  drawMarker(markers.light, LIGHT_COLOR, 'L')
  for (const key of POINT_KEYS) {
    drawMarker(markers[key], POINT_COLORS[key], POINT_LABELS[key])
  }

  // Screen-anchored overlays, unaffected by zoom/pan.
  calibCtx.setTransform(1, 0, 0, 1, 0, 0)
  if (frozenFrame) {
    calibCtx.save()
    calibCtx.fillStyle = 'rgba(15, 23, 42, 0.75)'
    calibCtx.fillRect(8, 8, 96, 24)
    calibCtx.fillStyle = '#f87171'
    calibCtx.font = 'bold 13px sans-serif'
    calibCtx.fillText('● FROZEN', 16, 25)
    calibCtx.restore()
  }
  if (view.scale > 1.01) {
    calibCtx.save()
    calibCtx.fillStyle = 'rgba(15, 23, 42, 0.75)'
    calibCtx.fillRect(width - 70, 8, 62, 24)
    calibCtx.fillStyle = '#93c5fd'
    calibCtx.font = 'bold 12px sans-serif'
    calibCtx.fillText(`${view.scale.toFixed(1)}x`, width - 60, 25)
    calibCtx.restore()
  }

  requestAnimationFrame(drawFrame)
}

function updateReadout() {
  const ring = ringGeometry()
  const lines = [
    `center: ${ring.cx.toFixed(0)}, ${ring.cy.toFixed(0)} px    limbus r: ${ring.limbusRadius.toFixed(1)} px    eye r: ${ring.eyeRadius.toFixed(1)} px`,
    `display vertical stretch: ${Number(verticalStretchInput.value).toFixed(3)}x (display only, not saved)`,
  ]
  let ready = true
  for (const which of ['forceps', 'light']) {
    const marker = markers[which]
    if (!marker) {
      lines.push(`${which.padEnd(7)}: not set`)
      ready = false
      continue
    }
    const angles = pixelToTrocarAngles(marker, ring)
    const flag = angles.outside ? '  (outside ring — clamped)' : ''
    lines.push(
      `${which.padEnd(7)}: rot_up ${angles.rot_up.toFixed(4)}  rot_clock ${angles.rot_clock.toFixed(4)}${flag}`,
    )
  }
  calibReadout.textContent = lines.join('\n')
  calibApply.disabled = !ready
  updatePickButtonStates()
}

function updatePointsReadout() {
  const lines = []
  let allSet = true
  for (const key of POINT_KEYS) {
    const marker = markers[key]
    if (!marker) {
      lines.push(`${key.padEnd(13)}: not set`)
      allSet = false
      continue
    }
    lines.push(`${key.padEnd(13)}: ${marker.x.toFixed(1)}, ${marker.y.toFixed(1)} px`)
  }
  pointsReadout.textContent = lines.join('\n')
  pointsPushButton.disabled = !allSet
  updatePickButtonStates()
}

function updatePickButtonStates() {
  for (const [key, button] of Object.entries(PICK_BUTTONS)) {
    button.classList.toggle('set', markers[key] !== null)
  }
}

function setPickTarget(which) {
  pickTarget = which
  for (const [key, button] of Object.entries(PICK_BUTTONS)) {
    button.classList.toggle('active', key === which)
  }
  if (which) {
    // A point-pick target (e.g. via keyboard shortcut) needs the section
    // visible even if it's currently collapsed.
    if (POINT_KEYS.includes(which)) manualPointsDetails.open = true
    setStatus(statusElFor(which), `Click the ${PICK_LABELS[which]} on the image.`, true)
  }
}

async function populateCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices()
  const cameras = devices.filter((device) => device.kind === 'videoinput')
  const previous = cameraSelect.value
  cameraSelect.innerHTML = ''
  cameras.forEach((camera, index) => {
    const option = document.createElement('option')
    option.value = camera.deviceId
    option.textContent = camera.label || `Camera ${index + 1}`
    cameraSelect.appendChild(option)
  })
  if (previous) cameraSelect.value = previous
}

async function startCamera() {
  try {
    stopCamera()
    if (frozenFrame) {
      // A new stream is about to replace the picture — an old frozen frame
      // would be stale and confusing to keep showing.
      frozenFrame.close?.()
      frozenFrame = null
    }
    const deviceId = cameraSelect.value || undefined
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId } } : true,
      audio: false,
    })
    calibVideo.srcObject = cameraStream
    await calibVideo.play()
    await populateCameras()
    cameraStopButton.disabled = false
    updateFreezeButtonState()
    setStatus(calibModeStatus, 'Camera started. Align the limbus to the ring, then set the trocars.', true)
  } catch (error) {
    setStatus(calibModeStatus, `Camera error: ${error.message}. On http a non-localhost host blocks camera access.`, false)
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop())
    cameraStream = null
  }
  calibVideo.srcObject = null
  cameraStopButton.disabled = true
  updateFreezeButtonState()
}

calibVideo.addEventListener('loadedmetadata', () => {
  if (calibVideo.videoWidth && calibVideo.videoHeight) {
    const resized = calibCanvas.width !== calibVideo.videoWidth || calibCanvas.height !== calibVideo.videoHeight
    calibCanvas.width = calibVideo.videoWidth
    calibCanvas.height = calibVideo.videoHeight
    if (resized) {
      // Native pixel space changed (new camera/resolution) — old marker
      // coordinates no longer point at the right place, a frozen snapshot at
      // the old size would draw distorted, and any pan/zoom no longer maps
      // to a valid region.
      for (const key of Object.keys(markers)) markers[key] = null
      if (frozenFrame) {
        frozenFrame.close?.()
        frozenFrame = null
        updateFreezeButtonState()
      }
      resetView()
    }
    updateReadout()
    updatePointsReadout()
  }
})

// Click-to-place a point vs. drag-to-pan share the same pointer gesture, so
// they're disambiguated by movement distance: pointerup with no meaningful
// movement places a point (if a pick target is active); pointerup after
// movement past DRAG_THRESHOLD only pans, no point placed. Pointer Events
// (not mouse-only) so this works the same for touch.
const DRAG_THRESHOLD_PX = 4
let dragState = null

function placePointAt(event) {
  if (!pickTarget) {
    setStatus(calibModeStatus, 'Choose a "Set ..." button first (or press its number key).', false)
    return
  }
  const canvasPt = canvasPointFromEvent(event)
  markers[pickTarget] = worldFromCanvasPoint(canvasPt)
  setPickTarget(null)
  updateReadout()
  updatePointsReadout()
}

calibCanvas.addEventListener('pointerdown', (event) => {
  if (event.button !== 0 && event.pointerType === 'mouse') return
  event.preventDefault()
  dragState = {
    pointerId: event.pointerId,
    startClientX: event.clientX,
    startClientY: event.clientY,
    startTx: view.tx,
    startTy: view.ty,
    moved: false,
  }
  calibCanvas.setPointerCapture(event.pointerId)
})

calibCanvas.addEventListener('pointermove', (event) => {
  if (!dragState || event.pointerId !== dragState.pointerId) return
  const dxScreen = event.clientX - dragState.startClientX
  const dyScreen = event.clientY - dragState.startClientY
  if (!dragState.moved && Math.hypot(dxScreen, dyScreen) > DRAG_THRESHOLD_PX) {
    dragState.moved = true
    calibCanvas.classList.add('panning')
  }
  if (!dragState.moved) return
  // Scale by the actual rendered image area (not the full element box, which
  // may include object-fit letterbox bars in focus mode) so a screen-pixel
  // drag maps to the same visual distance in canvas pixels.
  const content = canvasContentRect()
  const dx = dxScreen * (calibCanvas.width / content.width)
  const dy = dyScreen * (calibCanvas.height / content.height)
  view.tx = dragState.startTx + dx
  view.ty = dragState.startTy + dy
  clampView()
})

function endPointerInteraction(event) {
  if (!dragState || event.pointerId !== dragState.pointerId) return
  const wasDrag = dragState.moved
  calibCanvas.classList.remove('panning')
  try {
    calibCanvas.releasePointerCapture(dragState.pointerId)
  } catch {
    /* already released */
  }
  dragState = null
  if (!wasDrag) placePointAt(event)
}

calibCanvas.addEventListener('pointerup', endPointerInteraction)
calibCanvas.addEventListener('pointercancel', () => {
  dragState = null
  calibCanvas.classList.remove('panning')
})

// Wheel = zoom, centered on the cursor so the point under it stays put.
calibCanvas.addEventListener(
  'wheel',
  (event) => {
    event.preventDefault()
    const canvasPt = canvasPointFromEvent(event)
    const worldBefore = worldFromCanvasPoint(canvasPt)
    const factor = Math.exp(-event.deltaY * 0.0015)
    view.scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, view.scale * factor))
    view.tx = canvasPt.x - worldBefore.x * view.scale
    view.ty = canvasPt.y - worldBefore.y * view.scale
    clampView()
  },
  { passive: false },
)

zoomResetButton.addEventListener('click', resetView)

calibApply.addEventListener('click', async () => {
  const ring = ringGeometry()
  const forceps = pixelToTrocarAngles(markers.forceps, ring)
  const light = pixelToTrocarAngles(markers.light, ring)
  const f = calibrationForm.elements
  f.eye_center_px_0.value = ring.cx.toFixed(2)
  f.eye_center_px_1.value = ring.cy.toFixed(2)
  f.eye_radius_px.value = ring.eyeRadius.toFixed(2)
  f.forceps_rot_up.value = forceps.rot_up.toFixed(5)
  f.forceps_rot_clock.value = forceps.rot_clock.toFixed(5)
  f.light_rot_up.value = light.rot_up.toFixed(5)
  f.light_rot_clock.value = light.rot_clock.toFixed(5)
  try {
    const response = await fetch('/api/calibration', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(readCalibration()),
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || response.statusText)
    fillCalibration(data)
    setStatus(calibrationStatus, 'Calibration updated from live calibration.', true)
    try {
      const push = await pushCalibrationToStream()
      if (push?.stream_result?.error) {
        setStatus(calibModeStatus, 'Trocars applied. The forceps tips need a processed frame from this camera to reconstruct — upload/process one to see them.', true)
      } else {
        setStatus(calibModeStatus, 'Saved and applied to the live stream — the viz should update now.', true)
      }
    } catch (pushError) {
      setStatus(calibModeStatus, `Saved, but live stream update failed: ${pushError.message}`, false)
    }
  } catch (error) {
    setStatus(calibModeStatus, error.message, false)
  }
})

// Push hand-clicked tip/shadow points straight to the stream, merged with the
// saved calibration server-side (/api/manual-points), entirely bypassing the
// segmentation service. Lets the viz be exercised end-to-end before the YOLO
// model has trained weights.
pointsPushButton.addEventListener('click', async () => {
  const payload = {}
  for (const key of POINT_KEYS) {
    payload[`${key}_px`] = [markers[key].x, markers[key].y]
  }
  try {
    const response = await fetch('/api/manual-points', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || response.statusText)
    setStatus(pointsStatus, 'Points pushed to the live stream — check the viz.', true)
  } catch (error) {
    setStatus(pointsStatus, error.message, false)
  }
})

pointsClearButton.addEventListener('click', () => {
  for (const key of POINT_KEYS) markers[key] = null
  updatePointsReadout()
  setStatus(pointsStatus, 'Points cleared.', true)
})

cameraStartButton.addEventListener('click', startCamera)
cameraStopButton.addEventListener('click', stopCamera)
cameraSelect.addEventListener('change', () => { if (cameraStream) startCamera() })
ringRadiusInput.addEventListener('input', updateReadout)
verticalStretchInput.addEventListener('input', () => {
  applyVerticalStretch()
  updateReadout()
})
stretchResetButton.addEventListener('click', () => {
  verticalStretchInput.value = '1'
  applyVerticalStretch()
  updateReadout()
})
pickForcepsButton.addEventListener('click', () => setPickTarget('forceps'))
pickLightButton.addEventListener('click', () => setPickTarget('light'))
pickLeftTipButton.addEventListener('click', () => setPickTarget('left_tip'))
pickRightTipButton.addEventListener('click', () => setPickTarget('right_tip'))
pickLeftShadowButton.addEventListener('click', () => setPickTarget('left_shadow'))
pickRightShadowButton.addEventListener('click', () => setPickTarget('right_shadow'))
freezeButton.addEventListener('click', toggleFreeze)

// ---- Focus mode (in-page, not real browser/OS fullscreen) -----------------
// A CSS fixed overlay that covers the viewport without ever calling the
// Fullscreen API — no browser chrome changes, no permission prompt, stays a
// normal window. All the same controls stay available (ring/stretch, trocars,
// manual points), just laid out to fit without scrolling.
let manualPointsWasOpenBeforeFocusMode = false

function enterFocusMode() {
  if (labelingStage.classList.contains('focus-mode')) return
  labelingStage.classList.add('focus-mode')
  focusModeButton.textContent = 'Exit focus mode'
  manualPointsWasOpenBeforeFocusMode = manualPointsDetails.open
  manualPointsDetails.open = true
  document.body.style.overflow = 'hidden'
}

function exitFocusMode() {
  if (!labelingStage.classList.contains('focus-mode')) return
  labelingStage.classList.remove('focus-mode')
  focusModeButton.textContent = 'Focus mode'
  manualPointsDetails.open = manualPointsWasOpenBeforeFocusMode
  document.body.style.overflow = ''
}

function toggleFocusMode() {
  if (labelingStage.classList.contains('focus-mode')) {
    exitFocusMode()
  } else {
    enterFocusMode()
  }
}

focusModeButton.addEventListener('click', toggleFocusMode)

// ---- Keyboard shortcuts ----------------------------------------------------
// Number keys jump straight to a pick target (auto-expanding the manual
// points section if needed) — the whole point being to avoid reaching for a
// mouse/button while your eyes are on the image.
const HOTKEYS = {
  1: 'forceps',
  2: 'light',
  3: 'left_tip',
  4: 'right_tip',
  5: 'left_shadow',
  6: 'right_shadow',
}

document.addEventListener('keydown', (event) => {
  const tag = event.target.tagName
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return

  if (HOTKEYS[event.key]) {
    event.preventDefault()
    setPickTarget(HOTKEYS[event.key])
    return
  }
  if (event.key === 'f' || event.key === 'F') {
    event.preventDefault()
    toggleFreeze()
    return
  }
  if (event.key === 'g' || event.key === 'G') {
    event.preventDefault()
    toggleFocusMode()
    return
  }
  if (event.key === 'Escape') {
    if (labelingStage.classList.contains('focus-mode')) {
      exitFocusMode()
    } else if (pickTarget) {
      setPickTarget(null)
    }
  }
})

applyVerticalStretch()
updateReadout()
updatePointsReadout()
updateFreezeButtonState()
requestAnimationFrame(drawFrame)

loadCalibration().catch((error) => setStatus(calibrationStatus, error.message, false))
loadServiceStatus().catch((error) => {
  serviceStatus.textContent = error.message
  serviceStatus.className = 'service-status error'
})
