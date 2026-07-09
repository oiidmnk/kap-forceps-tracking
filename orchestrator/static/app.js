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
    setStatus(calibrationStatus, 'Calibration saved.', true)
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

loadCalibration().catch((error) => setStatus(calibrationStatus, error.message, false))
loadServiceStatus().catch((error) => {
  serviceStatus.textContent = error.message
  serviceStatus.className = 'service-status error'
})
