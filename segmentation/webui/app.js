const $ = (selector) => document.querySelector(selector)
const $$ = (selector) => [...document.querySelectorAll(selector)]

const state = { runs: [], selectedId: null, filter: 'all', kind: 'training', timer: null }
const activeStatuses = new Set(['queued', 'running', 'cancelling'])

function escapeHtml(value) {
  const div = document.createElement('div')
  div.textContent = String(value)
  return div.innerHTML
}

function titleCase(value) { return String(value).replaceAll('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase()) }
function kindLabel(kind) { return kind === 'training' ? 'Model training' : kind === 'synthetic' ? 'Synthetic data' : kind === 'dataset_split' ? 'Dataset split' : kind === 'prediction' ? 'Media prediction' : 'Video mask' }
function kindCategory(kind) { return kind === 'training' ? 'model' : kind === 'synthetic' ? 'generated' : kind === 'dataset_split' ? 'split' : kind === 'prediction' ? 'inference' : 'masked' }
function formatDate(value, includeDate = false) {
  if (!value) return '—'
  const date = new Date(value)
  return new Intl.DateTimeFormat(undefined, includeDate ? { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' } : { hour: '2-digit', minute: '2-digit' }).format(date)
}
function formatDuration(start, finish) {
  if (!start) return '—'
  const seconds = Math.max(0, Math.floor((new Date(finish || Date.now()) - new Date(start)) / 1000))
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = seconds % 60
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`
}
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) }
  if (!(options.body instanceof FormData)) headers['Content-Type'] = 'application/json'
  const response = await fetch(path, { ...options, headers })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join('; ') : data.detail
    throw new Error(detail || response.statusText)
  }
  return data
}

function renderRunList() {
  const visible = state.runs.filter((run) => {
    if (state.filter === 'all') return true
    if (state.filter === 'running') return activeStatuses.has(run.status)
    return run.status === 'completed'
  })
  $('#run-list').innerHTML = visible.map((run) => `
    <button class="run-item ${run.id === state.selectedId ? 'active' : ''}" data-run-id="${run.id}">
      <span class="run-item-top"><strong>${escapeHtml(run.name)}</strong><time>${formatDate(run.created_at)}</time></span>
      <span class="run-item-meta"><span>${kindCategory(run.kind)}</span><span class="mini-status ${run.status}">${escapeHtml(run.status)}</span></span>
    </button>`).join('')
  $('#empty-runs').classList.toggle('hidden', state.runs.length > 0)
  $$('.run-item').forEach((button) => button.addEventListener('click', () => selectRun(button.dataset.runId)))
}

async function refreshRuns() {
  try {
    state.runs = await api('/api/runs')
    $('#runner-status').textContent = 'Ready'
    renderRunList()
    if (state.selectedId) await refreshDetail()
  } catch (error) {
    $('#runner-status').textContent = 'Offline'
    console.error(error)
  }
}

async function selectRun(id) {
  state.selectedId = id
  renderRunList()
  $('#welcome').classList.add('hidden')
  $('#run-detail').classList.remove('hidden')
  await refreshDetail()
}

function renderMetrics(run) {
  const entries = Object.entries(run.summary || {})
  $('#metrics').innerHTML = entries.map(([key, value]) => `<div class="metric"><small title="${escapeHtml(key)}">${escapeHtml(titleCase(key))}</small><strong>${escapeHtml(value)}</strong></div>`).join('')
}

function renderArtifacts(run) {
  const artifacts = run.artifacts || []
  const images = artifacts.filter((item) => item.type === 'image')
  const videos = artifacts.filter((item) => item.type === 'video')
  const files = artifacts.filter((item) => !['image', 'video'].includes(item.type))
  $('#artifact-count').textContent = `${artifacts.length} artifact${artifacts.length === 1 ? '' : 's'}`
  const signature = JSON.stringify([run.id, ...artifacts.map((item) => [item.path, item.size, item.type])])
  const gallery = $('#gallery')
  if (gallery.dataset.signature !== signature) {
    gallery.innerHTML = videos.map((item) => `<video src="${item.url}" controls preload="metadata" aria-label="${escapeHtml(item.name)}"></video>`).join('') + images.map((item) => `<a href="${item.url}" target="_blank" title="${escapeHtml(item.path)}"><img src="${item.url}" alt="${escapeHtml(item.name)}" loading="lazy"></a>`).join('')
    $('#files').innerHTML = files.slice(0, 40).map((item) => `<div class="file-row"><a href="${item.url}" target="_blank">${escapeHtml(item.path)}</a><small>${formatBytes(item.size)}</small></div>`).join('')
    gallery.dataset.signature = signature
  }
  $('#results-empty').classList.toggle('hidden', artifacts.length > 0 || Object.keys(run.summary || {}).length > 0)
}

function parameterValue(value) {
  if (value === null || value === '') return 'auto'
  if (Array.isArray(value)) return value.length ? value.join(', ') : 'none'
  if (typeof value === 'object') return `${value.minimum} — ${value.maximum}`
  return String(value)
}

function renderDetail(run) {
  $('#detail-kind').textContent = kindLabel(run.kind)
  $('#detail-id').textContent = run.id
  $('#detail-name').textContent = run.name
  $('#detail-status').textContent = titleCase(run.status)
  $('#detail-status-dot').className = `status-dot ${run.status}`
  $('#detail-started').textContent = formatDate(run.started_at, true)
  $('#detail-duration').textContent = formatDuration(run.started_at, run.finished_at)
  const progress = run.progress ?? 0
  $('#progress-value').textContent = run.progress === null ? '—' : `${Math.round(progress)}%`
  $('#progress-bar').style.width = `${progress}%`
  $('#cancel-button').classList.toggle('hidden', !activeStatuses.has(run.status))
  $('#cancel-button').disabled = run.status === 'cancelling'
  $('#parameter-list').innerHTML = Object.entries(run.parameters).map(([key, value]) => `<div><dt>${escapeHtml(titleCase(key))}</dt><dd>${escapeHtml(parameterValue(value))}</dd></div>`).join('')
  const consoleEl = $('#console-log')
  const wasNearBottom = consoleEl.scrollHeight - consoleEl.scrollTop - consoleEl.clientHeight < 50
  consoleEl.textContent = run.log || 'Waiting for output…'
  if (wasNearBottom) consoleEl.scrollTop = consoleEl.scrollHeight
  renderMetrics(run)
  renderArtifacts(run)
}

async function refreshDetail() {
  if (!state.selectedId) return
  try { renderDetail(await api(`/api/runs/${state.selectedId}`)) } catch (error) { console.error(error) }
}

function setKind(kind) {
  state.kind = kind
  $$('.kind-option').forEach((button) => button.classList.toggle('active', button.dataset.kind === kind))
  $('#training-fields').classList.toggle('hidden', kind !== 'training')
  $('#synthetic-fields').classList.toggle('hidden', kind !== 'synthetic')
  $('#dataset-split-fields').classList.toggle('hidden', kind !== 'dataset_split')
  $('#prediction-fields').classList.toggle('hidden', kind !== 'prediction')
  $('#video-mask-fields').classList.toggle('hidden', kind !== 'video_mask')
  $('#form-error').textContent = ''
}

function refreshDatasetOptions() {
  const runs = state.runs.filter((run) => ['synthetic', 'dataset_split'].includes(run.kind) && run.status === 'completed')
  const options = runs.map((run) => `<option value="${run.id}">${escapeHtml(run.name)} · ${run.kind === 'synthetic' ? 'generated' : 'split'} · ${formatDate(run.created_at, true)}</option>`).join('')
  $('#run-form').elements.dataset_run_id.innerHTML = `<option value="">Use dataset config below</option>${options}`
  $('#run-form').elements.source_run_id.innerHTML = `<option value="">Select a completed dataset run</option>${options}`
  const modelOptions = state.runs.filter((run) => run.kind === 'training' && run.status === 'completed').map((run) => `<option value="${run.id}">${escapeHtml(run.name)} · ${formatDate(run.created_at, true)}</option>`).join('')
  $('#run-form').elements.model_run_id.innerHTML = `<option value="">Select a completed training run</option>${modelOptions}`
  const maskedOptions = state.runs.filter((run) => run.kind === 'video_mask' && run.status === 'completed').map((run) => `<option value="${run.id}">${escapeHtml(run.name)} · ${formatDate(run.created_at, true)}</option>`).join('')
  $('#run-form').elements.masked_video_run_id.innerHTML = `<option value="">Upload media or use a source path below</option>${maskedOptions}`
}

function openDialog(kind = 'training', sourceRun = null) {
  $('#run-form').reset()
  refreshDatasetOptions()
  setKind(kind)
  if (sourceRun) populateForm(sourceRun)
  $('#run-dialog').showModal()
}

function populateForm(run) {
  const form = $('#run-form')
  setKind(run.kind)
  form.elements.name.value = `${run.name} copy`
  for (const [key, value] of Object.entries(run.parameters)) {
    if (run.kind === 'dataset_split' && key === 'seed') {
      form.elements.split_seed.value = value
      continue
    }
    if (run.kind === 'prediction') {
      const predictionFields = {
        source: 'prediction_source', confidence: 'prediction_confidence', max_detections: 'prediction_max_detections',
        imgsz: 'prediction_imgsz', device: 'prediction_device', preprocess_preset: 'prediction_preprocess_preset',
        scene_filter: 'prediction_scene_filter', temporal_filter: 'prediction_temporal_filter',
      }
      const field = form.elements[predictionFields[key]]
      if (field) {
        if (field.type === 'checkbox') field.checked = Boolean(value)
        else field.value = value ?? ''
        continue
      }
    }
    if (run.kind === 'video_mask') {
      const maskFields = {
        source: 'mask_source', circle: 'mask_circle', inner_threshold: 'mask_inner_threshold', probe: 'mask_probe',
        radius: 'mask_radius', track: 'mask_track', smooth: 'mask_smooth', erode: 'mask_erode', threshold: 'mask_threshold',
        size: 'mask_size', crf: 'mask_crf',
      }
      const field = form.elements[maskFields[key]]
      if (field) {
        if (field.type === 'checkbox') field.checked = Boolean(value)
        else field.value = value ?? ''
        continue
      }
    }
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      if (form.elements[`${key}_min`]) form.elements[`${key}_min`].value = value.minimum
      if (form.elements[`${key}_max`]) form.elements[`${key}_max`].value = value.maximum
    } else if (form.elements[key]) {
      if (form.elements[key].type === 'checkbox') form.elements[key].checked = Boolean(value)
      else form.elements[key].value = Array.isArray(value) ? value.join(', ') : value ?? ''
    }
  }
}

function readNumber(form, name) { return Number(form.elements[name].value) }
function readRange(form, name) { return { minimum: readNumber(form, `${name}_min`), maximum: readNumber(form, `${name}_max`) } }
function readParameters(form, uploadedSource = null) {
  if (state.kind === 'training') return {
    model: form.elements.model.value.trim(), config: form.elements.config.value.trim(),
    epochs: readNumber(form, 'epochs'), imgsz: readNumber(form, 'imgsz'), batch: readNumber(form, 'batch'),
    patience: readNumber(form, 'patience'), device: form.elements.device.value.trim() || null,
    preprocess_preset: form.elements.preprocess_preset.value.trim() || null,
    rebuild_preprocessed: form.elements.rebuild_preprocessed.checked,
    dataset_run_id: form.elements.dataset_run_id.value || null,
  }
  if (state.kind === 'dataset_split') return {
    source_run_id: form.elements.source_run_id.value,
    train_ratio: readNumber(form, 'train_ratio'),
    seed: readNumber(form, 'split_seed'),
  }
  if (state.kind === 'prediction') return {
    model_run_id: form.elements.model_run_id.value,
    masked_video_run_id: form.elements.masked_video_run_id.value || null,
    source: uploadedSource || form.elements.prediction_source.value.trim(),
    confidence: readNumber(form, 'prediction_confidence'),
    max_detections: readNumber(form, 'prediction_max_detections'),
    imgsz: readNumber(form, 'prediction_imgsz'),
    device: form.elements.prediction_device.value.trim() || null,
    preprocess_preset: form.elements.prediction_preprocess_preset.value.trim() || null,
    scene_filter: form.elements.prediction_scene_filter.checked,
    temporal_filter: form.elements.prediction_temporal_filter.checked,
  }
  if (state.kind === 'video_mask') return {
    source: uploadedSource || form.elements.mask_source.value.trim(),
    circle: form.elements.mask_circle.value,
    inner_threshold: readNumber(form, 'mask_inner_threshold'),
    probe: readNumber(form, 'mask_probe'), radius: readNumber(form, 'mask_radius'),
    track: form.elements.mask_track.checked, smooth: readNumber(form, 'mask_smooth'),
    erode: readNumber(form, 'mask_erode'), threshold: readNumber(form, 'mask_threshold'),
    size: readNumber(form, 'mask_size'), crf: readNumber(form, 'mask_crf'),
  }
  const seed = form.elements.seed.value.trim()
  const csv = (name) => form.elements[name].value.split(',').map((value) => value.trim()).filter(Boolean)
  return {
    count: readNumber(form, 'count'), preview: readNumber(form, 'preview'), width: readNumber(form, 'width'), height: readNumber(form, 'height'),
    val_fraction: readNumber(form, 'val_fraction'), workers: readNumber(form, 'workers'), seed: seed === '' ? null : Number(seed), prefix: form.elements.prefix.value.trim(),
    backgrounds: csv('backgrounds'), background_rotation: readNumber(form, 'background_rotation'), image_rotations: csv('image_rotations').map(Number),
    axis_roll: readNumber(form, 'axis_roll'), shadow_axis_roll: readNumber(form, 'shadow_axis_roll'), circular_mask: form.elements.circular_mask.checked,
    shadow_scale: readRange(form, 'shadow_scale'), tip_scale: readRange(form, 'tip_scale'), shadow_opacity: readRange(form, 'shadow_opacity'),
    shadow_blur: readRange(form, 'shadow_blur'), forceps_opacity: readRange(form, 'forceps_opacity'), forceps_blur: readRange(form, 'forceps_blur'),
  }
}

$('#run-form').addEventListener('submit', async (event) => {
  event.preventDefault()
  const form = event.currentTarget
  const button = $('#launch-button')
  button.disabled = true
  $('#form-error').textContent = ''
  try {
    let uploadedSource = null
    if (state.kind === 'prediction' || state.kind === 'video_mask') {
      const useMaskedRun = state.kind === 'prediction' && form.elements.masked_video_run_id.value
      const media = state.kind === 'prediction' ? form.elements.media_file.files[0] : form.elements.mask_media_file.files[0]
      if (media && !useMaskedRun) {
        button.querySelector('span').textContent = 'Uploading…'
        const uploadBody = new FormData()
        uploadBody.append('media', media)
        uploadedSource = (await api('/api/uploads', { method: 'POST', body: uploadBody })).source
      }
    }
    button.querySelector('span').textContent = 'Launching…'
    const run = await api('/api/runs', { method: 'POST', body: JSON.stringify({ kind: state.kind, name: form.elements.name.value, parameters: readParameters(form, uploadedSource) }) })
    $('#run-dialog').close()
    await refreshRuns()
    await selectRun(run.id)
  } catch (error) { $('#form-error').textContent = error.message } finally { button.disabled = false; button.querySelector('span').textContent = 'Launch run' }
})

$('#cancel-button').addEventListener('click', async () => {
  if (!state.selectedId || !confirm('Stop this run? Partial artifacts will be kept.')) return
  try { renderDetail(await api(`/api/runs/${state.selectedId}/cancel`, { method: 'POST' })) } catch (error) { alert(error.message) }
})
$('#rerun-button').addEventListener('click', async () => { if (state.selectedId) openDialog(state.runs.find((run) => run.id === state.selectedId)?.kind, await api(`/api/runs/${state.selectedId}`)) })
$('#copy-log').addEventListener('click', async () => { await navigator.clipboard.writeText($('#console-log').textContent); $('#copy-log').textContent = 'Copied'; setTimeout(() => { $('#copy-log').textContent = 'Copy' }, 1000) })
$('#new-run-button').addEventListener('click', () => openDialog())
$('#welcome-new-run').addEventListener('click', () => openDialog())
$$('[data-quick-kind]').forEach((button) => button.addEventListener('click', () => openDialog(button.dataset.quickKind)))
$$('.kind-option').forEach((button) => button.addEventListener('click', () => setKind(button.dataset.kind)))
$('#run-form').elements.dataset_run_id.addEventListener('change', (event) => {
  if (!event.target.value) return
  const form = $('#run-form')
  if (form.elements.model.value === 'yolo11n-seg.pt') form.elements.model.value = 'yolo11n-pose.pt'
  if (form.elements.config.value === 'configs/forceps_seg.yaml') form.elements.config.value = 'configs/forceps_pose.yaml'
})
$$('.dialog-close, .dialog-cancel').forEach((button) => button.addEventListener('click', () => $('#run-dialog').close()))
$$('.filter').forEach((button) => button.addEventListener('click', () => { state.filter = button.dataset.filter; $$('.filter').forEach((item) => item.classList.toggle('active', item === button)); renderRunList() }))

refreshRuns()
state.timer = setInterval(refreshRuns, 2000)
