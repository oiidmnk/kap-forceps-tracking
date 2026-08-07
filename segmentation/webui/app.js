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
function kindLabel(kind) { return { training: 'Pose training', segmentation_training: 'Segmentation training', synthetic: 'Synthetic data', dataset_split: 'Dataset split', prediction: 'Media prediction', video_mask: 'Video mask', classical_roi: 'Classical ROI' }[kind] || titleCase(kind) }
function kindCategory(kind) { return { training: 'pose model', segmentation_training: 'segment model', synthetic: 'generated', dataset_split: 'split', prediction: 'inference', video_mask: 'masked', classical_roi: 'opencv' }[kind] || kind }
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
    const selectedWasActive = activeStatuses.has(state.runs.find((run) => run.id === state.selectedId)?.status)
    state.runs = await api('/api/runs')
    $('#runner-status').textContent = 'Ready'
    renderRunList()
    const selectedIsActive = activeStatuses.has(state.runs.find((run) => run.id === state.selectedId)?.status)
    if (state.selectedId && (selectedWasActive || selectedIsActive)) await refreshDetail()
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
  $('#delete-button').classList.toggle('hidden', activeStatuses.has(run.status))
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
  const sections = {
    training: $('#training-fields'),
    segmentation_training: $('#segmentation-training-fields'),
    synthetic: $('#synthetic-fields'),
    dataset_split: $('#dataset-split-fields'),
    prediction: $('#prediction-fields'),
    video_mask: $('#video-mask-fields'),
    classical_roi: $('#classical-roi-fields'),
  }
  for (const [sectionKind, section] of Object.entries(sections)) {
    const active = sectionKind === kind
    section.classList.toggle('hidden', !active)
    section.querySelectorAll('input, select, textarea').forEach((control) => { control.disabled = !active })
  }
  syncStartingModelField()
  $('#form-error').textContent = ''
}

function refreshDatasetOptions() {
  const runs = state.runs.filter((run) => ['synthetic', 'dataset_split'].includes(run.kind) && run.status === 'completed')
  const renderDatasetOptions = (items) => items.map((run) => `<option value="${run.id}">${escapeHtml(run.name)} · ${run.dataset_format || 'pose'} · ${formatDate(run.created_at, true)}</option>`).join('')
  $('#run-form').elements.dataset_run_id.innerHTML = `<option value="">Use dataset config below</option>${renderDatasetOptions(runs.filter((run) => (run.dataset_format || 'pose') === 'pose'))}`
  $('#run-form').elements.segmentation_dataset_run_id.innerHTML = `<option value="">Use dataset config below</option>${renderDatasetOptions(runs.filter((run) => run.dataset_format === 'segment'))}`
  $('#run-form').elements.source_run_id.innerHTML = `<option value="">Select a completed dataset run</option>${renderDatasetOptions(runs)}`
  const modelOptions = state.runs.filter((run) => run.kind === 'training' && run.status === 'completed').map((run) => `<option value="${run.id}">${escapeHtml(run.name)} · ${formatDate(run.created_at, true)}</option>`).join('')
  $('#run-form').elements.starting_model_run_id.innerHTML = `<option value="">Use checkpoint or model YAML below</option>${modelOptions}`
  $('#run-form').elements.model_run_id.innerHTML = `<option value="">Select a completed pose training run</option>${modelOptions}`
  const segmentationOptions = state.runs.filter((run) => run.kind === 'segmentation_training' && run.status === 'completed').map((run) => `<option value="${run.id}">${escapeHtml(run.name)} · ${formatDate(run.created_at, true)}</option>`).join('')
  $('#run-form').elements.segmentation_starting_model_run_id.innerHTML = `<option value="">Use checkpoint below</option>${segmentationOptions}`
  $('#run-form').elements.segmentation_model_run_id.innerHTML = `<option value="">Run pose on the full frame</option>${segmentationOptions}`
  const maskedOptions = state.runs.filter((run) => run.kind === 'video_mask' && run.status === 'completed').map((run) => `<option value="${run.id}">${escapeHtml(run.name)} · ${formatDate(run.created_at, true)}</option>`).join('')
  $('#run-form').elements.masked_video_run_id.innerHTML = `<option value="">Upload media or use a source path below</option>${maskedOptions}`
  $('#run-form').elements.realism_video_run_id.innerHTML = `<option value="">Upload a video or use a source path below</option>${maskedOptions}`
}

function openDialog(kind = 'training', sourceRun = null) {
  $('#run-form').reset()
  $('#run-form').elements.model.disabled = false
  $('#run-form').elements.segmentation_model.disabled = false
  refreshDatasetOptions()
  setKind(kind)
  if (sourceRun) populateForm(sourceRun)
  syncStartingModelField()
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
        segmentation_confidence: 'prediction_segmentation_confidence', roi_padding: 'prediction_roi_padding',
      }
      const field = form.elements[predictionFields[key]]
      if (field) {
        if (field.type === 'checkbox') field.checked = Boolean(value)
        else field.value = value ?? ''
        continue
      }
    }
    if (run.kind === 'segmentation_training') {
      const trainingFields = {
        model: 'segmentation_model', config: 'segmentation_config', starting_model_run_id: 'segmentation_starting_model_run_id',
        dataset_run_id: 'segmentation_dataset_run_id', epochs: 'segmentation_epochs', imgsz: 'segmentation_imgsz',
        batch: 'segmentation_batch', patience: 'segmentation_patience', device: 'segmentation_device',
        preprocess_preset: 'segmentation_preprocess_preset', rebuild_preprocessed: 'segmentation_rebuild_preprocessed',
      }
      const field = form.elements[trainingFields[key]]
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
    if (run.kind === 'classical_roi') {
      const classicalFields = {
        source: 'classical_source', forceps_max_saturation: 'classical_forceps_max_saturation',
        forceps_max_value: 'classical_forceps_max_value', canny_low: 'classical_canny_low',
        canny_high: 'classical_canny_high', temporal_smoothing: 'classical_temporal_smoothing',
        temporal_alpha: 'classical_temporal_alpha', temporal_max_gap: 'classical_temporal_max_gap',
      }
      const field = form.elements[classicalFields[key]]
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
function syncStartingModelField() {
  const form = $('#run-form')
  form.elements.model.disabled = state.kind !== 'training' || Boolean(form.elements.starting_model_run_id.value)
  form.elements.segmentation_model.disabled = state.kind !== 'segmentation_training' || Boolean(form.elements.segmentation_starting_model_run_id.value)
}
function readParameters(form, uploadedSource = null) {
  if (state.kind === 'training') return {
    model: form.elements.model.value.trim(), config: form.elements.config.value.trim(),
    starting_model_run_id: form.elements.starting_model_run_id.value || null,
    epochs: readNumber(form, 'epochs'), imgsz: readNumber(form, 'imgsz'), batch: readNumber(form, 'batch'),
    patience: readNumber(form, 'patience'), device: form.elements.device.value.trim() || null,
    preprocess_preset: form.elements.preprocess_preset.value.trim() || null,
    rebuild_preprocessed: form.elements.rebuild_preprocessed.checked,
    dataset_run_id: form.elements.dataset_run_id.value || null,
  }
  if (state.kind === 'segmentation_training') return {
    model: form.elements.segmentation_model.value.trim(), config: form.elements.segmentation_config.value.trim(),
    starting_model_run_id: form.elements.segmentation_starting_model_run_id.value || null,
    epochs: readNumber(form, 'segmentation_epochs'), imgsz: readNumber(form, 'segmentation_imgsz'), batch: readNumber(form, 'segmentation_batch'),
    patience: readNumber(form, 'segmentation_patience'), device: form.elements.segmentation_device.value.trim() || null,
    preprocess_preset: form.elements.segmentation_preprocess_preset.value.trim() || null,
    rebuild_preprocessed: form.elements.segmentation_rebuild_preprocessed.checked,
    dataset_run_id: form.elements.segmentation_dataset_run_id.value || null,
  }
  if (state.kind === 'dataset_split') return {
    source_run_id: form.elements.source_run_id.value,
    train_ratio: readNumber(form, 'train_ratio'),
    seed: readNumber(form, 'split_seed'),
  }
  if (state.kind === 'prediction') return {
    model_run_id: form.elements.model_run_id.value,
    segmentation_model_run_id: form.elements.segmentation_model_run_id.value || null,
    masked_video_run_id: form.elements.masked_video_run_id.value || null,
    source: uploadedSource || form.elements.prediction_source.value.trim(),
    confidence: readNumber(form, 'prediction_confidence'),
    max_detections: readNumber(form, 'prediction_max_detections'),
    imgsz: readNumber(form, 'prediction_imgsz'),
    device: form.elements.prediction_device.value.trim() || null,
    preprocess_preset: form.elements.prediction_preprocess_preset.value.trim() || null,
    scene_filter: form.elements.prediction_scene_filter.checked,
    temporal_filter: form.elements.prediction_temporal_filter.checked,
    segmentation_confidence: readNumber(form, 'prediction_segmentation_confidence'),
    roi_padding: readNumber(form, 'prediction_roi_padding'),
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
  if (state.kind === 'classical_roi') return {
    source: uploadedSource || form.elements.classical_source.value.trim(),
    forceps_max_saturation: readNumber(form, 'classical_forceps_max_saturation'),
    forceps_max_value: readNumber(form, 'classical_forceps_max_value'),
    canny_low: readNumber(form, 'classical_canny_low'),
    canny_high: readNumber(form, 'classical_canny_high'),
    temporal_smoothing: form.elements.classical_temporal_smoothing.checked,
    temporal_alpha: readNumber(form, 'classical_temporal_alpha'),
    temporal_max_gap: readNumber(form, 'classical_temporal_max_gap'),
  }
  const seed = form.elements.seed.value.trim()
  const csv = (name) => form.elements[name].value.split(',').map((value) => value.trim()).filter(Boolean)
  const realismVideoRunId = form.elements.realism_video_run_id.value
  return {
    label_format: form.elements.label_format.value,
    count: readNumber(form, 'count'), preview: readNumber(form, 'preview'), width: readNumber(form, 'width'), height: readNumber(form, 'height'),
    val_fraction: readNumber(form, 'val_fraction'), workers: readNumber(form, 'workers'), seed: seed === '' ? null : Number(seed), prefix: form.elements.prefix.value.trim(),
    backgrounds: csv('backgrounds'),
    realism_video_run_id: realismVideoRunId || null,
    realism_video: realismVideoRunId ? null : uploadedSource || form.elements.realism_video.value.trim() || null,
    video_backgrounds: readNumber(form, 'video_backgrounds'), video_samples: readNumber(form, 'video_samples'),
    video_degradation: form.elements.video_degradation.checked,
    background_rotation: readNumber(form, 'background_rotation'), image_rotations: csv('image_rotations').map(Number),
    axis_roll: readNumber(form, 'axis_roll'), shadow_axis_roll: readNumber(form, 'shadow_axis_roll'), circular_mask: form.elements.circular_mask.checked,
    shadow_scale: readRange(form, 'shadow_scale'), tip_scale: readRange(form, 'tip_scale'), shadow_opacity: readRange(form, 'shadow_opacity'),
    shadow_blur: readRange(form, 'shadow_blur'), forceps_blur: readRange(form, 'forceps_blur'),
    forceps_contrast: readRange(form, 'forceps_contrast'), shadow_correlation: readNumber(form, 'shadow_correlation'),
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
    if (['prediction', 'video_mask', 'synthetic', 'classical_roi'].includes(state.kind)) {
      const useMaskedRun = state.kind === 'prediction'
        ? form.elements.masked_video_run_id.value
        : state.kind === 'synthetic' && form.elements.realism_video_run_id.value
      const media = state.kind === 'prediction'
        ? form.elements.media_file.files[0]
        : state.kind === 'video_mask'
          ? form.elements.mask_media_file.files[0]
          : state.kind === 'synthetic'
            ? form.elements.realism_video_file.files[0]
            : form.elements.classical_media_file.files[0]
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
$('#delete-button').addEventListener('click', async () => {
  if (!state.selectedId) return
  const run = state.runs.find((item) => item.id === state.selectedId)
  if (!confirm(`Delete “${run?.name || state.selectedId}”? All artifacts and logs for this run will be permanently removed.`)) return
  const button = $('#delete-button')
  button.disabled = true
  try {
    await api(`/api/runs/${state.selectedId}`, { method: 'DELETE' })
    state.selectedId = null
    $('#run-detail').classList.add('hidden')
    $('#welcome').classList.remove('hidden')
    await refreshRuns()
  } catch (error) {
    alert(error.message)
  } finally {
    button.disabled = false
  }
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
$('#run-form').elements.starting_model_run_id.addEventListener('change', syncStartingModelField)
$('#run-form').elements.segmentation_starting_model_run_id.addEventListener('change', syncStartingModelField)
$$('.dialog-close, .dialog-cancel').forEach((button) => button.addEventListener('click', () => $('#run-dialog').close()))
$$('.filter').forEach((button) => button.addEventListener('click', () => { state.filter = button.dataset.filter; $$('.filter').forEach((item) => item.classList.toggle('active', item === button)); renderRunList() }))

async function pollRuns() {
  await refreshRuns()
  state.timer = setTimeout(pollRuns, 2000)
}

pollRuns()
