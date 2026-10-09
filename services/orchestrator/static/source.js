// Source selection + eye-circle editor. Talks to the detector through /api/source/*.
(() => {
  const $ = (sel) => document.querySelector(sel)
  const select = $('#source-select')
  const statusEl = $('#source-status')
  const canvas = $('#circle-canvas'), ctx = canvas.getContext('2d')
  const readout = $('#circle-readout')
  const applyBtn = $('#circle-apply'), autoBtn = $('#circle-auto'), clearBtn = $('#circle-clear')
  const drawBtn = $('#circle-draw'), revertBtn = $('#circle-revert')
  const normView = $('#normalized-view')
  const detectorUrl = $('#detector-url')
  const uploadForm = $('#source-upload-form'), uploadInput = $('#source-upload')
  const uploadButton = $('#source-upload-button'), uploadStatus = $('#source-upload-status')

  document.querySelectorAll('[data-service-port]').forEach((link) => {
    const url = new URL(link.href)
    url.hostname = location.hostname
    url.protocol = location.protocol
    link.href = url.href
  })

  let frame = null            // ImageBitmap of the last raw frame
  let frameSize = null        // [w, h] of the raw source frame
  let circle = null           // working circle {cx, cy, r} in raw-frame px
  let savedCircle = null      // what the detector currently uses
  let drag = null, dirty = false
  let drawing = false
  let editorRevision = 0
  let sourceRequestId = 0
  let savingCircle = false
  let loaded = false
  let switchingSource = false

  const setStatus = (msg, ok = true) => { statusEl.textContent = msg; statusEl.className = `status ${ok ? 'ok' : 'error'}` }
  async function api(path, opts) {
    const r = await fetch(path, opts)
    const body = await r.json().catch(() => ({}))
    if (!r.ok) throw new Error(body.detail || r.statusText)
    return body
  }
  const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  const sourceLabel = (source) => source.includes('://') || /^\d+$/.test(source) ? source : source.split(/[\\/]/).pop()

  // ---- source list ------------------------------------------------------
  function fillSources(sources, current) {
    select.innerHTML = ''
    const add = (value, label, group) => {
      let g = select.querySelector(`optgroup[label="${group}"]`)
      if (!g) { g = document.createElement('optgroup'); g.label = group; select.appendChild(g) }
      const o = document.createElement('option'); o.value = value; o.textContent = label; g.appendChild(o)
    }
    for (const f of sources.files || []) add(f, f.split('/').pop(), 'Video files')
    for (const c of sources.cameras || []) add(String(c.index), `Camera ${c.index}` + (c.width ? ` (${c.width}x${c.height})` : ' (in use)'), 'Capture devices')
    if (current && ![...select.options].some((o) => o.value === current)) add(current, sourceLabel(current), 'Current')
    select.value = current || ''
  }

  async function loadSource() {
    const revision = editorRevision
    const requestId = ++sourceRequestId
    const data = await api('/api/source')
    if (requestId !== sourceRequestId || revision !== editorRevision || drag || drawing || savingCircle) return
    detectorUrl.value = data.detector_url
    if (data.error) { setStatus(data.error, false); return }
    fillSources(data.sources, data.config.source)
    savedCircle = data.config.circle
    if (!dirty) circle = savedCircle ? { ...savedCircle } : null
    updateReadout()
    setStatus(data.config.source ? `source: ${sourceLabel(data.config.source)} · background: ${data.config.background}` : 'no source selected', !!data.config.source)
    if (!loaded) { loaded = true; normView.src = '/api/source/live.mjpg' }
  }

  async function useSource(value) {
    if (!value) return
    switchingSource = true
    select.disabled = true
    try { await post('/api/source/config', { source: value }); setStatus('source switched, background relearns', true); await loadSource() }
    catch (e) { setStatus(e.message, false) }
    finally { switchingSource = false; select.disabled = false }
  }
  select.addEventListener('change', () => useSource(select.value))
  uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault()
    const file = uploadInput.files[0]
    if (!file) return
    const showUploadStatus = (message, ok) => {
      uploadStatus.textContent = message
      uploadStatus.className = `status ${ok ? 'ok' : 'error'}`
    }
    if (!file.name.toLowerCase().endsWith('.mp4')) { showUploadStatus('Choose an MP4 video.', false); return }
    if (!file.size || file.size > 512 * 1024 * 1024) { showUploadStatus('Choose a non-empty MP4 up to 512 MiB.', false); return }
    uploadButton.disabled = true
    showUploadStatus('Uploading video…', true)
    try {
      const body = new FormData()
      body.append('file', file)
      await api('/api/source/upload', { method: 'POST', body })
      dirty = false; circle = null; savedCircle = null
      uploadForm.reset()
      await loadSource()
      showUploadStatus('Video uploaded. Live processing started; draw and apply the eye circle.', true)
    } catch (error) { showUploadStatus(error.message, false) }
    finally { uploadButton.disabled = false }
  })
  $('#source-reset').addEventListener('click', async () => {
    try {
      await post('/api/source/reset', {})
      setStatus('reset: source restarted, background discarded and relearning', true)
      await loadSource()
    } catch (e) { setStatus(e.message, false) }
  })
  $('#source-scan').addEventListener('click', async () => {
    setStatus('scanning capture devices…', true)
    try {
      const sources = await api('/api/source/scan')
      fillSources(sources, sources.current)
      setStatus(sources.cameras.length ? `${sources.cameras.length} device(s) found` : 'no devices found (is the detector running on the host?)', sources.cameras.length > 0)
    } catch (e) { setStatus(e.message, false) }
  })
  $('#detector-url-save').addEventListener('click', async () => {
    try { await api('/api/source/detector-url', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url: detectorUrl.value }) }); normView.src = '/api/source/live.mjpg?' + Date.now(); await loadSource() }
    catch (e) { setStatus(e.message, false) }
  })

  // ---- circle editor ------------------------------------------------------
  const scale = () => (frameSize ? canvas.width / frameSize[0] : 1)
  function toRaw(ev) {
    const rect = canvas.getBoundingClientRect()
    const k = frameSize[0] / rect.width
    return { x: (ev.clientX - rect.left) * k, y: (ev.clientY - rect.top) * k, k }
  }
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (frame) ctx.drawImage(frame, 0, 0, canvas.width, canvas.height)
    else { ctx.fillStyle = '#9aa4b2'; ctx.font = '16px sans-serif'; ctx.fillText('waiting for a frame from the detector…', 20, 40) }
    if (circle && frameSize) {
      const s = scale()
      ctx.save()
      ctx.beginPath(); ctx.rect(0, 0, canvas.width, canvas.height)
      ctx.arc(circle.cx * s, circle.cy * s, circle.r * s, 0, Math.PI * 2, true)
      ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fill('evenodd')
      ctx.beginPath(); ctx.arc(circle.cx * s, circle.cy * s, circle.r * s, 0, Math.PI * 2)
      ctx.lineWidth = 2; ctx.strokeStyle = dirty ? '#f59e0b' : '#22c55e'; ctx.stroke()
      ctx.beginPath(); ctx.arc(circle.cx * s, circle.cy * s, 4, 0, Math.PI * 2); ctx.fillStyle = ctx.strokeStyle; ctx.fill()
      const handleRadius = 6 * canvas.width / canvas.getBoundingClientRect().width
      for (const handle of handles()) {
        ctx.beginPath(); ctx.arc(handle.x * s, handle.y * s, handleRadius, 0, Math.PI * 2)
        ctx.fillStyle = '#fff'; ctx.fill(); ctx.stroke()
      }
      ctx.restore()
    }
  }
  function updateReadout() {
    readout.textContent = circle ? `c=(${circle.cx.toFixed(0)}, ${circle.cy.toFixed(0)})  r=${circle.r.toFixed(0)} · ${dirty ? 'Unsaved adjustments' : 'Applied'}` : 'no circle'
    applyBtn.disabled = !(dirty && circle) || !!drag || drawing || savingCircle
    revertBtn.disabled = !dirty && !drawing
    revertBtn.hidden = !dirty && !drawing
    drawBtn.setAttribute('aria-pressed', String(drawing))
    canvas.style.cursor = drawing || !circle ? 'crosshair' : 'default'
    draw()
  }

  function handles() {
    return [
      { x: circle.cx + circle.r, y: circle.cy, cursor: 'ew-resize' },
      { x: circle.cx - circle.r, y: circle.cy, cursor: 'ew-resize' },
      { x: circle.cx, y: circle.cy + circle.r, cursor: 'ns-resize' },
      { x: circle.cx, y: circle.cy - circle.r, cursor: 'ns-resize' },
    ]
  }
  function hitHandle(point) {
    return circle && handles().find((handle) => Math.hypot(point.x - handle.x, point.y - handle.y) <= 14 * point.k)
  }
  function markChanged() {
    editorRevision++
    dirty = circle === null || savedCircle === null ? circle !== savedCircle :
      circle.cx !== savedCircle.cx || circle.cy !== savedCircle.cy || circle.r !== savedCircle.r
    updateReadout()
  }
  function finishDrag(cancel = false) {
    if (!drag) return
    const previous = drag
    drag = null
    if (cancel || !previous.started) circle = previous.original
    if (canvas.hasPointerCapture(previous.pointerId)) canvas.releasePointerCapture(previous.pointerId)
    markChanged()
  }
  drawBtn.addEventListener('click', () => {
    finishDrag(true)
    drawing = !drawing
    editorRevision++
    updateReadout()
    canvas.focus({ preventScroll: true })
  })
  revertBtn.addEventListener('click', () => {
    finishDrag(true)
    circle = savedCircle ? { ...savedCircle } : null
    drawing = false
    markChanged()
  })
  canvas.addEventListener('pointerdown', (ev) => {
    if (!frameSize || ev.button !== 0 || drag) return
    canvas.focus({ preventScroll: true })
    const p = toRaw(ev)
    const original = circle ? { ...circle } : null
    let mode
    if (drawing || !circle) mode = 'draw'
    else if (hitHandle(p)) mode = 'resize'
    else if (Math.hypot(p.x - circle.cx, p.y - circle.cy) < circle.r) mode = 'move'
    else return
    ev.preventDefault()
    canvas.setPointerCapture(ev.pointerId)
    drag = { mode, original, start: p, pointerId: ev.pointerId, started: false }
    editorRevision++
    updateReadout()
  })
  canvas.addEventListener('pointermove', (ev) => {
    const p = frameSize && toRaw(ev)
    if (!p) return
    if (!drag) {
      const handle = hitHandle(p)
      canvas.style.cursor = drawing || !circle ? 'crosshair' : handle ? handle.cursor : Math.hypot(p.x - circle.cx, p.y - circle.cy) < circle.r ? 'move' : 'default'
      return
    }
    if (ev.pointerId !== drag.pointerId) return
    if (!drag.started && Math.hypot(p.x - drag.start.x, p.y - drag.start.y) < 3 * p.k) return
    drag.started = true
    const gain = ev.shiftKey ? 0.2 : 1
    if (drag.mode === 'draw') {
      circle = { cx: drag.start.x, cy: drag.start.y, r: Math.max(20, Math.hypot(p.x - drag.start.x, p.y - drag.start.y)) }
      drawing = false
    } else if (drag.mode === 'move') {
      circle.cx = drag.original.cx + (p.x - drag.start.x) * gain
      circle.cy = drag.original.cy + (p.y - drag.start.y) * gain
    } else {
      const distance = Math.hypot(p.x - drag.original.cx, p.y - drag.original.cy)
      const startDistance = Math.hypot(drag.start.x - drag.original.cx, drag.start.y - drag.original.cy)
      circle.r = Math.max(20, drag.original.r + (distance - startDistance) * gain)
    }
    markChanged()
  })
  canvas.addEventListener('pointerup', (ev) => { if (drag?.pointerId === ev.pointerId) finishDrag() })
  canvas.addEventListener('pointercancel', (ev) => { if (drag?.pointerId === ev.pointerId) finishDrag() })
  canvas.addEventListener('lostpointercapture', (ev) => { if (drag?.pointerId === ev.pointerId) finishDrag() })
  canvas.addEventListener('keydown', (ev) => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return
    if (ev.key === 'Escape') {
      ev.preventDefault()
      finishDrag(true)
      drawing = false
      editorRevision++
      updateReadout()
      return
    }
    if (!circle || drag || drawing) return
    const step = ev.shiftKey ? 10 : 1
    switch (ev.key) {
      case 'ArrowLeft': circle.cx -= step; break
      case 'ArrowRight': circle.cx += step; break
      case 'ArrowUp': circle.cy -= step; break
      case 'ArrowDown': circle.cy += step; break
      case '+': case '=': circle.r += 1; break
      case '-': case '−': case '_': circle.r = Math.max(20, circle.r - 1); break
      default: return
    }
    ev.preventDefault()
    markChanged()
  })

  applyBtn.addEventListener('click', async () => {
    if (!circle || drag || drawing || savingCircle) return
    const appliedCircle = { ...circle }
    const revision = editorRevision
    savingCircle = true
    sourceRequestId++
    updateReadout()
    try {
      const res = await post('/api/source/config', { circle: appliedCircle })
      savedCircle = res.config.circle
      if (revision === editorRevision && !drag && !drawing) circle = { ...savedCircle }
      markChanged()
      setStatus('circle applied · eye calibration set to the normalised square · background relearns', true)
    } catch (e) { setStatus(e.message, false) }
    finally { savingCircle = false; updateReadout() }
  })
  autoBtn.addEventListener('click', async () => {
    const revision = editorRevision
    try {
      const detectedCircle = await api('/api/source/autocircle')
      if (revision !== editorRevision || drag || savingCircle) return
      circle = detectedCircle; drawing = false; markChanged(); canvas.focus({ preventScroll: true })
    }
    catch (e) { setStatus(e.message, false) }
  })
  clearBtn.addEventListener('click', async () => {
    if (savingCircle) return
    const revision = editorRevision
    savingCircle = true
    sourceRequestId++
    updateReadout()
    try {
      await post('/api/source/config', { clear_circle: true })
      savedCircle = null
      if (revision === editorRevision && !drag) { circle = null; drawing = false }
      markChanged()
      setStatus('no circle: centre crop of the frame', true)
    }
    catch (e) { setStatus(e.message, false) }
    finally { savingCircle = false; updateReadout() }
  })

  // ---- preview loop ---------------------------------------------------------
  async function refreshFrame() {
    try {
      const r = await fetch('/api/source/preview.jpg', { cache: 'no-store' })
      if (r.ok) {
        const w = Number(r.headers.get('X-Frame-Width')), h = Number(r.headers.get('X-Frame-Height'))
        const bmp = await createImageBitmap(await r.blob())
        if (drag) bmp.close()
        else {
          if (frame) frame.close()
          frame = bmp
          if (!frameSize || frameSize[0] !== w || frameSize[1] !== h) { frameSize = [w, h]; canvas.height = Math.round(canvas.width * h / w) }
          draw()
        }
      }
    } catch (_) { /* detector offline: keep last frame */ }
    setTimeout(refreshFrame, drag ? 400 : 700)
  }
  loadSource().catch((e) => setStatus(e.message, false))
  setInterval(() => { if (!dirty && !drag && !drawing && !savingCircle && !switchingSource) loadSource().catch(() => {}) }, 5000)
  refreshFrame()
})()
