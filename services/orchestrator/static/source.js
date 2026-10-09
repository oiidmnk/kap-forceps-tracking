// Source selection + eye-circle editor. Talks to the detector through /api/source/*.
(() => {
  const $ = (sel) => document.querySelector(sel)
  const select = $('#source-select'), custom = $('#source-custom')
  const statusEl = $('#source-status')
  const canvas = $('#circle-canvas'), ctx = canvas.getContext('2d')
  const readout = $('#circle-readout')
  const applyBtn = $('#circle-apply'), autoBtn = $('#circle-auto'), clearBtn = $('#circle-clear')
  const normView = $('#normalized-view')
  const detectorUrl = $('#detector-url')

  let frame = null            // ImageBitmap of the last raw frame
  let frameSize = null        // [w, h] of the raw source frame
  let circle = null           // working circle {cx, cy, r} in raw-frame px
  let savedCircle = null      // what the detector currently uses
  let drag = null, dirty = false
  let loaded = false

  const setStatus = (msg, ok = true) => { statusEl.textContent = msg; statusEl.className = `status ${ok ? 'ok' : 'error'}` }
  async function api(path, opts) {
    const r = await fetch(path, opts)
    const body = await r.json().catch(() => ({}))
    if (!r.ok) throw new Error(body.detail || r.statusText)
    return body
  }
  const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

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
    if (current && ![...select.options].some((o) => o.value === current)) add(current, current, 'Current')
    select.value = current || ''
    custom.value = ''
  }

  async function loadSource() {
    const data = await api('/api/source')
    detectorUrl.value = data.detector_url
    if (data.error) { setStatus(data.error, false); return }
    fillSources(data.sources, data.config.source)
    savedCircle = data.config.circle
    if (!dirty) circle = savedCircle ? { ...savedCircle } : null
    updateReadout()
    setStatus(data.config.source ? `source: ${data.config.source} · background: ${data.config.background}` : 'no source selected', !!data.config.source)
    if (!loaded) { loaded = true; normView.src = '/api/source/live.mjpg' }
  }

  $('#source-use').addEventListener('click', async () => {
    const value = custom.value.trim() || select.value
    if (!value) return
    try { await post('/api/source/config', { source: value }); setStatus('source switched, background relearns', true); await loadSource() }
    catch (e) { setStatus(e.message, false) }
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
      ctx.restore()
    }
  }
  function updateReadout() {
    readout.textContent = circle ? `c=(${circle.cx.toFixed(0)}, ${circle.cy.toFixed(0)})  r=${circle.r.toFixed(0)}` : 'no circle'
    applyBtn.disabled = !(dirty && circle)
    draw()
  }

  canvas.addEventListener('pointerdown', (ev) => {
    if (!frameSize) return
    canvas.setPointerCapture(ev.pointerId)
    const p = toRaw(ev)
    if (circle) {
      const d = Math.hypot(p.x - circle.cx, p.y - circle.cy)
      if (Math.abs(d - circle.r) < 14 * p.k) { drag = { mode: 'resize' }; return }
      if (d < circle.r) { drag = { mode: 'move', dx: circle.cx - p.x, dy: circle.cy - p.y }; return }
    }
    circle = { cx: p.x, cy: p.y, r: 20 }; drag = { mode: 'resize' }; dirty = true; updateReadout()
  })
  canvas.addEventListener('pointermove', (ev) => {
    if (!drag) return
    const p = toRaw(ev)
    if (drag.mode === 'move') { circle.cx = p.x + drag.dx; circle.cy = p.y + drag.dy }
    else circle.r = Math.max(20, Math.hypot(p.x - circle.cx, p.y - circle.cy))
    dirty = true; updateReadout()
  })
  const end = () => { drag = null }
  canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end)
  canvas.addEventListener('wheel', (ev) => {
    if (!circle) return
    ev.preventDefault()
    circle.r = Math.max(20, circle.r * (ev.deltaY < 0 ? 1.03 : 0.97)); dirty = true; updateReadout()
  }, { passive: false })

  applyBtn.addEventListener('click', async () => {
    try {
      const res = await post('/api/source/config', { circle })
      savedCircle = res.config.circle; dirty = false; updateReadout()
      setStatus('circle applied · eye calibration set to the normalised square · background relearns', true)
    } catch (e) { setStatus(e.message, false) }
  })
  autoBtn.addEventListener('click', async () => {
    try { circle = await api('/api/source/autocircle'); dirty = true; updateReadout() }
    catch (e) { setStatus(e.message, false) }
  })
  clearBtn.addEventListener('click', async () => {
    try { await post('/api/source/config', { clear_circle: true }); circle = null; savedCircle = null; dirty = false; updateReadout(); setStatus('no circle: centre crop of the frame', true) }
    catch (e) { setStatus(e.message, false) }
  })

  // ---- preview loop ---------------------------------------------------------
  async function refreshFrame() {
    try {
      const r = await fetch('/api/source/preview.jpg', { cache: 'no-store' })
      if (r.ok) {
        const w = Number(r.headers.get('X-Frame-Width')), h = Number(r.headers.get('X-Frame-Height'))
        const bmp = await createImageBitmap(await r.blob())
        frame = bmp
        if (!frameSize || frameSize[0] !== w || frameSize[1] !== h) { frameSize = [w, h]; canvas.height = Math.round(canvas.width * h / w) }
        draw()
      }
    } catch (_) { /* detector offline: keep last frame */ }
    setTimeout(refreshFrame, drag ? 400 : 700)
  }
  loadSource().catch((e) => setStatus(e.message, false))
  setInterval(() => { if (!dirty && !drag) loadSource().catch(() => {}) }, 5000)
  refreshFrame()
})()
