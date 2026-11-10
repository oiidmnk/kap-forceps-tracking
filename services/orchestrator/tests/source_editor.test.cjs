const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const vm = require('node:vm')

function editorHarness() {
  const elements = new Map()
  function element() {
    const listeners = new Map()
    return {
      listeners, style: {}, width: 640, height: 360,
      addEventListener(name, callback) { listeners.set(name, callback) },
      setAttribute() {}, focus() {}, setPointerCapture() {}, releasePointerCapture() {},
      hasPointerCapture() { return false },
      getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 360 } },
    }
  }
  const selectElement = (selector) => {
    if (!elements.has(selector)) elements.set(selector, element())
    return elements.get(selector)
  }
  const context = vm.createContext({
    $: selectElement,
    canvas: selectElement('#circle-canvas'),
    ctx: new Proxy({}, { get: () => () => {} }),
    readout: selectElement('#circle-readout'),
    applyBtn: selectElement('#circle-apply'),
    autoBtn: selectElement('#circle-auto'),
    clearBtn: selectElement('#circle-clear'),
    drawBtn: selectElement('#circle-draw'),
    revertBtn: selectElement('#circle-revert'),
    rollBtn: selectElement('#roll-mark'),
    rollResetBtn: selectElement('#roll-reset'),
    rollReadout: selectElement('#roll-readout'),
    frame: null, frameSize: [1920, 1080],
    circle: { cx: 600, cy: 450, r: 300 },
    savedCircle: { cx: 600, cy: 450, r: 300 },
    dirty: false, drawing: false, drag: null,
    editorRevision: 0, sourceRequestId: 0, savingCircle: false,
    marking: false, rollPoints: [],
  })
  const source = fs.readFileSync('services/orchestrator/static/source.js', 'utf8')
  vm.runInContext(source.slice(source.indexOf('  const scale ='), source.indexOf('  // ---- preview loop')), context)
  return {
    context, elements,
    fire(type, properties = {}, selector = '#circle-canvas') {
      const event = { button: 0, pointerId: 1, preventDefault() { this.prevented = true }, ...properties }
      const result = elements.get(selector).listeners.get(type)?.(event)
      return result && typeof result.then === 'function' ? result : event
    },
  }
}

test('keyboard nudges use source pixels, with coarse arrows and a minimum radius', () => {
  const editor = editorHarness()
  assert.equal(editor.fire('keydown', { key: 'ArrowRight' }).prevented, true)
  editor.fire('keydown', { key: 'ArrowUp', shiftKey: true })
  assert.equal(editor.context.circle.cx, 601)
  assert.equal(editor.context.circle.cy, 440)
  editor.fire('keydown', { key: '+', shiftKey: true })
  assert.equal(editor.context.circle.r, 301)
  editor.fire('keydown', { key: '-' })
  assert.equal(editor.context.circle.r, 300)
  editor.context.circle.r = 20
  editor.fire('keydown', { key: '-' })
  assert.equal(editor.context.circle.r, 20)
  assert.equal(editor.fire('keydown', { key: 'ArrowLeft', metaKey: true }).prevented, undefined)
  assert.equal(editor.context.circle.cx, 601)
})

test('resize keeps the grab offset and ignores movement below the drag threshold', () => {
  const editor = editorHarness()
  editor.fire('pointerdown', { clientX: 308, clientY: 150 })
  editor.fire('pointermove', { clientX: 310, clientY: 150 })
  assert.equal(editor.context.circle.r, 300)
  editor.fire('pointermove', { clientX: 318, clientY: 150 })
  assert.equal(editor.context.circle.r, 330)
  editor.fire('keydown', { key: 'Escape' })
  assert.equal(editor.context.circle.r, 300)
  assert.equal(editor.context.dirty, false)
})

test('outside clicks preserve the circle; explicit redraw and Revert are safe', () => {
  const editor = editorHarness()
  editor.fire('pointerdown', { clientX: 500, clientY: 300 })
  assert.equal(editor.context.drag, null)
  editor.fire('click', {}, '#circle-draw')
  editor.fire('pointerdown', { clientX: 500, clientY: 300 })
  editor.fire('pointermove', { clientX: 540, clientY: 300 })
  editor.fire('pointerup')
  assert.equal(editor.context.circle.cx, 1500)
  assert.equal(editor.context.circle.r, 120)
  assert.equal(editor.context.dirty, true)
  editor.fire('click', {}, '#circle-revert')
  assert.equal(editor.context.circle.cx, 600)
  assert.equal(editor.context.circle.r, 300)
  assert.equal(editor.context.dirty, false)
})

test('interrupted fine dragging retains edits and scrolling does not resize', () => {
  const editor = editorHarness()
  editor.fire('pointerdown', { clientX: 200, clientY: 150 })
  editor.fire('pointermove', { clientX: 220, clientY: 150, shiftKey: true })
  assert.equal(editor.context.circle.cx, 612)
  editor.fire('pointercancel')
  assert.equal(editor.context.circle.cx, 612)
  assert.equal(editor.elements.get('#circle-canvas').listeners.has('wheel'), false)
})

test('losing pointer capture preserves a newly drawn circle', () => {
  const editor = editorHarness()
  editor.fire('click', {}, '#circle-draw')
  editor.fire('pointerdown', { clientX: 500, clientY: 300 })
  editor.fire('pointermove', { clientX: 540, clientY: 300 })
  editor.fire('lostpointercapture')
  assert.equal(editor.context.circle.cx, 1500)
  assert.equal(editor.context.circle.r, 120)
  assert.equal(editor.context.dirty, true)
  assert.equal(editor.context.drag, null)
})

test('late Apply response preserves edits made while saving', async () => {
  const editor = editorHarness()
  let resolveSave
  editor.context.post = () => new Promise((resolve) => { resolveSave = resolve })
  editor.context.setStatus = () => {}
  editor.fire('keydown', { key: 'ArrowRight' })
  const saving = editor.elements.get('#circle-apply').listeners.get('click')()
  editor.fire('keydown', { key: 'ArrowRight' })
  resolveSave({ config: { circle: { cx: 601, cy: 450, r: 300 } } })
  await saving
  assert.equal(editor.context.circle.cx, 602)
  assert.equal(editor.context.savedCircle.cx, 601)
  assert.equal(editor.context.dirty, true)
  assert.equal(editor.context.savingCircle, false)
})

test('poll responses cannot replace an active drag or changes since the request started', async () => {
  const editor = editorHarness()
  const source = fs.readFileSync('services/orchestrator/static/source.js', 'utf8')
  vm.runInContext(source.slice(source.indexOf('  async function loadSource()'), source.indexOf('  async function useSource(')), editor.context)
  editor.context.detectorUrl = {}
  editor.context.fillSources = () => {}
  editor.context.setStatus = () => {}
  editor.context.sourceLabel = (source) => source
  editor.context.loaded = true
  let resolvePoll
  editor.context.api = () => new Promise((resolve) => { resolvePoll = resolve })
  const data = { config: { circle: { cx: 600, cy: 450, r: 300 }, source: 'video' } }
  const firstPoll = editor.context.loadSource()
  editor.fire('pointerdown', { clientX: 200, clientY: 150 })
  resolvePoll(data)
  await firstPoll
  editor.fire('pointermove', { clientX: 220, clientY: 150 })
  editor.fire('pointerup')
  assert.equal(editor.context.circle.cx, 660)
  const secondPoll = editor.context.loadSource()
  editor.fire('click', {}, '#circle-revert')
  resolvePoll({ config: { ...data.config, circle: { cx: 555, cy: 450, r: 300 } } })
  await secondPoll
  assert.equal(editor.context.circle.cx, 600)
})

test('Mark trocars turns two clicks into an eye-roll request without touching the circle', async () => {
  const editor = editorHarness()
  const posted = []
  editor.context.post = (path, body) => { posted.push([path, body]); return Promise.resolve({ roll_deg: 1.5, reference_deg: 3.3, line_deg: 4.8 }) }
  editor.context.setStatus = () => {}
  editor.fire('click', {}, '#roll-mark')
  assert.equal(editor.context.marking, true)
  editor.fire('pointerdown', { clientX: 40, clientY: 240 })
  editor.fire('pointerdown', { clientX: 280, clientY: 250 })
  await new Promise((resolve) => setImmediate(resolve))
  assert.equal(JSON.stringify(posted), JSON.stringify([['/api/eye-roll', { points: [[120, 720], [840, 750]] }]]))
  assert.equal(editor.context.marking, false)
  assert.equal(editor.context.drag, null)
  assert.equal(JSON.stringify(editor.context.circle), JSON.stringify({ cx: 600, cy: 450, r: 300 }))
  assert.equal(editor.context.dirty, false)
  assert.equal(editor.elements.get('#roll-readout').textContent, 'roll +1.5° · line 4.8° vs 3.3°')
})

test('Escape cancels marking trocars', () => {
  const editor = editorHarness()
  editor.context.setStatus = () => {}
  editor.fire('click', {}, '#roll-mark')
  editor.fire('pointerdown', { clientX: 40, clientY: 240 })
  editor.fire('keydown', { key: 'Escape' })
  assert.equal(editor.context.marking, false)
  assert.equal(editor.context.rollPoints.length, 0)
})
