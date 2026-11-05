/** Install: npm ci --prefix writeup/figures/viz
 * Chromium: cd writeup/figures/viz && npx playwright install chromium
 * Run from repo root: node writeup/figures/viz/capture.mjs [--url URL] [--out DIRECTORY]
 */
import { chromium } from 'playwright';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import path from 'node:path';
import { poseFrame } from '../../../services/viz/src/feed/mockGenerator.js';

// Comparable before/after states: insertion mm, lateral u/v, half jaw angle and roll rad.
const STATES = [
  ['01_far_overview',       14.36, 0,   0, 0.1,  'Overview', 'Fern; Scope in Bereitschaft'],
  ['02_near_overview',      18.36, 0,   0, 0.1,  'Overview', 'Nah; Scope aktiv'],
  ['03_warning_overview',   21.16, 0,   0, 0.1,  'Overview', 'Warnbereich'],
  ['04_critical_overview',  22,    0,   0, 0.1,  'Overview', 'Kritischer Abstand (Tiefenlimit 22 mm)'],
  ['05_warning_surgeon',   21.16, 0,   0, 0.1,  'Surgeon',  'Warnbereich aus Operateursperspektive'],
  ['06_warning_sagittal',  21.16, 0,   0, 0.1,  'Sagittal', 'Warnbereich im Sagittalschnitt'],
  ['07_warning_fundus',    21.16, 0,   0, 0.1,  'Fundus',   'Warnbereich in Fundusansicht'],
  ['08_approach_trend',    17.36, 0,   0, 0.1,  'Overview', 'Annäherung über 4 s von ca. 5 auf 1.5 mm', 'trend'],
  ['09_nodata_live',       21.16, 0,   0, 0.1,  'Overview', 'Live tracking ohne Messdaten', 'live'],
  ['10_display_menu_open', 21.16, 0,   0, 0.1,  'Overview', 'Display-Menü geöffnet', 'display'],
  ['11_legend_and_nav',    21.16, 0,   0, 0.1,  'Overview', 'Legende und Navigation'],
  ['12_jaws_closed',      18.36, 0,   0, 0.02, 'Overview', 'Backen geschlossen'],
  ['13_jaws_wide',        18.36, 0,   0, 0.4,  'Overview', 'Backen maximal geöffnet'],
  ['14_lateral_sway',     17.42, 0.5, 0, 0.1,  'Overview', 'Seitliche Auslenkung; Schatten/Landepunkt versetzt'],
  ['15_asymmetric_near',   18,    0.5, 0, 0.3,  'Overview', 'Asymmetrisch: L näher (≈1.3 vs 1.9 mm), beide Near retina'],
  ['16_asymmetric_critical', 19.15, 0.5, 0, 0.3, 'Overview', 'Asymmetrisch: L kritisch, R Near retina'],
  ['17_live_critical',      22,    0,   0, 0.1, 'Overview', 'Live: stetiger kritischer Rahmen und Live-Genauigkeitshinweis', 'live-critical'],
  ['18_live_near',          21.16, 0,   0, 0.1, 'Overview', 'Live: Near retina', 'live-near'],
  ['19_live_held',          21.16, 0,   0, 0.1, 'Overview', 'Live: Signal gap, letzter Wert gehalten (300 ms)', 'live-held'],
  ['20_live_ghost_last',    22,    0,   0, 0.1, 'Overview', 'Live: Ghost mit letzter kritischer Messung (1200 ms)', 'live-ghost'],
];
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i += 2)
  if (!['--url', '--out'].includes(args[i]) || !args[i + 1]) throw Error('Usage: capture.mjs [--url URL] [--out DIRECTORY]');
const option = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const base = option('--url', 'http://localhost:5173');
const out = path.resolve(option('--out', 'writeup/figures/viz/baseline'));
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, args: [
  ...(process.platform === 'darwin' ? ['--use-angle=metal'] : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']),
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
] });
const pageOptions = { viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 };
let page = await browser.newPage(pageOptions);
const errors = [];
let streamTimer;
page.on('pageerror', e => errors.push(e.message));
// Optional recovery after a renderer timeout; preserves earlier manifest entries.
const from = process.env.CAPTURE_FROM || '';
const records = from ? (await readFile(path.join(out, 'README.md'), 'utf8').catch(error => { if (error.code === 'ENOENT') return ''; throw error; })).split('\n').filter(line => line.startsWith('- ') && (process.env.CAPTURE_ONLY ? !line.startsWith(`- ${process.env.CAPTURE_ONLY}`) : line.slice(2, 4) < from.slice(0, 2)) || line.startsWith('- crop_')) : [];
const distance = () => page.locator('.distance-value > span').first().innerText();
async function capture(file, description, pose, view, selector, source = 'debug', details = '') {
  const d = await distance();
  const before = pose.depth;
  const clip = selector ? await page.locator(selector).boundingBox() : undefined;
  if (selector && !clip) throw Error(`Crop not found: ${selector}`);
  if (file === '08_approach_trend') {
    // CDP avoids Playwright's additional screenshot preparation frames while moving.
    const cdp = await page.context().newCDPSession(page);
    const shot = await cdp.send('Page.captureScreenshot', { format: 'png', fromSurface: true, clip: { x: 0, y: 0, width: 1600, height: 1000, scale: 2 } });
    await writeFile(path.join(out, `${file}.png`), Buffer.from(shot.data, 'base64'));
    await cdp.detach();
  } else await page.screenshot({ timeout: 120000, path: path.join(out, `${file}.png`), ...(clip ? { clip } : {}) });
  const afterDistance = await distance();
  if (file === '08_approach_trend') pose.depth = `${before} → ${await page.evaluate(() => window.__vizDebug.getPose().depth)}`;
  records.push(`- ${file}.png: ${description}${file === '08_approach_trend' ? '; Skript: depth startet bei 17.36 mm, +0.0875 mm alle 100 ms (0.875 mm/s), Zielaufnahme ab Schritt 40 (depth 20.86 mm); folgende Pose ist nur das Aufnahmefenster' : ''}; Quelle ${source}; Pose ${JSON.stringify(pose)}; angezeigter Abstand ${file === '08_approach_trend' ? `${d} → ${afterDistance} (Messungen vor/nach Screenshot; siehe PNG)` : d} mm; Ansicht ${view}${details ? `; ${details}` : ''}.`);
  console.log(`${file}: ${d} mm`);
}
try {
  for (const [name, depth, u, v, half, view, description, action] of STATES.filter(state => (!from || state[0] >= from) && (!process.env.CAPTURE_ONLY || state[0] === process.env.CAPTURE_ONLY))) {
    if (action?.startsWith('live-')) {
      await page.close();
      page = await browser.newPage(pageOptions);
      page.on('pageerror', e => errors.push(e.message));
      // Install before app startup: resetting performance.now later would leave
      // the pipeline's readout scheduling timestamps on the old clock.
      await page.clock.install();
    }
    const pose = { depth, u, v, half, roll: 0 };
    const url = new URL(base);
    url.searchParams.set('pose', Object.entries(pose).map(([k, v]) => `${k}:${v}`).join(','));
    await page.goto(url.href, { waitUntil: 'networkidle' });
    await page.bringToFront();
    await page.getByLabel('Data source', { exact: true }).selectOption('debug');
    await page.waitForFunction(() => !!window.__vizDebug);
    await page.evaluate(p => window.__vizDebug.setPose(p), pose);
    await page.getByRole('button', { name: view, exact: true }).click();
    await page.waitForTimeout(1500);
    if (!await page.locator('canvas').first().evaluate(c => !!(c.getContext('webgl2') || c.getContext('webgl'))))
      throw Error('WebGL unavailable');
    const mockedLive = action?.startsWith('live-');
    let liveDetails = '';
    if (mockedLive) {
      const frame = poseFrame({ depth, wobbleU: u, wobbleV: v, half, roll: 0 });
      let count = 0;
      let connected = false;
      await page.routeWebSocket(/\/ws(?:\?.*)?$/, ws => {
        connected = true;
        const send = () => ws.send(JSON.stringify({ ...frame, t: ++count / 30, confidence: 0.99, calibrated: true }));
        send();
        streamTimer = setInterval(send, 1000 / 30);
        ws.onClose(() => clearInterval(streamTimer));
      });
      // Observe receipt time without changing the application's feed or parser.
      await page.evaluate(() => {
        const NativeWebSocket = window.WebSocket;
        window.WebSocket = class extends NativeWebSocket {
          constructor(...args) {
            super(...args);
            this.addEventListener('message', () => { window.__captureLastReceipt = performance.now(); });
          }
        };
      });
      await page.getByLabel('Data source', { exact: true }).selectOption('live');
      await page.waitForTimeout(1500);
      if (!connected || count < 30) throw Error('Mock /ws not connected or insufficient frames');
      if (!Number.isFinite(parseFloat(await distance()))) throw Error('Live mock produced no measurement');
      if (action === 'live-held' || action === 'live-ghost') {
        clearInterval(streamTimer);
        // Freeze the application clock, then advance to the exact receipt age.
        // Screenshot preparation cannot push the 200–400 ms held window into ghost.
        await page.clock.pauseAt(new Date(Date.now() + 50));
        const age = await page.evaluate(() => performance.now() - window.__captureLastReceipt);
        const target = action === 'live-held' ? 300 : 1200;
        if (!Number.isFinite(age) || age > target) throw Error(`Missed live capture window: ${age} ms`);
        await page.clock.runFor(target - age);
        liveDetails = `Messalter ${target} ms, Browser-Uhr für PNG angehalten`;
      }
      if (action === 'live-critical') {
        const animations = await page.locator('.scene-alarm').evaluate(el => [el, ...el.querySelectorAll('*')].flatMap(node => node.getAnimations().map(a => a.animationName)));
        if (animations.includes('alarm-pulse')) throw Error('Live critical alarm still pulses');
        await page.locator('.distance-qualifier').filter({ hasText: /(?:Recorded sim runs: Critical|Sim: Critical shown)/ }).waitFor();
      }
      if (action === 'live-held') {
        await page.getByText('Signal gap · holding last value', { exact: true }).waitFor();
        if (!await page.locator('.distance-value.measurement-held').count()) throw Error('Held values are not dimmed');
      }
      if (action === 'live-ghost') {
        await page.locator('.distance-status').filter({ hasText: /No measurement · last: Critical proximity .* mm, \d+ s ago/ }).waitFor();
        if (Number.isFinite(parseFloat(await distance()))) throw Error('Ghost still shows a current measurement');
      }
      liveDetails += `${liveDetails ? '; ' : ''}30-Hz-WebSocket-Mock, ${count} Frames, Status ${(await page.locator('.distance-status').innerText()).replace(/\s+/g, ' ')}, Hinweis ${await page.locator('.distance-qualifier').innerText()}`;
    }
    if (action === 'live') {
      await page.getByLabel('Data source', { exact: true }).selectOption('live');
      await page.waitForTimeout(1800);
      if (Number.isFinite(parseFloat(await distance()))) throw Error('Live data present: cannot capture no-data state');
    }
    if (action === 'display') {
      await page.getByRole('button', { name: /Display/ }).click();
      await page.waitForTimeout(1500);
    }
    if (action === 'trend') {
      // Fixed 100 ms steps continue during screenshot encoding. Capture at step 40;
      // later steps hold the same velocity, preventing an artificial stopped trend.
      for (let attempt = 0; attempt < 4; attempt++) {
      await page.evaluate(() => window.__vizDebug.setPose({ depth: 17.36 }));
      await page.waitForTimeout(1500);
      await page.evaluate(() => {
        window.__captureStep = 0;
        window.__captureTimer = setInterval(() => {
          window.__captureStep++;
          window.__vizDebug.setPose({ depth: 17.36 + window.__captureStep * 0.0875 });
        }, 100);
      });
      await page.waitForFunction(() => window.__captureStep >= 40, { }, { polling: 20 });
      try {
        await page.getByText(/Approaching · |Contact in ≈/).first().waitFor({ timeout: 1500 });
        break;
      } catch (error) {
        await page.evaluate(() => clearInterval(window.__captureTimer));
        if (attempt === 3) throw error;
      }
      }
      pose.depth = await page.evaluate(() => window.__vizDebug.getPose().depth);
    }
    await capture(name, description, pose, view, undefined, mockedLive || action === 'live' ? 'live' : 'debug', liveDetails);
    if (mockedLive) {
      clearInterval(streamTimer);
      await page.clock.resume();
    }
    if (action === 'trend') await page.evaluate(() => clearInterval(window.__captureTimer));
    if (name === '03_warning_overview')
      for (const [crop, selector] of Object.entries({ distance: '.distance-panel', approach: '.approach-panel', instrument: '.instrument-panel', scope: '.scope-panel' }))
        await capture(`crop_${crop}`, 'Warnbereich, Panel-Ausschnitt', pose, view, selector);
  }
  await writeFile(path.join(out, 'README.md'), `# UI screenshots (${path.basename(out)})\n\n${path.basename(out) === 'baseline' ? 'UI before the review iteration (old version).' : 'Current UI; the previous version with identical poses is in ../baseline/.'}\n\nAufruf: \`node writeup/figures/viz/capture.mjs --url ${base} --out ${out}\`\n\n1600 × 1000 CSS-Pixel, DPR 2 (3200 × 2000 PNG); Chromium ${browser.version()}, ${process.platform === 'darwin' ? 'ANGLE Metal' : 'SwiftShader'}. Jeder Zustand wird frisch mit URL-Pose geladen und mindestens 1.5 s beruhigt. Trend: 100-ms-Pose-Schritte, Aufnahme während laufender Bewegung nach ca. 4 s; Timing und Hundertstel können variieren. Kritischer Abstand ist durch depth ≤ 22 auf ca. 0.36 mm begrenzt. Backenöffnung und laterale Auslenkung verändern die tatsächliche Distanz.\n\nLive 17–20: Playwright page.routeWebSocket fängt /ws ab (kein Backend). Der unveränderte poseFrame-Generator liefert mm-Koordinaten; der Mock sendet ca. 30 Hz mit streng steigendem t = Frame/30, confidence 0.99 und calibrated true. Nach 1.5 s Stream wird direkt aufgenommen (17/18) oder das Senden bei offener Verbindung gestoppt (19/20). Die Browser-Uhr wird bei 300/1200 ms seit letzter Nachricht für den Screenshot angehalten. Ghost zeigt keinen aktuellen Abstand; last bezieht sich auf die letzte Messung, die Sekundenanzeige zählt ab Ghost-Beginn.\n\n${records.sort((a,b) => a.localeCompare(b)).join('\n')}\n\nJavaScript-Laufzeitfehler: ${errors.length ? errors.join('; ') : 'keine'}.\n`);
} finally { clearInterval(streamTimer); await browser.close(); }
