#!/usr/bin/env node
/** Real Chromium checks for the editor and its independently served Web bundle.
 * Run the Python editor first, then:
 * node scripts/check_web.mjs --url http://127.0.0.1:8080 --output /tmp/web-check
 * No API credentials or remote inference requests are used.
 */
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';
import path from 'node:path';
import fs from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {createServer} from 'node:http';
import {execFileSync} from 'node:child_process';

const require = createRequire(import.meta.url);
const {chromium} = require('playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
if (args.includes('--help')) {
  console.log('node scripts/check_web.mjs --url http://127.0.0.1:8080 --output NEW_DIRECTORY [--project FILE] [--image FILE]');
  process.exit(0);
}
const options = {};
for (let index = 0; index < args.length; index += 2) {
  if (!['--url', '--output', '--project', '--image'].includes(args[index]) || !args[index + 1]) {
    throw new Error('Expected --url, --output, --project or --image followed by a value.');
  }
  options[args[index].slice(2)] = args[index + 1];
}
if (!options.output) throw new Error('--output must name a new directory.');
const baseUrl = options.url || 'http://127.0.0.1:8080';
const origin = new URL(baseUrl);
if (!['http:', 'https:'].includes(origin.protocol) || !['localhost', '127.0.0.1', '[::1]'].includes(origin.hostname)) {
  throw new Error('This check requires a local HTTP editor URL.');
}
const output = path.resolve(options.output);
const fixtureProject = path.resolve(options.project || path.join(root, 'samples/results/alignment/project.l2split'));
const fixtureImage = path.resolve(options.image || path.join(root, 'samples/original_character.png'));
await fs.access(fixtureProject);
await fs.access(fixtureImage);
await fs.mkdir(output); // Refuse to overwrite previous evidence or user files.

const report = {version: 1, startedAt: new Date().toISOString(), status: 'running',
  browser: 'Chromium', checks: [], artifacts: [], consoleErrors: [], unexpectedResponses: [],
  remoteInferenceRequests: 0};
let browser, context, page, standaloneServer, sessionToken = null;
let activeCheck = 'bootstrap';
const timeout = 30000;
const sanitize = value => String(value).replace(/sk-[A-Za-z0-9_-]+/g, '[credential]')
  .replace(/(?:\/workspace|\/tmp)\/[^\s'"<>]+/g, '[local file]')
  .replace(/(?:session|token|authorization)[=:]\S+/gi, '[private value]').slice(0, 500);
const writeReport = async () => {
  report.finishedAt = new Date().toISOString();
  report.passedChecks = report.checks.filter(check => check.status === 'passed').length;
  report.skippedChecks = report.checks.filter(check => check.status === 'skipped').length;
  await fs.writeFile(path.join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
};
async function check(name, callback) {
  activeCheck = name;
  const started = Date.now();
  const details = await callback();
  const status = details?.skipped ? 'skipped' : 'passed';
  report.checks.push({name, status, durationMs: Date.now() - started, ...(details ? {details} : {})});
  console.log(`${status === 'passed' ? 'PASS' : 'SKIP'} ${name}`);
}
const testid = id => page.getByTestId(id);
async function apiState() {
  return page.evaluate(async token => {
    const response = await fetch('/api/state', {headers: {'X-Session-Token': token}, signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error(`State request failed: HTTP ${response.status}`);
    return response.json();
  }, sessionToken);
}
async function imageHash(view, id) {
  return page.evaluate(async ({token, view, id}) => {
    const response = await fetch(`/api/image/${view}.png${id ? `?part_id=${encodeURIComponent(id)}` : ''}`, {
      headers: {'X-Session-Token': token}, signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error(`Image request failed: HTTP ${response.status}`);
    const digest = await crypto.subtle.digest('SHA-256', await response.arrayBuffer());
    return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  }, {token: sessionToken, view, id});
}
const maskHash = id => imageHash('mask', id);
async function waitState(predicate) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    const state = await apiState();
    if (predicate(state)) return state;
    await page.waitForTimeout(80);
  }
  throw new Error('Editor state did not reach the expected condition within 30 seconds.');
}
async function changed(action) {
  await waitEditorReady();
  const revision = (await apiState()).revision;
  await action();
  const state = await waitState(state => state.revision > revision);
  await waitEditorReady();
  return state;
}
async function waitEditorReady() {
  await page.waitForFunction(() => !document.querySelector('[data-testid="add-part"]').disabled &&
    document.querySelector('#canvas-loading').hidden, undefined, {timeout});
}
function frameFingerprint(canvas) {
  const context = canvas.getContext('2d');
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  let hash = 2166136261, nontransparent = 0;
  const colors = new Set();
  for (let index = 0; index < pixels.length; index += 4) {
    for (let channel = 0; channel < 4; channel++) hash = Math.imul(hash ^ pixels[index + channel], 16777619);
    if (pixels[index + 3]) nontransparent++;
    if (index % 100 === 0) colors.add(`${pixels[index]},${pixels[index + 1]},${pixels[index + 2]},${pixels[index + 3]}`);
  }
  return {hash: (hash >>> 0).toString(16), nontransparent, sampledColors: colors.size,
    width: canvas.width, height: canvas.height};
}
async function editorFrame() {
  await page.waitForTimeout(100);
  return testid('editor-canvas').evaluate(frameFingerprint);
}
async function slider(id, value) {
  await testid(id).evaluate((element, value) => {
    element.value = String(value);
    element.dispatchEvent(new Event('input', {bubbles: true}));
  }, value);
}
async function screenshot(name, target = page) {
  await target.screenshot({path: path.join(output, name), fullPage: true});
  report.artifacts.push(name);
}
async function download(id, filename) {
  const waiting = page.waitForEvent('download', {timeout});
  await testid(id).click();
  const file = await waiting;
  if (await file.failure()) throw new Error(`${id} download failed.`);
  const destination = path.join(output, filename);
  await file.saveAs(destination);
  assert((await fs.stat(destination)).size > 0, `${id} produced an empty download.`);
  report.artifacts.push(filename);
  return destination;
}
async function openDetails(selector) {
  const details = page.locator(selector);
  if (!await details.evaluate(element => element.open)) await details.locator('summary').click();
}
async function canvasStroke(points) {
  // Fit resets pan/zoom. The editor publishes its current source-to-CSS transform
  // as canvas data attributes for coordinate-aware integrations and accessibility.
  await testid('fit').click();
  const state = await apiState();
  const transform = await testid('editor-canvas').evaluate((canvas, size) => {
    const rect = canvas.getBoundingClientRect();
    const data = canvas.dataset;
    const scale = Number(data.scale || data.viewScale);
    const offsetX = Number(data.offsetX || data.viewOffsetX);
    const offsetY = Number(data.offsetY || data.viewOffsetY);
    if (Number.isFinite(scale) && scale > 0 && Number.isFinite(offsetX) && Number.isFinite(offsetY)) {
      return {left: rect.left, top: rect.top, scale, offsetX, offsetY};
    }
    // Canvas transform is public state on the drawing context, not a test hook.
    const matrix = canvas.getContext('2d').getTransform();
    if (matrix.a !== 1 || matrix.d !== 1 || matrix.e || matrix.f) {
      return {left: rect.left, top: rect.top, scale: matrix.a * rect.width / canvas.width,
        offsetX: matrix.e * rect.width / canvas.width, offsetY: matrix.f * rect.height / canvas.height};
    }
    throw new Error('Editor canvas does not expose its source coordinate transform.');
  }, state.size);
  const project = ([x, y]) => [transform.left + transform.offsetX + x * transform.scale,
    transform.top + transform.offsetY + y * transform.scale];
  const [startX, startY] = project(points[0]);
  await page.mouse.move(startX, startY);
  await page.mouse.down();
  for (const point of points.slice(1)) await page.mouse.move(...project(point), {steps: 3});
  await page.mouse.up();
}
function zipExtract(filename, destination) {
  const code = `import pathlib,sys,zipfile
root=pathlib.Path(sys.argv[2]).resolve()
with zipfile.ZipFile(sys.argv[1]) as archive:
    names=archive.namelist()
    assert len(names)==len(set(names)), 'Duplicate archive names'
    assert len(names)<=512, 'Too many archive entries'
    assert sum(member.file_size for member in archive.infolist())<=256*1024*1024, 'Archive too large'
    for member in archive.infolist():
        name=member.filename.replace('\\\\','/')
        target=(root/name).resolve()
        assert not name.startswith('/') and ':' not in name and root in target.parents, 'Unsafe archive path'
        assert ((member.external_attr>>16)&0o170000)!=0o120000, 'Archive symlinks forbidden'
    archive.extractall(root)
`;
  execFileSync(process.env.PYTHON || 'python3', ['-c', code, filename, destination], {timeout: 15000});
}
async function serveBundle(directory) {
  const mime = {'.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.png': 'image/png'};
  const server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
      const relative = pathname === '/' ? 'index.html' : pathname.slice(1);
      const filename = path.resolve(directory, relative);
      if (!filename.startsWith(directory + path.sep)) {response.writeHead(403);response.end();return;}
      const data = await fs.readFile(filename);
      response.writeHead(200, {'Content-Type': mime[path.extname(filename)] || 'application/octet-stream'});
      response.end(data);
    } catch {response.writeHead(404);response.end();}
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return {server, url: `http://127.0.0.1:${server.address().port}/`};
}

try {
  const executable = process.env.CHROMIUM_PATH || '/usr/bin/chromium';
  browser = await chromium.launch({headless: true,
    ...(existsSync(executable) ? {executablePath: executable} : {}), args: ['--no-sandbox', '--disable-dev-shm-usage']});
  context = await browser.newContext({viewport: {width: 1600, height: 1100}, acceptDownloads: true});
  context.setDefaultTimeout(timeout);
  page = await context.newPage();
  const observe = target => {
    target.on('pageerror', () => report.consoleErrors.push('Unhandled browser JavaScript error.'));
    target.on('console', message => {
      const resource = message.location().url;
      if (resource && new URL(resource).pathname === '/favicon.ico') return;
      if (message.type() === 'error') report.consoleErrors.push(sanitize(message.text()));
    });
  };
  observe(page);
  page.on('request', request => {
    const pathname = new URL(request.url()).pathname;
    if (request.method() === 'POST' && ['/api/proposals/gpt', '/api/proposals/alignment'].includes(pathname)) {
      report.remoteInferenceRequests++;
    }
  });
  page.on('response', async response => {
    const pathname = new URL(response.url()).pathname;
    if (pathname === '/api/session' && response.ok()) sessionToken = (await response.json()).session;
    if (response.status() >= 400 && pathname !== '/favicon.ico') {
      report.unexpectedResponses.push({route: pathname, status: response.status()});
    }
  });
  await check('editor bootstrap', async () => {
    await page.goto(baseUrl, {waitUntil: 'domcontentloaded'});
    await testid('sample-alignment').waitFor({state: 'visible'});
    const end = Date.now() + timeout;
    while (!sessionToken && Date.now() < end) await page.waitForTimeout(50);
    assert(sessionToken, 'Editor did not initialize an authenticated session.');
    assert.equal((await apiState()).parts.length, 0);
  });
  await check('sample renders real character textures', async () => {
    await testid('sample-alignment').click();
    const state = await waitState(state => state.parts.length === 9);
    await waitEditorReady();
    await page.waitForFunction(() => document.querySelector('#editor-canvas')?.width > 100);
    const frame = await editorFrame();
    assert(frame.nontransparent > 1000 && frame.sampledColors > 100, 'Character textures are absent from the canvas.');
    return {parts: state.parts.length, frame};
  });
  await check('animation controls change rendered pixels', async () => {
    for (const selector of ['#idle', '#auto-blink', '#pointer-tracking']) await page.locator(selector).uncheck();
    if (await testid('playback').getAttribute('aria-pressed') === 'true') await testid('playback').click();
    let previous = await editorFrame();
    const frames = {};
    for (const [id, value] of [['angle-x', .75], ['eye-open', 0], ['mouth-open', .9]]) {
      await slider(id, value);
      const next = await editorFrame();
      assert.notEqual(next.hash, previous.hash, `${id} did not change the canvas.`);
      frames[id] = next.hash;
      previous = next;
    }
    await testid('expression').selectOption('cry');
    const crying = await editorFrame();
    assert.notEqual(crying.hash, previous.hash, 'Expression did not change rendered pixels.');
    frames.cry = crying.hash;
    await screenshot('editor_expression.png');
    await testid('expression').selectOption('neutral');
    for (const [id, value] of [['angle-x', 0], ['eye-open', 1], ['mouth-open', 0]]) await slider(id, value);
    await testid('motion').selectOption('nod');
    await testid('playback').click();
    const moving = await editorFrame();
    await page.waitForTimeout(350);
    const later = await editorFrame();
    assert.notEqual(moving.hash, later.hash, 'Playing a motion did not change animation frames.');
    await testid('playback').click();
    await testid('motion').selectOption('idle');
    return {frames, motionFrames: [moving.hash, later.hash]};
  });
  await check('runtime preserves opaque and translucent alpha without mesh seams', async () => {
    const results = await page.evaluate(async () => {
      const results = [];
      const cases = [
        {role: 'static', parameters: {}, margin: 0, tolerance: 0},
        {role: 'face', parameters: {angleZ: .8}, margin: 11, tolerance: 0},
        {role: 'hair_back', parameters: {angleX: .8}, margin: 11, tolerance: 1},
        {role: 'body', parameters: {breath: .8}, margin: 11, tolerance: 0},
      ];
      for (const alpha of [255, 128]) for (const test of cases) {
        const source = document.createElement('canvas'); source.width = source.height = 64;
        const context = source.getContext('2d'), pixels = context.createImageData(64, 64);
        for (let index = 0; index < pixels.data.length; index += 4) pixels.data.set([61, 109, 153, alpha], index);
        context.putImageData(pixels, 0, 0);
        const canvas = document.createElement('canvas'); canvas.width = canvas.height = 64;
        const model = {version: 1, canvas: [64, 64], config: {meshResolution: 4},
          layers: [{id: 'alpha-check', bounds: [0, 0, 64, 64], role: test.role, visible: true}]};
        const character = new Live2DWeb.Character(canvas, model, {idle: false, autoBlink: true,
          pointerTracking: false, assetUrls: {'alpha-check': source.toDataURL()}});
        await character.load();character.setParameters(test.parameters);character.render(0);
        const rendered = canvas.getContext('2d').getImageData(0, 0, 64, 64).data;
        let changed = 0, maxDifference = 0;
        for (let y = test.margin; y < 64 - test.margin; y++) for (let x = test.margin; x < 64 - test.margin; x++) {
          const difference = Math.abs(rendered[(y * 64 + x) * 4 + 3] - alpha);
          maxDifference = Math.max(maxDifference, difference);
          if (difference > test.tolerance) changed++;
        }
        results.push({alpha, role: test.role, mismatchedPixels: changed, maxDifference,
          tolerance: test.tolerance});character.destroy();
      }
      return results;
    });
    for (const result of results) assert.equal(result.mismatchedPixels, 0, `${result.role} alpha ${result.alpha} has mesh seams.`);
    return {cases: results};
  });
  let bundlePath;
  await check('animated model downloads as a standalone Web bundle', async () => {
    bundlePath = await download('export-web', 'character-web.zip');
    return {bytes: (await fs.stat(bundlePath)).size};
  });

  await check('image and project upload through real file controls', async () => {
    await changed(() => testid('projectimage-input').setInputFiles(fixtureImage));
    assert((await apiState()).size.every(value => value > 100));
    await changed(() => testid('project-input').setInputFiles(fixtureProject));
    assert.equal((await apiState()).parts.length, 9);
  });
  await check('default relative adjustment preserves aligned artwork and can be undone', async () => {
    const before = await apiState();
    const original = await imageHash('composite');
    await openDetails('details:has(#apply-adjustment)');
    for (const [id, value] of [['adjust-scale', '1'], ['adjust-angle', '0'], ['adjust-x', '0'], ['adjust-y', '0']]) {
      assert.equal(await page.locator(`#${id}`).inputValue(), value);
    }
    await page.locator('#apply-adjustment').click();
    await page.waitForFunction(() => !document.querySelector('[data-testid="job-accept"]').disabled, undefined, {timeout});
    await testid('job-accept').click();
    const accepted = await waitState(state => state.revision > before.revision);
    await waitEditorReady();
    assert.deepEqual(accepted.parts.map(part => part.id), before.parts.map(part => part.id));
    assert.equal(await imageHash('composite'), original, 'Default adjustment changed source artwork.');
    await changed(() => testid('undo').click());
    assert.equal(await imageHash('composite'), original);
    assert.deepEqual((await apiState()).parts, before.parts);
  });
  let editedPart;
  await check('brush add erase undo redo preserve stable part IDs', async () => {
    const added = await changed(() => testid('add-part').click());
    editedPart = added.parts.find(part => !part.bounds);
    assert(editedPart, 'New empty part was not created.');
    await testid('view-mode').selectOption('overlay');
    await openDetails('#mask-controls');
    await slider('brush-radius', 24);
    const [width, height] = added.size;
    const center = [Math.round(width * .45), Math.round(height * .45)];
    const initial = await maskHash(editedPart.id);
    await changed(() => canvasStroke([center, [center[0] + 50, center[1]]]));
    const painted = await maskHash(editedPart.id);
    assert.notEqual(painted, initial);
    const bounds = (await apiState()).parts.find(part => part.id === editedPart.id).bounds;
    assert(Math.abs(bounds[0] - (center[0] - 24)) <= 2 && Math.abs(bounds[1] - (center[1] - 24)) <= 2,
      'Pointer stroke did not reach the expected source-image coordinates.');
    await page.locator('#mask-erase').click();
    await changed(() => canvasStroke([center]));
    const erased = await maskHash(editedPart.id);
    assert.notEqual(erased, painted);
    await changed(() => testid('undo').click());
    assert.equal(await maskHash(editedPart.id), painted);
    await changed(() => testid('redo').click());
    assert.equal(await maskHash(editedPart.id), erased);
    assert((await apiState()).parts.some(part => part.id === editedPart.id));
  });
  await check('lasso and hidden masks are editable independently', async () => {
    await page.locator('#mask-add').click();
    await testid('mask-tool').selectOption('lasso');
    await testid('mask-target').selectOption('hidden');
    const initialVisible = await maskHash(editedPart.id);
    const [width, height] = (await apiState()).size;
    const x = Math.round(width * .2), y = Math.round(height * .2);
    await changed(() => canvasStroke([[x, y], [x + 60, y], [x + 60, y + 60], [x, y + 60]]));
    const state = await apiState();
    assert(state.parts.find(part => part.id === editedPart.id).hidden_bounds);
    assert.equal(await maskHash(editedPart.id), initialVisible);
    await testid('mask-target').selectOption('visible');
    await testid('mask-tool').selectOption('brush');
  });
  await check('rename role visibility and order remain editable', async () => {
    await testid('part-name').fill('Browser edited part');
    await testid('part-kind').fill('accessory');
    await changed(() => testid('apply-part').click());
    assert.equal((await apiState()).parts.find(part => part.id === editedPart.id).name, 'Browser edited part');
    await changed(() => testid('role-select').selectOption('accessory_body'));
    assert.equal((await apiState()).runtime_config.bindings[editedPart.id].role, 'accessory_body');
    const row = page.locator(`[data-part-id="${editedPart.id}"]`);
    const visibility = row.locator('.visibility');
    await changed(() => visibility.click());
    assert.equal((await apiState()).parts.find(part => part.id === editedPart.id).visible, false);
    await changed(() => visibility.click());
    const before = (await apiState()).parts.map(part => part.id);
    await changed(() => page.locator('#move-down').click());
    assert.notDeepEqual((await apiState()).parts.map(part => part.id), before);
    await changed(() => page.locator('#move-up').click());
    assert.deepEqual((await apiState()).parts.map(part => part.id), before);
  });
  await check('declining remote upload creates no inference request', async () => {
    await openDetails('details:has([data-testid="propose-gpt"])');
    const before = report.remoteInferenceRequests;
    await page.locator('#gpt-model').fill('gpt-4.1');
    let shown = false;
    page.once('dialog', dialog => {shown = true; return dialog.dismiss();});
    await testid('propose-gpt').click();
    await page.waitForTimeout(150);
    assert.equal(report.remoteInferenceRequests, before);
    assert.equal(before, 0);
    assert(shown, 'Remote image upload consent was not shown.');
  });
  await check('color proposals are previewed rejected and explicitly accepted', async () => {
    const before = await apiState();
    const preservedMask = await maskHash(editedPart.id);
    await testid('auto-color').click();
    await testid('job-accept').waitFor({state: 'visible'});
    await page.waitForFunction(() => !document.querySelector('[data-testid="job-accept"]').disabled, undefined, {timeout});
    assert.deepEqual((await apiState()).parts, before.parts);
    await testid('job-reject').click();
    await waitState(state => !state.active_job);
    assert.deepEqual((await apiState()).parts, before.parts);
    assert.equal(await maskHash(editedPart.id), preservedMask);
    await testid('auto-color').click();
    await page.waitForFunction(() => !document.querySelector('[data-testid="job-accept"]').disabled, undefined, {timeout});
    await testid('job-accept').click();
    const adopted = await waitState(state => state.parts.length > before.parts.length);
    for (const part of before.parts) assert(adopted.parts.some(candidate => candidate.id === part.id));
    assert.equal(await maskHash(editedPart.id), preservedMask);
    return {originalParts: before.parts.length, adoptedParts: adopted.parts.length};
  });
  await check('all export downloads and project round trip preserve edits', async () => {
    await openDetails('.secondary-export');
    const before = await apiState();
    const masks = await Promise.all(before.parts.map(part => maskHash(part.id)));
    const archive = await download('export-project', 'edited-project.l2split');
    const png = await download('export-png', 'edited-png.zip');
    const psd = await download('export-psd', 'edited-layers.psd');
    assert.equal((await fs.readFile(archive)).subarray(0, 2).toString(), 'PK');
    assert.equal((await fs.readFile(png)).subarray(0, 2).toString(), 'PK');
    assert.equal((await fs.readFile(psd)).subarray(0, 4).toString(), '8BPS');
    await changed(() => testid('project-input').setInputFiles(archive));
    const after = await apiState();
    assert.deepEqual(after.parts, before.parts);
    assert.deepEqual(after.runtime_config, before.runtime_config);
    assert.deepEqual(await Promise.all(after.parts.map(part => maskHash(part.id))), masks);
    await screenshot('editor_mask.png');
    return {parts: after.parts.length, stableIdsPreserved: true, masksPreserved: true};
  });

  await check('exported model runs on an independent HTTP server', async () => {
    const bundleDirectory = path.join(output, 'standalone');
    zipExtract(bundlePath, bundleDirectory);
    const serving = await serveBundle(bundleDirectory);
    standaloneServer = serving.server;
    const standalone = await context.newPage();
    observe(standalone);
    const requests = [];
    standalone.on('request', request => requests.push(request.url()));
    await standalone.goto(serving.url, {waitUntil: 'domcontentloaded'});
    await standalone.waitForFunction(() => window.character?.loaded);
    assert(requests.every(request => new URL(request).origin === new URL(serving.url).origin),
      'Standalone bundle made a request outside its independent server.');
    await standalone.evaluate(() => {
      character.stop().setIdle(false).setAutoBlink(false).setPointerTracking(false).setMotion('idle');
      character.setParameters({angleX: 0, angleY: 0, angleZ: 0, eyeOpen: 1, mouthOpen: 0, breath: 0});
      character.setExpression('neutral'); character.render(0);
    });
    const responsiveCanvases = [];
    for (const viewport of [{width: 1600, height: 1100}, {width: 880, height: 680}]) {
      await standalone.setViewportSize(viewport);
      await standalone.waitForFunction(() => {
        const canvas = document.querySelector('#character'), rect = canvas.getBoundingClientRect();
        const ratio = Math.min(window.devicePixelRatio || 1, 8192 / rect.width, 8192 / rect.height);
        return Math.abs(canvas.width - rect.width * ratio) <= 1 &&
          Math.abs(canvas.height - rect.height * ratio) <= 1;
      }, undefined, {timeout});
      const dimensions = await standalone.locator('#character').evaluate(canvas => {
        const rect = canvas.getBoundingClientRect();
        return {css: {width: Number(rect.width.toFixed(2)), height: Number(rect.height.toFixed(2))},
          backing: {width: canvas.width, height: canvas.height}, devicePixelRatio: window.devicePixelRatio || 1};
      });
      const xScale = dimensions.backing.width / dimensions.css.width;
      const yScale = dimensions.backing.height / dimensions.css.height;
      assert(Math.abs(xScale - yScale) < .01,
        'CSS stretched the character canvas with different horizontal and vertical scales.');
      await standalone.evaluate(() => {character.setParameters({angleX: 0});character.render(0);});
      const normal = await standalone.locator('#character').evaluate(frameFingerprint);
      assert(normal.nontransparent > 1000 && normal.sampledColors > 100,
        'Resized canvas stopped rendering real character textures.');
      await standalone.evaluate(() => {character.setParameters({angleX: .55});character.render(0);});
      const moved = await standalone.locator('#character').evaluate(frameFingerprint);
      assert.notEqual(moved.hash, normal.hash, 'Animation parameters stopped working after a viewport resize.');
      responsiveCanvases.push({viewport, ...dimensions, frameHashes: [normal.hash, moved.hash]});
    }
    assert.notDeepEqual(responsiveCanvases[0].backing, responsiveCanvases[1].backing,
      'Canvas backing dimensions did not react to the viewport change.');
    await standalone.setViewportSize({width: 1600, height: 1100});
    await standalone.waitForFunction(() => {
      const canvas = document.querySelector('#character'), rect = canvas.getBoundingClientRect();
      return Math.abs(canvas.width / canvas.height - rect.width / rect.height) < .005;
    }, undefined, {timeout});
    await standalone.evaluate(() => {character.setParameters({angleX: 0});character.render(0);});
    const initial = await standalone.locator('#character').evaluate(frameFingerprint);
    assert(initial.nontransparent > 1000 && initial.sampledColors > 100);
    await standalone.locator('#angleX').evaluate(element => {
      element.value = '.8'; element.dispatchEvent(new Event('input', {bubbles: true})); character.render(0);
    });
    const angled = await standalone.locator('#character').evaluate(frameFingerprint);
    assert.notEqual(angled.hash, initial.hash);
    await standalone.evaluate(() => {character.setParameters({eyeOpen: 0, mouthOpen: .8});character.render(0);});
    const blinkMouth = await standalone.locator('#character').evaluate(frameFingerprint);
    assert.notEqual(blinkMouth.hash, angled.hash);
    await standalone.locator('#expression').selectOption('cry');
    await standalone.evaluate(() => character.render(0));
    const expression = await standalone.locator('#character').evaluate(frameFingerprint);
    assert.notEqual(expression.hash, blinkMouth.hash);
    await standalone.locator('#motion').selectOption('nod');
    await standalone.evaluate(() => character.render(0));
    const motionBefore = await standalone.locator('#character').evaluate(frameFingerprint);
    await standalone.evaluate(() => character.render(500));
    const motionAfter = await standalone.locator('#character').evaluate(frameFingerprint);
    assert.notEqual(motionBefore.hash, motionAfter.hash);
    await screenshot('standalone_expression.png', standalone);
    const video = await standalone.evaluate(async () => {
      const canvas = document.querySelector('#character');
      if (!canvas.captureStream || typeof MediaRecorder === 'undefined') return null;
      const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find(type => MediaRecorder.isTypeSupported(type));
      if (!mime) return null;
      character.setParameters({angleX: 0, angleY: 0, angleZ: 0, eyeOpen: 1, mouthOpen: .2, breath: 0});
      character.setExpression('smile').setMotion('nod').setAutoBlink(true).setIdle(true).start();
      const stream = canvas.captureStream(24), chunks = [];
      const recorder = new MediaRecorder(stream, {mimeType: mime, videoBitsPerSecond: 1600000});
      const result = new Promise((resolve, reject) => {
        const watchdog = setTimeout(() => {
          character.stop();stream.getTracks().forEach(track => track.stop());
          reject(new Error('Canvas recording exceeded its 15 second limit.'));
        }, 15000);
        recorder.ondataavailable = event => {if (event.data.size) chunks.push(event.data);};
        recorder.onerror = () => {clearTimeout(watchdog);reject(new Error('Canvas recording failed.'));};
        recorder.onstop = async () => {
          clearTimeout(watchdog);
          const bytes = new Uint8Array(await new Blob(chunks, {type: mime}).arrayBuffer());
          character.stop();stream.getTracks().forEach(track => track.stop());resolve(Array.from(bytes));
        };
      });
      recorder.start();setTimeout(() => recorder.stop(), 3500);
      return result;
    });
    if (video) {
      assert(video.length > 1000, 'Animation recording is empty.');
      await fs.writeFile(path.join(output, 'character_motion.webm'), Buffer.from(video));
      report.artifacts.push('character_motion.webm');
    }
    await standalone.close();
    return {parts: 9, independentServer: true, responsiveCanvases, frameHashes: [initial.hash, angled.hash, blinkMouth.hash,
      expression.hash, motionBefore.hash, motionAfter.hash], recording: video ? 'character_motion.webm' : 'unsupported; skipped'};
  });
  await check('exported model also opens directly as a local file', async () => {
    const local = await context.newPage();
    observe(local);
    try {
      await local.goto(pathToFileURL(path.join(output, 'standalone/index.html')).href);
      await local.waitForFunction(() => window.character?.loaded);
      await screenshot('standalone_file.png', local);
      return {loadedTextures: await local.evaluate(() => character.layers.length)};
    } catch (error) {
      if (String(error.message).includes('ERR_BLOCKED_BY_ADMINISTRATOR')) {
        return {skipped: 'This Chromium installation blocks file:// navigation by policy; the independent HTTP bundle passed.'};
      }
      throw error;
    } finally {await local.close();}
  });
  await check('browser has no unexpected API failures or JavaScript errors', async () => {
    assert.deepEqual(report.unexpectedResponses, []);
    assert.deepEqual(report.consoleErrors, []);
    assert.equal(report.remoteInferenceRequests, 0);
  });
  report.status = 'passed';
} catch (error) {
  report.status = 'failed';
  report.checks.push({name: activeCheck, status: 'failed', error: sanitize(error.message)});
  console.error(`FAIL ${activeCheck}: ${sanitize(error.message)}`);
  if (page) {
    try {await screenshot('failure.png');} catch { /* Closed/crashed page has no screenshot. */ }
  }
  process.exitCode = 1;
} finally {
  sessionToken = null;
  if (browser) await browser.close();
  if (standaloneServer) await new Promise(resolve => standaloneServer.close(resolve));
  await writeReport();
}
