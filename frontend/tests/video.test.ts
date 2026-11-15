import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readFileSync, mkdtempSync, mkdirSync, rmSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import * as api from '../src/api.ts'
import * as upload from '../src/upload.ts'
const require = createRequire(import.meta.url)
const tempRoot = fileURLToPath(new URL('../.tmp/', import.meta.url))
mkdirSync(tempRoot, {recursive: true})
const compiled = mkdtempSync(join(tempRoot, 'ui-test-'))
process.on('exit', () => rmSync(compiled, {recursive: true, force: true}))
execFileSync(process.execPath, [require.resolve('typescript/package.json').replace('package.json', 'bin/tsc'), '--ignoreConfig', '--jsx', 'react-jsx', '--module', 'commonjs', '--noCheck', '--moduleResolution', 'bundler', '--target', 'ES2022', '--esModuleInterop', '--skipLibCheck', '--outDir', compiled, 'src/App.tsx', 'src/VideoAnalysis.tsx'], {cwd: new URL('..', import.meta.url)})
function component(name: string, extra: Record<string, unknown> = {}) {
  const code = readFileSync(join(compiled, `${name}.js`), 'utf8')
  const exports: Record<string, React.ComponentType<any>> = {}
  new Function('require', 'exports', code)((id: string) => id in extra ? extra[id] : id === './api' ? api : id === './upload' ? upload : require(id), exports)
  return exports
}
const video = component('VideoAnalysis')
const fixture = {success: true, status: 'analyzed', predicted_label: 'fake', average_fake_score: .8, average_real_score: .2,
  threshold: .5, sampled_frames: 3, successfully_analyzed_frames: 3, frames_without_faces: 0, faces_analyzed: 3,
  frames_above_threshold: 3, percentage_frames_above_threshold: 100, decoding_failures: 0, processing_time_seconds: .4, duration_seconds: 3,
  model_name: 'unit-fixture', model_version: 'test', warning: 'Experimental, not proof of authenticity.', limitations: ['Not calibrated.'],
  frame_level_results: [0, 1, 2].map(i => ({frame_index: i, timestamp_seconds: i, timestamp_method: 'frame_index_over_fps_estimate', status: 'analyzed', average_fake_score: .8, error: null,
    faces: [{bounding_box: [1, 2, 3, 4], crop_box: [1, 2, 4, 5], detection_confidence: .99, status: 'analyzed', probabilities: {real: .2, fake: .8}, predicted_label: 'fake'}]}))}

test('video validation: formats, size, empty files and duration', () => {
  for (const name of ['x.mp4', 'x.mov', 'x.avi', 'x.mkv']) assert.equal(upload.validateVideo(new File(['x'], name)), null)
  assert.match(upload.validateVideo(new File(['x'], 'bad.webm'))!, /MP4/)
  assert.match(upload.validateVideo(new File([], 'empty.mp4'))!, /nonempty/)
  assert.match(upload.validateVideo(new File([new Uint8Array(50 * 1024 * 1024 + 1)], 'big.mp4'))!, /50 MB/)
  assert.equal(upload.validateDuration(120), null)
  for (const n of [121, Infinity, NaN, 0]) assert.match(upload.validateDuration(n)!, /120 seconds/)
})

test('video API uploads multipart field and validates results', async t => {
  const file = new File(['unit'], 'small.avi', {type: 'video/x-msvideo'})
  const controller = new AbortController()
  t.mock.method(globalThis, 'fetch', async (url: string, init: RequestInit) => {
    assert.equal(url, '/api/v1/detect/video'); assert.equal(init.signal, controller.signal)
    assert.equal((init.body as FormData).get('file'), file)
    return Response.json(fixture)
  })
  assert.deepEqual(await api.predictVideo(file, controller.signal), fixture)
  assert.equal(api.isVideoPrediction({...fixture, average_fake_score: NaN}), false)
  assert.equal(api.isVideoPrediction({...fixture, status: 'insufficient_analysis'}), false)
})

test('video API errors, malformed results and aborts are surfaced', async t => {
  const file = new File(['x'], 'a.mp4'), signal = new AbortController().signal
  t.mock.method(globalThis, 'fetch', async () => Response.json({error: {message: 'Duration exceeds 120 seconds.'}}, {status: 400}))
  await assert.rejects(api.predictVideo(file, signal), /120 seconds/)
  t.mock.method(globalThis, 'fetch', async () => Response.json({success: true}))
  await assert.rejects(api.predictVideo(file, signal), /Unexpected video/)
  t.mock.method(globalThis, 'fetch', async () => { throw new DOMException('Aborted', 'AbortError') })
  await assert.rejects(api.predictVideo(file, signal), {name: 'AbortError'})
})

test('video results render scores, frames, faces and uncertainty', () => {
  const html = renderToStaticMarkup(React.createElement(video.VideoResults, {result: fixture}))
  for (const text of ['Experimental video assessment', '80.0%', '3 / 3', 'Frames without faces', 'Frame 0', 'Face 1', 'not proof', 'Not calibrated.']) assert.ok(html.includes(text), text)
  const insufficient = renderToStaticMarkup(React.createElement(video.VideoResults, {result: {...fixture, status: 'insufficient_analysis', predicted_label: null}}))
  assert.ok(insufficient.includes('Insufficient analysis'))
})

test('navigation and video upload controls render', () => {
  const app = component('App', {'./VideoAnalysis': {__esModule: true, default: video.default}})
  const html = renderToStaticMarkup(React.createElement(app.default))
  for (const label of ['Image Analysis', 'Video Analysis', 'About']) assert.ok(html.includes(label))
  const screen = renderToStaticMarkup(React.createElement(video.default, {ready: true}))
  for (const label of ['Browse video', 'Upload video', 'Analyze Video', 'aria-busy="false"']) assert.ok(screen.includes(label))
})


// Exercise the real event handlers with a small hook host, without adding a DOM dependency.
function hookHost(name: string, overrides: Record<string, unknown> = {}) {
  const slots: any[] = []; let cursor = 0
  const hooks = {...React,
    useState(initial: unknown) { const i = cursor++; if (!(i in slots)) slots[i] = initial; return [slots[i], (value: any) => { slots[i] = typeof value === 'function' ? value(slots[i]) : value }] },
    useRef(initial: unknown) { const i = cursor++; if (!(i in slots)) slots[i] = {current: initial}; return slots[i] },
    useEffect() {},
  }
  const loaded = component(name, {react: hooks, ...overrides})
  return {render(props: Record<string, unknown> = {}) { cursor = 0; return loaded.default(props) }}
}
function find(root: any, predicate: (node: any) => boolean): any {
  if (!root || typeof root !== 'object') return undefined
  if (predicate(root)) return root
  const children = root.props?.children
  for (const child of Array.isArray(children) ? children.flat(Infinity) : [children]) { const match = find(child, predicate); if (match) return match }
}

test('mode switches show the image, video and About screens', () => {
  const host = hookHost('App', {'./VideoAnalysis': {__esModule: true, default: video.default}})
  let tree = host.render()
  find(tree, n => n.type === 'button' && n.props.children === 'Video Analysis').props.onClick()
  tree = host.render()
  assert.ok(find(tree, n => n.type === video.default))
  assert.equal(find(tree, n => n.props?.className === 'workspace').props.hidden, true)
  find(tree, n => n.type === 'button' && n.props.children === 'About').props.onClick()
  tree = host.render()
  assert.equal(find(tree, n => n.props?.id === 'about').props.hidden, false)
  find(tree, n => n.type === 'button' && n.props.children === 'Image Analysis').props.onClick()
  assert.equal(find(host.render(), n => n.props?.className === 'workspace').props.hidden, false)
})

test('video upload, processing state, result and API failure handlers', async () => {
  let resolve: (value: unknown) => void = () => {}
  let reject: (reason: Error) => void = () => {}
  // Local client override controls network timing only; all UI handlers remain real.
  const fakeApi = {...api, predictVideo: () => new Promise((yes, no) => { resolve = yes; reject = no })}
  const host = hookHost('VideoAnalysis', {'./api': fakeApi})
  let tree = host.render({ready: true})
  const uploadInput = () => find(tree, n => n.type === 'input')
  uploadInput().props.onChange({target: {files: [new File(['x'], 'bad.webm')], value: ''}})
  tree = host.render({ready: true})
  assert.match(find(tree, n => n.props?.role === 'alert').props.children, /MP4/)
  uploadInput().props.onChange({target: {files: [new File(['x'], 'good.mp4')], value: ''}})
  tree = host.render({ready: true})
  const action = () => find(tree, n => n.props?.className === 'analyze-button')
  assert.equal(action().props.disabled, false)
  const pending = action().props.onClick()
  tree = host.render({ready: true})
  assert.equal(action().props.disabled, true)
  assert.ok(find(tree, n => n.props?.['aria-busy'] === true))
  resolve(fixture); await new Promise(yes => setImmediate(yes)); await pending
  tree = host.render({ready: true})
  assert.ok(find(tree, n => n.type === video.VideoResults || n.props?.result === fixture))
  action().props.onClick(); tree = host.render({ready: true})
  reject(new Error('Video analysis unavailable.')); await new Promise(yes => setImmediate(yes))
  tree = host.render({ready: true})
  assert.match(find(tree, n => n.props?.role === 'alert').props.children, /unavailable/)
  assert.equal(action().props.disabled, false)
})
