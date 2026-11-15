import assert from 'node:assert/strict'
import { test } from 'node:test'
import { isPrediction, predict } from '../src/api.ts'
import { validateImage } from '../src/upload.ts'

// Controlled API fixtures only; these are never shipped as application predictions.
const fixture = { success: true, predicted_label: 'fake', probabilities: { real: .25, fake: .75 },
  threshold: .5, model_name: 'unit-fixture', model_version: 'unit-test', warning: 'not proof' }

test('prediction validation rejects malformed probabilities and missing metadata', () => {
  assert.equal(isPrediction(fixture), true)
  for (const value of [null, {}, { ...fixture, probabilities: { real: .5, fake: .7 } },
    { ...fixture, probabilities: { real: NaN, fake: .5 } }, { ...fixture, probabilities: { real: -1, fake: 2 } },
    { ...fixture, predicted_label: 'authentic' }, { ...fixture, warning: null }]) assert.equal(isPrediction(value), false)
})

test('upload validation accepts image types and rejects empty, oversize and wrong types', () => {
  for (const type of ['image/jpeg', 'image/png', 'image/webp']) assert.equal(validateImage(new File(['x'], 'image', { type })), null)
  assert.match(validateImage(new File(['x'], 'bad', { type: 'text/plain' }))!, /JPEG/)
  assert.match(validateImage(new File([], 'empty', { type: 'image/png' }))!, /nonempty/)
  assert.match(validateImage(new File([new Uint8Array(10 * 1024 * 1024 + 1)], 'large', { type: 'image/png' }))!, /10 MB/)
})

test('API sends multipart field and abort signal, then returns checked prediction', async t => {
  const file = new File(['fixture'], 'image.png', { type: 'image/png' })
  const controller = new AbortController()
  t.mock.method(globalThis, 'fetch', async (url: string, init: RequestInit) => {
    assert.equal(url, '/api/v1/detect/image'); assert.equal(init.method, 'POST')
    assert.equal(init.signal, controller.signal); assert.equal((init.body as FormData).get('file'), file)
    return Response.json(fixture)
  })
  assert.deepEqual(await predict(file, controller.signal), fixture)
})

test('API errors preserve the structured server message for retry feedback', async t => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ success: false, error: { message: 'Model unavailable.' } }, { status: 503 }))
  await assert.rejects(predict(new File(['x'], 'image'), new AbortController().signal), /Model unavailable/)
})

test('API rejects malformed successful responses', async t => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ success: true }))
  await assert.rejects(predict(new File(['x'], 'image'), new AbortController().signal), /Unexpected prediction/)
})

test('API propagates network and abort errors', async t => {
  t.mock.method(globalThis, 'fetch', async () => { throw new DOMException('Aborted', 'AbortError') })
  await assert.rejects(predict(new File(['x'], 'image'), new AbortController().signal), { name: 'AbortError' })
})
