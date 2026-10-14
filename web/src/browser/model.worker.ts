import { Tokenizer } from '@huggingface/tokenizers'
import * as ort from 'onnxruntime-web/webgpu'
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url'
import mjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url'
import { encodeRequest } from './encoding'
import { PreparationProgress } from './progress'
import { sha256, unpackWeights, validateManifest, type Manifest } from './assets'
import { revision, type DecisionRequest, type Result } from '../contract'

type Message = {
  type: string
  assetBase?: string
  manifestSha?: string
  requestId?: string
  request?: DecisionRequest
}
const scope = globalThis as unknown as {
  onmessage: ((event: MessageEvent<Message>) => void) | null
  postMessage(value: unknown): void
  location: Location
}
const send = (value: unknown) => scope.postMessage(value)
let manifest: Manifest | null = null,
  tokenizer: Tokenizer | null = null,
  cache: Cache | null = null
let assetBase = '',
  manifestSha = '',
  busy = false,
  progress: PreparationProgress | null = null,
  fromCache = true
class BrowserInputError extends Error {}

function status(status: string, stage: string) {
  send({
    type: 'status',
    status,
    stage,
    ...progress?.snapshot(),
    fromCache,
  })
}
function url(file: string) {
  return new URL(file, assetBase).href
}

async function savedBytes(file: string, repairInvalid = false): Promise<Uint8Array | null> {
  const expected = manifest!.files[file]
  if (!expected) throw new Error('Model file is absent from the verified manifest.')
  const response = await cache!.match(url(file))
  if (!response) return null
  const value = new Uint8Array(await response.arrayBuffer())
  if (value.byteLength !== expected.bytes || (await sha256(value)) !== expected.sha256) {
    await cache!.delete(url(file))
    if (!repairInvalid)
      throw new Error('A saved model file failed verification. Download the model again to repair it.')
    return null
  }
  return value
}

async function bytes(file: string, downloading = false): Promise<Uint8Array> {
  if (!downloading) {
    const value = await savedBytes(file)
    if (!value) throw new Error('Saved model files are missing. Download the model again.')
    return value
  }
  const expected = manifest!.files[file]
  if (!expected) throw new Error('Model file is absent from the verified manifest.')
  status('downloading', `Downloading ${file}`)
  const response = await fetch(url(file), { credentials: 'omit', referrerPolicy: 'no-referrer' })
  if (!response.ok || !response.body)
    throw new Error(`Model download failed (${response.status}). Check your connection and retry.`)
  let value: Uint8Array
  const reader = response.body.getReader(),
    chunks: Uint8Array[] = []
  let count = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    count += value.byteLength
    if (count > expected.bytes) {
      await reader.cancel()
      throw new Error('A model download exceeded its verified size.')
    }
    chunks.push(value)
    progress!.received(file, count)
    status('downloading', `Downloading ${file}`)
  }
  progress!.finished(file)
  status('verifying', `Verifying and saving ${file}`)
  value = new Uint8Array(count)
  let offset = 0
  for (const chunk of chunks) {
    value.set(chunk, offset)
    offset += chunk.byteLength
  }
  if (count !== expected.bytes || (await sha256(value)) !== expected.sha256)
    throw new Error(
      'A model download failed verification. Retry to fetch the correct pinned file.',
    )
  try {
    await cache!.put(url(file), new Response(new Uint8Array(value).buffer))
  } catch {
    throw new Error(
      'The browser could not save the model. Free local storage or allow storage for this site, then retry.',
    )
  }
  progress!.verified(file)
  return value!
}

async function prepare(message: Message) {
  if (!message.assetBase || !message.manifestSha || !/^[a-f0-9]{64}$/.test(message.manifestSha))
    throw new Error('Browser model files are not configured.')
  const base = new URL(message.assetBase, scope.location.href)
  if (base.protocol !== 'https:' && base.origin !== scope.location.origin)
    throw new Error('Model downloads require HTTPS.')
  if (!base.pathname.endsWith('/')) throw new Error('Invalid model download location.')
  if (
    !globalThis.crypto?.subtle ||
    !globalThis.caches ||
    !globalThis.WebAssembly ||
    typeof BigInt64Array === 'undefined'
  )
    throw new Error(
      'This browser cannot run or save this model. Try a current desktop browser with local storage enabled.',
    )
  assetBase = base.href
  manifestSha = message.manifestSha
  progress = null
  fromCache = true
  status('checking', 'Checking model files and local storage')
  const response = await fetch(url('manifest.json'), {
    credentials: 'omit',
    referrerPolicy: 'no-referrer',
  })
  if (!response.ok)
    throw new Error('Could not download the model manifest. Check your connection and retry.')
  const raw = new Uint8Array(await response.arrayBuffer())
  if (raw.byteLength > 256 * 1024 || (await sha256(raw)) !== manifestSha)
    throw new Error('Model manifest does not match the pinned browser release.')
  manifest = validateManifest(JSON.parse(new TextDecoder().decode(raw)))
  try {
    cache = await caches.open(`cadence-browser-${manifestSha}`)
  } catch {
    throw new Error(
      'This browser cannot save model files. Enable local storage or try another desktop browser.',
    )
  }
  progress = new PreparationProgress(manifest.files)
  const missing: string[] = []
  status('checking', 'Verifying saved model files')
  for (const file of Object.keys(manifest.files)) {
    const reused = (await savedBytes(file, true)) !== null
    progress.checked(file, reused)
    if (!reused) missing.push(file)
    status('checking', `Checked saved file ${file}`)
  }
  fromCache = missing.length === 0
  for (const file of missing) await bytes(file, true)
  if (!progress.snapshot().filesReady) throw new Error('Model files are not fully verified.')
  status('loading', 'All model files verified; preparing tokenizer')
  tokenizer = new Tokenizer(
    JSON.parse(new TextDecoder().decode(await bytes('tokenizer.json'))),
    JSON.parse(new TextDecoder().decode(await bytes('tokenizer_config.json'))),
  )
  ort.env.wasm.numThreads = 1
  ort.env.wasm.proxy = false
  ort.env.wasm.wasmPaths = {
    wasm: new URL(wasmUrl, scope.location.href).href,
    mjs: new URL(mjsUrl, scope.location.href).href,
  }
  // WASM is the qualified physical-browser path. WebGPU remains experimental.
  send({ type: 'ready', provider: 'wasm', fromCache })
}

async function run(request: DecisionRequest): Promise<Result> {
  if (!manifest || !tokenizer || !cache) throw new Error('Download and prepare the model first.')
  const began = performance.now()
  let prepared: ReturnType<typeof encodeRequest>
  try {
    prepared = encodeRequest(tokenizer, request, { maxState: 512, maxTokens: 512 })
  } catch (error) {
    throw new BrowserInputError((error as Error).message)
  }
  const { encoding, optionIds } = prepared
  const n = encoding.ids.length,
    d = manifest.hiddenSize,
    r = manifest.rotaryWidth
  let hidden = new Float32Array(n * d)
  const words = new Uint32Array(hidden.buffer)
  for (const shard of manifest.embeddingShards) {
    const wanted = encoding.ids.flatMap((id, i) =>
      id >= shard.rowStart && id < shard.rowStart + shard.rows ? [i] : [],
    )
    if (!wanted.length) continue
    status('running', 'Reading context and possible answers')
    const data = new DataView((await bytes(shard.file)).buffer)
    for (const i of wanted)
      for (let j = 0; j < d; j++)
        words[i * d + j] =
          data.getUint16(((encoding.ids[i] - shard.rowStart) * d + j) * 2, true) << 16
  }
  const rotary = new Float32Array((await bytes(manifest.rotary)).buffer)
  const cos = rotary.slice(0, n * r),
    sin = rotary.slice(manifest.maxTokens * r, manifest.maxTokens * r + n * r)
  const mask = new Float32Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) mask[i * n + j] = -3.4028234663852886e38
  for (let index = 0; index < manifest.graphs.length; index++) {
    const graph = manifest.graphs[index]
    status(
      'running',
      index < 24 ? `Evaluating model layer ${index + 1} of 24` : 'Comparing answers',
    )
    const weights = unpackWeights(await bytes(graph.weights), graph.decodedBytes, graph.packing)
    const session = await ort.InferenceSession.create(await bytes(graph.graph), {
      executionProviders: ['wasm'],
      externalData: [{ path: graph.externalPath, data: weights }],
      graphOptimizationLevel: 'all',
    })
    const data = {
      hidden,
      cos,
      sin,
      mask,
      options: BigInt64Array.from(encoding.opt_idx[0], BigInt),
      decide: BigInt64Array.from(encoding.decide_idx, BigInt),
    }
    const feeds: Record<string, ort.Tensor> = {}
    try {
      for (const key of graph.inputs) {
        const value = data[key as keyof typeof data]
        const dims =
          key === 'hidden'
            ? [1, n, d]
            : key === 'cos' || key === 'sin'
              ? [1, n, r]
              : key === 'mask'
                ? [1, 1, n, n]
                : [value.length]
        feeds[key] = new ort.Tensor(
          key === 'options' || key === 'decide' ? 'int64' : 'float32',
          value,
          dims,
        )
      }
      const outputs = await session.run(feeds)
      try {
        if (graph.name === 'readout') {
          const logits = Array.from((await outputs.logits.getData()) as Float32Array)
          if (logits.length !== optionIds.length || logits.some((z) => !Number.isFinite(z)))
            throw new Error('The browser model returned invalid scores.')
          const maximum = Math.max(...logits),
            exp = logits.map((z) => Math.exp(z - maximum)),
            sum = exp.reduce((a, b) => a + b, 0)
          const probabilities = Object.fromEntries(optionIds.map((id, i) => [id, exp[i] / sum]))
          const kind = request.questions.decision.type
          return {
            kind,
            native_option_ids: optionIds,
            probabilities,
            value:
              kind === 'noul'
                ? probabilities.true
                : kind === 'score'
                  ? optionIds.reduce((v, id) => v + Number(id) * probabilities[id], 0)
                  : null,
            model: 'foolish-bandit/cadence-0.8b',
            revision,
            device: 'browser',
            profile: 'browser_fp32',
            artifact_manifest_sha256: manifestSha,
            tokens: n,
            seconds: (performance.now() - began) / 1000,
            truncated: false,
          }
        }
        hidden = new Float32Array((await outputs.output.getData()) as Float32Array)
      } finally {
        Object.values(outputs).forEach((t) => t.dispose())
      }
    } finally {
      Object.values(feeds).forEach((t) => t.dispose())
      await session.release()
    }
  }
  throw new Error('The browser model did not produce a decision.')
}

scope.onmessage = (event) => {
  const message = event.data
  if (busy) {
    send({
      type: 'error',
      requestId: message.requestId,
      message: 'Wait for the current model operation to finish.',
    })
    return
  }
  busy = true
  void (async () => {
    try {
      if (message.type === 'prepare') await prepare(message)
      else if (message.type === 'run' && message.request)
        send({ type: 'result', requestId: message.requestId, result: await run(message.request) })
      else if (message.type === 'clear-cache') {
        await caches.delete(`cadence-browser-${manifestSha}`)
        cache = null
        tokenizer = null
        manifest = null
        send({ type: 'cleared' })
      } else throw new Error('Unknown browser model operation.')
    } catch (error) {
      send({
        type: error instanceof BrowserInputError ? 'input-error' : 'error',
        requestId: message.requestId,
        message: (error as Error).message,
      })
    } finally {
      busy = false
    }
  })()
}
