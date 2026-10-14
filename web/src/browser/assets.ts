import { revision } from '../contract'

export type Packing = {
  offset: number
  bytes: number
  decodedOffset: number
  decodedBytes: number
  encoding: string
}
export type Graph = {
  name: string
  graph: string
  weights: string
  externalPath: string
  decodedBytes: number
  packing: Packing[]
  inputs: string[]
}
export type Manifest = {
  format: string
  model: string
  revision: string
  maxTokens: number
  hiddenSize: number
  vocabSize: number
  rotaryWidth: number
  embeddingShards: { file: string; rowStart: number; rows: number }[]
  rotary: string
  graphs: Graph[]
  files: Record<string, { bytes: number; sha256: string }>
  assetBytesExcludingManifest: number
}

export function unpackWeights(
  packed: Uint8Array,
  decodedBytes: number,
  parts: Packing[],
): Uint8Array {
  if (
    !Number.isSafeInteger(decodedBytes) ||
    decodedBytes <= 0 ||
    decodedBytes > 256 * 1024 * 1024 ||
    decodedBytes % 4
  )
    throw new Error('Invalid decoded weight size.')
  const decoded = new Uint8Array(decodedBytes)
  const words = new Uint32Array(decoded.buffer)
  const view = new DataView(packed.buffer, packed.byteOffset, packed.byteLength)
  let inputOffset = 0,
    outputOffset = 0
  for (const part of parts) {
    if (
      ![part.offset, part.bytes, part.decodedOffset, part.decodedBytes].every(
        Number.isSafeInteger,
      ) ||
      part.offset !== inputOffset ||
      part.decodedOffset !== outputOffset ||
      part.bytes <= 0 ||
      part.decodedBytes <= 0 ||
      inputOffset + part.bytes > packed.byteLength ||
      outputOffset + part.decodedBytes > decodedBytes
    )
      throw new Error('Invalid packed weight boundaries.')
    if (
      part.encoding === 'bf16-bits' &&
      part.bytes % 2 === 0 &&
      part.decodedBytes === part.bytes * 2
    ) {
      for (let i = 0; i < part.bytes / 2; i++)
        words[outputOffset / 4 + i] = view.getUint16(inputOffset + i * 2, true) << 16
    } else if (
      part.encoding === 'fp32-bits' &&
      part.bytes % 4 === 0 &&
      part.decodedBytes === part.bytes
    ) {
      decoded.set(packed.subarray(inputOffset, inputOffset + part.bytes), outputOffset)
    } else throw new Error('Unknown or inconsistent packed weight encoding.')
    inputOffset += part.bytes
    outputOffset += part.decodedBytes
  }
  if (inputOffset !== packed.byteLength || outputOffset !== decodedBytes)
    throw new Error('Incomplete packed weight data.')
  return decoded
}

export function validateManifest(value: unknown): Manifest {
  const m = value as Manifest
  if (
    !m ||
    m.format !== 'cadence-browser-layer-stack-v1' ||
    m.model !== 'foolish-bandit/cadence-0.8b' ||
    m.revision !== revision ||
    m.maxTokens !== 512 ||
    m.hiddenSize !== 1024 ||
    m.vocabSize !== 248320 ||
    m.rotaryWidth !== 64 ||
    !Array.isArray(m.graphs) ||
    m.graphs.length !== 25 ||
    !Array.isArray(m.embeddingShards) ||
    m.embeddingShards.length !== 31 ||
    !m.files
  )
    throw new Error('Model files do not match this version of the demo.')
  for (const [name, file] of Object.entries(m.files)) {
    if (
      !/^[A-Za-z0-9_.-]+$/.test(name) ||
      !file ||
      !Number.isSafeInteger(file.bytes) ||
      file.bytes <= 0 ||
      file.bytes > 64 * 1024 * 1024 ||
      !/^[a-f0-9]{64}$/.test(file.sha256)
    )
      throw new Error('Invalid model file manifest.')
  }
  if (
    Object.values(m.files).reduce((n, f) => n + f.bytes, 0) !== m.assetBytesExcludingManifest ||
    m.assetBytesExcludingManifest > 1.7e9
  )
    throw new Error('Invalid model download size.')
  m.graphs.forEach((g, i) => {
    if (
      g.name !== (i < 24 ? `layer-${String(i).padStart(2, '0')}` : 'readout') ||
      !m.files[g.graph] ||
      !m.files[g.weights] ||
      g.externalPath !== `${g.name}.fp32` ||
      !Array.isArray(g.packing) ||
      !Array.isArray(g.inputs)
    )
      throw new Error('Invalid model layer manifest.')
  })
  let row = 0
  for (const s of m.embeddingShards) {
    if (
      s.rowStart !== row ||
      !Number.isSafeInteger(s.rows) ||
      s.rows <= 0 ||
      s.rows > 8192 ||
      m.files[s.file]?.bytes !== s.rows * m.hiddenSize * 2
    )
      throw new Error('Invalid model embedding shards.')
    row += s.rows
  }
  if (
    row !== m.vocabSize ||
    !m.files[m.rotary] ||
    m.files[m.rotary].bytes !== 2 * m.maxTokens * m.rotaryWidth * 4
  )
    throw new Error('Invalid model position data.')
  return m
}

export async function sha256(bytes: Uint8Array): Promise<string> {
  const hash = await crypto.subtle.digest('SHA-256', new Uint8Array(bytes).buffer)
  return Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, '0')).join('')
}
