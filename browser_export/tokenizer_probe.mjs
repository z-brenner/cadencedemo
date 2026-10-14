/** Actual official JS tokenizer versus frozen native encoder fixtures; no inference/downloads. */
import assert from 'node:assert/strict'
import { readFile, writeFile, stat, mkdtemp } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { pathToFileURL } from 'node:url'
import { resolve, join } from 'node:path'
import { tmpdir } from 'node:os'
import { build } from 'esbuild'

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, argument, i, all) => {
  if (argument.startsWith('--')) pairs.push([argument.slice(2), all[i + 1]])
  return pairs
}, []))
if (!args['tokenizer-dir']) throw new Error('Provide --tokenizer-dir with the exact cached pinned tokenizer files.')
const libraryDir = resolve(args['library-dir'] ?? '/tmp/cadence-tokenizer-probe/node_modules/@huggingface/tokenizers')
const libraryManifest = JSON.parse(await readFile(join(libraryDir, 'package.json'), 'utf8'))
assert.equal(libraryManifest.version, '0.2.0', 'Only the reviewed library version is qualified.')
const { Tokenizer } = await import(pathToFileURL(join(libraryDir, 'dist/tokenizers.mjs')))
const files = ['tokenizer.json', 'tokenizer_config.json']
const bytes = await Promise.all(files.map((name) => readFile(join(args['tokenizer-dir'], name))))
const hashes = bytes.map((value) => createHash('sha256').update(value).digest('hex'))
assert.deepEqual(hashes, ['fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927', 'e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de'], 'Tokenizer files must match the pinned upstream snapshot.')
const tokenizer = new Tokenizer(JSON.parse(bytes[0]), JSON.parse(bytes[1]))
const temp = await mkdtemp(join(tmpdir(), 'cadence-encoding-'))
const encodingPath = join(temp, 'encoding.mjs')
await build({ entryPoints: [resolve('web/src/browser/encoding.ts')], outfile: encodingPath, bundle: true, platform: 'browser', format: 'esm', target: 'es2022' })
const { encodeRequest } = await import(pathToFileURL(encodingPath))
const references = [args.reference ?? 'docs/browser-native-reference.json', ...(args.extra ? [args.extra] : [])]
const checks = []
const browserCases = []
for (const path of references) {
  const reference = JSON.parse(await readFile(path, 'utf8'))
  for (const fixture of reference.cases) {
    const result = encodeRequest(tokenizer, fixture.request)
    assert.equal(result.record.state, fixture.nativeRecord.state, `${fixture.name}: rendered state`)
    assert.deepEqual(result.record.questions[0].options, fixture.nativeRecord.questions[0].options, `${fixture.name}: option texts`)
    assert.deepEqual(result.encoding, fixture.encoding, `${fixture.name}: every encoding field`)
    if (fixture.cpuFp32) assert.deepEqual(result.optionIds, fixture.cpuFp32.native_option_ids)
    browserCases.push(fixture)
    checks.push({ name: fixture.name, tokens: result.encoding.ids.length, tokenIdsExact: true, readoutIndicesExact: true, allEncodingFieldsExact: true })
  }
}
let browserVerification = null
if (args.browser === 'true') {
  const { chromium } = await import('@playwright/test')
  const browser = await chromium.launch({ channel: 'chrome' })
  try {
    const page = await browser.newPage()
    let networkRequests = 0
    page.on('request', () => { networkRequests += 1 })
    const bundled = await build({ stdin: { contents: `import { Tokenizer } from ${JSON.stringify(join(libraryDir, 'dist/tokenizers.mjs'))}; import { encodeRequest } from ${JSON.stringify(resolve('web/src/browser/encoding.ts'))}; globalThis.cadenceTokenizerProbe = { Tokenizer, encodeRequest };`, resolveDir: process.cwd() }, bundle: true, platform: 'browser', format: 'iife', target: 'es2022', write: false })
    await page.addScriptTag({ content: bundled.outputFiles[0].text })
    browserVerification = await page.evaluate(({ tokenizerFiles, cases }) => {
      const { Tokenizer, encodeRequest } = globalThis.cadenceTokenizerProbe
      const tokenizer = new Tokenizer(...tokenizerFiles)
      for (const fixture of cases) {
        const result = encodeRequest(tokenizer, fixture.request)
        if (JSON.stringify(result.encoding) !== JSON.stringify(fixture.encoding)) throw new Error(`${fixture.name}: browser encoding differs`)
        if (result.record.state !== fixture.nativeRecord.state) throw new Error(`${fixture.name}: browser state differs`)
      }
      return { userAgent: navigator.userAgent, casesChecked: cases.length, allExact: true, modelInferenceRun: false }
    }, { tokenizerFiles: bytes.map((value) => JSON.parse(value)), cases: browserCases })
    browserVerification.networkRequests = networkRequests
  } finally {
    await browser.close()
  }
}
const report = {
  tokenizerLibrary: '@huggingface/tokenizers', tokenizerLibraryVersion: libraryManifest.version,
  tokenizerLibraryUnpackedBytes: 360962, tokenizerLibraryModuleBytes: (await stat(join(libraryDir, 'dist/tokenizers.mjs'))).size,
  tokenizerModel: 'Qwen/Qwen3.5-0.8B-Base', tokenizerRevision: 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68',
  tokenizerFiles: files.map((name, i) => ({ name, bytes: bytes[i].length, sha256: hashes[i] })),
  executionEnvironment: 'Node.js with official browser-compatible pure JS tokenizer',
  modelInferenceRun: false, nativeCasesChecked: checks.length, allExact: true, checks,
  browserVerification,
  limitations: ['One question per causal row; no option isolation.', 'Canonical browser JSON wire values; no raw numeric lexeme preservation.', 'JSON nesting above 128 levels is rejected. Numeric integers beyond the safe JavaScript range and nonfinite numbers are rejected.', 'Tokenizer parity alone does not establish browser model inference parity.'],
}
if (args.output) await writeFile(args.output, JSON.stringify(report, null, 2) + '\n')
console.log(JSON.stringify({ ...report, checks: undefined }, null, 2))
