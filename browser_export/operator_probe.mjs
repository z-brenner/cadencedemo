/** Run synthetic ORT Web operators in real Chrome, with one WASM thread. */
import { createServer } from 'node:http'
import { readFile, stat, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'node:path'

const require = createRequire(path.join(process.cwd(), 'package.json'))
const { chromium } = require('playwright')
const tooling = process.env.PROBE_TOOLING || '/tmp/cadence-browser-probe'
const graphs = process.env.PROBE_GRAPHS || path.join(tooling, 'graphs')
const graphCases = JSON.parse(await readFile(path.join(graphs, 'cases.json'), 'utf8'))
const trainedLayerWeightsLoaded = graphCases.some(c => c.name.startsWith('trained_delta_layer_'))
const ortDist = path.join(tooling, 'node_modules/onnxruntime-web/dist')
const output = process.env.PROBE_OUTPUT || '/tmp/cadence-browser-probe/operator-evidence.json'
const softwareGpu = process.env.PROBE_SOFTWARE_GPU === '1'
const verbose = process.env.PROBE_VERBOSE === '1'
const caseFilter = process.env.PROBE_CASE || ''
const mime = { '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' }
const harness = `<!doctype html><title>Cadence synthetic operator probe</title>
<h1>Cadence synthetic operator probe</h1><pre id="result">Running small graphs…</pre>
<script type="module">
import * as ort from '/ort/ort.webgpu.min.mjs';
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = '/ort/';
ort.env.logLevel = ${JSON.stringify(verbose ? 'verbose' : 'warning')};
const allCases = await (await fetch('/graphs/cases.json')).json();
const cases = allCases.filter(c => !${JSON.stringify(caseFilter)} || c.name === ${JSON.stringify(caseFilter)});
window.runProbe = async (provider, disableCpuFallback = false) => {
  const result = {provider, disableCpuFallback, cases: []};
  const selectedCases = disableCpuFallback && cases[0]?.name === 'matmul' ? [...cases,
    {...cases[0],name:'forced_cpu_matmul_control',forceCpuNodeName:'matmul'}] : cases;
  for (const c of selectedCases) {
    const row = {name:c.name, operator:c.operator, bytes:c.bytes, sha256:c.sha256};
    if (c.sourceFixtureSha256) row.sourceFixtureSha256 = c.sourceFixtureSha256;
    let session;
    let phase = 'session-create';
    const start = performance.now();
    try {
      session = await ort.InferenceSession.create('/graphs/' + c.file,
        {executionProviders:[c.forceCpuNodeName ? {name:provider,forceCpuNodeNames:[c.forceCpuNodeName]} : provider],
          graphOptimizationLevel:'disabled', logSeverityLevel:${verbose ? 0 : 2}, logVerbosityLevel:${verbose ? 1 : 0},
          ...(disableCpuFallback ? {extra:{session:{disable_cpu_ep_fallback:'1'}}} : {})});
      row.sessionCreated = true;
      row.sessionCreateMs = performance.now() - start;
      phase = 'run';
      const feeds = {};
      for (const [name,t] of Object.entries(c.feeds)) {
        const data = t.dataFile ? new Float32Array(await (await fetch('/graphs/'+t.dataFile)).arrayBuffer()) :
          t.type === 'int64' ? BigInt64Array.from(t.data,BigInt) : Float32Array.from(t.data);
        feeds[name] = new ort.Tensor(t.type,data,t.dims);
      }
      const runStart = performance.now();
      const tensors = await session.run(feeds);
      row.runMs = performance.now() - runStart;
      row.outputs = {};
      row.errorsByOutput = {};
      for (const [name,t] of Object.entries(tensors)) {
        const data = Array.from(await t.getData(),Number);
        if (!data.every(Number.isFinite)) throw new Error('Nonfinite output: '+name);
        row.outputs[name] = {type:t.type,dims:t.dims,
          ...(c.expectedFiles ? {elements:data.length,firstValues:data.slice(0,8)} : {data})};
        if (c.expectedFiles) {
          const expected = new Float32Array(await (await fetch('/graphs/'+c.expectedFiles[name])).arrayBuffer());
          if (expected.length !== data.length) throw new Error('Reference output length mismatch: '+name);
          let maxAbs=0,maxRelative=0,sumError2=0,sumRef2=0;
          for (let i=0;i<data.length;i++) {
            const error=Math.abs(data[i]-expected[i]);
            maxAbs=Math.max(maxAbs,error);
            maxRelative=Math.max(maxRelative,error/Math.max(Math.abs(expected[i]),1e-6));
            sumError2+=error*error; sumRef2+=expected[i]*expected[i];
          }
          row.errorsByOutput[name]={maxAbsError:maxAbs,maxRelativeError:maxRelative,relativeDenominatorFloor:1e-6,
            relativeL2Error:Math.sqrt(sumError2/Math.max(sumRef2,1e-30))};
        }
      }
      row.maxAbsError = c.expectedFiles ? Math.max(...Object.values(row.errorsByOutput).map(m=>m.maxAbsError)) :
        c.expected ? Math.max(...Object.entries(c.expected).flatMap(([name,want]) =>
          want.map((v,i)=>Math.abs(v-row.outputs[name].data[i])))) : null;
      row.passed = row.maxAbsError !== null && row.maxAbsError <= 1e-5;
    } catch (error) {
      row.sessionCreated ??= false;
      row.passed = false;
      row.failurePhase = phase;
      row.error = String(error);
    } finally {
      if (session) await session.release();
      row.elapsedMs = performance.now() - start;
    }
    result.cases.push(row);
  }
  document.querySelector('#result').textContent = JSON.stringify(result,null,2);
  return result;
};
window.probeReady = true;
</script>`

const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname
    response.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
    response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
    if (pathname === '/') {
      response.setHeader('Content-Type', 'text/html'); response.end(harness); return
    }
    const root = pathname.startsWith('/ort/') ? ortDist : graphs
    const relative = pathname.replace(/^\/(ort|graphs)\//, '')
    const file = path.resolve(root, relative)
    if (!file.startsWith(root + path.sep)) throw new Error('Invalid local path')
    response.setHeader('Content-Type', mime[path.extname(file)] || 'application/octet-stream')
    response.setHeader('Content-Length', (await stat(file)).size)
    response.end(await readFile(file))
  } catch { response.writeHead(404); response.end('Not found') }
})
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
let browser
try {
  browser = await chromium.launch({channel:'chrome',headless:true,
    args: softwareGpu ? ['--enable-unsafe-webgpu','--use-angle=swiftshader','--enable-unsafe-swiftshader'] : []})
  const page = await browser.newPage({viewport:{width:1000,height:750}})
  const logs = []
  page.on('console', msg => logs.push({type:msg.type(),text:msg.text()}))
  page.on('pageerror', error => logs.push({type:'pageerror',text:String(error)}))
  const url = `http://127.0.0.1:${server.address().port}/`
  await page.goto(url)
  await page.waitForFunction(() => window.probeReady === true, {timeout:30000})
  const capability = await page.evaluate(async () => {
    const adapter = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
    return {userAgent:navigator.userAgent,webgpuApi:!!navigator.gpu,adapterAvailable:!!adapter,
      adapterInfo:adapter?.info ? {vendor:adapter.info.vendor,architecture:adapter.info.architecture,
        device:adapter.info.device,description:adapter.info.description,isFallbackAdapter:adapter.info.isFallbackAdapter} : null,
      features:adapter ? Array.from(adapter.features):[],
      limits:adapter ? {maxBufferSize:adapter.limits.maxBufferSize,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize}:null};
  })
  const runs = []
  runs.push(await page.evaluate(() => window.runProbe('wasm')))
  if (capability.adapterAvailable) {
    runs.push(await page.evaluate(() => window.runProbe('webgpu')))
    runs.push(await page.evaluate(() => window.runProbe('webgpu', true)))
  }
  else runs.push({provider:'webgpu',blocked:'navigator.gpu.requestAdapter() returned no adapter',cases:[]})
  const ortPackage = JSON.parse(await readFile(path.join(tooling,'node_modules/onnxruntime-web/package.json'),'utf8'))
  const evidence = {kind:trainedLayerWeightsLoaded ? 'trained-layer-browser-probe' : 'synthetic-browser-operator-probe',
    modelWeightsLoaded:trainedLayerWeightsLoaded,fullCadenceModelLoaded:false,trainedLayerWeightsLoaded,
    createdAt:new Date().toISOString(),ortWebVersion:ortPackage.version,browserVersion:browser.version(),
    browserAutomation:'Playwright; Browser skill absent',softwareGpuRequested:softwareGpu,
    wasmThreads:1,graphOptimizationLevel:'disabled',url,capability,runs,logs}
  await writeFile(output,JSON.stringify(evidence,null,2)+'\n')
  console.log(JSON.stringify({output,adapterAvailable:capability.adapterAvailable,
    adapterArchitecture:capability.adapterInfo?.architecture,
    runs:runs.map(r=>({provider:r.provider,blocked:r.blocked,disableCpuFallback:r.disableCpuFallback,
      passed:r.cases.filter(c=>c.passed).length,total:r.cases.length,
      failures:r.cases.filter(c=>!c.passed).map(c=>c.name)}))},null,2))
} finally {
  if (browser) await browser.close()
  await new Promise(resolve => server.close(resolve))
}
