/** Real full-stack browser parity probe; no mocked model or remote inference. */
import { createServer } from 'node:http'
import { createReadStream } from 'node:fs'
import { readFile, stat, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'node:path'
const require = createRequire(path.join(process.cwd(), 'package.json'))
const { chromium } = require('playwright')
const assets = process.env.CADENCE_BROWSER_ASSETS || '/tmp/cadence-browser-export-v1'
const ortDist = '/tmp/cadence-browser-probe/node_modules/onnxruntime-web/dist'
const output = process.env.STACK_PROBE_OUTPUT || '/tmp/cadence-full-browser-parity.json'
const softwareGpu = process.env.PROBE_SOFTWARE_GPU === '1'
const provider = softwareGpu ? 'webgpu' : 'wasm'
const harness = `<!doctype html><title>Cadence full model — real inference probe</title>
<h1>Cadence full model — real inference probe</h1><pre id="status">Preparing…</pre>
<script type="module">
import * as ort from '/ort/ort.webgpu.min.mjs';
ort.env.wasm.numThreads=1; ort.env.wasm.proxy=false; ort.env.wasm.wasmPaths='/ort/';
ort.env.logLevel='warning';
const status = message => document.querySelector('#status').textContent=message;
window.runStack=async()=>{
 const began=performance.now();
 const manifest=await (await fetch('/assets/manifest.json')).json();
 const reference=await(await fetch('/reference.json')).json();
 const rotary=new Float32Array(await(await fetch('/assets/'+manifest.rotary)).arrayBuffer());
 const inputs=reference.cases.map(c=>{
   const ids=c.encoding.ids, n=ids.length, d=manifest.hiddenSize, r=manifest.rotaryWidth;
   const hidden=new Float32Array(n*d); const bits=new Uint32Array(hidden.buffer);
   const mask=new Float32Array(n*n);
   for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)mask[i*n+j]=-3.4028234663852886e38;
   return {name:c.name,n,hidden,cos:rotary.slice(0,n*r),
     sin:rotary.slice(manifest.maxTokens*r,manifest.maxTokens*r+n*r),mask,
     options:BigInt64Array.from(c.encoding.opt_idx[0],BigInt),decide:BigInt64Array.from(c.encoding.decide_idx,BigInt)};
 });
 for(const shard of manifest.embeddingShards){
   if(!reference.cases.some(c=>c.encoding.ids.some(id=>id>=shard.rowStart&&id<shard.rowStart+shard.rows)))continue;
   const embedding=new Uint16Array(await(await fetch('/assets/'+shard.file)).arrayBuffer());
   for(let c=0;c<inputs.length;c++){
     const ids=reference.cases[c].encoding.ids,d=manifest.hiddenSize,bits=new Uint32Array(inputs[c].hidden.buffer);
     if(reference.cases[c].encoding.pos.some((p,i)=>p!==i))throw Error('Only one causal row is supported');
     for(let i=0;i<ids.length;i++)if(ids[i]>=shard.rowStart&&ids[i]<shard.rowStart+shard.rows)
       for(let j=0;j<d;j++)bits[i*d+j]=embedding[(ids[i]-shard.rowStart)*d+j]<<16;
   }
 }
 const stages=[]; const results=[];
 for(const graph of manifest.graphs){
   const start=performance.now(); status('Loading '+graph.name);
   const packed=new Uint8Array(await(await fetch('/assets/'+graph.weights)).arrayBuffer());
   const decoded=new Uint8Array(graph.decodedBytes); const decodedBits=new Uint32Array(decoded.buffer);
   for(const part of graph.packing){
     if(part.encoding==='bf16-bits'){
       const shorts=new Uint16Array(packed.buffer,part.offset,part.bytes/2);
       for(let i=0;i<shorts.length;i++)decodedBits[part.decodedOffset/4+i]=shorts[i]<<16;
     }else decoded.set(packed.subarray(part.offset,part.offset+part.bytes),part.decodedOffset);
   }
   const graphBytes=new Uint8Array(await(await fetch('/assets/'+graph.graph)).arrayBuffer());
   const session=await ort.InferenceSession.create(graphBytes,{executionProviders:['${provider}'],
     externalData:[{path:graph.externalPath,data:decoded}],graphOptimizationLevel:'all'});
   const loaded=performance.now(); let runs=0;
   try{
     for(let i=0;i<inputs.length;i++){
       const c=inputs[i]; status(graph.name+' · '+c.name);
       const feeds={};
       for(const key of graph.inputs){
         const data=c[key];
         const dims=key==='hidden'?[1,c.n,manifest.hiddenSize]:key==='cos'||key==='sin'?[1,c.n,manifest.rotaryWidth]:key==='mask'?[1,1,c.n,c.n]:[data.length];
         feeds[key]=new ort.Tensor(key==='options'||key==='decide'?'int64':'float32',data,dims);
       }
       const runStart=performance.now(); const tensors=await session.run(feeds); runs+=performance.now()-runStart;
       if(graph.name==='readout'){
         const logits=Array.from(await tensors.logits.getData()); const max=Math.max(...logits);
         const exps=logits.map(x=>Math.exp(x-max)); const sum=exps.reduce((a,b)=>a+b,0); const probabilities=exps.map(x=>x/sum);
         const native=reference.cases[i].cpuFp32; const wantP=native.native_option_ids.map(id=>native.probabilities[id]),wantZ=native.native_option_ids.map(id=>native.logits[id]);
         if(logits.length!==wantZ.length||logits.some(z=>!Number.isFinite(z)))throw Error('Invalid model logits');
         results.push({name:c.name,logits,probabilities,
           maxLogitDrift:Math.max(...logits.map((x,j)=>Math.abs(x-wantZ[j]))),
           maxProbabilityDrift:Math.max(...probabilities.map((x,j)=>Math.abs(x-wantP[j]))),
           sameLeadingIndex:probabilities.indexOf(Math.max(...probabilities))===wantP.indexOf(Math.max(...wantP)),
           nativeOptionIds:native.native_option_ids,
           value:native.kind==='noul'?probabilities[1]:native.kind==='score'?probabilities.reduce((a,p,j)=>a+p*j,0):null,
           nativeValue:native.value??null});
       }else c.hidden=new Float32Array(await tensors.output.getData());
       for(const t of Object.values(tensors))t.dispose();
       for(const t of Object.values(feeds))t.dispose();
     }
   }finally{await session.release();}
   stages.push({name:graph.name,loadMs:loaded-start,runAllFixturesMs:runs,totalMs:performance.now()-start});
 }
 const result={inference:'real full Cadence',provider:'${provider}',softwareGpu:${softwareGpu},
   revision:manifest.revision,profile:manifest.profile,assetBytes:manifest.assetBytesExcludingManifest,
   totalMs:performance.now()-began,userAgent:navigator.userAgent,stages,results,
   memory:performance.memory?{usedJSHeapSize:performance.memory.usedJSHeapSize,totalJSHeapSize:performance.memory.totalJSHeapSize}:null};
 result.tolerances={maxAbsoluteLogitDrift:1e-4,maxAbsoluteProbabilityDrift:1e-5,maxAbsoluteDerivedValueDrift:1e-5};
 result.passed=results.length===reference.cases.length&&results.every(c=>c.maxLogitDrift<=1e-4&&c.maxProbabilityDrift<=1e-5&&c.sameLeadingIndex&&(c.value===null||Math.abs(c.value-c.nativeValue)<=1e-5));
 status(JSON.stringify(result,null,2)); return result;
}; window.ready=true;
</script>`
const server = createServer(async (req, res) => {
  try {
    const pathname = new URL(req.url, 'http://localhost').pathname
    if (pathname === '/favicon.ico') {
      res.writeHead(204)
      res.end()
      return
    }
    if (pathname === '/') {
      res.setHeader('Content-Type', 'text/html')
      res.end(harness)
      return
    }
    if (pathname === '/reference.json') {
      res.setHeader('Content-Type', 'application/json')
      res.end(await readFile('docs/browser-native-reference.json'))
      return
    }
    const root = pathname.startsWith('/ort/') ? ortDist : assets
    const relative = pathname.replace(/^\/(assets|ort)\//, '')
    const file = path.resolve(root, relative)
    if (!file.startsWith(root + path.sep)) throw Error('Invalid task asset path')
    res.setHeader('Content-Length', (await stat(file)).size)
    res.setHeader(
      'Content-Type',
      file.endsWith('.mjs')
        ? 'text/javascript'
        : file.endsWith('.wasm')
          ? 'application/wasm'
          : file.endsWith('.json')
            ? 'application/json'
            : 'application/octet-stream',
    )
    createReadStream(file).pipe(res)
  } catch {
    res.writeHead(404)
    res.end('Not found')
  }
})
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
let browser
try {
  browser = await chromium.launch({
    channel: 'chrome',
    headless: true,
    args: softwareGpu
      ? ['--enable-unsafe-webgpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
      : [],
  })
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } })
  const logs = []
  page.on('pageerror', (e) => logs.push(String(e)))
  page.on('console', (m) => {
    if (m.type() === 'error') logs.push(m.text())
  })
  await page.goto('http://127.0.0.1:' + server.address().port)
  await page.waitForFunction(() => window.ready === true)
  const result = await page.evaluate(() => window.runStack())
  result.errors = logs
  await writeFile(output, JSON.stringify(result, null, 2) + '\n')
  await page.screenshot({ path: output.replace(/\.json$/, '.png'), fullPage: true })
  console.log(
    JSON.stringify({
      output,
      totalMs: result.totalMs,
      provider,
      result: result.results.map((c) => ({
        name: c.name,
        maxLogitDrift: c.maxLogitDrift,
        maxProbabilityDrift: c.maxProbabilityDrift,
        sameLeadingIndex: c.sameLeadingIndex,
      })),
      errors: logs,
    }),
  )
  if (!result.passed || logs.length) throw Error('Full stack parity or browser errors failed')
} finally {
  if (browser) await browser.close()
  await new Promise((resolve) => server.close(resolve))
}
