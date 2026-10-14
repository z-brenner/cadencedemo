/** Exercise the production guided UI and real browser worker with local assets. */
import { createServer } from 'node:http'
import { createReadStream } from 'node:fs'
import { readFile, stat, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'node:path'
const require = createRequire(path.join(process.cwd(), 'package.json'))
const { chromium, expect } = require('@playwright/test')
const AxeBuilder = require('@axe-core/playwright').default
const root = '/tmp/cadence-real-browser-ui',
  assets = '/tmp/cadence-browser-export-v1'
const server = createServer(async (req, res) => {
  try {
    const pathname = new URL(req.url, 'http://localhost').pathname
    const base = pathname.startsWith('/model/') ? assets : root
    const relative = pathname.startsWith('/model/')
      ? pathname.slice(7)
      : pathname === '/'
        ? 'index.html'
        : pathname.slice(1)
    const file = path.resolve(base, relative)
    if (!file.startsWith(base + path.sep)) throw Error('Invalid task path')
    res.setHeader(
      'Content-Type',
      file.endsWith('.js') || file.endsWith('.mjs')
        ? 'text/javascript'
        : file.endsWith('.wasm')
          ? 'application/wasm'
          : file.endsWith('.json')
            ? 'application/json'
            : file.endsWith('.css')
              ? 'text/css'
              : file.endsWith('.html')
                ? 'text/html'
                : file.endsWith('.svg')
                  ? 'image/svg+xml'
                  : 'application/octet-stream',
    )
    res.setHeader('Content-Length', (await stat(file)).size)
    createReadStream(file).pipe(res)
  } catch {
    res.writeHead(404)
    res.end('Not found')
  }
})
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
let browser, sampler
let peakRssBytes = 0
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true })
  const browserSession = await browser.newBrowserCDPSession()
  sampler = setInterval(async () => {
    try {
      const { processInfo } = await browserSession.send('SystemInfo.getProcessInfo')
      const sizes = await Promise.all(
        processInfo.map(async (p) => {
          try {
            const text = await readFile('/proc/' + p.id + '/status', 'utf8')
            return Number(text.match(/^VmRSS:\s+(\d+) kB/m)?.[1] ?? 0) * 1024
          } catch {
            return 0
          }
        }),
      )
      peakRssBytes = Math.max(
        peakRssBytes,
        sizes.reduce((a, b) => a + b, 0),
      )
    } catch {
      /* Owned Chrome may be closing. */
    }
  }, 500)
  const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } })
  const page = await context.newPage()
  const requests = [],
    errors = []
  page.on('request', (r) => requests.push(new URL(r.url()).pathname))
  page.on('pageerror', (e) => errors.push(String(e)))
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text())
  })
  const address = 'http://127.0.0.1:' + server.address().port
  await page.goto(address)
  await expect(page.getByRole('button', { name: 'Download model', exact: true })).toBeVisible()
  await page.waitForTimeout(500)
  if (requests.some((p) => p.startsWith('/model/') || p.endsWith('.wasm')))
    throw Error('Model/runtime downloaded automatically')
  const downloadStart = performance.now()
  await page.getByRole('button', { name: 'Download model', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Get decision', exact: true })).toBeEnabled({
    timeout: 180000,
  })
  const downloadMs = performance.now() - downloadStart,
    results = []
  for (const name of ['Play music', 'Is this spam?', 'Rate a reply']) {
    await page.getByRole('button', { name, exact: true }).click()
    await page.getByRole('button', { name: 'Get decision', exact: true }).click()
    await expect(page.getByText('Raw JSON', { exact: true })).toBeVisible({ timeout: 120000 })
    await page.getByText('Raw JSON', { exact: true }).click()
    const result = JSON.parse(await page.locator('details.raw pre').innerText())
    if (
      result.device !== 'browser' ||
      result.profile !== 'browser_fp32' ||
      result.truncated !== false ||
      Object.values(result.probabilities).some((p) => !Number.isFinite(p))
    )
      throw Error('Invalid real browser result')
    results.push({ example: name, result })
    await page.getByText('Raw JSON', { exact: true }).click()
  }
  await writeFile(
    '/tmp/cadence-real-browser-ui-results.json',
    JSON.stringify(results, null, 2) + '\n',
  )
  const axeDesktop = (await new AxeBuilder({ page }).analyze()).violations
  await page.screenshot({ path: '/tmp/cadence-real-browser-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  const axeMobile = (await new AxeBuilder({ page }).analyze()).violations
  await page.screenshot({ path: '/tmp/cadence-real-browser-mobile.png', fullPage: true })
  await page.reload()
  await page.getByRole('button', { name: 'Download model', exact: true }).click()
  await expect(page.getByText(/Verified saved files were reused/)).toBeVisible({ timeout: 180000 })
  const cachedVerified = true
  const preparesBefore = requests.filter((p) => p.endsWith('/model/manifest.json')).length
  await page.getByLabel('Context', { exact: true }).fill('a '.repeat(600))
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.getByRole('alert').filter({ hasText: '512' })).toBeVisible({ timeout: 10000 })
  await expect(page.getByRole('button', { name: 'Get decision', exact: true })).toBeEnabled()
  await page.getByLabel('Context', { exact: true }).fill('Can you play some jazz for dinner?')
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.getByText('Raw JSON', { exact: true })).toBeVisible({ timeout: 120000 })
  const inputRecoveryWithoutPreparation =
    requests.filter((p) => p.endsWith('/model/manifest.json')).length === preparesBefore
  if (!inputRecoveryWithoutPreparation) throw Error('Input recovery re-prepared the model')
  // Corrupt only a task-owned cache file. The worker must reject before inference.
  await page.evaluate(async () => {
    const name = (await caches.keys()).find((k) => k.startsWith('cadence-browser-'))
    const cache = await caches.open(name)
    const request = (await cache.keys()).find((r) =>
      new URL(r.url).pathname.endsWith('/layer-00.weights'),
    )
    await cache.put(request, new Response(new Uint8Array([0]).buffer))
  })
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.getByText(/saved model file failed verification/i).first()).toBeVisible({
    timeout: 30000,
  })
  const corruptCacheRejected = true
  const evidence = {
    inference: 'real full Cadence browser worker and production guided UI',
    browser: await browser.version(),
    downloadMs,
    cachedVerified,
    corruptCacheRejected,
    inputRecoveryWithoutPreparation,
    overflow,
    axeDesktop,
    axeMobile,
    results,
    requests,
    browserProcessPeakRssBytes: peakRssBytes,
    memoryMeasurement:
      'Sum of owned Chrome process RSS sampled every 500 ms; shared pages may be counted more than once.',
    apiCalls: requests.filter((p) => p.startsWith('/api/')),
    errors,
  }
  await writeFile('docs/browser-real-ui-smoke.json', JSON.stringify(evidence, null, 2) + '\n')
  console.log(
    JSON.stringify({
      downloadMs,
      cachedVerified,
      corruptCacheRejected,
      results: results.map((x) => ({
        example: x.example,
        tokens: x.result.tokens,
        seconds: x.result.seconds,
      })),
      overflow,
      axeDesktop: axeDesktop.length,
      axeMobile: axeMobile.length,
      apiCalls: evidence.apiCalls,
      errors,
    }),
  )
  if (
    overflow ||
    axeDesktop.length ||
    axeMobile.length ||
    errors.length ||
    evidence.apiCalls.length
  )
    throw Error('Real browser UI checks failed')
} finally {
  clearInterval(sampler)
  if (browser) await browser.close()
  await new Promise((resolve) => server.close(resolve))
}
