import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
const manifestSha = 'a'.repeat(64)
const revision = '17cdf36034677e406033fbf97637b4db1984ca70'
const apiCalls: string[] = []
const errors: string[] = []

test.beforeEach(async ({ page }) => {
  apiCalls.length = 0
  errors.length = 0
  page.on('request', (request) => {
    if (request.url().includes('/api/')) apiCalls.push(request.url())
  })
  page.on('pageerror', (error) => errors.push(error.message))
  await page.addInitScript(
    ({ revision, manifestSha }) => {
      const state = window as unknown as {
        __cadenceWorkers: any[]
        __cadenceHoldResult: boolean
        __cadenceBadManifest: boolean
      }
      state.__cadenceWorkers = []
      state.__cadenceHoldResult = false
      state.__cadenceBadManifest = false
      class MockWorker {
        onmessage: ((event: { data: any }) => void) | null = null
        onerror: (() => void) | null = null
        messages: any[] = []
        terminated = false
        constructor() {
          state.__cadenceWorkers.push(this)
        }
        terminate() {
          this.terminated = true
        }
        postMessage(message: any) {
          this.messages.push(message)
          if (message.type === 'prepare') {
            setTimeout(
              () =>
                this.onmessage?.({
                  data: {
                    type: 'status',
                    status: 'downloading',
                    downloadedBytes: 25_000_000,
                    totalBytes: 100_000_000,
                    stage: 'Downloading verified model files',
                    fromCache: false,
                  },
                }),
              0,
            )
          }
          if (message.type === 'run' && !state.__cadenceHoldResult) {
            const q = message.request.questions.decision
            const ids =
              q.type === 'choice'
                ? Object.keys(q.criteria)
                : q.type === 'noul'
                  ? ['false', 'true']
                  : q.criteria.map((_: string, i: number) => String(i))
            const probabilities = Object.fromEntries(
              ids.map((id: string, i: number) => [id, i === 0 ? 0.8 : 0.2 / (ids.length - 1)]),
            )
            const value =
              q.type === 'noul'
                ? probabilities.true
                : q.type === 'score'
                  ? ids.reduce((sum: number, id: string) => sum + Number(id) * probabilities[id], 0)
                  : null
            setTimeout(
              () =>
                this.onmessage?.({
                  data: {
                    type: 'result',
                    requestId: message.requestId,
                    result: {
                      kind: q.type,
                      native_option_ids: ids,
                      probabilities,
                      value,
                      tokens: 48,
                      seconds: 0.045,
                      truncated: false,
                      model: 'foolish-bandit/cadence-0.8b',
                      revision,
                      device: 'browser',
                      profile: 'browser_fp32',
                      artifact_manifest_sha256: state.__cadenceBadManifest
                        ? 'b'.repeat(64)
                        : manifestSha,
                    },
                  },
                }),
              0,
            )
          }
        }
      }
      Object.defineProperty(window, 'Worker', { value: MockWorker })
    },
    { revision, manifestSha },
  )
  await page.goto('/?preview=ui-test')
})

async function ready(page: import('@playwright/test').Page, fromCache = false) {
  await expect(page.getByRole('progressbar', { name: 'Model file download' })).toHaveAttribute(
    'value',
    '25000000',
  )
  await page.evaluate(() => {
    ;(window as any).__cadenceWorkers.at(-1).onmessage({
      data: {
        type: 'status',
        status: 'loading',
        stage: 'Preparing tokenizer and runtime',
        downloadedBytes: 100_000_000,
        totalBytes: 100_000_000,
      },
    })
  })
  await expect(
    page.getByText('Preparing model files and tokenizer…', { exact: true }),
  ).toBeVisible()
  await page.evaluate((fromCache) => {
    const workers = (window as any).__cadenceWorkers
    workers.at(-1).onmessage({ data: { type: 'ready', provider: 'wasm', fromCache } })
  }, fromCache)
  await expect(page.getByRole('button', { name: 'Get decision', exact: true })).toBeEnabled()
}

test('model preparation requires a click and preserves typed browser decisions', async ({
  page,
}, testInfo) => {
  await expect(page.getByText(/UI test preview/)).toBeVisible()
  await expect(page.getByText(/1.54 GB/)).toBeVisible()
  await expect(page.getByText(/download host sees your IP/)).toBeVisible()
  await expect(page.getByRole('button', { name: 'Get decision', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: 'Is this spam?', exact: true }).click()
  expect(await page.evaluate(() => (window as any).__cadenceWorkers.length)).toBe(0)
  await page.getByRole('button', { name: 'Download model', exact: true }).click()
  await expect(page.getByRole('progressbar', { name: 'Model file download' })).toHaveAttribute(
    'value',
    '25000000',
  )
  await expect(page.getByText('25.0 MB of 100.0 MB (25%)', { exact: true })).toBeVisible()
  await ready(page)
  for (const [name, kind] of [
    ['Choose an answer', 'choice'],
    ['Yes or no', 'noul'],
    ['Rate on a scale', 'score'],
  ]) {
    await page.getByRole('button', { name, exact: true }).click()
    await page.getByRole('button', { name: 'Get decision', exact: true }).click()
    await expect(page.getByText('Mock result · UI test only', { exact: true })).toBeVisible()
    const actual = await page.evaluate(() =>
      (window as any).__cadenceWorkers.at(-1).messages.at(-1),
    )
    expect(actual.request.questions.decision.type).toBe(kind)
  }
  await page.getByText('Technical result details', { exact: true }).click()
  await expect(
    page.getByText(
      'Timing includes reading saved files, preparing layers and computing this decision.',
      { exact: true },
    ),
  ).toBeVisible()
  await expect(page.getByText(/qualified against the CPU FP32 reference/)).toBeVisible()
  await page.getByLabel('Context', { exact: true }).fill('Updated context')
  await expect(
    page.getByText('You changed the inputs. Get a new decision to update this result.'),
  ).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
  await page.evaluate(() => {
    ;(document.activeElement as HTMLElement)?.blur()
    window.scrollTo(0, 0)
  })
  await page.screenshot({
    path: `/tmp/cadence-browser-mocked-${testInfo.project.name}.png`,
    fullPage: true,
  })
  expect(apiCalls).toEqual([])
  expect(errors).toEqual([])
})

test('cancel and retry reuse only completed verified files and ignore retired workers', async ({
  page,
}) => {
  await page.getByRole('button', { name: 'Download model', exact: true }).click()
  await expect(page.getByText('25.0 MB of 100.0 MB (25%)', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Cancel preparation', exact: true }).click()
  await expect(page.getByText(/Retry reuses only completed/)).toBeVisible()
  expect(await page.evaluate(() => (window as any).__cadenceWorkers[0].terminated)).toBe(true)
  await page.evaluate(() =>
    (window as any).__cadenceWorkers[0].onmessage({
      data: { type: 'ready', provider: 'wasm', fromCache: false },
    }),
  )
  await expect(page.getByRole('button', { name: 'Get decision', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: 'Retry model preparation', exact: true }).click()
  await ready(page, true)
  await expect(page.getByText(/Verified saved files were reused/)).toBeVisible()
  expect(apiCalls).toEqual([])
})

test('clearing saved files touches only the configured model cache and starts no worker', async ({
  page,
}) => {
  await page.evaluate(async (manifestSha) => {
    await caches.open(`cadence-browser-${manifestSha}`)
    await caches.open('cadence-unrelated-ui-test')
  }, manifestSha)
  await page.getByText('Model and saved files', { exact: true }).click()
  await page.getByRole('button', { name: 'Clear saved model', exact: true }).click()
  await expect(
    page.getByText('Saved files for this model release cleared.', { exact: true }),
  ).toBeVisible()
  expect(
    await page.evaluate(
      async (manifestSha) => ({
        cleared: !(await caches.has(`cadence-browser-${manifestSha}`)),
        retained: await caches.has('cadence-unrelated-ui-test'),
        workers: (window as any).__cadenceWorkers.length,
      }),
      manifestSha,
    ),
  ).toEqual({ cleared: true, retained: true, workers: 0 })
  expect(apiCalls).toEqual([])
})

test('device errors and invalid model provenance never fall back to an API', async ({ page }) => {
  await page.getByRole('button', { name: 'Download model', exact: true }).click()
  await page.evaluate(() =>
    (window as any).__cadenceWorkers
      .at(-1)
      .onmessage({ data: { type: 'error', message: 'This device does not have enough memory.' } }),
  )
  await expect(page.getByRole('alert')).toContainText('not have enough memory')
  await expect(page.getByRole('button', { name: 'Local setup guide', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Retry model preparation', exact: true }).click()
  await ready(page)
  await page.evaluate(() => {
    ;(window as any).__cadenceBadManifest = true
  })
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.locator('.error')).toContainText('Cadence could not verify this result.')
  await expect(page.locator('.distribution')).toHaveCount(0)
  await expect(
    page.getByRole('button', { name: 'Retry model preparation', exact: true }),
  ).toBeVisible()
  expect(apiCalls).toEqual([])
})

test('canceling an in-flight decision settles the form and requires preparation again', async ({
  page,
}) => {
  await page.getByRole('button', { name: 'Download model', exact: true }).click()
  await ready(page)
  await page.evaluate(() => {
    ;(window as any).__cadenceHoldResult = true
  })
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.getByLabel('Context', { exact: true })).toBeDisabled()
  await page.getByRole('button', { name: 'Cancel decision', exact: true }).click()
  await expect(page.getByLabel('Context', { exact: true })).toBeEnabled()
  await expect(page.getByRole('button', { name: 'Get decision', exact: true })).toBeDisabled()
  await expect(page.locator('.distribution')).toHaveCount(0)
  expect(apiCalls).toEqual([])
})

test('browser limits stay visible and oversized token requests show an error', async ({ page }) => {
  await expect(page.getByText('3 / 16', { exact: true })).toBeVisible()
  await page.getByText('Model and saved files', { exact: true }).click()
  await expect(page.getByText(/512 packed tokens/)).toBeVisible()
  for (let i = 3; i < 16; i++)
    await page.getByRole('button', { name: 'Add option', exact: false }).click()
  await expect(page.getByRole('button', { name: 'Add option', exact: false })).toBeDisabled()
  await page.getByRole('button', { name: 'Download model', exact: true }).click()
  await ready(page)
  await page.evaluate(() => {
    ;(window as any).__cadenceHoldResult = true
  })
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await page.evaluate(() => {
    const worker = (window as any).__cadenceWorkers.at(-1)
    const requestId = worker.messages.at(-1).requestId
    worker.onmessage({
      data: {
        type: 'input-error',
        requestId,
        message:
          'This request exceeds the 512 packed-token browser limit. Shorten the context or answers.',
      },
    })
  })
  await expect(page.locator('.error')).toContainText('512 packed-token')
  await expect(page.locator('.distribution')).toHaveCount(0)
  expect(apiCalls).toEqual([])
})

test('a recoverable oversized input can be shortened and rerun without preparing another worker', async ({
  page,
}) => {
  await page.getByRole('button', { name: 'Download model', exact: true }).click()
  await ready(page, true)
  await page.getByLabel('Context', { exact: true }).fill('word '.repeat(600))
  await page.evaluate(() => {
    ;(window as any).__cadenceHoldResult = true
  })
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await page.evaluate(() => {
    const worker = (window as any).__cadenceWorkers[0]
    worker.onmessage({
      data: { type: 'input-error', requestId: 'retired-request', message: 'Old input error' },
    })
  })
  await expect(page.getByRole('button', { name: 'Cancel decision', exact: true })).toBeVisible()
  await expect(page.getByLabel('Context', { exact: true })).toBeDisabled()
  await page.evaluate(() => {
    const worker = (window as any).__cadenceWorkers[0]
    worker.onmessage({
      data: {
        type: 'input-error',
        requestId: worker.messages.at(-1).requestId,
        message:
          'This request exceeds the 512 packed-token browser limit. Shorten the context or answers.',
      },
    })
  })
  await expect(page.locator('.error')).toContainText('512 packed-token')
  await expect(page.locator('.distribution')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Get decision', exact: true })).toBeEnabled()
  await expect(page.getByText(/Verified saved files were reused/)).toBeVisible()
  await page.getByText('Model and saved files', { exact: true }).click()
  await expect(page.getByText('Runtime: WASM', { exact: true })).toBeVisible()
  await page.getByLabel('Context', { exact: true }).fill('Can you play jazz?')
  await expect(page.locator('.error')).toHaveCount(0)
  await expect(page.locator('.model-error')).toHaveCount(0)
  await page.evaluate(() => {
    ;(window as any).__cadenceHoldResult = false
  })
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.locator('.answer')).toHaveText('Play music')
  expect(
    await page.evaluate(() => {
      const workers = (window as any).__cadenceWorkers
      return {
        count: workers.length,
        terminated: workers[0].terminated,
        prepares: workers[0].messages.filter((message: any) => message.type === 'prepare').length,
        runs: workers[0].messages.filter((message: any) => message.type === 'run').length,
      }
    }),
  ).toEqual({ count: 1, terminated: false, prepares: 1, runs: 2 })
  await expect(
    page.getByRole('button', { name: 'Retry model preparation', exact: true }),
  ).toHaveCount(0)
  expect(apiCalls).toEqual([])
})
