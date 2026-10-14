import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
const revision = '17cdf36034677e406033fbf97637b4db1984ca70'
const model = 'foolish-bandit/cadence-0.8b'
const cudaRuntime = { device: 'cuda', profile: 'cuda_bf16_fp32' }
const capturedErrors = new WeakMap<object, string[]>()
test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  capturedErrors.set(page, errors)
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text())
  })
  await page.route('**/api/health', (route) =>
    route.fulfill({
      json: {
        status: 'ready',
        busy: false,
        inference: 'real',
        device: 'cuda',
        profile: 'cuda_bf16_fp32',
        limits: { options: 32 },
        revision,
      },
    }),
  )
  await page.route('**/api/predict', async (route) => {
    const request = route.request().postDataJSON()
    const q = request.questions.decision
    const keys =
      q.type === 'choice'
        ? Object.keys(q.criteria)
        : q.type === 'noul'
          ? ['false', 'true']
          : q.criteria.map((_: unknown, i: number) => String(i))
    const probabilities = Object.fromEntries(
      keys.map((k: string, i: number) => [k, i === 0 ? 0.8 : 0.2 / (keys.length - 1)]),
    )
    const value =
      q.type === 'noul'
        ? probabilities.true
        : q.type === 'score'
          ? keys.reduce((v: number, k: string) => v + Number(k) * probabilities[k], 0)
          : null
    await route.fulfill({
      json: {
        kind: q.type,
        native_option_ids: keys,
        probabilities,
        value,
        tokens: 48,
        seconds: 0.045,
        truncated: false,
        model,
        revision,
        ...cudaRuntime,
      },
    })
  })
  await page.goto('/?preview=ui-test')
})
test('choice, Boolean and ordinal runs preserve native request semantics', async ({ page }) => {
  await page.getByRole('button', { name: 'Run decision', exact: true }).click()
  await expect(page.getByText('Highest-probability option')).toBeVisible()
  await expect(page.getByText('80.0%', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Boolean', exact: true }).click()
  const booleanRequest = page.waitForRequest((r) => r.url().endsWith('/api/predict'))
  await page.getByRole('button', { name: 'Run decision', exact: true }).click()
  expect((await booleanRequest).postDataJSON().questions.decision.type).toBe('noul')
  await expect(page.getByText('Probability of true', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Score', exact: true }).click()
  await page.getByRole('button', { name: 'Run decision', exact: true }).click()
  await expect(page.getByText('Expected level', { exact: true })).toBeVisible()
  await expect(page.getByText('0.50 / 4', { exact: true })).toBeVisible()
  await page.getByLabel('State', { exact: true }).fill('Different context')
  await expect(
    page.getByText('Inputs changed. This result belongs to your previous run.'),
  ).toBeVisible()
})
test('JSON validation, loading, recovery and raw result', async ({ page }) => {
  await page.getByRole('button', { name: 'Classify support intent', exact: true }).click()
  await page.getByLabel('State', { exact: true }).fill('{bad json')
  await page.getByRole('button', { name: 'Run decision', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('not valid JSON')
  await page.getByRole('button', { name: 'Play music', exact: true }).click()
  await page.route('**/api/predict', (route) =>
    route.fulfill({
      status: 503,
      json: { detail: 'Model could not load. Check local CUDA setup.' },
    }),
  )
  await page.getByRole('button', { name: 'Run decision', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Check local CUDA')
  await expect(page.getByRole('button', { name: 'Run decision', exact: true })).toBeEnabled()
  await page.unroute('**/api/predict')
  let release: () => void = () => {}
  const waiting = new Promise<void>((resolve) => {
    release = resolve
  })
  await page.route('**/api/predict', async (route) => {
    await waiting
    await route.fulfill({
      json: {
        kind: 'choice',
        native_option_ids: ['play_music', 'book_restaurant', 'get_weather'],
        probabilities: { play_music: 0.8, book_restaurant: 0.1, get_weather: 0.1 },
        value: null,
        tokens: 48,
        seconds: 0.045,
        truncated: false,
        model,
        revision,
        ...cudaRuntime,
      },
    })
  })
  await page.getByRole('button', { name: 'Run decision', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Running decision…' })).toBeDisabled()
  await expect(page.getByLabel('State', { exact: true })).toBeDisabled()
  release()
  await expect(page.getByText('Highest-probability option')).toBeVisible()
  await page.getByText('Raw JSON', { exact: true }).click()
  await expect(page.locator('.raw pre')).toContainText(revision)
})
test('option editing, setup dialog keyboard and accessibility', async ({ page }) => {
  await page.getByRole('button', { name: 'Add option', exact: false }).click()
  await expect(page.getByLabel('Option 4 ID')).toBeVisible()
  await page.getByLabel('Option 4 ID').fill('play_music')
  await page.getByRole('button', { name: 'Run decision', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('unique ID')
  await page.getByRole('button', { name: 'Remove option 4' }).click()
  await page.getByRole('button', { name: 'Setup guide' }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).not.toBeVisible()
  const results = await new AxeBuilder({ page }).analyze()
  expect(results.violations).toEqual([])
})
test('responsive screenshots explicitly identify mocked preview', async ({ page }, testInfo) => {
  const consoleErrors = capturedErrors.get(page)!
  await expect(page).toHaveTitle('Cadence — typed decision playground')
  await expect(page.getByRole('heading', { name: 'Give context. Get a decision.' })).toBeVisible()
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)
  await expect(page.getByText(/UI test preview/)).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: `docs/preview-${testInfo.project.name}-empty.png`, fullPage: true })
  await page.getByRole('button', { name: 'Run decision', exact: true }).click()
  await expect(page.getByText('Mock result · UI test only')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({
    path: `docs/preview-${testInfo.project.name}-result.png`,
    fullPage: true,
  })
  expect(consoleErrors).toEqual([])
})

test('CPU profile is reported and the editor respects its option limit', async ({ page }) => {
  await page.route('**/api/health', (route) =>
    route.fulfill({
      json: {
        status: 'ready',
        busy: false,
        inference: 'real',
        revision,
        device: 'cpu',
        profile: 'cpu_fp32',
        limits: { options: 16 },
      },
    }),
  )
  await page.route('**/api/predict', (route) => {
    const q = route.request().postDataJSON().questions.decision
    const keys = Object.keys(q.criteria)
    return route.fulfill({
      json: {
        kind: q.type,
        native_option_ids: keys,
        probabilities: Object.fromEntries(keys.map((k) => [k, 1 / keys.length])),
        value: null,
        tokens: 48,
        seconds: 0.4,
        truncated: false,
        model,
        revision,
        ...cudaRuntime,
        device: 'cpu',
        profile: 'cpu_fp32',
      },
    })
  })
  await page.reload()
  await expect(page.locator('footer')).toContainText('CPU FP32 (experimental)')
  await expect(page.getByText('3 / 16', { exact: true })).toBeVisible()
  for (let i = 3; i < 16; i++)
    await page.getByRole('button', { name: 'Add option', exact: false }).click()
  await expect(page.getByRole('button', { name: 'Add option', exact: false })).toBeDisabled()
  await page.getByRole('button', { name: 'Run decision', exact: true }).click()
  await expect(page.getByText(/precision differs from the public CUDA profile/)).toBeVisible()
})
