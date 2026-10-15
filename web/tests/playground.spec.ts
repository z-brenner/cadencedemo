import { chooseExample } from './helpers'
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
  await page.getByText('Customize inputs', { exact: false }).click()
})
test('choice, Boolean and ordinal runs preserve native request semantics', async ({ page }) => {
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.getByText('Cadence favors', { exact: true })).toBeVisible()
  await expect(page.getByText('80.0%', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Yes or no', exact: true }).click()
  const booleanRequest = page.waitForRequest((r) => r.url().endsWith('/api/predict'))
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  expect((await booleanRequest).postDataJSON().questions.decision.type).toBe('noul')
  await expect(page.getByText('Model probability of Yes', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Rate on a scale', exact: true }).click()
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.getByText('Average model rating', { exact: true })).toBeVisible()
  await expect(page.getByText('0.50 / 4', { exact: true })).toBeVisible()
  await page.getByLabel('Context', { exact: true }).fill('Different context')
  await expect(
    page.getByText('You changed the inputs. Get a new decision to update this result.'),
  ).toBeVisible()
})
test('JSON validation, loading, recovery and raw result', async ({ page }) => {
  await chooseExample(page, 'Route a review request')
  await page.getByLabel('Context', { exact: true }).fill('{bad json')
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('not valid JSON')
  await chooseExample(page, 'Route a document')
  await page.route('**/api/predict', (route) =>
    route.fulfill({
      status: 503,
      json: { detail: 'Model could not load. Check local CUDA setup.' },
    }),
  )
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Check local CUDA')
  await expect(page.getByRole('button', { name: 'Get decision', exact: true })).toBeEnabled()
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
        native_option_ids: ['confidentiality_agreement', 'services_agreement', 'meeting_notes'],
        probabilities: { confidentiality_agreement: 0.8, services_agreement: 0.1, meeting_notes: 0.1 },
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
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Running decision…' })).toBeDisabled()
  await expect(page.getByLabel('Context', { exact: true })).toBeDisabled()
  release()
  await expect(page.getByText('Cadence favors', { exact: true })).toBeVisible()
  await page.getByText('Raw JSON', { exact: true }).click()
  await expect(page.locator('.raw pre')).toContainText(revision)
})
test('option editing, setup dialog keyboard and accessibility', async ({ page }) => {
  await page.getByRole('button', { name: 'Add option', exact: false }).click()
  await page.getByText('Advanced input details', { exact: true }).click()
  await expect(page.getByLabel('Option 4 ID')).toBeVisible()
  await page.getByLabel('Option 4 ID').fill('confidentiality_agreement')
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
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
  await expect(page).toHaveTitle('Cadence Demo — typed decisions')
  await expect(page.getByRole('heading', { name: 'Small model. Real decisions.' })).toBeVisible()
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)
  await expect(page.getByText(/UI test preview/)).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({
    path: `/tmp/cadence-guided-${testInfo.project.name}-empty.png`,
    fullPage: true,
  })
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.getByText('Mock result · UI test only')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({
    path: `/tmp/cadence-guided-${testInfo.project.name}-result.png`,
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
  await page.getByText('Customize inputs', { exact: false }).click()
  await expect(page.locator('footer')).toContainText('CPU FP32 (experimental)')
  await expect(page.getByText('3 / 16', { exact: true })).toBeVisible()
  for (let i = 3; i < 16; i++)
    await page.getByRole('button', { name: 'Add option', exact: false }).click()
  await expect(page.getByRole('button', { name: 'Add option', exact: false })).toBeDisabled()
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await page.getByText('Technical result details', { exact: true }).click()
  await expect(page.getByText(/precision differs from the public CUDA profile/)).toBeVisible()
})

test('multilingual and astral JSON reaches the API unchanged', async ({ page }) => {
  await chooseExample(page, 'Route a review request')
  const state = { text: '日'.repeat(3000), accent: 'é'.repeat(2000), emoji: '😀'.repeat(1000) }
  const source = JSON.stringify(state)
  await page.getByLabel('Context', { exact: true }).fill(source)
  const sent = page.waitForRequest((request) => request.url().endsWith('/api/predict'))
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  expect((await sent).postDataJSON().state).toEqual(state)
  await expect(page.getByText('Cadence favors', { exact: true })).toBeVisible()
  await expect(page.getByLabel('Context', { exact: true })).toHaveValue(source)
  expect(capturedErrors.get(page)).toEqual([])
})

test('selecting the active JSON format preserves context', async ({ page }) => {
  await chooseExample(page, 'Route a review request')
  const source = await page.getByLabel('Context', { exact: true }).inputValue()
  await page.getByText('Advanced input details', { exact: true }).click()
  await page.getByRole('button', { name: 'JSON', exact: true }).click()
  await expect(page.getByLabel('Context', { exact: true })).toHaveValue(source)
})

test('unsafe integer JSON is explained before a request can run', async ({ page }) => {
  await chooseExample(page, 'Route a review request')
  await page.getByLabel('Context', { exact: true }).fill('{"account_id":9007199254740993}')
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('safe range')
  await expect(page.getByRole('alert')).toContainText('strings')
})

test('guided inputs hide technical controls and retain exact native choices', async ({ page }) => {
  await expect(page.getByLabel('Context', { exact: true })).toBeVisible()
  await expect(page.getByLabel('Question', { exact: true })).toBeVisible()
  await expect(page.getByText('Possible answers', { exact: true })).toBeVisible()
  await expect(page.getByLabel('Option 1 ID')).not.toBeVisible()
  await expect(page.getByRole('button', { name: 'JSON', exact: true })).not.toBeVisible()
  const sent = page.waitForRequest((r) => r.url().endsWith('/api/predict'))
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  expect(Object.keys((await sent).postDataJSON().questions.decision.criteria)).toEqual([
    'confidentiality_agreement',
    'services_agreement',
    'meeting_notes',
  ])
  await expect(page.locator('.answer')).toHaveText('Confidentiality agreement')
  await expect(
    page.getByText(
      'These percentages describe the model’s output. They do not measure how likely the answer is to be correct.',
    ),
  ).toBeVisible()
})

test('an exact probability tie does not claim one leading answer', async ({ page }) => {
  await page.route('**/api/predict', (route) =>
    route.fulfill({
      json: {
        kind: 'choice',
        native_option_ids: ['confidentiality_agreement', 'services_agreement', 'meeting_notes'],
        probabilities: { confidentiality_agreement: 0.45, services_agreement: 0.45, meeting_notes: 0.1 },
        value: null,
        tokens: 48,
        seconds: 0.045,
        truncated: false,
        model,
        revision,
        ...cudaRuntime,
      },
    }),
  )
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.locator('.answer')).toHaveText('No single leading answer')
  await expect(page.getByText('45.0%', { exact: true })).toHaveCount(2)
})

test('guided form fits a narrow phone with advanced controls open', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 })
  await page.getByText('Advanced input details', { exact: true }).click()
  await expect(page.getByLabel('Option 1 ID')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Get decision', exact: true })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
})

test('adding answers after a removal keeps generated IDs unique and stable', async ({ page }) => {
  await page.getByRole('button', { name: 'Add option', exact: false }).click()
  await page.getByText('Advanced input details', { exact: true }).click()
  const keptId = await page.getByLabel('Option 4 ID').inputValue()
  await page.getByRole('button', { name: 'Remove option 3', exact: true }).click()
  await page.getByRole('button', { name: 'Add option', exact: false }).click()
  const ids = await page
    .locator('.native-ids input')
    .evaluateAll((inputs) => inputs.map((input) => (input as HTMLInputElement).value))
  expect(ids[2]).toBe(keptId)
  expect(new Set(ids).size).toBe(ids.length)
  await page.getByText('Advanced input details', { exact: true }).click()
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  await expect(page.locator('.answer')).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('colliding readable names identify exact native choices in the editor and results', async ({
  page,
}) => {
  await page.getByText('Advanced input details', { exact: true }).click()
  const ids = ['confidentiality_agreement', 'confidentiality agreement', 'Confidentiality agreement']
  for (let i = 0; i < ids.length; i++) {
    await page.getByLabel(`Option ${i + 1} ID`).fill(ids[i])
    await page.getByLabel(`Answer ${i + 1} description`).fill('')
  }
  await page.getByText('Advanced input details', { exact: true }).click()
  const labels = ids.map((id) => `Confidentiality agreement (ID: ${id})`)
  await expect(page.locator('.answer-name')).toHaveText(labels)
  const sent = page.waitForRequest((r) => r.url().endsWith('/api/predict'))
  await page.getByRole('button', { name: 'Get decision', exact: true }).click()
  expect(Object.keys((await sent).postDataJSON().questions.decision.criteria)).toEqual(ids)
  await expect(page.locator('.answer')).toHaveText(labels[0])
  await expect(page.locator('.probability-label > span')).toHaveText(labels)
  await chooseExample(page, 'Route a document')
  await expect(page.locator('.answer-name').first()).toHaveText('Confidentiality agreement')
  await expect(page.locator('.answer')).toHaveText(labels[0])
  await expect(
    page.getByText('You changed the inputs. Get a new decision to update this result.'),
  ).toBeVisible()
})

test('local backend health does not accept a browser execution profile', async ({ page }) => {
  await page.route('**/api/health', (route) =>
    route.fulfill({
      json: {
        status: 'ready',
        busy: false,
        inference: 'real',
        revision,
        device: 'browser',
        profile: 'browser_fp32',
        limits: { options: 16 },
      },
    }),
  )
  await page.reload()
  await expect(page.getByText('Backend not connected', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Download model', exact: true })).toHaveCount(0)
})
