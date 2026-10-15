import { chooseExample } from './helpers'
import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

const apiCalls: string[] = []
const errors: string[] = []
test.beforeEach(async ({ page }) => {
  apiCalls.length = 0
  errors.length = 0
  page.on('request', (request) => {
    if (request.url().includes('/api/')) apiCalls.push(request.url())
  })
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text())
  })
  await page.goto('./')
  await page.getByText('Customize inputs', { exact: false }).click()
})

test('static site inspects all three real request types without inference calls', async ({
  page,
}) => {
  await expect(page).toHaveTitle('Cadence Demo — typed decisions')
  await expect(
    page.getByText('Static playground · no model inference', { exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Get decision', exact: true })).toHaveCount(0)
  for (const [name, kind] of [
    ['Choose an answer', 'choice'],
    ['Yes or no', 'noul'],
    ['Rate on a scale', 'score'],
  ]) {
    await page.getByRole('button', { name, exact: true }).click()
    await page.getByRole('button', { name: 'Inspect request', exact: true }).click()
    const request = JSON.parse((await page.locator('.request-preview pre').textContent())!)
    expect(request.questions.decision.type).toBe(kind)
    await expect(page.getByText('No inference has run.', { exact: true })).toBeVisible()
    await expect(page.locator('.distribution')).toHaveCount(0)
  }
  expect(apiCalls).toEqual([])
  expect(errors).toEqual([])
})

test('copied JSON is the complete validated request', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.getByRole('button', { name: 'Inspect request', exact: true }).click()
  const shown = JSON.parse((await page.locator('.request-preview pre').textContent())!)
  await page.getByRole('button', { name: 'Copy JSON', exact: true }).click()
  await expect(page.getByText('Copied request JSON.', { exact: true })).toBeVisible()
  expect(JSON.parse(await page.evaluate(() => navigator.clipboard.readText()))).toEqual(shown)
  expect(apiCalls).toEqual([])
})

test('all six fictional legal-work presets inspect valid native requests', async ({ page }) => {
  await expect(page.locator('.scenario-note')).toContainText('general-purpose research model')
  await expect(page.locator('.scenario-note')).toContainText('does not draft documents or provide legal advice')
  for (const [name, kind] of [
    ['Route a document', 'choice'],
    ['Route a review request', 'choice'],
    ['Check a notice term', 'noul'],
    ['Check automatic renewal', 'noul'],
    ['Prioritize review', 'score'],
    ['Prioritize an archive item', 'score'],
  ]) {
    await chooseExample(page, name)
    await expect(page.locator('button.preset').filter({ hasText: name })).toHaveAttribute('aria-pressed', 'true')
    await page.getByRole('button', { name: 'Inspect request', exact: true }).click()
    const request = JSON.parse((await page.locator('.request-preview pre').textContent())!)
    expect(request.questions.decision.type).toBe(kind)
    if (name === 'Route a review request') expect(request.state.fictional).toBe(true)
    else expect(request.state).toMatch(/^Fictional /)
    await expect(page.getByText('No inference has run.', { exact: true })).toBeVisible()
  }
  expect(apiCalls).toEqual([])
  expect(errors).toEqual([])
})

test('static site validates input and explains separate local setup', async ({ page }) => {
  await chooseExample(page, 'Route a review request')
  await page.getByLabel('Context', { exact: true }).fill('{bad json')
  await page.getByRole('button', { name: 'Inspect request', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('not valid JSON')
  await page.getByRole('button', { name: 'Setup guide', exact: true }).click()
  await expect(page.getByRole('dialog')).toContainText(
    'Open the local playground in a separate tab',
  )
  await expect(page.getByRole('link', { name: 'Open local playground' })).toHaveAttribute(
    'href',
    'http://127.0.0.1:5173',
  )
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).not.toBeVisible()
  expect(apiCalls).toEqual([])
})

test('repository base path, navigation, mobile layout and accessibility', async ({
  page,
}, testInfo) => {
  await expect(page).toHaveURL(/\/cadencedemo\/$/)
  await expect(page.getByRole('heading', { name: 'Small model. Real decisions.' })).toBeVisible()
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)
  await page.getByRole('button', { name: 'Inspect request', exact: true }).click()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({
    path: `/tmp/cadence-guided-pages-${testInfo.project.name}.png`,
    fullPage: true,
  })
  await page.getByRole('link', { name: 'Cadence home' }).click()
  await expect(page).toHaveURL(/\/cadencedemo\/$/)
  await expect(
    page.getByText('Static playground · no model inference', { exact: true }),
  ).toBeVisible()
  expect(apiCalls).toEqual([])
  expect(errors).toEqual([])
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
  await page.getByRole('button', { name: 'Inspect request', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('safe range')
  await expect(page.getByRole('alert')).toContainText('strings')
})
