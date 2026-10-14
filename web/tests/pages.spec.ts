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
})

test('static site inspects all three real request types without inference calls', async ({
  page,
}) => {
  await expect(page).toHaveTitle('Cadence Demo — typed decisions')
  await expect(
    page.getByText('Static playground · no model inference', { exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Run decision', exact: true })).toHaveCount(0)
  for (const [name, kind] of [
    ['Choice', 'choice'],
    ['Boolean', 'noul'],
    ['Score', 'score'],
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

test('static site validates input and explains separate local setup', async ({ page }) => {
  await page.getByRole('button', { name: 'Classify support intent', exact: true }).click()
  await page.getByLabel('State', { exact: true }).fill('{bad json')
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
  await expect(
    page.getByRole('heading', { name: 'Give context. Get a decision.' }),
  ).toBeVisible()
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)
  await page.getByRole('button', { name: 'Inspect request', exact: true }).click()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
    true,
  )
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({
    path: `test-results/pages-${testInfo.project.name}.png`,
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
  await page.getByRole('button', { name: 'Classify support intent', exact: true }).click()
  const source = await page.getByLabel('State', { exact: true }).inputValue()
  await page.getByRole('button', { name: 'JSON', exact: true }).click()
  await expect(page.getByLabel('State', { exact: true })).toHaveValue(source)
})
