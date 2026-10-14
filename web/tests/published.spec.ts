/** Published build preflight: real assets/UI; no model download or inference. */
import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

test('published research demo exposes honest opt-in download before any model activity', async ({ page }, testInfo) => {
  const errors: string[] = [], modelRequests: string[] = [], apiRequests: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()) })
  page.on('request', request => {
    if (/huggingface|\.wasm|model\.worker|\.onnx|\.weights/.test(request.url())) modelRequests.push(request.url())
    if (request.url().includes('/api/')) apiRequests.push(request.url())
  })
  await page.goto('./', { waitUntil: 'networkidle' })
  await expect(page).toHaveTitle('Cadence Demo — typed decisions')
  await expect(page.getByText('Research project · interactive demo', { exact: true })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Run Cadence on this device' })).toBeVisible()
  await expect(page.locator('.model-preparation')).toContainText('1.54 GB')
  await expect(page.locator('.model-preparation')).toContainText('28 MB')
  await expect(page.locator('.model-preparation')).toContainText('Your inputs stay on this device')
  await expect(page.getByRole('button', { name: 'Download model', exact: true })).toBeEnabled()
  await expect(page.getByText('Not available yet', { exact: true })).toHaveCount(0)
  await expect(page.getByText('Static playground · no model inference', { exact: true })).toHaveCount(0)
  await expect(page.getByText(/UI test preview/)).toHaveCount(0)
  for (const name of ['Choose an answer', 'Yes or no', 'Rate on a scale']) {
    await page.getByRole('button', { name, exact: true }).click()
    await expect(page.getByLabel('Context', { exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Get decision', exact: true })).toBeDisabled()
  }
  await page.getByText('Model and saved files', { exact: true }).click()
  await expect(page.locator('.model-details')).toContainText('512 packed tokens')
  await expect(page.locator('.model-details')).toContainText('16 answers')
  await page.getByText('Advanced input details', { exact: true }).click()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([])
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)
  await page.screenshot({ path: `/tmp/cadence-published-preflight-${testInfo.project.name}.png`, fullPage: true })
  await page.getByRole('button', { name: 'Setup guide', exact: true }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).not.toBeVisible()
  expect(modelRequests).toEqual([])
  expect(apiRequests).toEqual([])
  expect(errors).toEqual([])
})
