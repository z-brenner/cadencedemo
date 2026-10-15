import type { Page } from '@playwright/test'

export async function chooseExample(page: Page, name: string) {
  const picker = page.getByLabel('Legal-work example', { exact: true })
  if (await picker.isVisible()) await picker.selectOption({ label: name })
  else await page.getByRole('button', { name, exact: true }).click()
}
