import { defineConfig } from '@playwright/test'
export default defineConfig({
  testDir: 'web/tests',
  testMatch: '**/browser.spec.ts',
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:5176',
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROME ? { channel: 'chrome' } : {}),
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1280, height: 1000 } } },
    { name: 'mobile', use: { viewport: { width: 390, height: 844 }, isMobile: true } },
  ],
  webServer: {
    command:
      './node_modules/.bin/vite preview --host 127.0.0.1 --outDir /tmp/cadence-browser-ui-dist --port 5176 --strictPort',
    url: 'http://127.0.0.1:5176',
    reuseExistingServer: true,
  },
})
