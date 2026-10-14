import { defineConfig } from '@playwright/test'
export default defineConfig({
  testDir: 'web/tests',
  testMatch: '**/published.spec.ts',
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:5178/cadencedemo/',
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROME ? { channel: 'chrome' } : {}),
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1280, height: 1000 } } },
    { name: 'mobile', use: { viewport: { width: 390, height: 844 }, isMobile: true } },
    { name: 'narrow', use: { viewport: { width: 320, height: 640 }, isMobile: true } },
  ],
  webServer: {
    command: 'npx vite preview --host 127.0.0.1 --outDir dist-pages --base /cadencedemo/ --port 5178 --strictPort',
    url: 'http://127.0.0.1:5178/cadencedemo/',
    reuseExistingServer: false,
  },
})
