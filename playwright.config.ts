import { defineConfig } from '@playwright/test'
const pages = process.env.PAGES_TEST === '1'
export default defineConfig({
  testDir: 'web/tests',
  testMatch: pages ? '**/pages.spec.ts' : '**/playground.spec.ts',
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: pages ? 'http://127.0.0.1:5175/cadencedemo/' : 'http://127.0.0.1:5174',
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROME ? { channel: 'chrome' } : {}),
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1280, height: 1000 } } },
    { name: 'mobile', use: { viewport: { width: 390, height: 844 }, isMobile: true } },
  ],
  webServer: {
    command: pages
      ? 'npx vite preview --host 127.0.0.1 --outDir dist-pages --base /cadencedemo/ --port 5175 --strictPort'
      : 'npm run preview -- --port 5174 --strictPort',
    url: pages ? 'http://127.0.0.1:5175/cadencedemo/' : 'http://127.0.0.1:5174',
    reuseExistingServer: true,
  },
})
