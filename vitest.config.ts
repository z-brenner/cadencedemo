import { defineConfig } from 'vitest/config'
export default defineConfig({
  test: {
    environment: 'node',
    include: ['web/tests/**/*.test.ts*', 'web/src/browser/**/*.test.ts'],
  },
})
