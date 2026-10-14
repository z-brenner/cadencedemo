// Run from the repository root: npx vitest run --config browser_export/tokenizer_probe.vitest.config.mjs
export default { test: { environment: 'node', include: ['web/src/browser/encoding.test.ts'] } }
