/** Browser UI test build: no actual model/hosting, explicit mock-only preview. */
import { build } from 'vite'
process.env.VITE_CADENCE_ASSET_BASE = 'https://example.invalid/model/'
process.env.VITE_CADENCE_MANIFEST_SHA256 = 'a'.repeat(64)
await build({ mode: 'pages', build: { outDir: '/tmp/cadence-browser-ui-dist' } })
