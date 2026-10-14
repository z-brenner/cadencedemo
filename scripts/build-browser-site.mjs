/** Build the public site from one reviewed immutable artifact configuration. */
import { readFile } from 'node:fs/promises'
import { build } from 'vite'
const release = JSON.parse(await readFile(new URL('../browser-release.json', import.meta.url), 'utf8'))
if (
  release.repository !== 'foolish-bandit/cadence-0.8b-browser' ||
  !/^[a-f0-9]{40}$/.test(release.revision) ||
  !/^[a-f0-9]{64}$/.test(release.manifestSha256) ||
  !Number.isSafeInteger(release.artifactBytes) ||
  release.artifactBytes <= 0 || release.artifactBytes > 1.7e9
) throw new Error('Invalid public browser artifact configuration.')
process.env.VITE_CADENCE_ASSET_BASE = `https://huggingface.co/${release.repository}/resolve/${release.revision}/`
process.env.VITE_CADENCE_MANIFEST_SHA256 = release.manifestSha256
await build({ mode: 'pages', base: '/cadencedemo/', build: { outDir: 'dist-pages' } })
