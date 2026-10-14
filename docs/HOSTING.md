# Public website and local inference

The [GitHub Pages research demo](https://z-brenner.github.io/cadencedemo/) serves the guided UI, a dedicated browser worker and ONNX runtime. Choosing **Download model** fetches approximately **1.54 GB** of public model files, plus **28 MB** of runtime, before enabling decisions. Nothing downloads automatically on page load. The complete trained text model evaluates choice, yes/no and ordinal rating questions on the device, without an inference API or Python backend. Inputs are not sent to Hugging Face; the download hosts receive the visitor's ordinary network information, including IP address.

[browser-release.json](../browser-release.json) pins `foolish-bandit/cadence-0.8b-browser@17705d964a94b323a3eab04a01e345c7446d6e54` and manifest SHA-256 `3b773382a7463f5b2bffcba7dd0d0896fbccff33e85ed65cedf60fea19529657`. [Public file verification](browser-public-artifact-verification.json) compares all 89 assets and the manifest with the validated local export. The public resolve/CDN paths permit CORS reads from Pages. The original checkpoint and all notices remain intact.

A desktop browser with substantial available memory is recommended. WASM FP32 runs in one dedicated worker; its token/answer limits and precision differences are explained in [browser qualification](BROWSER_FEASIBILITY.md). Mobile layout checks do not establish mobile hardware inference. Experimental outputs can be wrong.

Progress reports verified bytes. Cancel stops the worker; retry reuses only completed, verified cached files. Every cached file is reverified, corrupt files are rejected/repaired, and storage failures are explicit. CacheStorage can be unavailable or evicted. Context/history are not persisted by this application. No embedded credentials or remote inference fallback exist.

With both browser asset settings absent, `npm run build:pages` still creates the honest static request explorer: edit examples → inspect/copy JSON → local setup. It visibly explains that browser model downloads are unavailable in that build. The ordinary local API build remains separate.

## Run real inference locally

Follow the [README setup](../README.md#local-setup), including the pinned model dependencies and cache preparation for CPU. Run the loopback backend on port 8000 and `npm run dev` on port 5173. From the site's setup guide, open **http://127.0.0.1:5173** in a separate tab after starting both servers. On that local UI, choose/edit an example and run a decision. Vite proxies `/api` to the loopback backend, so browser requests share the local UI origin. Copied request JSON can also be sent to the documented local HTTP API. The default public CUDA profile and explicit CPU FP32 profile remain available locally.

The Pages URL does **not** connect to the local backend. The existing API grants no cross-origin CORS permission; its JSON POST preflight from `https://z-brenner.github.io` is rejected. HTTPS and local-network browser policies add platform-specific constraints: loopback has mixed-content exceptions, but those do not grant CORS access or guarantee local-network permission. We do not ask visitors to disable browser protections or expose their loopback service. See [MDN CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS) and [mixed-content guidance](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Mixed_content).

A server-based public inference service would need a separately provisioned model-capable backend, HTTPS and resource management. No such service is configured or purchased. Pages cannot execute Python/Torch, but it can serve the browser worker and runtime while the browser downloads pinned model assets from a separate public host and performs its own inference.

## Deployment and maintenance

`.github/workflows/pages.yml` first builds/tests the static fallback, then builds the browser-enabled site from the reviewed release configuration and checks its opt-in UI, then uploads **only `dist-pages/`** with `actions/upload-pages-artifact`. The main-branch deployment uses `actions/deploy-pages` and the `github-pages` environment. Build permissions are read-only; only the deployment job has Pages/OIDC write permissions. No repository sources, Python environment, weights, test artifacts or private files enter the Pages artifact. CI checks local, static fallback, mocked browser interactions and the published browser preflight before merge. CI does not download the heavy model or claim to run inference.

The production path is `/cadencedemo/`, including generated JavaScript, CSS, favicon and home navigation. Local development keeps `/`. Commands:

```bash
npm run build:browser
npm run test:published
# Preview the exact published browser build:
npx vite preview --host 127.0.0.1 --outDir dist-pages --base /cadencedemo/ --port 5175
```

The configuration follows [GitHub's custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) and [Vite's repository-site base-path guidance](https://vite.dev/guide/static-deploy.html#github-pages). Deployment is limited to this repository's public website and its approved browser artifact. It does not publish an inference endpoint or change the public model checkpoint.
