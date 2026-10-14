# Public website and local inference

The [GitHub Pages site](https://z-brenner.github.io/cadencedemo/) is a static request explorer. It provides editable examples for all three native query types, browser validation, request JSON inspection/copy and local setup guidance. It makes no inference requests, loads no model weights, and produces no decisions or probabilities. There are no mock results or embedded credentials. Inputs remain in the page until the user copies them; no history is persisted.

The isolated browser candidate runs the full trained model locally in a browser worker, after an explicit approximately 1.54 GB model download. It uses no inference API or Python backend. It is not yet enabled on the public site: external artifact publication is awaiting approval, and the actual immutable host revision/CORS behavior must be verified first. See [browser qualification and hosting requirements](BROWSER_FEASIBILITY.md). With both browser asset build settings absent, the existing static/local modes are preserved.

## Run real inference locally

Follow the [README setup](../README.md#local-setup), including the pinned model dependencies and cache preparation for CPU. Run the loopback backend on port 8000 and `npm run dev` on port 5173. From the site's setup guide, open **http://127.0.0.1:5173** in a separate tab after starting both servers. On that local UI, choose/edit an example and run a decision. Vite proxies `/api` to the loopback backend, so browser requests share the local UI origin. Copied request JSON can also be sent to the documented local HTTP API. The default public CUDA profile and explicit CPU FP32 profile remain available locally.

The Pages URL does **not** connect to the local backend. The existing API grants no cross-origin CORS permission; its JSON POST preflight from `https://z-brenner.github.io` is rejected. HTTPS and local-network browser policies add platform-specific constraints: loopback has mixed-content exceptions, but those do not grant CORS access or guarantee local-network permission. We do not ask visitors to disable browser protections or expose their loopback service. See [MDN CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS) and [mixed-content guidance](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Mixed_content).

A server-based public inference service would need a separately provisioned model-capable backend, HTTPS and resource management. No such service is configured or purchased. Pages cannot execute Python/Torch, but it can serve the browser worker and runtime while the browser downloads pinned model assets from a separate public host and performs its own inference.

## Deployment and maintenance

`.github/workflows/pages.yml` builds and browser-tests the static mode, then uploads **only `dist-pages/`** with `actions/upload-pages-artifact`. The main-branch deployment uses `actions/deploy-pages` and the `github-pages` environment. Build permissions are read-only; only the deployment job has Pages/OIDC write permissions. No repository sources, Python environment, weights, test artifacts or private files enter the Pages artifact. Existing CI remains and checks both local and static builds before merge.

The production path is `/cadencedemo/`, including generated JavaScript, CSS, favicon and home navigation. Local development keeps `/`. Commands:

```bash
npm run build:pages
npm run test:pages
# Preview the exact static build:
npx vite preview --host 127.0.0.1 --outDir dist-pages --base /cadencedemo/ --port 5175
```

The configuration follows [GitHub's custom Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) and [Vite's repository-site base-path guidance](https://vite.dev/guide/static-deploy.html#github-pages). Deployment is limited to this repository's public static website. It does not publish an inference endpoint or change the public model checkpoint.
