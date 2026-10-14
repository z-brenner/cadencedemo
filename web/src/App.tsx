import { useEffect, useRef, useState } from 'react'
import { defaultDraft, examples } from './examples'
import {
  makeRequest,
  deviceLabel,
  validProfile,
  type Device,
  revision,
  summarize,
  validateResult,
  type Draft,
  type Kind,
  type Result,
} from './contract'

type Health = {
  status: 'cold' | 'loading' | 'ready' | 'error'
  busy: boolean
  inference: string
  revision: string
  device: Device
  profile: string
  limits: { options: number }
}
type Saved = { result: Result; draft: Draft; request: ReturnType<typeof makeRequest> }
type Prepared = { draft: Draft; request: ReturnType<typeof makeRequest> }
const hosted = import.meta.env.MODE === 'pages'
const kindNames: Record<Kind, string> = { choice: 'Choice', noul: 'Boolean', score: 'Score' }
function Wave({ small = false }: { small?: boolean }) {
  return (
    <svg
      className={small ? 'wave small' : 'wave'}
      width="42"
      height="42"
      viewBox="0 0 42 42"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M2 22h5l3-7 4 13L20 5l6 31 4-19 3 10 3-5h4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  )
}

export default function App() {
  const [draft, setDraft] = useState<Draft>(() => defaultDraft('choice'))
  const [selected, setSelected] = useState('Play music')
  const [saved, setSaved] = useState<Saved | null>(null)
  const [prepared, setPrepared] = useState<Prepared | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [health, setHealth] = useState<Health | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [copyStatus, setCopyStatus] = useState('')
  const [setup, setSetup] = useState(false)
  const resultRef = useRef<HTMLHeadingElement>(null)
  const errorRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const setupRef = useRef<HTMLDialogElement>(null)
  const preview = new URLSearchParams(window.location.search).get('preview') === 'ui-test'

  useEffect(() => {
    if (hosted) return
    let mounted = true
    const check = async () => {
      try {
        const response = await fetch('/api/health', { signal: AbortSignal.timeout(5000) })
        if (!response.ok) throw new Error()
        const body = (await response.json()) as Health
        if (
          body.revision !== revision ||
          body.inference !== 'real' ||
          !validProfile(body.device, body.profile) ||
          body.limits?.options !== (body.device === 'cpu' ? 16 : 32)
        )
          throw new Error()
        if (mounted) setHealth(body)
      } catch {
        if (mounted) setHealth(null)
      }
    }
    void check()
    const interval = setInterval(() => void check(), busy ? 2000 : 10000)
    return () => {
      mounted = false
      clearInterval(interval)
    }
  }, [busy])
  useEffect(() => {
    if (!busy) return
    const start = Date.now()
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [busy])
  useEffect(() => () => abortRef.current?.abort(), [])
  useEffect(() => {
    if (setup) setupRef.current?.showModal()
    else setupRef.current?.close()
  }, [setup])

  function edit(patch: Partial<Draft>) {
    setDraft((d) => ({ ...d, ...patch }))
    setSelected('')
    setError('')
    setCopyStatus('')
  }
  function loadExample(name: string) {
    const example = examples.find((e) => e.name === name)!
    setDraft(structuredClone(example.draft))
    setSelected(name)
    setError('')
    setCopyStatus('')
  }
  function changeKind(kind: Kind) {
    setDraft(defaultDraft(kind))
    setSelected(examples.find((e) => e.draft.kind === kind)!.name)
    setError('')
    setCopyStatus('')
  }
  const previous = saved ?? prepared
  const stale = previous !== null && JSON.stringify(previous.draft) !== JSON.stringify(draft)

  const maxOptions = health?.limits.options ?? 32

  async function run() {
    let request: ReturnType<typeof makeRequest>
    try {
      request = makeRequest(draft, maxOptions)
    } catch (e) {
      setError((e as Error).message)
      setTimeout(() => errorRef.current?.focus(), 0)
      return
    }
    if (hosted) {
      setPrepared({ request, draft: structuredClone(draft) })
      setError('')
      setCopyStatus('')
      setTimeout(() => resultRef.current?.focus(), 0)
      return
    }
    setBusy(true)
    setError('')
    setSaved(null)
    setElapsed(0)
    setCopyStatus('')
    const snapshot = structuredClone(draft)
    const controller = new AbortController()
    abortRef.current = controller
    const timeout = setTimeout(() => controller.abort(), 300000)
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal: controller.signal,
      })
      let data: unknown
      try {
        data = await response.json()
      } catch {
        throw new Error(
          'The local backend did not return JSON. Start the backend using the setup guide.',
        )
      }
      if (!response.ok) {
        const detail = (data as { detail?: unknown }).detail
        throw new Error(
          typeof detail === 'string'
            ? detail
            : `Request failed (${response.status}). Check the backend terminal and retry.`,
        )
      }
      setSaved({ result: validateResult(data, request), draft: snapshot, request })
      setTimeout(() => resultRef.current?.focus(), 0)
    } catch (e) {
      const message = controller.signal.aborted
        ? 'Request timed out after 5 minutes. The backend may still be loading or computing; check its terminal before retrying.'
        : (e as Error).message === 'Failed to fetch'
          ? 'Could not reach the local backend. Start it using the setup guide, then retry.'
          : (e as Error).message
      setError(message)
      setTimeout(() => errorRef.current?.focus(), 0)
    } finally {
      clearTimeout(timeout)
      setBusy(false)
      abortRef.current = null
    }
  }
  async function copy() {
    if (!saved && !prepared) return
    try {
      await navigator.clipboard.writeText(
        saved
          ? JSON.stringify({ request: saved.request, result: saved.result }, null, 2)
          : JSON.stringify(prepared!.request),
      )
      setCopyStatus(saved ? 'Copied request and result.' : 'Copied request JSON.')
    } catch {
      setCopyStatus('Clipboard is unavailable. Open Raw JSON to select and copy the result.')
    }
  }
  const summary = saved ? summarize(saved.result) : null
  const status = hosted
    ? 'Static site · no inference'
    : health?.status === 'loading'
      ? 'Loading model'
      : health?.busy
        ? 'Backend busy'
        : health?.status === 'ready'
          ? 'Model ready'
          : health?.status === 'cold'
            ? 'Loads on first run'
            : health?.status === 'error'
              ? 'Model setup needs attention'
              : 'Backend not connected'

  return (
    <>
      <a className="skip" href="#workbench">
        Skip to playground
      </a>
      <header className="header">
        <a className="brand" href="./" aria-label="Cadence home">
          <Wave small />
          Cadence
          <span className="demo-badge">Demo</span>
        </a>
        <nav aria-label="Resources">
          <a
            href={`https://huggingface.co/foolish-bandit/cadence-0.8b/tree/${revision}`}
            target="_blank"
            rel="noreferrer"
          >
            Model card <span aria-hidden="true">↗</span>
          </a>
          <a href="https://github.com/z-brenner/cadencedemo" target="_blank" rel="noreferrer">
            GitHub <span aria-hidden="true">↗</span>
          </a>
        </nav>
      </header>
      <main>
        {hosted && (
          <aside className="hosted-banner" aria-label="Hosting and inference">
            <strong>Static playground · no model inference</strong>
            <p>
              Edit examples and inspect a validated request. Run Cadence with the local setup
              guide.
            </p>
            <button className="text-button" onClick={() => setSetup(true)}>
              Run locally →
            </button>
          </aside>
        )}
        {preview && (
          <div className="preview-banner" role="status">
            UI test preview — results are mocked by the automated test. No model inference.
          </div>
        )}
        <section className="intro">
          <h1>Give context. Get a decision.</h1>
          <p>
            {hosted
              ? 'Explore typed decision inputs. Run the model locally when you’re ready.'
              : 'Explore choice, Boolean and ordinal score queries with Cadence 0.8B.'}
          </p>
          <details className="about-demo">
            <summary>How to use this demo</summary>
            <p>
              Cadence is a small model that turns context and a question into a structured
              decision. It chooses an option, estimates the probability of true, or scores
              ordered levels. It does not generate chat replies.
            </p>
            <ol>
              <li>
                Pick an example. Edit its state (the context), instructions (your question), and
                possible answers.
              </li>
              <li>
                {hosted
                  ? 'Inspect and copy the request JSON. Use the setup guide to run these inputs in the local playground.'
                  : 'Run a decision. First use loads the model; later runs reuse it.'}
              </li>
              <li>
                Read the distribution as the model’s preference across your answers. A high
                probability does not establish correctness or certainty.
              </li>
            </ol>
            <p>
              This is a demo. The public website explores inputs; real inference runs locally.{' '}
              <a
                href="https://github.com/z-brenner/cadencedemo/blob/main/docs/BENCHMARK_AND_CPU_GUIDANCE.md"
                target="_blank"
                rel="noreferrer"
              >
                Evaluation and CPU limitations ↗
              </a>
            </p>
          </details>
        </section>
        <section className="presets" aria-label="Example presets">
          <span>Example presets</span>
          <div>
            {examples.map((e) => (
              <button
                key={e.name}
                type="button"
                className={selected === e.name ? 'preset active' : 'preset'}
                disabled={busy}
                onClick={() => loadExample(e.name)}
                aria-pressed={selected === e.name}
              >
                {e.name}
              </button>
            ))}
          </div>
        </section>
        <div className="workbench" id="workbench">
          <form
            className="editor panel"
            onSubmit={(e) => {
              e.preventDefault()
              void run()
            }}
          >
            <fieldset disabled={busy} className="edit-fields">
              <div className="toolbar">
                <fieldset className="control-group">
                  <legend>Query type</legend>
                  <div className="segmented">
                    {(['choice', 'noul', 'score'] as Kind[]).map((kind) => (
                      <button
                        key={kind}
                        type="button"
                        aria-pressed={draft.kind === kind}
                        className={draft.kind === kind ? 'selected' : ''}
                        onClick={() => changeKind(kind)}
                      >
                        {kindNames[kind]}
                      </button>
                    ))}
                  </div>
                </fieldset>
                <fieldset className="control-group">
                  <legend>State format</legend>
                  <div className="segmented format">
                    {(['text', 'json'] as const).map((format) => (
                      <button
                        key={format}
                        type="button"
                        aria-pressed={draft.format === format}
                        className={draft.format === format ? 'selected' : ''}
                        onClick={() =>
                          draft.format !== format &&
                          edit({
                            format,
                            state:
                              format === 'json'
                                ? JSON.stringify({ text: draft.state }, null, 2)
                                : draft.state,
                          })
                        }
                      >
                        {format === 'text' ? 'Text' : 'JSON'}
                      </button>
                    ))}
                  </div>
                </fieldset>
              </div>
              <div className="field">
                <div className="field-label">
                  <label htmlFor="state">State</label>
                  <span>{draft.state.length.toLocaleString()} / 12,000</span>
                </div>
                <textarea
                  id="state"
                  className="code"
                  value={draft.state}
                  maxLength={12000}
                  rows={4}
                  onChange={(e) => edit({ state: e.target.value })}
                  aria-describedby="state-hint"
                  spellCheck={draft.format === 'text'}
                />
                <p id="state-hint" className="hint">
                  {draft.format === 'json'
                    ? 'State is the context: use a JSON object or array of facts.'
                    : 'State is the text context you want Cadence to evaluate.'}
                </p>
              </div>
              <div className="field">
                <label htmlFor="instructions">Instructions</label>
                <textarea
                  id="instructions"
                  value={draft.instructions}
                  maxLength={2000}
                  rows={2}
                  aria-describedby="instructions-hint"
                  onChange={(e) => edit({ instructions: e.target.value })}
                />
                <p id="instructions-hint" className="hint">
                  The question or task you want the model to decide.
                </p>
              </div>
              <div className="field options">
                <div className="field-label">
                  <span className="label">
                    {draft.kind === 'score'
                      ? 'Ordered levels'
                      : draft.kind === 'noul'
                        ? 'Boolean criteria'
                        : 'Options'}
                  </span>
                  <span>
                    {draft.kind === 'score'
                      ? 'Lowest → highest'
                      : draft.kind === 'choice'
                        ? `${draft.options.length} / ${maxOptions}`
                        : ''}
                  </span>
                </div>
                {draft.kind !== 'score' && (
                  <p className="hint">
                    {draft.kind === 'noul'
                      ? 'Describe what false and true mean for your question.'
                      : 'Each option is a possible answer. IDs identify the answers; descriptions explain them.'}
                  </p>
                )}
                {draft.options.map((option, i) => (
                  <div className={`option-row ${draft.kind}`} key={i}>
                    <span className="option-index" aria-hidden="true">
                      {String(i + 1).padStart(2, '0')}
                    </span>
                    {draft.kind === 'choice' ? (
                      <input
                        className="code option-id"
                        aria-label={`Option ${i + 1} ID`}
                        value={option.id}
                        maxLength={64}
                        onChange={(e) =>
                          edit({
                            options: draft.options.map((o, j) =>
                              j === i ? { ...o, id: e.target.value } : o,
                            ),
                          })
                        }
                      />
                    ) : (
                      <span className="fixed-id">{draft.kind === 'score' ? i : option.id}</span>
                    )}
                    <input
                      aria-label={`${draft.kind === 'score' ? 'Level' : 'Option'} ${draft.kind === 'score' ? i : i + 1} description`}
                      value={option.description}
                      maxLength={500}
                      placeholder={
                        draft.kind === 'score'
                          ? 'Describe this level'
                          : 'Description (optional)'
                      }
                      onChange={(e) =>
                        edit({
                          options: draft.options.map((o, j) =>
                            j === i ? { ...o, description: e.target.value } : o,
                          ),
                        })
                      }
                    />
                    {draft.kind !== 'noul' && (
                      <button
                        className="remove"
                        type="button"
                        aria-label={`Remove ${draft.kind === 'score' ? 'level' : 'option'} ${draft.kind === 'score' ? i : i + 1}`}
                        disabled={draft.options.length <= 2}
                        onClick={() =>
                          edit({ options: draft.options.filter((_, j) => j !== i) })
                        }
                      >
                        <svg
                          width="16"
                          height="16"
                          viewBox="0 0 24 24"
                          fill="none"
                          aria-hidden="true"
                        >
                          <path
                            d="M4 6h16M9 6V3h6v3M7 6l1 15h8l1-15M10 10v7M14 10v7"
                            stroke="currentColor"
                            strokeWidth="1.6"
                            strokeLinecap="round"
                          />
                        </svg>
                      </button>
                    )}
                  </div>
                ))}
                {draft.kind !== 'noul' && (
                  <button
                    className="add secondary"
                    type="button"
                    disabled={draft.options.length >= maxOptions}
                    onClick={() =>
                      edit({
                        options: [
                          ...draft.options,
                          { id: `option_${draft.options.length + 1}`, description: '' },
                        ],
                      })
                    }
                  >
                    + &nbsp; Add {draft.kind === 'score' ? 'level' : 'option'}
                  </button>
                )}
                {draft.kind === 'score' && (
                  <p className="hint">
                    Levels are indexed from 0. The score is the probability-weighted mean of
                    these indices.
                  </p>
                )}
              </div>
            </fieldset>
            {error && (
              <div className="error" role="alert" tabIndex={-1} ref={errorRef}>
                {error}
              </div>
            )}
            <button className="run" type="submit" disabled={busy}>
              {busy ? (
                <>
                  <span className="spinner" aria-hidden="true" />{' '}
                  {health?.status === 'loading' ? 'Loading model…' : 'Running decision…'}
                </>
              ) : (
                <>
                  <span aria-hidden="true">{hosted ? '↗' : '▶'}</span>{' '}
                  {hosted ? 'Inspect request' : 'Run decision'}
                </>
              )}
            </button>
            {busy && (
              <p className="loading-note" role="status">
                {elapsed}s elapsed.{' '}
                {health?.status === 'loading'
                  ? 'First use downloads and loads the pinned model; this can take several minutes.'
                  : 'One decision is running on the local backend.'}
              </p>
            )}
          </form>
          <section className="result panel" aria-labelledby="result-title" aria-busy={busy}>
            <div className="result-header">
              <h2 id="result-title" ref={resultRef} tabIndex={-1}>
                {hosted ? 'Request' : 'Result'}
              </h2>
              {(saved || prepared) && (
                <button className="text-button" onClick={() => void copy()}>
                  Copy JSON
                </button>
              )}
            </div>
            {saved && summary ? (
              <div className="result-content">
                {stale && (
                  <p className="stale" role="status">
                    Inputs changed. This result belongs to your previous run.
                  </p>
                )}
                {preview && <p className="mock-result">Mock result · UI test only</p>}
                <p className="result-label">{summary.label}</p>
                <p className="answer">{summary.title}</p>
                {saved.result.kind === 'score' && (
                  <p className="hint">An expected level can fall between two descriptions.</p>
                )}
                <div className="distribution" aria-label="Model probability distribution">
                  {saved.result.native_option_ids.map((key, i) => {
                    const probability = saved.result.probabilities[key]
                    const option =
                      saved.result.kind === 'choice'
                        ? saved.draft.options.find((o) => o.id === key)!
                        : saved.draft.options[i]
                    return (
                      <div key={key} className="probability-row">
                        <div className="probability-label">
                          <span>{saved.result.kind === 'score' ? `Level ${key}` : key}</span>
                          <strong>{(probability * 100).toFixed(1)}%</strong>
                        </div>
                        <div className="bar" aria-hidden="true">
                          <div style={{ width: `${probability * 100}%` }} />
                        </div>
                        {option.description && <p>{option.description}</p>}
                      </div>
                    )
                  })}
                </div>
                <p className="interpretation">
                  Probabilities describe the model output, not a guarantee of correctness.
                </p>
                <div className="metadata">
                  <span>{saved.result.tokens} packed tokens</span>
                  <span>{(saved.result.seconds * 1000).toFixed(0)} ms model call</span>
                </div>
                <p className="hint">Timing excludes model loading and network overhead.</p>
                <p className="hint">
                  {deviceLabel(saved.result.device)}
                  {saved.result.device === 'cpu'
                    ? ' · precision differs from the public CUDA profile'
                    : ''}
                </p>
                <details className="raw">
                  <summary>Raw JSON</summary>
                  <pre>{JSON.stringify(saved.result, null, 2)}</pre>
                </details>
                <p className="hint" role="status">
                  {copyStatus}
                </p>
              </div>
            ) : prepared ? (
              <div className="result-content request-preview">
                {stale && (
                  <p className="stale" role="status">
                    Inputs changed. Inspect again to update this request.
                  </p>
                )}
                <p className="result-label">Validated typed request</p>
                <h3>Ready for local inference.</h3>
                <p>No inference has run.</p>
                <p className="hint">
                  Copy this JSON for the local API, or use these inputs in the local playground.
                  No probability or decision is produced on this site. Device and native token
                  limits are checked again locally.
                </p>
                <div className="raw">
                  <pre>{JSON.stringify(prepared.request, null, 2)}</pre>
                </div>
                <p className="hint" role="status">
                  {copyStatus}
                </p>
              </div>
            ) : (
              <div className="empty">
                <Wave />
                <h3>
                  {hosted
                    ? 'Explore a typed request.'
                    : busy
                      ? 'Evaluating your context…'
                      : 'Your decision starts here.'}
                </h3>
                <p>
                  {hosted
                    ? 'Inspect a request to see the JSON you can send to Cadence locally.'
                    : busy
                      ? 'The distribution will appear here when local inference finishes.'
                      : 'Run a query to see the model distribution.'}
                </p>
                {!busy && health === null && (
                  <button className="text-button" onClick={() => setSetup(true)}>
                    {hosted ? 'Local setup guide →' : 'Connect your local backend →'}
                  </button>
                )}
              </div>
            )}
            <div className="result-foot">
              <span
                className={`status-dot ${health?.status === 'ready' ? 'ready' : ''}`}
                aria-hidden="true"
              />
              <span>{status}</span>
            </div>
          </section>
        </div>
        <section className="guide" aria-labelledby="guide-title">
          <h2 id="guide-title">Three typed query kinds</h2>
          <div className="guide-columns">
            <div>
              <h3>Choice</h3>
              <p>
                Select the best option from a list of labeled choices. Use for intent
                classification, routing and tool selection.
              </p>
            </div>
            <div>
              <h3>Boolean</h3>
              <p>
                Answer yes or no to a question. Returns the probability of true, using the
                model’s native <code>noul</code> type.
              </p>
            </div>
            <div>
              <h3>Score</h3>
              <p>
                Assign an ordinal score over ordered levels, starting at 0. Use for preference,
                quality or relevance judgments.
              </p>
            </div>
          </div>
          <p className="guide-note">
            <span aria-hidden="true">ⓘ</span> Probabilities describe the model output, not a
            guarantee of correctness.
          </p>
        </section>
        <footer>
          <span>
            {hosted ? 'Static request explorer · ' : 'Local inference · '}
            {health
              ? health.device === 'cpu'
                ? 'CPU FP32 (experimental)'
                : 'CUDA'
              : 'CPU FP32 or CUDA'}
          </span>
          <button className="text-button" onClick={() => setSetup(true)}>
            Setup guide <span aria-hidden="true">→</span>
          </button>
        </footer>
      </main>
      <dialog
        ref={setupRef}
        onCancel={() => setSetup(false)}
        onClose={() => setSetup(false)}
        aria-labelledby="setup-title"
      >
        <div className="dialog-header">
          <h2 id="setup-title">Run Cadence locally</h2>
          <button
            className="secondary"
            onClick={() => setSetup(false)}
            aria-label="Close setup guide"
          >
            Close
          </button>
        </div>
        <p>
          Use Python 3.12. Choose the public CUDA profile or the experimental CPU FP32 port. For
          CPU, cache the pinned model files first and select CADENCE_DEVICE=cpu with
          HF_HUB_OFFLINE=1. CPU precision differs from the CUDA profile. MPS is unsupported.
        </p>
        {hosted && (
          <p>
            Open the local playground in a separate tab after starting both servers below. This
            GitHub Pages site does not connect to your local backend.
          </p>
        )}
        <ol>
          <li>Install the backend and model dependencies using the repository README.</li>
          <li>
            Start the backend on loopback:
            <pre>
              {health?.device === 'cpu' ? 'CADENCE_DEVICE=cpu HF_HUB_OFFLINE=1 ' : ''}python -m
              uvicorn backend.app:app --host 127.0.0.1 --port 8000
            </pre>
          </li>
          <li>
            Start this UI:<pre>npm ci{`\n`}npm run dev</pre>
          </li>
        </ol>
        {hosted && (
          <p>
            <a
              className="setup-link"
              href="http://127.0.0.1:5173"
              target="_blank"
              rel="noreferrer"
            >
              Open local playground
            </a>{' '}
            (after setup). The local UI proxies requests to the loopback API on the same origin.
          </p>
        )}
        <p>
          The first decision reconstructs the pinned model at revision{' '}
          <code>{revision.slice(0, 12)}</code>. Fresh use downloads roughly 1.8 GB of upstream
          weights. CPU requires those files already cached. The model stays loaded for
          subsequent requests.
        </p>
        <p>
          {hosted
            ? 'Inputs edited on this static site stay in this tab. For inference, use the local UI and backend. '
            : 'Text stays between this browser and your local backend. '}
          Initial setup contacts Hugging Face for model files. No request history is stored by
          the demo.
        </p>
        <a
          className="setup-link"
          href="https://github.com/z-brenner/cadencedemo#local-setup"
          target="_blank"
          rel="noreferrer"
        >
          Read complete setup and troubleshooting ↗
        </a>
      </dialog>
    </>
  )
}
