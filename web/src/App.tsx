import { useEffect, useRef, useState } from 'react'
import { defaultDraft, examples } from './examples'
import {
  makeRequest,
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
}
type Saved = { result: Result; draft: Draft; request: ReturnType<typeof makeRequest> }
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
    let mounted = true
    const check = async () => {
      try {
        const response = await fetch('/api/health', { signal: AbortSignal.timeout(5000) })
        if (!response.ok) throw new Error()
        const body = (await response.json()) as Health
        if (body.revision !== revision || body.inference !== 'real') throw new Error()
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
  const stale = saved !== null && JSON.stringify(saved.draft) !== JSON.stringify(draft)

  async function run() {
    let request: ReturnType<typeof makeRequest>
    try {
      request = makeRequest(draft)
    } catch (e) {
      setError((e as Error).message)
      setTimeout(() => errorRef.current?.focus(), 0)
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
    if (!saved) return
    try {
      await navigator.clipboard.writeText(
        JSON.stringify({ request: saved.request, result: saved.result }, null, 2),
      )
      setCopyStatus('Copied request and result.')
    } catch {
      setCopyStatus('Clipboard is unavailable. Open Raw JSON to select and copy the result.')
    }
  }
  const summary = saved ? summarize(saved.result) : null
  const status =
    health?.status === 'loading'
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
        {preview && (
          <div className="preview-banner" role="status">
            UI test preview — results are mocked by the automated test. No model inference.
          </div>
        )}
        <section className="intro">
          <h1>Give context. Get a decision.</h1>
          <p>Explore choice, Boolean and ordinal score queries with Cadence 0.8B.</p>
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
                    ? 'Use a JSON object or array. Fields are rendered as labeled context.'
                    : 'The context to evaluate. Cadence returns a decision, not generated text.'}
                </p>
              </div>
              <div className="field">
                <label htmlFor="instructions">Instructions</label>
                <textarea
                  id="instructions"
                  value={draft.instructions}
                  maxLength={2000}
                  rows={2}
                  onChange={(e) => edit({ instructions: e.target.value })}
                />
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
                        ? `${draft.options.length} / 32`
                        : ''}
                  </span>
                </div>
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
                        draft.kind === 'score' ? 'Describe this level' : 'Description (optional)'
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
                        onClick={() => edit({ options: draft.options.filter((_, j) => j !== i) })}
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
                    disabled={draft.options.length >= 32}
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
                    Levels are indexed from 0. The score is the probability-weighted mean of these
                    indices.
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
                  <span aria-hidden="true">▶</span> Run decision
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
                Result
              </h2>
              {saved && (
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
                <details className="raw">
                  <summary>Raw JSON</summary>
                  <pre>{JSON.stringify(saved.result, null, 2)}</pre>
                </details>
                <p className="hint" role="status">
                  {copyStatus}
                </p>
              </div>
            ) : (
              <div className="empty">
                <Wave />
                <h3>{busy ? 'Evaluating your context…' : 'Your decision starts here.'}</h3>
                <p>
                  {busy
                    ? 'The distribution will appear here when local inference finishes.'
                    : 'Run a query to see the model distribution.'}
                </p>
                {!busy && health === null && (
                  <button className="text-button" onClick={() => setSetup(true)}>
                    Connect your local backend →
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
                Answer yes or no to a question. Returns the probability of true, using the model’s
                native <code>noul</code> type.
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
          <span>Local inference · CUDA required</span>
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
          Use Python 3.12 and a compatible NVIDIA CUDA GPU. The published loader rejects CPU and
          MPS; CPU optimization is being validated separately.
        </p>
        <ol>
          <li>Install the backend and model dependencies using the repository README.</li>
          <li>
            Start the backend on loopback:
            <pre>python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000</pre>
          </li>
          <li>
            Start this UI:<pre>npm ci{`\n`}npm run dev</pre>
          </li>
        </ol>
        <p>
          The first decision loads the reviewed public runtime at revision{' '}
          <code>{revision.slice(0, 12)}</code>. Fresh use downloads roughly 1.8 GB of upstream
          weights. The model stays loaded for subsequent requests.
        </p>
        <p>
          Text stays between this browser and your local backend. Initial setup contacts Hugging
          Face for model files. No request history is stored by the demo.
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
