import { useEffect, useRef, useState } from 'react'
import { defaultDraft, examples } from './examples'
import { browserModelAvailable, useBrowserModel } from './browser/useBrowserModel'
import { downloadPercentage, downloadByteLabel } from './browser/progress'
import {
  makeRequest,
  deviceLabel,
  validProfile,
  type Device,
  revision,
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
const staticExplorer = import.meta.env.MODE === 'pages' && !browserModelAvailable
const kindNames: Record<Kind, string> = {
  choice: 'Choose an answer',
  noul: 'Yes or no',
  score: 'Rate on a scale',
}
function answerLabel(id: string) {
  if (id === 'true') return 'Yes'
  if (id === 'false') return 'No'
  const label = id.replaceAll('_', ' ')
  return label.charAt(0).toUpperCase() + label.slice(1)
}
function choiceLabel(id: string, ids: readonly string[]) {
  const label = answerLabel(id)
  const collisions = ids.filter((other) => answerLabel(other).toLowerCase() === label.toLowerCase())
  return collisions.length > 1 ? `${label} (ID: ${id})` : label
}
function nextOptionId(options: Draft['options']) {
  const used = new Set(options.map((option) => option.id))
  let index = 1
  while (used.has(`option_${index}`)) index += 1
  return `option_${index}`
}
function friendlySummary(result: Result) {
  if (result.kind === 'score')
    return {
      label: 'Average model rating',
      title: `${result.value!.toFixed(2)} / ${result.native_option_ids.length - 1}`,
    }
  if (result.kind === 'noul')
    return {
      label: 'Model probability of Yes',
      title: `${(result.value! * 100).toFixed(1)}%`,
    }
  const highest = Math.max(...Object.values(result.probabilities))
  const leaders = result.native_option_ids.filter((id) => result.probabilities[id] === highest)
  return {
    label: leaders.length === 1 ? 'Cadence favors' : 'Equal model preferences',
    title:
      leaders.length === 1
        ? choiceLabel(leaders[0], result.native_option_ids)
        : 'No single leading answer',
  }
}
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
  const browserModel = useBrowserModel()
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
    if (staticExplorer || browserModelAvailable) return
    let mounted = true
    const check = async () => {
      try {
        const response = await fetch('/api/health', { signal: AbortSignal.timeout(5000) })
        if (!response.ok) throw new Error()
        const body = (await response.json()) as Health
        if (
          body.revision !== revision ||
          body.inference !== 'real' ||
          (body.device !== 'cpu' && body.device !== 'cuda') ||
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

  const maxOptions = browserModelAvailable ? 16 : (health?.limits.options ?? 32)
  const draftOptionIds = draft.options.map((option) => option.id)

  async function run() {
    let request: ReturnType<typeof makeRequest>
    try {
      request = makeRequest(draft, maxOptions)
    } catch (e) {
      setError((e as Error).message)
      setTimeout(() => errorRef.current?.focus(), 0)
      return
    }
    if (staticExplorer) {
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
    if (browserModelAvailable) {
      try {
        const result = await browserModel.run(request)
        setSaved({ result, draft: snapshot, request })
        setTimeout(() => resultRef.current?.focus(), 0)
      } catch (cause) {
        const message = (cause as Error).message
        if (!message.startsWith('Browser operation canceled.')) {
          setError(message)
          setTimeout(() => errorRef.current?.focus(), 0)
        }
      } finally {
        setBusy(false)
      }
      return
    }
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
  const summary = saved ? friendlySummary(saved.result) : null
  const status = browserModelAvailable
    ? browserModel.status === 'ready'
      ? 'Model files ready on this device'
      : browserModel.status === 'running'
        ? 'Evaluating on this device'
        : 'Browser model ' + browserModel.status
    : staticExplorer
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
        {preview && (
          <div className="preview-banner" role="status">
            UI test preview — results are mocked by the automated test. No model inference.
          </div>
        )}
        <section className="intro">
          <p className="research-label">Research project · interactive demo</p>
          <h1>Give context. Get a decision.</h1>
          <p>
            {staticExplorer
              ? 'Try a familiar example, then make it your own. Explore how Cadence compares the answers you provide.'
              : browserModelAvailable
                ? 'Add context, ask a question, and compare answers on your device with Cadence 0.8B.'
                : 'Add context, ask a question, and compare the possible answers with Cadence 0.8B.'}
          </p>
          <p className="research-note">
            Experimental model outputs can be wrong. Test them for your use case before relying on
            them.
          </p>
          <details className="about-demo">
            <summary>How to use this demo</summary>
            <p>
              Cadence is a research project exploring small models for structured decisions. It turns
              context and a question into a structured decision. It chooses an option, estimates the
              probability of true, or scores ordered levels. It
              does not generate chat replies.
            </p>
            <ol>
              <li>Pick an example. Edit the context, question, and possible answers.</li>
              <li>
                {staticExplorer
                  ? 'Inspect and copy the request JSON. Use the setup guide to run these inputs in the local playground.'
                  : browserModelAvailable
                    ? 'Download the model, then get a decision. Later visits can reuse verified saved files.'
                    : 'Get a decision. First use loads the model; later runs reuse it.'}
              </li>
              <li>
                Read the distribution as the model’s preference across your answers. A high
                probability does not establish correctness or certainty.
              </li>
            </ol>
            <p>
              {browserModelAvailable
                ? 'This demo runs the trained model in your browser after you download it.'
                : 'This is a demo. The public website explores inputs; real inference runs locally.'}{' '}
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
        {staticExplorer && (
          <aside className="staticExplorer-banner" aria-labelledby="download-availability-title">
            <div>
              <div className="availability-heading">
                <h2 id="download-availability-title">Browser model download</h2>
                <span className="availability-status">Not available yet</span>
              </div>
              <p>
                Browser model files are awaiting publication. For now, explore requests below or
                run the model locally.
              </p>
              <p className="availability-mode">Static playground · no model inference</p>
            </div>
            <button className="secondary" onClick={() => setSetup(true)}>
              Run locally →
            </button>
          </aside>
        )}
        {browserModelAvailable && (
          <section className="model-preparation" aria-labelledby="model-preparation-title">
            <div className="model-preparation-heading">
              <h2 id="model-preparation-title">Run Cadence on this device</h2>
              {browserModel.status === 'ready' && <span className="ready-label">Files ready</span>}
            </div>
            <p>
              First use downloads about 1.54 GB of model files plus 28 MB of browser runtime. Your
              inputs stay on this device. The download host sees your IP address; your inputs are
              not sent to an inference API.
            </p>
            {browserModel.status === 'idle' && (
              <p className="hint">
                The model is downloaded only when you choose Download model. A desktop browser is
                recommended. Limited device memory can prevent execution.
              </p>
            )}
            {['checking', 'downloading', 'verifying', 'loading'].includes(browserModel.status) && (
              <div className="download-progress" role="status">
                <p>
                  {browserModel.status === 'downloading'
                    ? 'Downloading remaining model files…'
                    : browserModel.status === 'verifying'
                      ? 'Verifying and saving downloaded files…'
                      : browserModel.status === 'loading'
                        ? 'All model files verified. Preparing tokenizer…'
                        : browserModel.stage === 'Clearing saved model'
                          ? 'Clearing saved model…'
                          : 'Checking saved model files…'}
                </p>
                {browserModel.status === 'downloading' && browserModel.totalBytes > 0 ? (
                  <>
                    <progress
                      max={browserModel.totalBytes}
                      value={browserModel.downloadComplete
                        ? browserModel.downloadedBytes
                        : Math.min(browserModel.downloadedBytes, browserModel.totalBytes - 1)}
                      aria-label="Model file download"
                    />
                    <p>
                      {downloadByteLabel(browserModel.downloadedBytes, browserModel.totalBytes)} ({downloadPercentage(browserModel)!.toFixed(1)}%)
                    </p>
                  </>
                ) : (
                  <progress aria-label={browserModel.stage === 'Clearing saved model'
                    ? 'Clearing saved model'
                    : browserModel.status === 'checking'
                      ? 'Verifying saved model files'
                      : browserModel.status === 'loading'
                        ? 'Preparing tokenizer' : 'Verifying downloaded model files'} />
                )}
                {browserModel.status === 'checking' && browserModel.totalFiles > 0 && (
                  <p>{browserModel.checkedFiles} of {browserModel.totalFiles} files checked.</p>
                )}
                {browserModel.cachedBytes > 0 && (
                  <p className="hint">{(browserModel.cachedBytes / 1_000_000).toFixed(1)} MB of saved model files verified; excluded from the download total.</p>
                )}
                {browserModel.status === 'downloading' && (
                  <p className="hint">Progress measures this attempt’s remaining model download. Files are verified before use; the browser runtime loads on the first decision.</p>
                )}
              </div>
            )}
            {browserModel.status === 'ready' && (
              <p role="status">
                Model files are ready on this device. 100% of model files verified.
                {browserModel.fromCache ? ' Verified saved files were reused.' : ''} Get decision
                initializes the browser runtime and evaluates your question.
              </p>
            )}
            {browserModel.status === 'running' && (
              <p role="status">Evaluating your question on this device…</p>
            )}
            {browserModel.status === 'canceled' && (
              <p role="status">
                Canceled. Your inputs are unchanged. Retry reuses only completed, verified files; an
                unfinished file downloads again.
              </p>
            )}
            {browserModel.error && !error && (
              <p className="model-error" role="alert">
                {browserModel.error}
              </p>
            )}
            {browserModel.stage === 'Saved model cleared' && (
              <p role="status">Saved files for this model release cleared.</p>
            )}
            <div className="model-actions">
              {['idle', 'canceled', 'error'].includes(browserModel.status) && (
                <button
                  className="run"
                  type="button"
                  onClick={() => {
                    setError('')
                    browserModel.prepare()
                  }}
                >
                  {browserModel.status === 'idle' ? 'Download model' : 'Retry model preparation'}
                </button>
              )}
              {['checking', 'downloading', 'verifying', 'loading', 'running'].includes(browserModel.status) &&
                browserModel.stage !== 'Clearing saved model' && (
                  <button className="secondary" type="button" onClick={() => browserModel.cancel()}>
                    {browserModel.status === 'running' ? 'Cancel decision' : 'Cancel preparation'}
                  </button>
                )}
              {browserModel.status === 'error' && (
                <button className="text-button" type="button" onClick={() => setSetup(true)}>
                  Local setup guide
                </button>
              )}
            </div>
            <details className="model-details">
              <summary>Model and saved files</summary>
              <p className="hint">
                Verified files are saved in this browser. If local storage is unavailable,
                preparation stops with an error. Saved files can be cleared or evicted, so a later
                visit may need another download.
              </p>
              <p className="hint">
                The browser uses FP32 inference, up to 16 answers, and up to 512 packed tokens.
                Oversized requests are rejected; they are not silently shortened.
              </p>
              {browserModel.stage && <p className="hint">Current step: {browserModel.stage}</p>}
              {browserModel.provider && (
                <p className="hint">Runtime: {browserModel.provider.toUpperCase()}</p>
              )}
              <p className="hint">Model revision: {revision}</p>
              <button
                className="secondary"
                type="button"
                disabled={
                  busy || ['checking', 'downloading', 'verifying', 'loading'].includes(browserModel.status)
                }
                onClick={() => {
                  setError('')
                  void browserModel.clearCache()
                }}
              >
                Clear saved model
              </button>
            </details>
          </section>
        )}
        <section className="presets" aria-label="Example presets">
          <span>Try an example</span>
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
        <p className="scenario-note">
          Cadence compares the answers you provide. It does not play music, book tables, or check
          live weather.
        </p>
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
                  <legend>Answer format</legend>
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
              </div>
              <div className="field">
                <div className="field-label">
                  <label htmlFor="state">
                    <span className="step" aria-hidden="true">
                      1
                    </span>
                    Context
                  </label>
                  <span>{draft.state.length.toLocaleString()} / 12,000</span>
                </div>
                <textarea
                  id="state"
                  aria-label="Context"
                  className={draft.format === 'json' ? 'code' : undefined}
                  value={draft.state}
                  maxLength={12000}
                  rows={4}
                  onChange={(e) => edit({ state: e.target.value })}
                  aria-describedby="state-hint"
                  spellCheck={draft.format === 'text'}
                />
                <p id="state-hint" className="hint">
                  {draft.format === 'json'
                    ? 'This example uses structured context. Keep it as a JSON object or array.'
                    : 'What should Cadence consider? Describe a situation or paste a message.'}
                </p>
              </div>
              <div className="field">
                <label htmlFor="instructions">
                  <span className="step" aria-hidden="true">
                    2
                  </span>
                  Question
                </label>
                <textarea
                  id="instructions"
                  aria-label="Question"
                  value={draft.instructions}
                  maxLength={2000}
                  rows={2}
                  aria-describedby="instructions-hint"
                  onChange={(e) => edit({ instructions: e.target.value })}
                />
                <p id="instructions-hint" className="hint">
                  What should Cadence decide? Ask a question the answers below can resolve.
                </p>
              </div>
              <div className="field options">
                <div className="field-label">
                  <span className="label">
                    <span className="step" aria-hidden="true">
                      3
                    </span>
                    <span>
                      {draft.kind === 'score'
                        ? 'Rating levels'
                        : draft.kind === 'noul'
                          ? 'What Yes and No mean'
                          : 'Possible answers'}
                    </span>
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
                      ? 'Describe what a Yes or No answer means for your question.'
                      : 'Describe each answer so Cadence can compare them.'}
                  </p>
                )}
                {draft.options.map((option, i) => (
                  <div className={`option-row ${draft.kind}`} key={i}>
                    <span className="option-index" aria-hidden="true">
                      {String(i + 1).padStart(2, '0')}
                    </span>
                    <span className="answer-name">
                      {draft.kind === 'score'
                        ? `Level ${i}`
                        : draft.kind === 'choice'
                          ? choiceLabel(option.id, draftOptionIds)
                          : answerLabel(option.id)}
                    </span>
                    <textarea
                      className="answer-description"
                      rows={2}
                      aria-label={`${draft.kind === 'score' ? 'Level' : 'Answer'} ${draft.kind === 'score' ? i : i + 1} description`}
                      value={option.description}
                      maxLength={500}
                      placeholder={
                        draft.kind === 'score' ? 'Describe this level' : 'Describe this answer'
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
                    disabled={draft.options.length >= maxOptions}
                    onClick={() =>
                      edit({
                        options: [
                          ...draft.options,
                          { id: nextOptionId(draft.options), description: '' },
                        ],
                      })
                    }
                  >
                    + &nbsp; Add {draft.kind === 'score' ? 'level' : 'option'}
                  </button>
                )}
                {draft.kind === 'score' && (
                  <p className="hint">
                    Arrange levels from lowest to highest, starting at 0. The result averages the
                    model’s preference across these levels.
                  </p>
                )}
              </div>
            </fieldset>
            <details className="advanced-input">
              <summary>Advanced input details</summary>
              <p className="hint">These controls affect the exact request sent to the model.</p>
              <fieldset className="control-group">
                <legend>State format</legend>
                <div className="segmented format">
                  {(['text', 'json'] as const).map((format) => (
                    <button
                      key={format}
                      type="button"
                      disabled={busy}
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
              {draft.kind === 'choice' && (
                <div className="native-ids">
                  <p className="hint">
                    Answer IDs are part of the model input. Existing examples keep their original
                    IDs.
                  </p>
                  {draft.options.map((option, i) => (
                    <label key={i}>
                      Answer {i + 1} ID
                      <input
                        className="code"
                        aria-label={`Option ${i + 1} ID`}
                        value={option.id}
                        maxLength={64}
                        disabled={busy}
                        onChange={(e) =>
                          edit({
                            options: draft.options.map((o, j) =>
                              j === i ? { ...o, id: e.target.value } : o,
                            ),
                          })
                        }
                      />
                    </label>
                  ))}
                </div>
              )}
            </details>
            {error && (
              <div className="error" role="alert" tabIndex={-1} ref={errorRef}>
                {error}
              </div>
            )}
            <button
              className="run"
              type="submit"
              disabled={busy || (browserModelAvailable && browserModel.status !== 'ready')}
            >
              {busy ? (
                <>
                  <span className="spinner" aria-hidden="true" />{' '}
                  {health?.status === 'loading' ? 'Loading model…' : 'Running decision…'}
                </>
              ) : (
                <>
                  <span aria-hidden="true">{staticExplorer ? '↗' : '▶'}</span>{' '}
                  {staticExplorer ? 'Inspect request' : 'Get decision'}
                </>
              )}
            </button>
            {busy && (
              <p className="loading-note" role="status">
                {elapsed}s elapsed.{' '}
                {browserModelAvailable
                  ? 'The model is evaluating your inputs in this browser.'
                  : health?.status === 'loading'
                    ? 'First use downloads and loads the pinned model; this can take several minutes.'
                    : 'One decision is running on the local backend.'}
              </p>
            )}
          </form>
          <section className="result panel" aria-labelledby="result-title" aria-busy={busy}>
            <div className="result-header">
              <h2 id="result-title" ref={resultRef} tabIndex={-1}>
                {staticExplorer ? 'Request' : 'Result'}
              </h2>
              {prepared && (
                <button className="text-button" onClick={() => void copy()}>
                  Copy JSON
                </button>
              )}
            </div>
            {saved && summary ? (
              <div className="result-content">
                {stale && (
                  <p className="stale" role="status">
                    You changed the inputs. Get a new decision to update this result.
                  </p>
                )}
                {preview && <p className="mock-result">Mock result · UI test only</p>}
                <p className="result-label">{summary.label}</p>
                <p className="answer">{summary.title}</p>
                {saved.result.kind === 'score' && (
                  <p className="hint">
                    This is a weighted average across the levels below, so it can fall between them.
                  </p>
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
                          <span>
                            {saved.result.kind === 'score'
                              ? `Level ${key}`
                              : saved.result.kind === 'choice'
                                ? choiceLabel(key, saved.result.native_option_ids)
                                : answerLabel(key)}
                          </span>
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
                  These percentages describe the model’s output. They do not measure how likely the
                  answer is to be correct.
                </p>
                <details className="technical-result">
                  <summary>Technical result details</summary>
                  <div className="metadata">
                    <span>{saved.result.tokens} packed tokens</span>
                    <span>{(saved.result.seconds * 1000).toFixed(0)} ms model call</span>
                  </div>
                  <p className="hint">
                    {saved.result.device === 'browser'
                      ? 'Timing includes reading saved files, preparing layers and computing this decision.'
                      : 'Timing excludes model loading and network overhead.'}
                  </p>
                  <p className="hint">
                    {deviceLabel(saved.result.device)}
                    {saved.result.device === 'cpu'
                      ? ' · precision differs from the public CUDA profile'
                      : saved.result.device === 'browser'
                        ? ' · qualified against the CPU FP32 reference; precision differs from the public CUDA profile'
                        : ''}
                  </p>
                  <p className="hint">Pinned model revision: {saved.result.revision}</p>
                  <button className="text-button" onClick={() => void copy()}>
                    Copy JSON
                  </button>
                </details>
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
                <h3>Your inputs are ready.</h3>
                <p>No inference has run.</p>
                <p className="hint">
                  Copy this JSON for the local API, or use these inputs in the local playground. No
                  probability or decision is produced on this site. Device and native token limits
                  are checked again locally.
                </p>
                <details className="raw">
                  <summary>Request JSON</summary>
                  <pre>{JSON.stringify(prepared.request, null, 2)}</pre>
                </details>
                <p className="hint" role="status">
                  {copyStatus}
                </p>
              </div>
            ) : (
              <div className="empty">
                <Wave />
                <h3>
                  {staticExplorer
                    ? 'Explore a typed request.'
                    : busy
                      ? 'Evaluating your context…'
                      : 'Your result will appear here.'}
                </h3>
                <p>
                  {staticExplorer
                    ? 'Inspect a request to see the JSON you can send to Cadence locally.'
                    : busy
                      ? browserModelAvailable
                        ? 'The distribution will appear here when browser inference finishes.'
                        : 'The distribution will appear here when local inference finishes.'
                      : 'Choose an example or edit the inputs, then get a decision to compare the answers.'}
                </p>
                {!busy && !browserModelAvailable && health === null && (
                  <button className="text-button" onClick={() => setSetup(true)}>
                    {staticExplorer ? 'Local setup guide →' : 'Connect your local backend →'}
                  </button>
                )}
              </div>
            )}
            <div className="result-foot">
              <span
                className={`status-dot ${(browserModelAvailable ? browserModel.status === 'ready' : health?.status === 'ready') ? 'ready' : ''}`}
                aria-hidden="true"
              />
              <span>{status}</span>
            </div>
          </section>
        </div>
        <section className="guide" aria-labelledby="guide-title">
          <h2 id="guide-title">What can you try?</h2>
          <div className="guide-columns">
            <div>
              <h3>Choose an answer</h3>
              <p>
                Compare possible answers, such as which team should handle a support message.
                Cadence returns a preference for each answer.
              </p>
            </div>
            <div>
              <h3>Yes or no</h3>
              <p>
                Ask whether a message looks like spam. Cadence estimates the probability of Yes and
                No using the descriptions you provide.
              </p>
            </div>
            <div>
              <h3>Rate on a scale</h3>
              <p>
                Rate how helpful a reply is. Describe levels from lowest to highest, then see the
                model’s preference across the scale.
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
            Research demo ·{' '}
            {browserModelAvailable
              ? 'On-device browser inference · FP32'
              : staticExplorer
                ? 'Static request explorer · '
                : 'Local inference · '}
            {!browserModelAvailable &&
              (health
                ? health.device === 'cpu'
                  ? 'CPU FP32 (experimental)'
                  : 'CUDA'
                : 'CPU FP32 or CUDA')}
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
        {staticExplorer && (
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
        {staticExplorer && (
          <p>
            <a className="setup-link" href="http://127.0.0.1:5173" target="_blank" rel="noreferrer">
              Open local playground
            </a>{' '}
            (after setup). The local UI proxies requests to the loopback API on the same origin.
          </p>
        )}
        <p>
          The first decision reconstructs the pinned model at revision{' '}
          <code>{revision.slice(0, 12)}</code>. Fresh use downloads roughly 1.8 GB of upstream
          weights. CPU requires those files already cached. The model stays loaded for subsequent
          requests.
        </p>
        <p>
          {staticExplorer
            ? 'Inputs edited on this static site stay in this tab. For inference, use the local UI and backend. '
            : 'Text stays between this browser and your local backend. '}
          Initial setup contacts Hugging Face for model files. No request history is stored by the
          demo.
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
