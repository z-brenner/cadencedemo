export const revision = '17cdf36034677e406033fbf97637b4db1984ca70'
export type Kind = 'choice' | 'noul' | 'score'
export type Option = { id: string; description: string }
export type Draft = {
  kind: Kind
  format: 'text' | 'json'
  state: string
  instructions: string
  options: Option[]
}
export type DecisionRequest = {
  state: string | object
  questions: {
    decision: {
      type: Kind
      instructions: string
      criteria: Record<string, string | null> | string[]
    }
  }
}
export type Result = {
  kind: Kind
  native_option_ids: string[]
  probabilities: Record<string, number>
  value: number | null
  tokens: number
  seconds: number
  truncated: false
  model: string
  revision: string
}

export function makeRequest(draft: Draft): DecisionRequest {
  if (!draft.state.trim()) throw new Error('Add a state to evaluate.')
  if (draft.state.length > 12000) throw new Error('Keep the state within 12,000 characters.')
  if (!draft.instructions.trim()) throw new Error('Add instructions for this decision.')
  if (draft.instructions.length > 2000)
    throw new Error('Keep instructions within 2,000 characters.')
  let state: string | object = draft.state
  if (draft.format === 'json') {
    try {
      state = JSON.parse(draft.state, (_key, value) => {
        if (typeof value === 'number' && !Number.isFinite(value))
          throw new Error('Nonfinite JSON number')
        return value
      })
    } catch {
      throw new Error('State is not valid JSON. Use a JSON object or array.')
    }
    if (state === null || typeof state !== 'object' || !Object.keys(state).length)
      throw new Error('JSON state must be a nonempty object or array.')
    if (JSON.stringify(state).length > 12000)
      throw new Error('Keep JSON state within 12,000 serialized characters.')
  }
  if (draft.options.length < 2 || draft.options.length > 32)
    throw new Error('Use between 2 and 32 options or levels.')
  if (draft.options.some((o) => o.description.length > 500))
    throw new Error('Keep each description within 500 characters.')
  let criteria: Record<string, string | null> | string[]
  if (draft.kind === 'score') {
    if (draft.options.some((o) => !o.description.trim()))
      throw new Error('Describe every score level, from lowest to highest.')
    criteria = draft.options.map((o) => o.description)
  } else {
    if (draft.options.some((o) => !o.id.trim() || o.id.length > 64))
      throw new Error('Option IDs need 1–64 characters.')
    if (new Set(draft.options.map((o) => o.id)).size !== draft.options.length)
      throw new Error('Each option needs a unique ID.')
    if (
      draft.kind === 'noul' &&
      (draft.options.length !== 2 ||
        !draft.options.some((o) => o.id === 'false') ||
        !draft.options.some((o) => o.id === 'true'))
    )
      throw new Error('Boolean uses false and true criteria.')
    criteria = Object.fromEntries(draft.options.map((o) => [o.id, o.description || null]))
  }
  return {
    state,
    questions: { decision: { type: draft.kind, instructions: draft.instructions, criteria } },
  }
}

export function validateResult(data: unknown, request: DecisionRequest): Result {
  const r = data as Result
  const q = request.questions.decision
  const keys =
    q.type === 'choice'
      ? Object.keys(q.criteria)
      : q.type === 'noul'
        ? ['false', 'true']
        : (q.criteria as string[]).map((_, i) => String(i))
  if (
    !r ||
    r.kind !== q.type ||
    r.revision !== revision ||
    r.model !== 'foolish-bandit/cadence-0.8b' ||
    r.truncated !== false ||
    !Array.isArray(r.native_option_ids) ||
    JSON.stringify(r.native_option_ids) !== JSON.stringify(keys) ||
    !r.probabilities ||
    Object.keys(r.probabilities).length !== keys.length ||
    keys.some(
      (k) =>
        !Number.isFinite(r.probabilities[k]) || r.probabilities[k] < 0 || r.probabilities[k] > 1,
    ) ||
    Math.abs(keys.reduce((sum, k) => sum + r.probabilities[k], 0) - 1) > 0.00001 ||
    !Number.isFinite(r.seconds) ||
    r.seconds < 0 ||
    !Number.isInteger(r.tokens) ||
    r.tokens < 1
  )
    throw new Error('The backend returned an invalid result. Check the backend version and retry.')
  if (r.kind === 'score' || r.kind === 'noul') {
    const expected =
      r.kind === 'noul'
        ? r.probabilities.true
        : keys.reduce((sum, k) => sum + Number(k) * r.probabilities[k], 0)
    if (
      typeof r.value !== 'number' ||
      !Number.isFinite(r.value) ||
      Math.abs(r.value - expected) > 0.00001
    )
      throw new Error('The backend returned an inconsistent score. No result was displayed.')
  }
  return r
}

export function summarize(result: Result) {
  if (result.kind === 'score')
    return {
      label: 'Expected level',
      title: `${result.value!.toFixed(2)} / ${result.native_option_ids.length - 1}`,
    }
  if (result.kind === 'noul')
    return { label: 'Probability of true', title: `${(result.value! * 100).toFixed(1)}%` }
  const winner = result.native_option_ids.reduce((best, key) =>
    result.probabilities[key] > result.probabilities[best] ? key : best,
  )
  return { label: 'Highest-probability option', title: winner }
}
