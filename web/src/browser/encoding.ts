import { unicodeLength } from '../contract'

/** Structural interface implemented by @huggingface/tokenizers 0.2.0. */
export interface CadenceTokenizer {
  encode(text: string, options: { add_special_tokens: false }): { ids: number[] }
  token_to_id(token: string): number | undefined
}
export type BrowserQuestion = {
  type: 'choice' | 'noul' | 'score'
  instructions: string
  criteria?: Record<string, string | null> | string[]
}
export type BrowserRequest = { state: string | object; questions: Record<string, BrowserQuestion> }
export type NativeRecord = {
  state: string
  questions: { instr: string; options: string[]; label: 0 }[]
}
export type NativeEncoding = {
  ids: number[]
  seg: number[]
  pos: number[]
  opt: number[]
  option_isolation: false
  decide_idx: number[]
  opt_idx: number[][]
  labels: number[]
  state_truncated: false
}
export const delimiters = [
  '<|fim_prefix|>',
  '<|fim_middle|>',
  '<|box_start|>',
  '<|box_end|>',
  '<|fim_suffix|>',
] as const
export const tokenizerRevision = 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'

export function sanitizeCallerText(text: string): string {
  unicodeLength(text)
  return text.replace(/<\|([A-Za-z0-9_]+)\|>/g, '<¦$1¦>')
}

// Python str(float), for numbers transmitted through JSON.stringify first. The
// browser wire contract turns -0 and integer-valued doubles into JSON integers.
function pythonWireNumber(value: number): string {
  if (!Number.isFinite(value)) throw new Error('JSON numbers must be finite.')
  if (Number.isInteger(value)) {
    if (!Number.isSafeInteger(value))
      throw new Error('JSON integers must stay in the safe integer range.')
    return String(value)
  }
  const wire = String(value)
  const [mantissa, explicitExponent] = wire.toLowerCase().split('e')
  if (explicitExponent !== undefined) {
    const exponent = Number(explicitExponent)
    return `${mantissa}e${exponent < 0 ? '-' : '+'}${String(Math.abs(exponent)).padStart(2, '0')}`
  }
  const negative = value < 0 ? '-' : ''
  const absolute = negative ? wire.slice(1) : wire
  if (Math.abs(value) >= 0.0001) return wire
  const digits = absolute.slice(2)
  const first = digits.search(/[1-9]/)
  const significant = digits.slice(first)
  const exponent = -(first + 1)
  const coefficient =
    significant.length === 1 ? significant : `${significant[0]}.${significant.slice(1)}`
  return `${negative}${coefficient}e-${String(-exponent).padStart(2, '0')}`
}

// Python str.lstrip recognizes U+001C..001F but does not recognize U+FEFF.
const pythonLeadingWhitespace =
  /^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/u

/** Native api.render, applied to the canonical browser JSON wire value. */
export function renderState(value: unknown, indent = 0): string {
  if (indent > 128) throw new Error('JSON nesting exceeds the browser limit of 128 levels.')
  if (value === null) return ''
  if (typeof value === 'string') {
    unicodeLength(value)
    return value
  }
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'number') return pythonWireNumber(value)
  const pad = '  '.repeat(indent)
  if (Array.isArray(value))
    return value
      .map(
        (item) => `${pad}- ${renderState(item, indent + 1).replace(pythonLeadingWhitespace, '')}`,
      )
      .join('\n')
  if (typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype) {
    return Object.entries(value)
      .map(([key, item]) => {
        unicodeLength(key)
        return item !== null && typeof item === 'object'
          ? `${pad}${key}:\n${renderState(item, indent + 1)}`
          : `${pad}${key}: ${renderState(item)}`
      })
      .join('\n')
  }
  throw new Error('State contains an unsupported JSON value.')
}

function optionText(name: string, description: string | null | undefined): string {
  unicodeLength(name)
  if (description === null || description === undefined || description === '') return name
  if (typeof description !== 'string') throw new Error('Option descriptions must be text or null.')
  unicodeLength(description)
  return `${name}: ${description}`
}

/** Single causal row; strict limits never silently truncate caller input. */
export function encodeRequest(
  tokenizer: CadenceTokenizer,
  request: BrowserRequest,
  { maxState = 8192, maxTokens = 8192 }: { maxState?: number; maxTokens?: number } = {},
): { record: NativeRecord; encoding: NativeEncoding; questionId: string; optionIds: string[] } {
  const entries = Object.entries(request.questions)
  if (entries.length !== 1)
    throw new Error('Browser inference accepts exactly one question per request.')
  const [questionId, question] = entries[0]
  if (typeof question.instructions !== 'string')
    throw new Error('Question instructions must be text.')
  let optionIds: string[], options: string[]
  if (question.type === 'score') {
    if (!Array.isArray(question.criteria) || question.criteria.some((x) => typeof x !== 'string'))
      throw new Error('Score criteria must be an ordered list of descriptions.')
    options = question.criteria.map((x) => renderState(x))
    optionIds = options.map((_, i) => String(i))
  } else if (question.type === 'choice' || question.type === 'noul') {
    if (Array.isArray(question.criteria) || (question.type === 'choice' && !question.criteria))
      throw new Error('Choice and Boolean criteria must be an object.')
    const criteria = question.criteria ?? {}
    optionIds = question.type === 'noul' ? ['false', 'true'] : Object.keys(criteria)
    options = optionIds.map((key, i) =>
      optionText(question.type === 'noul' ? ['no', 'yes'][i] : key, criteria[key]),
    )
  } else throw new Error('Unsupported decision type.')
  if (options.length < 2 || options.length > 32)
    throw new Error('Use between 2 and 32 options or levels.')
  const record: NativeRecord = {
    state: renderState(request.state),
    questions: [{ instr: question.instructions, options, label: 0 }],
  }
  const idsFor = (text: string) =>
    tokenizer.encode(sanitizeCallerText(text), { add_special_tokens: false }).ids
  const special = delimiters.map((token) => {
    const id = tokenizer.token_to_id(token)
    if (id === undefined) throw new Error('Tokenizer lacks a required Cadence delimiter.')
    return id
  })
  const stateIds = [special[0], ...idsFor(record.state)]
  if (stateIds.length > maxState)
    throw new Error(`The state exceeds the ${maxState}-token browser limit.`)
  const instructionIds = [special[1], ...idsFor(question.instructions)]
  const spans = options.map((option) => [special[2], ...idsFor(option), special[3]])
  const branch = [...instructionIds, ...spans.flat(), special[4]]
  if (stateIds.length + branch.length > maxTokens)
    throw new Error(`The decision row exceeds the ${maxTokens}-token browser limit.`)
  const ids = [...stateIds, ...branch]
  let cursor = stateIds.length + instructionIds.length
  const ends = spans.map((span) => {
    cursor += span.length
    return cursor - 1
  })
  return {
    record,
    questionId,
    optionIds,
    encoding: {
      ids,
      seg: [...stateIds.map(() => 0), ...branch.map(() => 1)],
      pos: ids.map((_, i) => i),
      opt: [
        ...stateIds.map(() => -1),
        ...instructionIds.map(() => -1),
        ...spans.flatMap((span, i) => span.map(() => i)),
        -2,
      ],
      option_isolation: false,
      decide_idx: [ids.length - 1],
      opt_idx: [ends],
      labels: [0],
      state_truncated: false,
    },
  }
}
