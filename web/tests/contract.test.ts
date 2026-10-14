import { describe, expect, it } from 'vitest'
import { makeRequest, summarize, validateResult } from '../src/contract'
import unicodeCases from '../../tests/fixtures/unicode-state-limits.json'
import compactCases from '../../tests/fixtures/compact-json-lengths.json'
const base = {
  kind: 'choice' as const,
  format: 'text' as const,
  state: 'Play jazz.',
  instructions: 'Classify intent.',
  options: [
    { id: 'music', description: 'Play music.' },
    { id: 'weather', description: 'Check weather.' },
  ],
}
describe('native contract', () => {
  it('preserves semantic IDs and explicitly maps Boolean to noul', () => {
    expect(makeRequest(base).questions.decision.criteria).toEqual({
      music: 'Play music.',
      weather: 'Check weather.',
    })
    expect(
      makeRequest({
        ...base,
        kind: 'noul',
        options: [
          { id: 'false', description: 'No' },
          { id: 'true', description: 'Yes' },
        ],
      }).questions.decision.type,
    ).toBe('noul')
    expect(makeRequest({ ...base, kind: 'score' }).questions.decision.criteria).toEqual([
      'Play music.',
      'Check weather.',
    ])
  })
  it('parses JSON without flattening it and rejects invalid or primitive JSON', () => {
    expect(
      makeRequest({ ...base, format: 'json', state: '{"request":"Play jazz."}' }).state,
    ).toEqual({ request: 'Play jazz.' })
    for (const state of [
      '{oops',
      'null',
      'true',
      '"text"',
      '{}',
      '[]',
      '{"amount":1e400}',
      '{"nested":[{"value":-1e400}]}',
    ])
      expect(() => makeRequest({ ...base, format: 'json', state })).toThrow()
  })
  it('rejects blank, duplicate or reserved-looking IDs without prototype loss', () => {
    expect(() =>
      makeRequest({
        ...base,
        options: [
          { id: 'same', description: 'a' },
          { id: 'same', description: 'b' },
        ],
      }),
    ).toThrow(/unique/)
    expect(() => makeRequest({ ...base, state: ' ' })).toThrow(/state/i)
    const r = makeRequest({
      ...base,
      options: [
        { id: '__proto__', description: 'first' },
        { id: 'toString', description: 'second' },
      ],
    })
    expect(Object.keys(r.questions.decision.criteria)).toEqual(['__proto__', 'toString'])
  })
  it('describes a score as expected zero-based level, never confidence', () => {
    const result = {
      kind: 'score' as const,
      native_option_ids: ['0', '1', '2'],
      probabilities: { '0': 0.2, '1': 0.3, '2': 0.5 },
      value: 1.3,
      tokens: 20,
      seconds: 0.04,
      truncated: false as const,
      model: 'foolish-bandit/cadence-0.8b',
      revision: '17cdf36034677e406033fbf97637b4db1984ca70',
      device: 'cuda' as const,
      profile: 'cuda_bf16_fp32' as const,
    }
    expect(summarize(result).title).toBe('1.30 / 2')
    expect(summarize(result).label).toBe('Expected level')
    expect(() =>
      validateResult(
        { ...result, probabilities: { '0': 0.2, '1': 0.3, '2': NaN } },
        makeRequest({
          ...base,
          kind: 'score',
          options: [...base.options, { id: 'third', description: 'Great' }],
        }),
      ),
    ).toThrow()
  })
})

it('rejects inconsistent device metadata and respects CPU option limits', () => {
  const request = makeRequest(base)
  const result = {
    kind: 'choice',
    native_option_ids: ['music', 'weather'],
    probabilities: { music: 0.8, weather: 0.2 },
    value: null,
    tokens: 20,
    seconds: 0.04,
    truncated: false,
    model: 'foolish-bandit/cadence-0.8b',
    revision: '17cdf36034677e406033fbf97637b4db1984ca70',
    device: 'cpu',
    profile: 'cpu_fp32',
  }
  expect(validateResult(result, request).device).toBe('cpu')
  expect(() => validateResult({ ...result, profile: 'cuda_bf16_fp32' }, request)).toThrow()
  expect(() =>
    makeRequest(
      {
        ...base,
        options: Array.from({ length: 17 }, (_, i) => ({
          id: String(i),
          description: 'option',
        })),
      },
      16,
    ),
  ).toThrow(/16/)
})

for (const sample of unicodeCases) {
  it(`uses shared Unicode boundary: ${sample.name}`, () => {
    const text = sample.glyph.repeat(sample.repeat) + sample.padding
    const state = sample.format === 'json' ? JSON.stringify({ [sample.key!]: text }) : text
    expect(state.length).toBe(sample.units)
    const draft = { ...base, format: sample.format as 'text' | 'json', state }
    if (sample.accepted)
      expect(makeRequest(draft).state).toEqual(
        sample.format === 'json' ? JSON.parse(state) : text,
      )
    else expect(() => makeRequest(draft)).toThrow(/12,000/)
  })
}
it('uses compact JSON budget without spacing or exponent padding', () => {
  const state = JSON.stringify({ text: '日'.repeat(11980), n: 1e-7 })
  expect(state.length).toBe(12000)
  expect(makeRequest({ ...base, format: 'json', state }).state).toEqual(JSON.parse(state))
})
for (const sample of compactCases) {
  it(`measures compact JSON scalar syntax: ${sample.source}`, () => {
    const draft = { ...base, format: 'json' as const, state: sample.source }
    if (!sample.browserAccepted) {
      expect(() => makeRequest(draft)).toThrow(/safe range/)
      return
    }
    const request = makeRequest(draft)
    expect(JSON.stringify(request.state).length).toBe(sample.units)
  })
}
for (const [field, limit] of [
  ['instructions', 2000],
  ['id', 64],
  ['description', 500],
] as const) {
  it(`uses UTF-16 boundary for ${field}`, () => {
    const make = (text: string) =>
      field === 'instructions'
        ? { ...base, instructions: text }
        : {
            ...base,
            options: [
              {
                id: field === 'id' ? text : 'music',
                description: field === 'description' ? text : 'first',
              },
              base.options[1],
            ],
          }
    expect(() => makeRequest(make('😀'.repeat(limit / 2)))).not.toThrow()
    expect(() => makeRequest(make('😀'.repeat(limit / 2) + 'x'))).toThrow()
  })
}
it('rejects unpaired surrogates without altering Unicode text', () => {
  expect(() => makeRequest({ ...base, state: '\ud800' })).toThrow(/Unicode/)
  expect(() => makeRequest({ ...base, format: 'json', state: '{"text":"\\ud800"}' })).toThrow(
    /valid JSON/,
  )
  expect(makeRequest({ ...base, state: 'é日😀é' }).state).toBe('é日😀é')
})
it('keeps deeply nested bounded JSON unchanged', () => {
  const state = '['.repeat(500) + '0' + ']'.repeat(500)
  const request = makeRequest({ ...base, format: 'json', state })
  expect(JSON.stringify(request.state)).toBe(state)
})

it('rejects UTF-8 request bodies beyond the API byte limit', () => {
  const draft = {
    ...base,
    state: '日'.repeat(12000),
    instructions: '日'.repeat(2000),
    options: Array.from({ length: 32 }, (_, i) => ({
      id: String(i),
      description: '日'.repeat(500),
    })),
  }
  expect(() => makeRequest(draft)).toThrow(/64 KiB/)
})

it('accepts the exact UTF-8 body limit and rejects one additional byte', () => {
  const draft = {
    ...base,
    state: 'x',
    instructions: '日'.repeat(2000),
    options: Array.from({ length: 32 }, (_, i) => ({
      id: String(i),
      description: '日'.repeat(500),
    })),
  }
  const overhead = new TextEncoder().encode(JSON.stringify(makeRequest(draft))).length - 1
  const state = 'x'.repeat(65536 - overhead)
  expect(
    new TextEncoder().encode(JSON.stringify(makeRequest({ ...draft, state }))).length,
  ).toBe(65536)
  expect(() => makeRequest({ ...draft, state: state + 'x' })).toThrow(/64 KiB/)
})

it('rejects unsafe JSON integers instead of silently rounding context', () => {
  for (const state of [
    '{"account_id":9007199254740993}',
    '{"n":-9007199254740993}',
    '{"nested":[9007199254740992]}',
  ])
    expect(() => makeRequest({ ...base, format: 'json', state })).toThrow(/safe range/)
  const state =
    '{"max":9007199254740991,"min":-9007199254740991,"account_id":"9007199254740993"}'
  expect(makeRequest({ ...base, format: 'json', state }).state).toEqual(JSON.parse(state))
})
