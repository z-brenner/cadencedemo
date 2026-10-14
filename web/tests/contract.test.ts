import { describe, expect, it } from 'vitest'
import { makeRequest, summarize, validateResult } from '../src/contract'
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
