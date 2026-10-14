import { describe, expect, it } from 'vitest'
import { encodeRequest, renderState, sanitizeCallerText } from './encoding'

const tokenizer = {
  token_to_id: (token: string) =>
    ({
      '<|fim_prefix|>': 10,
      '<|fim_middle|>': 11,
      '<|box_start|>': 12,
      '<|box_end|>': 13,
      '<|fim_suffix|>': 14,
    })[token],
  encode: (text: string) => ({ ids: [...text].map((x) => x.codePointAt(0)!) }),
}

describe('native Cadence browser encoding', () => {
  it('renders typed JSON with Python scalar labels and nested indentation', () => {
    expect(
      renderState({ flag: true, absent: null, list: [' hi', { flag: false }], nested: { n: 2 } }),
    ).toBe('flag: True\nabsent: \nlist:\n  - hi\n  - flag: False\nnested:\n  n: 2')
  })
  it('matches Python scientific notation after the browser JSON wire normalization', () => {
    expect(renderState([0.00001, 0.0001, -1.23e-7, 5e-324, -0, 1e15])).toBe(
      '- 1e-05\n- 0.0001\n- -1.23e-07\n- 5e-324\n- 0\n- 1000000000000000',
    )
  })
  it('uses Python lstrip whitespace rather than JavaScript trim', () => {
    expect(renderState(['\u001chello', '\ufeffhello'])).toBe('- hello\n- \ufeffhello')
  })
  it('neutralizes caller controls in every text field', () => {
    expect(sanitizeCallerText('<|fim_suffix|> <|im_start|> <|A_12|>')).toBe(
      '<¦fim_suffix¦> <¦im_start¦> <¦A_12¦>',
    )
  })
  it('includes exact option IDs, order, boundaries, and readout positions', () => {
    const result = encodeRequest(tokenizer, {
      state: 'a',
      questions: {
        q: { type: 'choice', instructions: 'b', criteria: { first: null, second: '' } },
      },
    })
    expect(result.record.questions[0].options).toEqual(['first', 'second'])
    expect(result.encoding.decide_idx).toEqual([19])
    expect(result.encoding.opt_idx).toEqual([[10, 18]])
    expect(result.optionIds).toEqual(['first', 'second'])
    expect(result.encoding.ids[0]).toBe(10)
    expect(result.encoding.ids.at(-1)).toBe(14)
    expect(result.encoding.pos).toEqual(result.encoding.ids.map((_, i) => i))
  })
  it('maps Boolean display words and score descriptions to native output IDs', () => {
    expect(
      encodeRequest(tokenizer, {
        state: 'a',
        questions: {
          q: { type: 'noul', instructions: 'b', criteria: { true: 'present', false: 'absent' } },
        },
      }).record.questions[0].options,
    ).toEqual(['no: absent', 'yes: present'])
    const result = encodeRequest(tokenizer, {
      state: 'a',
      questions: { q: { type: 'score', instructions: 'b', criteria: ['low', 'high'] } },
    })
    expect(result.record.questions[0].options).toEqual(['low', 'high'])
    expect(result.optionIds).toEqual(['0', '1'])
  })
  it('rejects overflow and unsupported values instead of truncating or approximating', () => {
    expect(() => renderState({ x: Number.MAX_SAFE_INTEGER + 1 })).toThrow(/safe/)
    expect(() => renderState({ x: Infinity })).toThrow(/finite/)
    expect(() => renderState({ x: '\ud800' })).toThrow(/Unicode/)
    expect(() =>
      encodeRequest(
        tokenizer,
        { state: 'abc', questions: { q: { type: 'noul', instructions: 'b' } } },
        { maxState: 3 },
      ),
    ).toThrow(/state/)
    expect(() =>
      encodeRequest(
        tokenizer,
        { state: 'a', questions: { q: { type: 'noul', instructions: 'b' } } },
        { maxTokens: 5 },
      ),
    ).toThrow(/row/)
  })
})
