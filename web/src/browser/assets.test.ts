import { describe, expect, it } from 'vitest'
import { unpackWeights } from './assets'

describe('lossless browser weight decoding', () => {
  it('restores BF16 bits and retains trained FP32 adapter bits', () => {
    const packed = new Uint8Array([0x80, 0x3f, 0x00, 0x80, 0x01, 0x00, 0x80, 0x3f])
    const result = unpackWeights(packed, 12, [
      { offset: 0, bytes: 4, decodedOffset: 0, decodedBytes: 8, encoding: 'bf16-bits' },
      { offset: 4, bytes: 4, decodedOffset: 8, decodedBytes: 4, encoding: 'fp32-bits' },
    ])
    expect(Array.from(new Uint32Array(result.buffer))).toEqual([0x3f800000, 0x80000000, 0x3f800001])
  })
  it('rejects gaps, overlaps and malformed or oversized decoding', () => {
    const part = { offset: 0, bytes: 2, decodedOffset: 0, decodedBytes: 4, encoding: 'bf16-bits' }
    expect(() => unpackWeights(new Uint8Array(2), 8, [part])).toThrow()
    expect(() => unpackWeights(new Uint8Array(4), 8, [part, part])).toThrow()
    expect(() => unpackWeights(new Uint8Array(2), 2 ** 32, [part])).toThrow()
    expect(() => unpackWeights(new Uint8Array(2), 4, [{ ...part, encoding: 'unknown' }])).toThrow()
  })
})
