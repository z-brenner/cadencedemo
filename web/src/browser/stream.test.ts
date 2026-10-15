import { describe, expect, it } from 'vitest'
import { readModelStream, ProgressGate } from './stream'
import { sha256 } from './assets'

function stream(chunks: number[][], canceled?: () => void) {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(Uint8Array.from(chunk))
      controller.close()
    },
    cancel: canceled,
  })
}
describe('bounded model-file streaming', () => {
  it('assembles unequal chunks once and reports absolute counts through the final byte', async () => {
    const counts: number[] = []
    expect(
      await readModelStream(stream([[1, 2], [3], [4, 5, 6]]), 6, (n) => counts.push(n)),
    ).toEqual(Uint8Array.from([1, 2, 3, 4, 5, 6]))
    expect(counts).toEqual([2, 3, 6])
  })
  it('rejects a short stream and a stream exceeding the pinned size', async () => {
    await expect(readModelStream(stream([[1]]), 2, () => {})).rejects.toThrow('incomplete')
    await expect(readModelStream(stream([[1, 2], [3]]), 2, () => {})).rejects.toThrow('exceeded')
  })
  it('waits for EOF even when the expected byte count has arrived', async () => {
    let end!: () => void
    let settled = false
    const pending = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(Uint8Array.of(1))
        end = () => c.close()
      },
    })
    const read = readModelStream(pending, 1, () => {}).then((value) => {
      settled = true
      return value
    })
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(settled).toBe(false)
    end()
    expect(await read).toEqual(Uint8Array.of(1))
  })
  it('rejects failed network reads and invalid sizes without returning file bytes', async () => {
    const failed = new ReadableStream<Uint8Array>({
      start(c) {
        c.error(new Error('network interrupted'))
      },
    })
    await expect(readModelStream(failed, 2, () => {})).rejects.toThrow('network interrupted')
    await expect(readModelStream(stream([]), 0, () => {})).rejects.toThrow('size')
  })
})
it('limits progress frequency while forcing final/file transitions immediately', () => {
  const gate = new ProgressGate(100)
  expect([
    gate.allow(0),
    gate.allow(10),
    gate.allow(99),
    gate.allow(100),
    gate.allow(110, true),
  ]).toEqual([true, false, false, true, true])
})
it('hashes only the exact view, including a nonzero byte offset', async () => {
  const all = Uint8Array.from([99, 97, 98, 99, 88])
  expect(await sha256(all.subarray(1, 4))).toBe(
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
  )
  expect(await sha256(all)).not.toBe(await sha256(all.subarray(1, 4)))
})
