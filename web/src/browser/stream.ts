/** One pinned file, one assembly buffer. No cache writes happen before EOF/hash checks. */
export async function readModelStream(
  stream: ReadableStream<Uint8Array>,
  expectedBytes: number,
  received: (absoluteBytes: number) => void,
): Promise<Uint8Array<ArrayBuffer>> {
  if (
    !Number.isSafeInteger(expectedBytes) ||
    expectedBytes <= 0 ||
    expectedBytes > 64 * 1024 * 1024
  )
    throw new Error('Invalid model file size.')
  const value = new Uint8Array(expectedBytes)
  const reader = stream.getReader()
  let count = 0
  try {
    for (;;) {
      const chunk = await reader.read()
      if (chunk.done) break
      if (count + chunk.value.byteLength > expectedBytes)
        throw new Error('A model download exceeded its verified size.')
      value.set(chunk.value, count)
      count += chunk.value.byteLength
      received(count)
    }
    if (count !== expectedBytes) throw new Error('Model file stream is incomplete.')
    return value
  } catch (error) {
    await reader.cancel().catch(() => {})
    throw error
  } finally {
    reader.releaseLock()
  }
}

export class ProgressGate {
  private last: number | null = null
  constructor(private intervalMs = 100) {}
  allow(now: number, force = false): boolean {
    if (force || this.last === null || now - this.last >= this.intervalMs) {
      this.last = now
      return true
    }
    return false
  }
}
