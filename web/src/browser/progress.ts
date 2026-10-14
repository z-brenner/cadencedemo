/** One preparation attempt: cached verification and network transfer are separate. */
export type ProgressSnapshot = {
  downloadedBytes: number
  totalBytes: number
  cachedBytes: number
  verifiedBytes: number
  checkedFiles: number
  totalFiles: number
  downloadComplete: boolean
  filesReady: boolean
}
export function downloadPercentage(p: Pick<ProgressSnapshot, 'downloadedBytes' | 'totalBytes' | 'downloadComplete'>): number | null {
  if (p.totalBytes <= 0) return null
  if (p.downloadComplete) return 100
  return Math.min(99.9, Math.floor((p.downloadedBytes / p.totalBytes) * 1000) / 10)
}
export function downloadByteLabel(received: number, total: number): string {
  const scale = total >= 1_000_000 ? 1_000_000 : total >= 1000 ? 1000 : 1
  const unit = scale === 1_000_000 ? 'MB' : scale === 1000 ? 'kB' : 'B'
  const digits = scale === 1 ? 0 : 1
  return `${(received / scale).toFixed(digits)} ${unit} of ${(total / scale).toFixed(digits)} ${unit}`
}
export class PreparationProgress {
  private sizes: Map<string, number>
  private checkedFiles = new Set<string>()
  private cached = new Set<string>()
  private receivedBytes = new Map<string, number>()
  private ended = new Set<string>()
  private verifiedFiles = new Set<string>()
  constructor(files: Record<string, { bytes: number }>) {
    this.sizes = new Map(Object.entries(files).map(([name, meta]) => [name, meta.bytes]))
    if (!this.sizes.size || [...this.sizes.values()].some(n => !Number.isSafeInteger(n) || n <= 0) ||
        !Number.isSafeInteger([...this.sizes.values()].reduce((a,b)=>a+b,0)))
      throw new Error('Invalid progress file sizes.')
  }
  private size(file: string) {
    const size = this.sizes.get(file)
    if (size === undefined) throw new Error('Invalid progress operation.')
    return size
  }
  checked(file: string, reused: boolean) {
    this.size(file)
    if (this.checkedFiles.has(file)) throw new Error('File was already checked.')
    this.checkedFiles.add(file)
    if (reused) { this.cached.add(file); this.verifiedFiles.add(file) }
  }
  received(file: string, absoluteBytes: number) {
    const size = this.size(file)
    if (this.checkedFiles.size !== this.sizes.size || this.cached.has(file) || this.ended.has(file) ||
        !Number.isSafeInteger(absoluteBytes) || absoluteBytes < (this.receivedBytes.get(file) ?? 0) || absoluteBytes > size)
      throw new Error('Invalid streamed byte count.')
    this.receivedBytes.set(file, absoluteBytes)
  }
  finished(file: string) {
    if (this.receivedBytes.get(file) !== this.size(file) || this.ended.has(file))
      throw new Error('Model file stream is incomplete.')
    this.ended.add(file)
  }
  verified(file: string) {
    this.size(file)
    if (!this.ended.has(file) || this.verifiedFiles.has(file)) throw new Error('Model file is not ready for verification.')
    this.verifiedFiles.add(file)
  }
  snapshot(): ProgressSnapshot {
    const sum = (names: Iterable<string>) => [...names].reduce((n, file) => n + this.sizes.get(file)!, 0)
    const checkedAll = this.checkedFiles.size === this.sizes.size
    const missing = [...this.sizes.keys()].filter(name => !this.cached.has(name))
    return {
      downloadedBytes: [...this.receivedBytes.values()].reduce((a,b)=>a+b,0),
      totalBytes: checkedAll ? sum(missing) : 0,
      cachedBytes: sum(this.cached),
      verifiedBytes: sum(this.verifiedFiles),
      checkedFiles: this.checkedFiles.size,
      totalFiles: this.sizes.size,
      downloadComplete: checkedAll && missing.every(name => this.ended.has(name)),
      filesReady: this.verifiedFiles.size === this.sizes.size,
    }
  }
}
