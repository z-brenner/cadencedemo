import { describe, expect, it } from 'vitest'
import { PreparationProgress, downloadPercentage, downloadByteLabel } from './progress'
const sizes = { small: { bytes: 10 }, large: { bytes: 90 } }
function cold() {
  const p = new PreparationProgress(sizes)
  p.checked('small', false); p.checked('large', false)
  return p
}
describe('actual network byte progress', () => {
  it('weights unequal files by bytes and includes partial chunks exactly once', () => {
    const p = cold()
    p.received('small', 10); p.finished('small'); p.verified('small')
    p.received('large', 35)
    expect(p.snapshot().downloadedBytes).toBe(45)
    expect(downloadPercentage(p.snapshot())).toBe(45)
    p.received('large', 35)
    expect(p.snapshot().downloadedBytes).toBe(45)
    p.received('large', 45)
    expect(p.snapshot().downloadedBytes).toBe(55)
  })
  it('does not round near-complete transfers up to100 before the final stream ends', () => {
    const p = new PreparationProgress({file:{bytes:1000}})
    p.checked('file',false);p.received('file',996)
    expect(downloadPercentage(p.snapshot())).toBe(99.6)
    p.received('file',1000)
    expect(downloadPercentage(p.snapshot())).toBe(99.9)
    expect(p.snapshot().filesReady).toBe(false)
    p.finished('file')
    expect(downloadPercentage(p.snapshot())).toBe(100)
    expect(p.snapshot().filesReady).toBe(false)
    p.verified('file')
    expect(p.snapshot().filesReady).toBe(true)
  })
  it('excludes verified cached bytes from the remaining network denominator', () => {
    const p = new PreparationProgress(sizes)
    p.checked('small',true)
    expect(p.snapshot().totalBytes).toBe(0)
    p.checked('large',false);p.received('large',45)
    expect(p.snapshot()).toMatchObject({cachedBytes:10,totalBytes:90,downloadedBytes:45,verifiedBytes:10})
    expect(downloadPercentage(p.snapshot())).toBe(50)
  })
  it('represents an all-cached preparation as no download', () => {
    const p=new PreparationProgress(sizes)
    p.checked('small',true);p.checked('large',true)
    expect(p.snapshot()).toMatchObject({cachedBytes:100,totalBytes:0,downloadedBytes:0,filesReady:true})
    expect(downloadPercentage(p.snapshot())).toBeNull()
  })
  it('retries reuse only verified saved files and discard an unfinished transfer', () => {
    const p=cold();p.received('small',10);p.finished('small');p.verified('small');p.received('large',30)
    expect(p.snapshot().downloadedBytes).toBe(40)
    const retry=new PreparationProgress(sizes)
    retry.checked('small',true);retry.checked('large',false)
    expect(retry.snapshot()).toMatchObject({cachedBytes:10,totalBytes:90,downloadedBytes:0,verifiedBytes:10})
    retry.received('large',20)
    expect(retry.snapshot().downloadedBytes).toBe(20)
  })
  it('rejects byte overruns, decreasing totals, premature EOF and duplicate cache accounting', () => {
    const p=cold()
    expect(()=>p.received('large',91)).toThrow()
    p.received('large',20)
    expect(()=>p.received('large',19)).toThrow()
    expect(()=>p.finished('large')).toThrow()
    expect(()=>p.verified('large')).toThrow()
    expect(()=>p.checked('small',true)).toThrow()
    expect(()=>p.received('unknown',1)).toThrow()
  })
  it('counts100 only after every missing stream ended, even when received byte totals match', () => {
    const p=cold();p.received('small',10);p.finished('small');p.verified('small');p.received('large',90)
    expect(p.snapshot().downloadComplete).toBe(false)
    expect(downloadPercentage(p.snapshot())).toBe(99.9)
    p.finished('large')
    expect(p.snapshot().downloadComplete).toBe(true)
    expect(p.snapshot().verifiedBytes).toBe(10)
    p.verified('large')
    expect(p.snapshot().verifiedBytes).toBe(100)
  })
})


it('shows small remaining downloads in kB rather than two rounded zero MB amounts', () => {
  expect(downloadByteLabel(0, 16712)).toBe('0.0 kB of 16.7 kB')
  expect(downloadByteLabel(25000000, 100000000)).toBe('25.0 MB of 100.0 MB')
})
