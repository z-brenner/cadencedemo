import { useEffect, useRef, useState } from 'react'
import { validateResult, type DecisionRequest, type Result } from '../contract'

const assetBase = import.meta.env.VITE_CADENCE_ASSET_BASE?.trim() ?? ''
const manifestSha = import.meta.env.VITE_CADENCE_MANIFEST_SHA256?.trim() ?? ''
export const browserModelAvailable = Boolean(assetBase && manifestSha)
export type BrowserModelStatus =
  | 'idle'
  | 'checking'
  | 'downloading'
  | 'verifying'
  | 'loading'
  | 'ready'
  | 'running'
  | 'canceled'
  | 'error'
type Pending = {
  requestId: string
  request: DecisionRequest
  resolve: (result: Result) => void
  reject: (error: Error) => void
}

export function useBrowserModel() {
  const [status, setStatus] = useState<BrowserModelStatus>('idle')
  const [downloadedBytes, setDownloadedBytes] = useState(0)
  const [totalBytes, setTotalBytes] = useState(0)
  const [cachedBytes, setCachedBytes] = useState(0)
  const [checkedFiles, setCheckedFiles] = useState(0)
  const [totalFiles, setTotalFiles] = useState(0)
  const [downloadComplete, setDownloadComplete] = useState(false)
  const [stage, setStage] = useState('')
  const [error, setError] = useState('')
  const [provider, setProvider] = useState<'wasm' | 'webgpu' | null>(null)
  const [fromCache, setFromCache] = useState(false)
  const workerRef = useRef<Worker | null>(null)
  const pendingRef = useRef<Pending | null>(null)
  const statusRef = useRef<BrowserModelStatus>('idle')

  function updateStatus(next: BrowserModelStatus) {
    statusRef.current = next
    setStatus(next)
  }
  function stopWorker(message: string) {
    workerRef.current?.terminate()
    workerRef.current = null
    pendingRef.current?.reject(new Error(message))
    pendingRef.current = null
  }
  useEffect(
    () => () => {
      workerRef.current?.terminate()
      workerRef.current = null
      pendingRef.current?.reject(new Error('Browser model closed.'))
      pendingRef.current = null
    },
    [],
  )

  function prepare() {
    if (
      !browserModelAvailable ||
      ['checking', 'downloading', 'verifying', 'loading', 'running', 'ready'].includes(statusRef.current)
    )
      return
    stopWorker('Model preparation restarted.')
    setError('')
    setStage('Checking model files')
    setProvider(null)
    setFromCache(false)
    setDownloadedBytes(0)
    setTotalBytes(0)
    setCachedBytes(0)
    setCheckedFiles(0)
    setTotalFiles(0)
    setDownloadComplete(false)
    updateStatus('checking')
    try {
      if (!window.isSecureContext || typeof Worker === 'undefined')
        throw new Error(
          'This browser cannot run Cadence here. Use a current browser over HTTPS, or follow the local setup guide.',
        )
      const worker = new Worker(new URL('./model.worker.ts', import.meta.url), { type: 'module' })
      workerRef.current = worker
      worker.onmessage = (event: MessageEvent) => {
        if (workerRef.current !== worker) return
        const message = event.data
        if (message.type === 'status') {
          if (['checking', 'downloading', 'verifying', 'loading', 'running'].includes(message.status))
            updateStatus(message.status)
          if (Number.isFinite(message.downloadedBytes) && message.downloadedBytes >= 0)
            setDownloadedBytes(message.downloadedBytes)
          if (Number.isFinite(message.totalBytes) && message.totalBytes >= 0)
            setTotalBytes(message.totalBytes)
          if (Number.isSafeInteger(message.cachedBytes) && message.cachedBytes >= 0) setCachedBytes(message.cachedBytes)
          if (Number.isSafeInteger(message.checkedFiles) && message.checkedFiles >= 0) setCheckedFiles(message.checkedFiles)
          if (Number.isSafeInteger(message.totalFiles) && message.totalFiles >= 0) setTotalFiles(message.totalFiles)
          if (typeof message.downloadComplete === 'boolean') setDownloadComplete(message.downloadComplete)
          if (typeof message.stage === 'string') setStage(message.stage)
          if (typeof message.fromCache === 'boolean') setFromCache(message.fromCache)
        } else if (message.type === 'ready') {
          if (message.provider !== 'wasm' && message.provider !== 'webgpu') {
            fail('The browser runtime returned an unsupported provider.')
            return
          }
          setProvider(message.provider)
          setFromCache(message.fromCache === true)
          setStage('Cadence is ready')
          updateStatus('ready')
        } else if (message.type === 'result') {
          const pending = pendingRef.current
          if (!pending || message.requestId !== pending.requestId) return
          try {
            const result = validateResult(message.result, pending.request, manifestSha)
            pendingRef.current = null
            updateStatus('ready')
            pending.resolve(result)
          } catch {
            fail(
              'Cadence could not verify this result. No decision was displayed. Retry model preparation.',
            )
          }
        } else if (message.type === 'input-error') {
          const pending = pendingRef.current
          if (!pending || message.requestId !== pending.requestId) return
          pendingRef.current = null
          setStage('Model files ready for revised inputs')
          updateStatus('ready')
          pending.reject(
            new Error(
              typeof message.message === 'string' && message.message.trim()
                ? message.message
                : 'Cadence could not encode these inputs. Edit the context or answers and retry.',
            ),
          )
        } else if (message.type === 'error') {
          if (message.requestId && message.requestId !== pendingRef.current?.requestId) return
          fail(
            typeof message.message === 'string'
              ? message.message
              : 'The browser model could not complete this operation.',
          )
        } else if (message.type === 'cleared') {
          stopWorker('Saved model cleared.')
          setProvider(null)
          setFromCache(false)
          setDownloadedBytes(0)
          setTotalBytes(0)
      setCachedBytes(0)
      setCheckedFiles(0)
      setTotalFiles(0)
      setDownloadComplete(false)
          setStage('Saved model cleared')
          updateStatus('idle')
        }
      }
      worker.onerror = () => {
        if (workerRef.current === worker)
          fail(
            'Cadence stopped in this browser. This device may not have enough memory. Retry or use the local setup guide.',
          )
      }
      worker.postMessage({ type: 'prepare', assetBase, manifestSha })
    } catch (cause) {
      fail((cause as Error).message)
    }
  }
  function fail(message: string) {
    stopWorker(message)
    setError(message)
    setProvider(null)
    updateStatus('error')
  }
  function cancel() {
    stopWorker('Browser operation canceled. Your inputs are unchanged.')
    setProvider(null)
    setError('')
    setStage('Canceled')
    updateStatus('canceled')
  }
  function run(request: DecisionRequest): Promise<Result> {
    if (statusRef.current !== 'ready' || !workerRef.current || pendingRef.current)
      return Promise.reject(new Error('Prepare Cadence on this device before getting a decision.'))
    const requestId = crypto.randomUUID()
    updateStatus('running')
    return new Promise((resolve, reject) => {
      pendingRef.current = { requestId, request, resolve, reject }
      try {
        workerRef.current!.postMessage({ type: 'run', requestId, request })
      } catch (cause) {
        fail((cause as Error).message)
      }
    })
  }
  async function clearCache() {
    stopWorker('Saved model cleared. Your inputs are unchanged.')
    setProvider(null)
    setError('')
    setStage('Clearing saved model')
    updateStatus('checking')
    try {
      if (!('caches' in window))
        throw new Error('Saved model storage is unavailable in this browser.')
      await caches.delete(`cadence-browser-${manifestSha}`)
      setFromCache(false)
      setDownloadedBytes(0)
      setTotalBytes(0)
      setCachedBytes(0)
      setCheckedFiles(0)
      setTotalFiles(0)
      setDownloadComplete(false)
      setStage('Saved model cleared')
      updateStatus('idle')
    } catch (cause) {
      fail((cause as Error).message)
    }
  }
  return {
    status,
    downloadedBytes,
    totalBytes,
    cachedBytes,
    checkedFiles,
    totalFiles,
    downloadComplete,
    stage,
    error,
    provider,
    fromCache,
    prepare,
    cancel,
    run,
    clearCache,
  }
}
