import { safeLocal } from './utils'

/** The bundle is served either at / or behind a proxy at /asr/, so the API base
 *  is derived from the document's own path rather than hardcoded. */
export const apiBase = () => window.location.pathname.replace(/[^/]*$/, '')

export const apiUrl = (path: string) => apiBase() + path

export const KEY_STORAGE = 'nanotts.key'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

function authHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra)
  const key = safeLocal(KEY_STORAGE, '').trim()
  if (key) headers.set('Authorization', `Bearer ${key}`)
  return headers
}

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(apiUrl(path), { ...init, headers: authHeaders(init?.headers) })
  if (!response.ok) {
    let message = `HTTP ${response.status}`
    try {
      const body = (await response.json()) as { error?: { message?: string } }
      message = body.error?.message ?? message
    } catch {
      /* not a JSON error body */
    }
    throw new ApiError(message, response.status)
  }
  return response
}

export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  return (await apiFetch(path, init)).json() as Promise<T>
}

/* ---------------------------------------------------------------- types */

export interface Voice {
  id: string
  prefix_frames: number
  seconds: number
  bytes: number
  created: string
}

export interface WorkerStat {
  index: number
  node: number
  busy: boolean
  served: number
}

export interface Percentiles {
  p50: number
  p95: number
  p99: number
}

export interface Stats {
  requests: number
  errors: number
  queued: number
  workers: WorkerStat[]
  ttfb_ms: Percentiles
  total_ms: Percentiles
  audio_seconds: number
  rtf: number
}

export interface SpeechOptions {
  input: string
  voice: string
  response_format: 'wav' | 'mp3' | 'pcm'
  temperature: number
  eos_threshold: number
  seed: number
  first_chunk_frames: number
}

export interface SpeechResult {
  blob: Blob | null
  bytes: number
  ttfbMs: number
  totalMs: number
  seconds: number
  frames: number
}

export const SAMPLE_RATE = 24_000
export const FRAME_RATE = 12.5

export const listVoices = () => apiJson<{ data: Voice[] }>('v1/voices').then((r) => r.data)
export const fetchStats = () => apiJson<Stats>('stats')
export const health = () => apiJson<{ status: string }>('healthz')

export async function warmVoice(id: string, file: File) {
  const form = new FormData()
  form.append('id', id)
  form.append('file', file)
  return apiJson<{
    id: string
    prefix_frames: number
    seconds: number
    source_sample_rate: number
    warm_ms: number
  }>('v1/voices', { method: 'POST', body: form })
}

export const deleteVoice = (id: string) =>
  apiFetch(`v1/voices/${encodeURIComponent(id)}`, { method: 'DELETE' })

export const voiceStateUrl = (id: string) => apiUrl(`v1/voices/${encodeURIComponent(id)}/state`)

/** Streams the response so the measured latency is the first audio frame, not
 *  the last byte. Buffering the whole body would report a one-second wait for
 *  what is really a ~170 ms time to first sound. */
export async function synthesize(options: SpeechOptions): Promise<SpeechResult> {
  const { input, voice, response_format, ...rest } = options
  const started = performance.now()
  const response = await apiFetch('v1/audio/speech', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input, voice, response_format, nanotts: rest }),
  })

  const reader = response.body?.getReader()
  if (!reader) throw new Error('streaming is not supported by this browser')

  const chunks: Uint8Array[] = []
  let ttfbMs = 0
  let bytes = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (!ttfbMs) ttfbMs = performance.now() - started
    chunks.push(value)
    bytes += value.length
  }
  const totalMs = performance.now() - started

  let seconds = 0
  if (response_format === 'pcm') seconds = bytes / 4 / SAMPLE_RATE
  else if (response_format === 'wav') seconds = Math.max(0, (bytes - 44) / 2 / SAMPLE_RATE)

  const blob =
    response_format === 'pcm'
      ? null
      : new Blob(chunks as BlobPart[], {
          type: response_format === 'mp3' ? 'audio/mpeg' : 'audio/wav',
        })

  return { blob, bytes, ttfbMs, totalMs, seconds, frames: Math.round(seconds * FRAME_RATE) }
}
