import { createFileRoute } from '@tanstack/react-router'
import { Copy, PlaySolid } from 'iconoir-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Columns, Page, Section, Stack } from '@/components/layout/primitives'
import { Button } from '@/components/ui/button'
import { CodeBlock } from '@/components/ui/code'
import { Field } from '@/components/ui/field'
import { Input, Textarea } from '@/components/ui/input'
import { Meter, MeterRow, RtfScale } from '@/components/ui/meter'
import { Select } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import {
  apiUrl,
  FRAME_RATE,
  listVoices,
  synthesize,
  type SpeechOptions,
  type SpeechResult,
  type Voice,
} from '@/lib/api'
import { useDelayedFlag } from '@/lib/hooks'
import { useI18n } from '@/lib/i18n'
import { useNotifications } from '@/lib/notifications'
import { safeLocal, storeLocal } from '@/lib/utils'

const FORMATS = [
  { value: 'wav', label: 'wav' },
  { value: 'mp3', label: 'mp3' },
  { value: 'pcm', label: 'pcm' },
] as const

const DEFAULT_TEXT = 'Ст+арый з+амок сто+ит д+орого, а дв+ерь закрыв+ает зам+ок.'

function RouteComponent() {
  const { t } = useI18n()
  const { notify } = useNotifications()

  const [text, setText] = useState(() => safeLocal('xvibe.text', DEFAULT_TEXT))
  const [voice, setVoice] = useState('')
  const [format, setFormat] = useState<SpeechOptions['response_format']>('wav')
  const [temperature, setTemperature] = useState('0.5')
  const [eos, setEos] = useState('-4')
  const [seed, setSeed] = useState('0')
  const [firstChunk, setFirstChunk] = useState('2')

  const [voices, setVoices] = useState<Voice[] | null>(null)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<SpeechResult | null>(null)
  const [copied, setCopied] = useState(false)
  const audioUrl = useRef<string | null>(null)
  const [playable, setPlayable] = useState<string | null>(null)

  const voicesPending = useDelayedFlag(voices === null)

  useEffect(() => {
    listVoices()
      .then((data) => {
        setVoices(data)
        setVoice((current) => current || (data[0]?.id ?? ''))
      })
      .catch((error: unknown) => {
        setVoices([])
        notify('error', t('error.generic'), error instanceof Error ? error.message : String(error))
      })
  }, [notify, t])

  useEffect(
    () => () => {
      if (audioUrl.current) URL.revokeObjectURL(audioUrl.current)
    },
    [],
  )

  const options = useCallback(
    (): SpeechOptions => ({
      input: text,
      voice,
      response_format: format,
      temperature: Number(temperature),
      eos_threshold: Number(eos),
      seed: Number(seed) || 0,
      first_chunk_frames: Number(firstChunk) || 2,
    }),
    [text, voice, format, temperature, eos, seed, firstChunk],
  )

  const run = async () => {
    setRunning(true)
    storeLocal('xvibe.text', text)
    try {
      const outcome = await synthesize(options())
      setResult(outcome)
      if (audioUrl.current) URL.revokeObjectURL(audioUrl.current)
      if (outcome.blob) {
        audioUrl.current = URL.createObjectURL(outcome.blob)
        setPlayable(audioUrl.current)
      } else {
        audioUrl.current = null
        setPlayable(null)
        notify('info', t('synth.pcmNote'), `${outcome.bytes.toLocaleString()} B`)
      }
    } catch (error) {
      notify('error', t('error.generic'), error instanceof Error ? error.message : String(error))
    } finally {
      setRunning(false)
    }
  }

  const curl = [
    `curl -N -X POST ${window.location.origin}${apiUrl('v1/audio/speech')} \\`,
    `  -H 'Content-Type: application/json' \\`,
    `  -d '${JSON.stringify(
      {
        input: text,
        voice,
        response_format: format,
        xvibe: {
          temperature: Number(temperature),
          eos_threshold: Number(eos),
          seed: Number(seed) || 0,
          first_chunk_frames: Number(firstChunk) || 2,
        },
      },
      null,
      2,
    ).replace(/'/g, "'\\''")}' \\`,
    `  --output speech.${format === 'pcm' ? 'raw' : format}`,
  ].join('\n')

  const rtf = result && result.seconds > 0 ? result.totalMs / 1000 / result.seconds : null

  return (
    <Page>
      <Section>
        <Stack gap="md">
          <Field label={t('synth.text')} hint={t('synth.textHint')}>
            <Textarea value={text} onChange={(event) => setText(event.target.value)} rows={3} />
          </Field>

          <Columns count={3} gap="sm">
            <Field label={t('synth.voice')}>
              {voicesPending ? (
                <Skeleton className="h-8 w-full" />
              ) : (
                <Select
                  value={voice}
                  onChange={setVoice}
                  aria-label={t('synth.voice')}
                  options={(voices ?? []).map((item) => ({
                    value: item.id,
                    label: `${item.id} · ${item.seconds.toFixed(1)} s`,
                  }))}
                />
              )}
            </Field>
            <Field label={t('synth.format')}>
              <Select
                value={format}
                onChange={(value) => setFormat(value as SpeechOptions['response_format'])}
                options={FORMATS}
                aria-label={t('synth.format')}
              />
            </Field>
            <Field label={t('synth.temperature')}>
              <Input
                type="number"
                step="0.05"
                value={temperature}
                onChange={(event) => setTemperature(event.target.value)}
              />
            </Field>
            <Field label={t('synth.eos')}>
              <Input
                type="number"
                step="0.5"
                value={eos}
                onChange={(event) => setEos(event.target.value)}
              />
            </Field>
            <Field label={t('synth.seed')}>
              <Input
                type="number"
                value={seed}
                onChange={(event) => setSeed(event.target.value)}
              />
            </Field>
            <Field label={t('synth.firstChunk')}>
              <Input
                type="number"
                min="1"
                value={firstChunk}
                onChange={(event) => setFirstChunk(event.target.value)}
              />
            </Field>
          </Columns>

          <Stack direction="row" gap="sm" align="center">
            <Button variant="primary" onClick={run} loading={running} disabled={!voice}>
              {!running ? <PlaySolid width={13} height={13} /> : null}
              {running ? t('synth.running') : t('synth.run')}
            </Button>
          </Stack>

          <MeterRow>
            <Meter
              label={t('metric.ttfb')}
              value={result ? `${Math.round(result.ttfbMs)} ms` : '—'}
              sub={t('metric.ttfbSub')}
              loading={running}
            />
            <Meter
              label={t('metric.total')}
              value={result ? `${Math.round(result.totalMs)} ms` : '—'}
              sub={t('metric.totalSub')}
              loading={running}
            />
            <Meter
              label={t('metric.audio')}
              value={result ? `${result.seconds.toFixed(2)} s` : '—'}
              sub={result ? `${result.frames} ${t('metric.frames')}` : `${FRAME_RATE} fps`}
              loading={running}
            />
            <Meter
              label={t('metric.rtf')}
              value={rtf == null ? '—' : rtf.toFixed(3)}
              sub={rtf == null ? t('metric.realtime') : `${(1 / rtf).toFixed(2)}× realtime`}
              loading={running}
            />
          </MeterRow>
          <RtfScale rtf={rtf} />

          {playable ? (
            <audio src={playable} controls className="w-full" autoPlay />
          ) : null}
        </Stack>
      </Section>

      <Section
        title={t('synth.request')}
        actions={
          <Button
            size="sm"
            onClick={() => {
              void navigator.clipboard.writeText(curl).then(
                () => {
                  setCopied(true)
                  window.setTimeout(() => setCopied(false), 1400)
                },
                () => notify('error', t('error.generic')),
              )
            }}
          >
            <Copy width={13} height={13} />
            {copied ? t('synth.copied') : t('synth.copy')}
          </Button>
        }
      >
        <CodeBlock>{curl}</CodeBlock>
      </Section>
    </Page>
  )
}

export const Route = createFileRoute('/')({
  component: RouteComponent,
  staticData: { header: { titleKey: 'synth.title', descriptionKey: 'synth.description' } },
})
