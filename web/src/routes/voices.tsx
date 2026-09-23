import { createFileRoute } from '@tanstack/react-router'
import { Download, Trash } from 'iconoir-react'
import { useCallback, useEffect, useState } from 'react'
import { Columns, Page, Section, Stack } from '@/components/layout/primitives'
import { Button } from '@/components/ui/button'
import { Dialog } from '@/components/ui/dialog'
import { Field } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { LinkButton } from '@/components/ui/link-button'
import { SkeletonRows } from '@/components/ui/skeleton'
import { Table, TableEmpty, Td, Th } from '@/components/ui/table'
import { deleteVoice, listVoices, voiceStateUrl, warmVoice, type Voice } from '@/lib/api'
import { useDelayedFlag } from '@/lib/hooks'
import { useI18n } from '@/lib/i18n'
import { useNotifications } from '@/lib/notifications'
import { formatBytes } from '@/lib/utils'

const ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/

function RouteComponent() {
  const { t, locale } = useI18n()
  const { notify } = useNotifications()

  const [voices, setVoices] = useState<Voice[] | null>(null)
  const [id, setId] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [warming, setWarming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pendingDelete, setPendingDelete] = useState<Voice | null>(null)
  const [deleting, setDeleting] = useState(false)

  const pending = useDelayedFlag(voices === null)

  const reload = useCallback(() => {
    listVoices()
      .then(setVoices)
      .catch((e: unknown) => {
        setVoices([])
        notify('error', t('error.generic'), e instanceof Error ? e.message : String(e))
      })
  }, [notify, t])

  useEffect(reload, [reload])

  const warm = async () => {
    setError(null)
    if (!ID_PATTERN.test(id.trim())) return setError(t('voices.badId'))
    if (!file) return setError(t('voices.noFile'))
    setWarming(true)
    try {
      const result = await warmVoice(id.trim(), file)
      notify(
        'success',
        t('voices.warmed'),
        `${result.prefix_frames} × 80 ms · ${result.seconds.toFixed(2)} s · ${Math.round(result.warm_ms)} ms`,
      )
      setId('')
      setFile(null)
      reload()
    } catch (e) {
      notify('error', t('error.generic'), e instanceof Error ? e.message : String(e))
    } finally {
      setWarming(false)
    }
  }

  const confirmDelete = async () => {
    if (!pendingDelete) return
    setDeleting(true)
    try {
      await deleteVoice(pendingDelete.id)
      notify('success', t('voices.delete'), pendingDelete.id)
      setPendingDelete(null)
      reload()
    } catch (e) {
      notify('error', t('error.generic'), e instanceof Error ? e.message : String(e))
    } finally {
      setDeleting(false)
    }
  }

  return (
    <Page>
      <Section title={t('voices.warmTitle')} description={t('voices.warmHint')}>
        <Stack gap="md">
          <Columns count={2} gap="sm">
            <Field label={t('voices.id')} error={error ?? undefined}>
              <Input
                value={id}
                placeholder="my_voice"
                onChange={(event) => setId(event.target.value)}
              />
            </Field>
            <Field label={t('voices.file')}>
              <Input
                type="file"
                accept="audio/wav,.wav"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </Field>
          </Columns>
          <Stack direction="row">
            <Button variant="primary" onClick={warm} loading={warming}>
              {warming ? t('voices.warming') : t('voices.warm')}
            </Button>
          </Stack>
        </Stack>
      </Section>

      <Section title={t('voices.title')}>
        {pending ? (
          <SkeletonRows rows={3} cols={5} />
        ) : (
          <Table>
            <thead>
              <tr>
                <Th>ID</Th>
                <Th numeric>{t('voices.prefix')}</Th>
                <Th numeric>{t('voices.size')}</Th>
                <Th>{t('voices.created')}</Th>
                <Th />
              </tr>
            </thead>
            <tbody>
              {(voices ?? []).length === 0 ? (
                <TableEmpty colSpan={5}>{t('voices.empty')}</TableEmpty>
              ) : (
                (voices ?? []).map((item) => (
                  <tr key={item.id}>
                    <Td className="font-medium">{item.id}</Td>
                    <Td numeric>
                      {item.prefix_frames} · {item.seconds.toFixed(2)} s
                    </Td>
                    <Td numeric>{formatBytes(item.bytes)}</Td>
                    <Td className="tabular font-mono text-[12px] text-muted">
                      {new Date(item.created).toLocaleString(locale === 'ru' ? 'ru-RU' : 'en-GB')}
                    </Td>
                    <Td className="text-right whitespace-nowrap">
                      <Stack direction="row" gap="xs" justify="end">
                        <LinkButton
                          href={voiceStateUrl(item.id)}
                          download={`${item.id}.safetensors`}
                        >
                          <Download width={12} height={12} />
                          {t('voices.download')}
                        </LinkButton>
                        <Button
                          size="sm"
                          variant="ghost"
                          aria-label={t('voices.delete')}
                          onClick={() => setPendingDelete(item)}
                          className="text-danger-text"
                        >
                          <Trash width={12} height={12} />
                        </Button>
                      </Stack>
                    </Td>
                  </tr>
                ))
              )}
            </tbody>
          </Table>
        )}
      </Section>

      {/* Destructive and irreversible: one of the few cases that earns a centre
          modal rather than a sheet or an inline confirm. */}
      <Dialog
        open={pendingDelete !== null}
        onOpenChange={(open) => !open && setPendingDelete(null)}
        title={t('voices.deleteTitle')}
        description={`${pendingDelete?.id ?? ''} — ${t('voices.deleteBody')}`}
        footer={
          <>
            <Button onClick={() => setPendingDelete(null)}>{t('voices.cancel')}</Button>
            <Button variant="danger" onClick={confirmDelete} loading={deleting}>
              {t('voices.delete')}
            </Button>
          </>
        }
      />
    </Page>
  )
}

export const Route = createFileRoute('/voices')({
  component: RouteComponent,
  staticData: { header: { titleKey: 'voices.title', descriptionKey: 'voices.description' } },
})
