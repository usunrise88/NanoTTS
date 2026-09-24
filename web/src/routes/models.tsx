import { createFileRoute } from '@tanstack/react-router'
import { CloudDownload, Play, Trash } from 'iconoir-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Page, Section, Stack } from '@/components/layout/primitives'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ProgressRow } from '@/components/ui/progress-row'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableEmpty, Td, Th } from '@/components/ui/table'
import {
  activateModel,
  cancelInstall,
  fetchRegistry,
  installModel,
  removeModel,
  type Registry,
} from '@/lib/api'
import { useDelayedFlag } from '@/lib/hooks'
import { useI18n } from '@/lib/i18n'
import { useModel } from '@/lib/model'
import { useNotifications } from '@/lib/notifications'

const mb = (bytes: number) =>
  bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`

function RouteComponent() {
  const { t } = useI18n()
  const { notify } = useNotifications()
  const { refresh: refreshModel } = useModel()

  const [registry, setRegistry] = useState<Registry | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const pending = useDelayedFlag(registry === null)
  const timer = useRef<number | null>(null)

  const load = useCallback(async () => {
    try {
      setRegistry(await fetchRegistry())
    } catch (error) {
      setRegistry((r) => r ?? ({ enabled: false } as Registry))
      if (registry === null)
        notify('error', t('error.generic'), error instanceof Error ? error.message : String(error))
    }
  }, [notify, t, registry])

  // Polled only while something is downloading. A registry that is idle does
  // not change on its own, and a console that keeps asking anyway is a console
  // that shows up in the server's request count for no reason.
  useEffect(() => {
    void load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const active = registry?.jobs?.some((j) => j.state === 'running')
    if (!active) {
      if (timer.current) window.clearTimeout(timer.current)
      return
    }
    timer.current = window.setTimeout(() => void load(), 1200)
    return () => {
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [registry, load])

  const run = async (label: string, id: string, action: () => Promise<unknown>) => {
    setBusy(id)
    try {
      await action()
      await load()
      await refreshModel()
      notify('success', label)
    } catch (error) {
      notify('error', t('error.generic'), error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(null)
    }
  }

  if (pending) return <Page><Skeleton className="h-40 w-full" /></Page>
  if (registry && !registry.enabled)
    return (
      <Page>
        <Section title={t('models.disabled.title')}>
          <p className="text-[13px] text-muted">{t('models.disabled.body')}</p>
        </Section>
      </Page>
    )

  const jobs = registry?.jobs ?? []
  const installed = registry?.installed ?? []
  const available = registry?.available ?? []

  return (
    <Page>
      {jobs.length ? (
        <Section title={t('models.jobs')}>
          <Stack gap="sm">
            {jobs.slice(0, 4).map((job) => (
              <ProgressRow
                key={job.id}
                title={job.model}
                state={job.state}
                stateLabel={t(`models.job.${job.state}` as never)}
                detail={[
                  job.stage,
                  job.file,
                  job.steps ? `${job.step}/${job.steps}` : '',
                  job.total ? `${Math.round((job.done / job.total) * 100)}%` : '',
                ]
                  .filter(Boolean)
                  .join(' · ')}
                fraction={
                  job.total
                    ? job.done / job.total
                    : job.steps
                      ? job.step / job.steps
                      : null
                }
                error={job.error || undefined}
                cancelLabel={t('models.cancel')}
                onCancel={() =>
                  void run(t('models.cancelled'), job.id, () => cancelInstall(job.id))
                }
              />
            ))}
          </Stack>
        </Section>
      ) : null}

      <Section title={t('models.installed')}>
        <Table>
          <thead>
            <tr>
              <Th>{t('models.col.model')}</Th>
              <Th>{t('models.col.arch')}</Th>
              <Th>{t('models.col.rate')}</Th>
              <Th>{t('models.col.voices')}</Th>
              <Th>{t('models.col.size')}</Th>
              <Th />
            </tr>
          </thead>
          <tbody>
            {installed.map((m) => (
              <tr key={m.id}>
                <Td>
                  <Stack direction="row" gap="sm" align="center">
                    <span className="font-medium text-text">{m.id}</span>
                    {m.active ? <Badge tone="accent">{t('models.active')}</Badge> : null}
                  </Stack>
                </Td>
                <Td>
                  <span className="text-muted">{m.architecture}</span>
                </Td>
                <Td>{(m.sample_rate / 1000).toFixed(1)} kHz</Td>
                <Td>{m.voices}</Td>
                <Td>{mb(m.bytes)}</Td>
                <Td>
                  <Stack direction="row" gap="sm" align="center">
                    {m.active ? null : (
                      <Button
                        size="sm"
                        variant="primary"
                        loading={busy === m.id}
                        onClick={() =>
                          void run(t('models.switched'), m.id, async () => {
                            const r = await activateModel(m.id)
                            return r
                          })
                        }
                      >
                        <Play width={13} height={13} />
                        {t('models.activate')}
                      </Button>
                    )}
                    <Button
                      size="sm"
                      disabled={m.active || busy === m.id}
                      onClick={() => void run(t('models.removed'), m.id, () => removeModel(m.id))}
                    >
                      <Trash width={13} height={13} />
                    </Button>
                  </Stack>
                </Td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Section>

      <Section title={t('models.available')} description={t('models.available.hint')}>
        <Table>
          <thead>
            <tr>
              <Th>{t('models.col.model')}</Th>
              <Th>{t('models.col.langs')}</Th>
              <Th>{t('models.col.arch')}</Th>
              <Th>{t('models.col.size')}</Th>
              <Th />
            </tr>
          </thead>
          <tbody>
            {available.map((m) => (
              <tr key={m.id}>
                <Td>
                  <Stack direction="row" gap="sm" align="center">
                    <span className="font-medium text-text">{m.title}</span>
                    <span className="text-[12px] text-faint">{m.repo}</span>
                  </Stack>
                </Td>
                <Td>{m.languages}</Td>
                <Td>
                  <span className="text-muted">{m.architecture}</span>
                </Td>
                <Td>~{mb(m.approx_bytes)}</Td>
                <Td>
                  <Button
                    size="sm"
                    loading={busy === m.id}
                    onClick={() => void run(t('models.installing'), m.id, () => installModel(m.id))}
                  >
                    <CloudDownload width={13} height={13} />
                    {t('models.install')}
                  </Button>
                </Td>
              </tr>
            ))}
            {available.length === 0 ? (
              <TableEmpty colSpan={5}>{t('models.allInstalled')}</TableEmpty>
            ) : null}
          </tbody>
        </Table>
      </Section>
    </Page>
  )
}

export const Route = createFileRoute('/models')({
  component: RouteComponent,
  staticData: { header: { titleKey: 'models.title', descriptionKey: 'models.description' } },
})
