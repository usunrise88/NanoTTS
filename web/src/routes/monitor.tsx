import { createFileRoute } from '@tanstack/react-router'
import { useCallback, useEffect, useState } from 'react'
import { Columns, Page, Section, Stack } from '@/components/layout/primitives'
import { Badge } from '@/components/ui/badge'
import { SkeletonRows } from '@/components/ui/skeleton'
import { Table, Td, Th } from '@/components/ui/table'
import { fetchStats, type Stats } from '@/lib/api'
import { useDelayedFlag, useInterval } from '@/lib/hooks'
import { useI18n } from '@/lib/i18n'

function RouteComponent() {
  const { t } = useI18n()
  const [stats, setStats] = useState<Stats | null>(null)
  const pending = useDelayedFlag(stats === null)

  const load = useCallback(() => {
    fetchStats()
      .then(setStats)
      .catch(() => {
        /* the connection indicator in the shell already reports this */
      })
  }, [])

  useEffect(load, [load])
  // Refreshed rather than loaded: once the table is on screen it updates in
  // place, so no skeleton ever appears again for this route.
  useInterval(load, 3000)

  return (
    <Page>
      <Columns count={2}>
        <Stack gap="md">
          <Section title={t('monitor.workers')}>
            {pending ? (
              <SkeletonRows rows={4} cols={4} />
            ) : (
              <Table>
                <thead>
                  <tr>
                    <Th>#</Th>
                    <Th numeric>{t('monitor.node')}</Th>
                    <Th>{t('monitor.status')}</Th>
                    <Th numeric>{t('monitor.served')}</Th>
                  </tr>
                </thead>
                <tbody>
                  {(stats?.workers ?? []).map((worker) => (
                    <tr key={worker.index}>
                      <Td>{worker.index}</Td>
                      <Td numeric>{worker.node}</Td>
                      <Td>
                        <Badge tone={worker.busy ? 'accent' : 'neutral'}>
                          {worker.busy ? t('monitor.busy') : t('monitor.idle')}
                        </Badge>
                      </Td>
                      <Td numeric>{worker.served}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Section>

          <Section title={t('monitor.latency')}>
            {pending ? (
              <SkeletonRows rows={2} cols={4} />
            ) : (
              <Table>
                <thead>
                  <tr>
                    <Th />
                    <Th numeric>p50</Th>
                    <Th numeric>p95</Th>
                    <Th numeric>p99</Th>
                  </tr>
                </thead>
                <tbody>
                  {(
                    [
                      ['TTFB', stats?.ttfb_ms],
                      [t('monitor.request'), stats?.total_ms],
                    ] as const
                  ).map(([label, value]) => (
                    <tr key={label}>
                      <Td>{label}</Td>
                      <Td numeric>{value ? Math.round(value.p50) : '—'}</Td>
                      <Td numeric>{value ? Math.round(value.p95) : '—'}</Td>
                      <Td numeric>{value ? Math.round(value.p99) : '—'}</Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Section>
        </Stack>

        <Section title={t('monitor.counters')}>
          {pending ? (
            <SkeletonRows rows={5} cols={2} />
          ) : (
            <Table>
              <tbody>
                {(
                  [
                    [t('monitor.requests'), stats?.requests ?? 0],
                    [t('monitor.errors'), stats?.errors ?? 0],
                    [t('monitor.queued'), stats?.queued ?? 0],
                    [t('monitor.synthesized'), `${(stats?.audio_seconds ?? 0).toFixed(1)} s`],
                    [t('monitor.rtfTotal'), stats?.rtf ? stats.rtf.toFixed(3) : '—'],
                  ] as const
                ).map(([label, value]) => (
                  <tr key={label}>
                    <Td className="text-muted">{label}</Td>
                    <Td numeric>{value}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Section>
      </Columns>
    </Page>
  )
}

export const Route = createFileRoute('/monitor')({
  component: RouteComponent,
  staticData: { header: { titleKey: 'monitor.title', descriptionKey: 'monitor.description' } },
})
