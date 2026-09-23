import { createRootRoute, Outlet, useMatches } from '@tanstack/react-router'
import { Connection } from '@/components/app/connection'
import { LocaleSwitch } from '@/components/app/locale-switch'
import { Nav } from '@/components/app/nav'
import { NotificationHistory } from '@/components/app/notification-history'
import { ThemeToggle } from '@/components/app/theme-toggle'
import { Toaster } from '@/components/ui/toaster'
import { I18nProvider, useI18n, type MessageKey } from '@/lib/i18n'
import { NotificationProvider } from '@/lib/notifications'
import { ThemeProvider } from '@/lib/theme'

/** Page headings live in the route's static data, not in the page body. The
 *  shell renders them, so every page gets identical heading typography and
 *  spacing whether or not its author thought about it. */
declare module '@tanstack/react-router' {
  interface StaticDataRouteOption {
    header?: { titleKey: MessageKey; descriptionKey?: MessageKey }
  }
}

function PageHeader() {
  const { t } = useI18n()
  const matches = useMatches()
  const header = [...matches].reverse().find((m) => m.staticData.header)?.staticData.header
  if (!header) return null
  return (
    <header className="mb-5">
      <h1 className="text-lg font-semibold tracking-tight text-text">{t(header.titleKey)}</h1>
      {header.descriptionKey ? (
        <p className="mt-0.5 text-[13px] text-muted">{t(header.descriptionKey)}</p>
      ) : null}
    </header>
  )
}

function Shell() {
  const { t } = useI18n()
  return (
    <div className="flex h-full flex-col">
      <div className="sticky top-0 z-30 border-b border-border bg-bg">
        <div className="mx-auto flex w-full max-w-[1120px] flex-wrap items-center gap-3 px-4 py-2.5">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-semibold tracking-tight">xVibeTTS</span>
            <span className="hidden text-xs text-faint sm:inline">{t('app.subtitle')}</span>
          </div>
          <span className="mx-1 hidden h-4 w-px bg-border sm:block" />
          <Nav />
          <div className="ml-auto flex items-center gap-1.5">
            <Connection />
            <NotificationHistory />
            <LocaleSwitch />
            <ThemeToggle />
          </div>
        </div>
      </div>

      <main className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[1120px] px-4 py-6">
          <PageHeader />
          <Outlet />
        </div>
      </main>
      <Toaster />
    </div>
  )
}

export const Route = createRootRoute({
  component: () => (
    <ThemeProvider>
      <I18nProvider>
        <NotificationProvider>
          <Shell />
        </NotificationProvider>
      </I18nProvider>
    </ThemeProvider>
  ),
})
