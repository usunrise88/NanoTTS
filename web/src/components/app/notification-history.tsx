import { Bell, CheckCircle, InfoCircle, WarningCircle } from 'iconoir-react'
import { Button } from '@/components/ui/button'
import { Popover } from '@/components/ui/popover'
import { useI18n } from '@/lib/i18n'
import { useNotifications, type NoticeLevel } from '@/lib/notifications'
import { cn } from '@/lib/utils'

const ICON: Record<NoticeLevel, typeof Bell> = {
  success: CheckCircle,
  error: WarningCircle,
  info: InfoCircle,
}

const TONE: Record<NoticeLevel, string> = {
  success: 'text-accent-text',
  error: 'text-danger-text',
  info: 'text-muted',
}

/** Toasts vanish; this is where they stay. Without it a toast could not honestly
 *  be the surface for anything the operator might need to re-read. */
export function NotificationHistory() {
  const { notices, clear } = useNotifications()
  const { t, locale } = useI18n()

  return (
    <Popover
      trigger={
        <Button variant="ghost" size="icon" aria-label={t('notif.title')} className="relative">
          <Bell width={15} height={15} />
          {notices.length > 0 ? (
            <span className="absolute top-1.5 right-1.5 size-1.5 rounded-full bg-accent" />
          ) : null}
        </Button>
      }
      className="w-[min(22rem,calc(100vw-2rem))]"
    >
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <span className="text-[11px] font-medium tracking-wide text-faint uppercase">
          {t('notif.title')}
        </span>
        {notices.length > 0 ? (
          <Button variant="ghost" size="sm" onClick={clear} className="h-6 px-1.5 text-[11px]">
            {t('notif.clear')}
          </Button>
        ) : null}
      </div>

      {notices.length === 0 ? (
        <p className="px-3 py-6 text-center text-[13px] text-faint">{t('notif.empty')}</p>
      ) : (
        <ul className="max-h-80 overflow-y-auto py-1">
          {notices.map((notice) => {
            const Icon = ICON[notice.level]
            return (
              <li key={notice.id} className="flex gap-2.5 px-3 py-2">
                <Icon
                  width={14}
                  height={14}
                  className={cn('mt-0.5 shrink-0', TONE[notice.level])}
                />
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] text-text">{notice.title}</p>
                  {notice.detail ? (
                    <p className="mt-0.5 text-xs break-words text-muted">{notice.detail}</p>
                  ) : null}
                </div>
                <time className="tabular shrink-0 font-mono text-[11px] text-faint">
                  {notice.at.toLocaleTimeString(locale === 'ru' ? 'ru-RU' : 'en-GB', {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </time>
              </li>
            )
          })}
        </ul>
      )}
    </Popover>
  )
}
