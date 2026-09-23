import { createContext, use, useCallback, useMemo, useState, type ReactNode } from 'react'
import { toast as sonner } from 'sonner'

export type NoticeLevel = 'success' | 'error' | 'info'

export interface Notice {
  id: string
  level: NoticeLevel
  title: string
  detail?: string
  at: Date
}

interface NotificationValue {
  notices: Notice[]
  notify: (level: NoticeLevel, title: string, detail?: string) => void
  clear: () => void
}

const NotificationContext = createContext<NotificationValue | null>(null)

const LIMIT = 50

/** Toasts are for after-the-fact acknowledgement and they disappear, so every
 *  one is also recorded here. The history dropdown is what makes a toast an
 *  acceptable surface for anything the operator might want to read twice. */
export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notices, setNotices] = useState<Notice[]>([])

  const notify = useCallback((level: NoticeLevel, title: string, detail?: string) => {
    const notice: Notice = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      level,
      title,
      at: new Date(),
      ...(detail ? { detail } : {}),
    }
    setNotices((prev) => [notice, ...prev].slice(0, LIMIT))
    const options = detail ? { description: detail } : undefined
    if (level === 'success') sonner.success(title, options)
    else if (level === 'error') sonner.error(title, options)
    else sonner(title, options)
  }, [])

  const clear = useCallback(() => setNotices([]), [])

  const value = useMemo(() => ({ notices, notify, clear }), [notices, notify, clear])
  return <NotificationContext value={value}>{children}</NotificationContext>
}

export function useNotifications(): NotificationValue {
  const value = use(NotificationContext)
  if (!value) throw new Error('useNotifications must be used inside NotificationProvider')
  return value
}
