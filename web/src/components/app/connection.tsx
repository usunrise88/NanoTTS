import { Key } from 'iconoir-react'
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Field } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Popover } from '@/components/ui/popover'
import { ApiError, health, KEY_STORAGE } from '@/lib/api'
import { useI18n } from '@/lib/i18n'
import { safeLocal, storeLocal, cn } from '@/lib/utils'

type Status = 'checking' | 'online' | 'offline' | 'auth'

const DOT: Record<Status, string> = {
  checking: 'bg-faint',
  online: 'bg-accent',
  offline: 'bg-danger',
  auth: 'bg-warn-text',
}

/** Connection state plus the bearer key, in the least blocking surface that
 *  fits: a popover in the bar rather than a settings page or a startup modal. */
export function Connection() {
  const { t } = useI18n()
  const [status, setStatus] = useState<Status>('checking')
  const [key, setKey] = useState(() => safeLocal(KEY_STORAGE, ''))

  useEffect(() => {
    let alive = true
    health()
      .then(() => alive && setStatus('online'))
      .catch((error: unknown) => {
        if (!alive) return
        setStatus(error instanceof ApiError && error.status === 401 ? 'auth' : 'offline')
      })
    return () => {
      alive = false
    }
  }, [])

  const label = t(`conn.${status === 'checking' ? 'checking' : status === 'online' ? 'online' : status === 'auth' ? 'auth' : 'offline'}` as const)

  return (
    <Popover
      trigger={
        <Button variant="ghost" size="sm" className="gap-2 px-2">
          <span className={cn('size-1.5 shrink-0 rounded-full', DOT[status])} />
          <span className="font-mono text-[11px] text-muted">{label}</span>
          <Key width={13} height={13} className="text-faint" />
        </Button>
      }
      className="w-[min(22rem,calc(100vw-2rem))] p-3"
    >
      <Field label={t('key.label')} hint={t('key.hint')}>
        <Input
          type="password"
          value={key}
          placeholder={t('key.placeholder')}
          autoComplete="off"
          onChange={(event) => setKey(event.target.value)}
        />
      </Field>
      <div className="mt-3 flex justify-end">
        <Button
          variant="primary"
          size="sm"
          onClick={() => {
            storeLocal(KEY_STORAGE, key.trim())
            setStatus('checking')
            health()
              .then(() => setStatus('online'))
              .catch((error: unknown) =>
                setStatus(error instanceof ApiError && error.status === 401 ? 'auth' : 'offline'),
              )
          }}
        >
          {t('key.save')}
        </Button>
      </div>
    </Popover>
  )
}
