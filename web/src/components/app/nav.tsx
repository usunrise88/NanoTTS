import { Link } from '@tanstack/react-router'
import { useI18n } from '@/lib/i18n'
import { cn } from '@/lib/utils'

const ITEMS = [
  { to: '/', key: 'nav.synth' },
  { to: '/voices', key: 'nav.voices' },
  { to: '/models', key: 'nav.models' },
  { to: '/monitor', key: 'nav.monitor' },
] as const

export function Nav() {
  const { t } = useI18n()
  return (
    <nav className="flex items-center gap-0.5">
      {ITEMS.map((item) => (
        <Link
          key={item.to}
          to={item.to}
          activeOptions={{ exact: item.to === '/' }}
          className={cn(
            'rounded-[var(--radius-control)] px-2.5 py-1 text-[13px] text-muted hover:bg-raised',
          )}
          activeProps={{ className: 'bg-raised text-text font-medium' }}
        >
          {t(item.key)}
        </Link>
      ))}
    </nav>
  )
}
