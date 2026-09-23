import { HalfMoon, Computer, SunLight } from 'iconoir-react'
import { Button } from '@/components/ui/button'
import { Popover } from '@/components/ui/popover'
import { useI18n } from '@/lib/i18n'
import { useTheme, type Theme } from '@/lib/theme'
import { cn } from '@/lib/utils'

const ICONS = { light: SunLight, dark: HalfMoon, system: Computer } as const

export function ThemeToggle() {
  const { theme, resolved, setTheme } = useTheme()
  const { t } = useI18n()
  const Current = resolved === 'dark' ? HalfMoon : SunLight

  const options: { value: Theme; label: string }[] = [
    { value: 'light', label: t('theme.light') },
    { value: 'dark', label: t('theme.dark') },
    { value: 'system', label: t('theme.system') },
  ]

  return (
    <Popover
      trigger={
        <Button variant="ghost" size="icon" aria-label={t('theme.system')}>
          <Current width={15} height={15} />
        </Button>
      }
      className="p-1"
    >
      <div className="flex w-36 flex-col">
        {options.map((option) => {
          const Icon = ICONS[option.value]
          return (
            <button
              key={option.value}
              type="button"
              onClick={() => setTheme(option.value)}
              className={cn(
                'flex items-center gap-2 rounded-[var(--radius-control)] px-2 py-1.5',
                'text-left text-[13px] hover:bg-raised',
                theme === option.value ? 'text-text' : 'text-muted',
              )}
            >
              <Icon width={14} height={14} />
              {option.label}
            </button>
          )
        })}
      </div>
    </Popover>
  )
}
