import { Button } from '@/components/ui/button'
import { LOCALES, useI18n } from '@/lib/i18n'
import { cn } from '@/lib/utils'

/** Two locales only, so a segmented control beats a dropdown: the alternative
 *  is visible without opening anything. */
export function LocaleSwitch() {
  const { locale, setLocale } = useI18n()
  return (
    <div className="flex items-center rounded-[var(--radius-control)] border border-border p-px">
      {LOCALES.map((value) => (
        <Button
          key={value}
          variant="ghost"
          size="sm"
          aria-pressed={locale === value}
          onClick={() => setLocale(value)}
          className={cn(
            'h-6 px-2 font-mono text-[11px] uppercase',
            locale === value ? 'bg-raised text-text' : 'text-faint',
          )}
        >
          {value}
        </Button>
      ))}
    </div>
  )
}
