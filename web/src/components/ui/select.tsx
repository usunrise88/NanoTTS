import { Select as BaseSelect } from '@base-ui-components/react/select'
import { Check, NavArrowDown } from 'iconoir-react'
import { cn } from '@/lib/utils'

export interface SelectOption {
  value: string
  label: string
}

/** Wraps Base UI so every select in the app shares one trigger shape, one popup
 *  surface and one indicator. Pages never touch the parts directly. */
export function Select({
  value,
  onChange,
  options,
  className,
  'aria-label': ariaLabel,
}: {
  value: string
  onChange: (value: string) => void
  options: readonly SelectOption[]
  className?: string
  'aria-label'?: string
}) {
  return (
    <BaseSelect.Root
      value={value}
      onValueChange={(next) => onChange(String(next))}
      items={options as SelectOption[]}
    >
      <BaseSelect.Trigger
        aria-label={ariaLabel}
        className={cn(
          'flex h-8 w-full items-center justify-between gap-2 rounded-[var(--radius-control)]',
          'border border-border bg-surface px-2.5 text-[13px] text-text',
          'hover:border-border-strong data-[popup-open]:border-border-strong',
          className,
        )}
      >
        <BaseSelect.Value className="truncate" />
        <BaseSelect.Icon className="shrink-0 text-faint">
          <NavArrowDown width={14} height={14} />
        </BaseSelect.Icon>
      </BaseSelect.Trigger>
      <BaseSelect.Portal>
        <BaseSelect.Positioner sideOffset={4} alignItemWithTrigger={false} className="z-50">
          <BaseSelect.Popup
            className={cn(
              'max-h-72 min-w-[var(--anchor-width)] overflow-y-auto rounded-[var(--radius-panel)]',
              'border border-border bg-surface p-1 shadow-lg shadow-black/10',
              'data-[starting-style]:opacity-0 data-[ending-style]:opacity-0',
            )}
            style={{ transition: 'opacity 120ms ease' }}
          >
            <BaseSelect.List>
              {options.map((option) => (
                <BaseSelect.Item
                  key={option.value}
                  value={option.value}
                  className={cn(
                    'flex cursor-default items-center gap-2 rounded-[var(--radius-control)]',
                    'py-1.5 pr-2 pl-1.5 text-[13px] text-text outline-none',
                    'data-[highlighted]:bg-raised',
                  )}
                >
                  <span className="flex size-3.5 shrink-0 items-center justify-center text-accent-text">
                    <BaseSelect.ItemIndicator>
                      <Check width={13} height={13} />
                    </BaseSelect.ItemIndicator>
                  </span>
                  <BaseSelect.ItemText className="truncate">{option.label}</BaseSelect.ItemText>
                </BaseSelect.Item>
              ))}
            </BaseSelect.List>
          </BaseSelect.Popup>
        </BaseSelect.Positioner>
      </BaseSelect.Portal>
    </BaseSelect.Root>
  )
}
