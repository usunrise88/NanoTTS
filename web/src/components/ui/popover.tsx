import { Popover as BasePopover } from '@base-ui-components/react/popover'
import type { ReactElement, ReactNode } from 'react'
import { cn } from '@/lib/utils'

/** The middle rung of the surface ladder: heavier than inline, lighter than a
 *  sheet, and it never blocks the page behind it. */
export function Popover({
  trigger,
  children,
  align = 'end',
  className,
}: {
  trigger: ReactElement<Record<string, unknown>>
  children: ReactNode
  align?: 'start' | 'center' | 'end'
  className?: string
}) {
  return (
    <BasePopover.Root>
      <BasePopover.Trigger render={trigger} />
      <BasePopover.Portal>
        <BasePopover.Positioner sideOffset={6} align={align} className="z-50">
          <BasePopover.Popup
            className={cn(
              'rounded-[var(--radius-panel)] border border-border bg-surface',
              'shadow-lg shadow-black/10 outline-none',
              'data-[starting-style]:opacity-0 data-[ending-style]:opacity-0',
              className,
            )}
            style={{ transition: 'opacity 120ms ease' }}
          >
            {children}
          </BasePopover.Popup>
        </BasePopover.Positioner>
      </BasePopover.Portal>
    </BasePopover.Root>
  )
}
