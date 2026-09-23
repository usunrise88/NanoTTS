import { Dialog as BaseDialog } from '@base-ui-components/react/dialog'
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

const backdrop =
  'fixed inset-0 z-40 bg-overlay data-[starting-style]:opacity-0 data-[ending-style]:opacity-0'

/** Centre modal. Reserved for true interruptions — destructive confirmations and
 *  choices the operator cannot skip. Everything else belongs in a sheet, a
 *  popover, or inline. */
export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description?: string
  children?: ReactNode
  footer?: ReactNode
}) {
  return (
    <BaseDialog.Root open={open} onOpenChange={onOpenChange}>
      <BaseDialog.Portal>
        <BaseDialog.Backdrop className={backdrop} style={{ transition: 'opacity 140ms ease' }} />
        <BaseDialog.Popup
          className={cn(
            'fixed top-1/2 left-1/2 z-50 w-[min(28rem,calc(100vw-2rem))]',
            '-translate-x-1/2 -translate-y-1/2 rounded-[var(--radius-panel)]',
            'border border-border bg-surface p-5 shadow-xl shadow-black/20 outline-none',
            'data-[starting-style]:opacity-0 data-[ending-style]:opacity-0',
          )}
          style={{ transition: 'opacity 140ms ease' }}
        >
          <BaseDialog.Title className="text-sm font-semibold text-text">{title}</BaseDialog.Title>
          {description ? (
            <BaseDialog.Description className="mt-1.5 text-[13px] text-muted">
              {description}
            </BaseDialog.Description>
          ) : null}
          {children ? <div className="mt-4">{children}</div> : null}
          {footer ? <div className="mt-5 flex justify-end gap-2">{footer}</div> : null}
        </BaseDialog.Popup>
      </BaseDialog.Portal>
    </BaseDialog.Root>
  )
}

/** Side sheet. The default home for anything that needs room but must not
 *  interrupt: detail panels, longer forms, settings. */
export function Sheet({
  open,
  onOpenChange,
  title,
  description,
  children,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description?: string
  children: ReactNode
}) {
  return (
    <BaseDialog.Root open={open} onOpenChange={onOpenChange}>
      <BaseDialog.Portal>
        <BaseDialog.Backdrop className={backdrop} style={{ transition: 'opacity 140ms ease' }} />
        <BaseDialog.Popup
          className={cn(
            'fixed inset-y-0 right-0 z-50 flex w-[min(30rem,100vw)] flex-col',
            'border-l border-border bg-surface p-5 shadow-xl shadow-black/20 outline-none',
            'data-[starting-style]:opacity-0 data-[ending-style]:opacity-0',
          )}
          style={{ transition: 'opacity 140ms ease' }}
        >
          <BaseDialog.Title className="text-sm font-semibold text-text">{title}</BaseDialog.Title>
          {description ? (
            <BaseDialog.Description className="mt-1.5 text-[13px] text-muted">
              {description}
            </BaseDialog.Description>
          ) : null}
          <div className="mt-4 min-h-0 flex-1 overflow-y-auto">{children}</div>
        </BaseDialog.Popup>
      </BaseDialog.Portal>
    </BaseDialog.Root>
  )
}
