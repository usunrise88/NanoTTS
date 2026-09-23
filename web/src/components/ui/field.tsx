import { Field as BaseField } from '@base-ui-components/react/field'
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/** Every labelled control goes through here, so label typography, spacing and
 *  the error slot are defined once instead of per form. */
export function Field({
  label,
  hint,
  error,
  className,
  children,
}: {
  label: string
  hint?: string
  error?: string
  className?: string
  children: ReactNode
}) {
  return (
    <BaseField.Root className={cn('flex min-w-0 flex-col gap-1.5', className)}>
      <BaseField.Label className="text-xs font-medium text-muted">{label}</BaseField.Label>
      {children}
      {hint && !error ? <span className="text-xs text-faint">{hint}</span> : null}
      {error ? <span className="text-xs text-danger-text">{error}</span> : null}
    </BaseField.Root>
  )
}
