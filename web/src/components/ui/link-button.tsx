import type { ComponentProps } from 'react'
import { cn } from '@/lib/utils'

/** A download has to be a real anchor for the browser to save the file, so it
 *  cannot reuse Button. This keeps it visually identical to a secondary one. */
export function LinkButton({ className, ...props }: ComponentProps<'a'>) {
  return (
    <a
      {...props}
      className={cn(
        'inline-flex h-7 items-center gap-1.5 rounded-[var(--radius-control)]',
        'border border-border px-2.5 text-xs text-text no-underline hover:bg-raised',
        className,
      )}
    />
  )
}
