import type { ComponentProps, ReactNode } from 'react'
import { cn } from '@/lib/utils'

export function Table({ className, ...props }: ComponentProps<'table'>) {
  return (
    <div className="overflow-x-auto">
      <table {...props} className={cn('w-full border-collapse text-[13px]', className)} />
    </div>
  )
}

export function Th({ className, numeric, ...props }: ComponentProps<'th'> & { numeric?: boolean }) {
  return (
    <th
      {...props}
      className={cn(
        'border-b border-border pb-2 pr-3 text-left text-[11px] font-medium tracking-wide text-faint uppercase',
        numeric && 'text-right',
        className,
      )}
    />
  )
}

export function Td({ className, numeric, ...props }: ComponentProps<'td'> & { numeric?: boolean }) {
  return (
    <td
      {...props}
      className={cn(
        'border-b border-border py-2 pr-3 align-middle',
        numeric && 'tabular text-right font-mono',
        className,
      )}
    />
  )
}

/** Consistent empty state for every table: one row, centred, muted. */
export function TableEmpty({ colSpan, children }: { colSpan: number; children: ReactNode }) {
  return (
    <tr>
      <td colSpan={colSpan} className="border-b border-border py-6 text-center text-faint">
        {children}
      </td>
    </tr>
  )
}
