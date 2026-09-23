import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { Skeleton } from './skeleton'

/** A read-out, not a stat card: small label, large monospaced figure, optional
 *  unit line. Used in a row so the figures line up across the set. */
export function Meter({
  label,
  value,
  sub,
  loading,
}: {
  label: string
  value: ReactNode
  sub?: ReactNode
  loading?: boolean
}) {
  return (
    <div className="bg-surface px-3 py-2.5">
      <div className="text-[11px] tracking-wide text-faint uppercase">{label}</div>
      {loading ? (
        <Skeleton className="mt-1 h-6 w-16" />
      ) : (
        <div className="tabular font-mono text-lg leading-7 font-semibold tracking-tight">
          {value}
        </div>
      )}
      {sub ? <div className="font-mono text-[11px] text-faint">{sub}</div> : null}
    </div>
  )
}

export function MeterRow({ children }: { children: ReactNode }) {
  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-[var(--radius-panel)] border border-border bg-border sm:grid-cols-4">
      {children}
    </div>
  )
}

/** The realtime line is the point of the scale, so it is drawn, not implied. */
export function RtfScale({ rtf }: { rtf: number | null }) {
  const pct = rtf == null ? 0 : Math.min(100, (rtf / 2) * 100)
  return (
    <div>
      <div className="relative h-2 overflow-hidden rounded-full border border-border bg-raised">
        <div
          className={cn('h-full', rtf != null && rtf > 1 ? 'bg-danger' : 'bg-accent')}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="relative mt-1 flex justify-between font-mono text-[10px] text-faint">
        <span>0</span>
        <span>1.0</span>
        <span>2.0</span>
      </div>
    </div>
  )
}
