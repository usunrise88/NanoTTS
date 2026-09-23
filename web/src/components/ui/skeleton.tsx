import { cn } from '@/lib/utils'

/** Sized to the real content so nothing shifts when the data lands, muted so it
 *  reads as absence rather than as a component, and pulsing softly — the pulse
 *  is switched off globally under prefers-reduced-motion. */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn('rounded-[var(--radius-control)] bg-raised', className)}
      style={{ animation: 'app-pulse 1.8s ease-in-out infinite' }}
    />
  )
}

export function SkeletonText({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cn('flex flex-col gap-2', className)}>
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} className={cn('h-3.5', i === lines - 1 ? 'w-2/3' : 'w-full')} />
      ))}
    </div>
  )
}

export function SkeletonRows({ rows = 4, cols = 4 }: { rows?: number; cols?: number }) {
  return (
    <div className="flex flex-col gap-px">
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} className="grid gap-3 py-2.5" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
          {Array.from({ length: cols }, (_, c) => (
            <Skeleton key={c} className={cn('h-3.5', c === 0 ? 'w-24' : 'w-16 justify-self-end')} />
          ))}
        </div>
      ))}
    </div>
  )
}
