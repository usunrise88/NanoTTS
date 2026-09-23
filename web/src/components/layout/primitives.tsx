import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/* The only three layout components in the app. Routes compose these and never
 * reach for raw spacing, which is what keeps every page on the same rhythm
 * without anyone having to remember the numbers. */

const GAP = {
  xs: 'gap-1.5',
  sm: 'gap-2.5',
  md: 'gap-4',
  lg: 'gap-6',
} as const

export type Gap = keyof typeof GAP

/** Root of every route. Owns the page gutter and the vertical rhythm between
 *  sections; the header above it belongs to the shell, not to the page. */
export function Page({ children, gap = 'md' }: { children: ReactNode; gap?: Gap }) {
  return <div className={cn('flex flex-col', GAP[gap])}>{children}</div>
}

/** A titled surface. The title is a section label, not a page heading — page
 *  headings come from the route's static data. */
export function Section({
  title,
  description,
  actions,
  children,
  className,
}: {
  title?: string
  description?: string
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section
      className={cn(
        'rounded-[var(--radius-panel)] border border-border bg-surface p-4',
        className,
      )}
    >
      {title || actions ? (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div className="min-w-0">
            {title ? (
              <h3 className="text-[11px] font-medium tracking-wide text-faint uppercase">
                {title}
              </h3>
            ) : null}
            {description ? <p className="mt-1 text-[13px] text-muted">{description}</p> : null}
          </div>
          {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
        </header>
      ) : null}
      {children}
    </section>
  )
}

/** Flow in one direction with a token gap. `wrap` is the only escape hatch and
 *  it still cannot introduce a custom spacing value. */
export function Stack({
  direction = 'column',
  gap = 'sm',
  wrap,
  align,
  justify,
  className,
  children,
}: {
  direction?: 'row' | 'column'
  gap?: Gap
  wrap?: boolean
  align?: 'start' | 'center' | 'end' | 'baseline'
  justify?: 'start' | 'between' | 'end'
  className?: string
  children: ReactNode
}) {
  return (
    <div
      className={cn(
        'flex min-w-0',
        direction === 'row' ? 'flex-row' : 'flex-col',
        GAP[gap],
        wrap && 'flex-wrap',
        align === 'center' && 'items-center',
        align === 'start' && 'items-start',
        align === 'end' && 'items-end',
        align === 'baseline' && 'items-baseline',
        justify === 'between' && 'justify-between',
        justify === 'end' && 'justify-end',
        className,
      )}
    >
      {children}
    </div>
  )
}

/** Responsive column set. Pages pick a count, not a breakpoint. */
export function Columns({
  count = 2,
  gap = 'md',
  children,
}: {
  count?: 2 | 3 | 4
  gap?: Gap
  children: ReactNode
}) {
  const cols = {
    2: 'md:grid-cols-2',
    3: 'sm:grid-cols-2 lg:grid-cols-3',
    4: 'sm:grid-cols-2 lg:grid-cols-4',
  }[count]
  return <div className={cn('grid grid-cols-1 items-start', cols, GAP[gap])}>{children}</div>
}
