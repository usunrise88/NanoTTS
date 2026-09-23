import { cn } from '@/lib/utils'

/** Monospaced block with its own scroll container, so a long command never
 *  widens the page. */
export function CodeBlock({ children, className }: { children: string; className?: string }) {
  return (
    <pre
      className={cn(
        'overflow-x-auto rounded-[var(--radius-control)] border border-border bg-raised',
        'p-3 font-mono text-[12px] leading-relaxed',
        className,
      )}
    >
      {children}
    </pre>
  )
}
