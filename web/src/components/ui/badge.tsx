import { cva, type VariantProps } from 'class-variance-authority'
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

const badge = cva(
  'inline-flex items-center gap-1 rounded-full border px-2 py-px font-mono text-[11px] leading-5',
  {
    variants: {
      tone: {
        neutral: 'border-border text-muted',
        accent: 'border-transparent bg-accent-surface text-accent-text',
        danger: 'border-transparent bg-danger-surface text-danger-text',
      },
    },
    defaultVariants: { tone: 'neutral' },
  },
)

export function Badge({
  tone,
  className,
  children,
}: VariantProps<typeof badge> & { className?: string; children: ReactNode }) {
  return <span className={cn(badge({ tone }), className)}>{children}</span>
}
