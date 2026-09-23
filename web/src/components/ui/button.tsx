import { cva, type VariantProps } from 'class-variance-authority'
import type { ComponentProps } from 'react'
import { cn } from '@/lib/utils'
import { Spinner } from './spinner'

const button = cva(
  'inline-flex items-center justify-center gap-1.5 rounded-[var(--radius-control)] ' +
    'border font-medium whitespace-nowrap select-none ' +
    'disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        primary: 'border-transparent bg-accent text-on-accent hover:bg-accent-hover',
        secondary: 'border-border bg-surface text-text hover:bg-raised',
        ghost: 'border-transparent text-muted hover:bg-raised hover:text-text',
        danger: 'border-transparent bg-danger text-white hover:opacity-90',
      },
      size: {
        sm: 'h-7 px-2.5 text-xs',
        md: 'h-8 px-3 text-[13px]',
        lg: 'h-9 px-4 text-sm',
        icon: 'size-8 p-0',
      },
    },
    defaultVariants: { variant: 'secondary', size: 'md' },
  },
)

interface ButtonProps extends ComponentProps<'button'>, VariantProps<typeof button> {
  /** Shows the inline spinner and disables the control. Never a full-screen
   *  overlay: an action that is running should hold its own place. */
  loading?: boolean
}

export function Button({ className, variant, size, loading, children, ...props }: ButtonProps) {
  return (
    <button
      type="button"
      {...props}
      disabled={props.disabled || loading}
      className={cn(button({ variant, size }), className)}
    >
      {loading ? <Spinner /> : null}
      {children}
    </button>
  )
}
