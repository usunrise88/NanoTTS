import type { ComponentProps } from 'react'
import { cn } from '@/lib/utils'

const base =
  'w-full rounded-[var(--radius-control)] border border-border bg-surface px-2.5 text-[13px] ' +
  'text-text placeholder:text-faint hover:border-border-strong ' +
  'disabled:cursor-not-allowed disabled:opacity-60'

export function Input({ className, ...props }: ComponentProps<'input'>) {
  return <input {...props} className={cn(base, 'h-8', className)} />
}

export function Textarea({ className, ...props }: ComponentProps<'textarea'>) {
  return <textarea {...props} className={cn(base, 'min-h-24 resize-y py-2 leading-relaxed', className)} />
}
