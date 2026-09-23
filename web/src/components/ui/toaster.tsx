import { Toaster as Sonner } from 'sonner'
import { useTheme } from '@/lib/theme'

/** Stacked, auto-dismissing, bottom-right. Toasts acknowledge what already
 *  happened; anything the operator must act on gets a real surface instead. */
export function Toaster() {
  const { resolved } = useTheme()
  return (
    <Sonner
      theme={resolved}
      position="bottom-right"
      closeButton
      duration={4500}
      toastOptions={{
        classNames: {
          toast:
            'border border-border bg-surface text-text rounded-[var(--radius-panel)] text-[13px]',
          description: 'text-muted',
        },
      }}
    />
  )
}
