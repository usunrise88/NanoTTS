import { cn } from '@/lib/utils'

/** The only spinner in the app: thin, monochrome, inline, sized to the text it
 *  sits beside. Full-screen spinners are deliberately impossible here. */
export function Spinner({ className }: { className?: string }) {
  return (
    <span
      role="status"
      aria-label="loading"
      className={cn(
        'inline-block size-3.5 shrink-0 rounded-full border-[1.5px] border-current',
        'border-r-transparent align-[-2px] opacity-70',
        className,
      )}
      style={{ animation: 'app-spin 700ms linear infinite' }}
    />
  )
}
