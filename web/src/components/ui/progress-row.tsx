import { Xmark } from 'iconoir-react'
import { Badge } from './badge'
import { Button } from './button'

/** One long-running background job, with its bar.
 *
 *  A model download runs for minutes, so it cannot be a toast: by the time
 *  someone wonders whether it is still going, the toast has gone. It lives on
 *  the page instead, and keeps its last state after it finishes so a failure
 *  can be read by whoever was not watching.
 */
export interface ProgressRowProps {
  title: string
  state: 'running' | 'done' | 'failed' | 'cancelled'
  stateLabel: string
  detail: string
  /** 0..1, or null when the size is not known yet. */
  fraction: number | null
  error?: string
  cancelLabel?: string
  onCancel?: () => void
}

export function ProgressRow({
  title,
  state,
  stateLabel,
  detail,
  fraction,
  error,
  cancelLabel,
  onCancel,
}: ProgressRowProps) {
  const running = state === 'running'
  const pct = Math.round((fraction ?? 0) * 100)
  return (
    <div className="rounded-[var(--radius-control)] border border-border bg-raised px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13px] font-medium text-text">{title}</span>
        <Badge tone={state === 'failed' ? 'danger' : running ? 'accent' : 'neutral'}>
          {stateLabel}
        </Badge>
        <span className="text-[12px] text-muted">{detail}</span>
        {running && onCancel ? (
          <Button size="sm" className="ml-auto" onClick={onCancel}>
            <Xmark width={13} height={13} />
            {cancelLabel}
          </Button>
        ) : null}
      </div>
      {running ? (
        <div className="mt-2 h-1 overflow-hidden rounded-full bg-border" aria-hidden>
          <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
        </div>
      ) : null}
      {error ? <p className="mt-1.5 text-[12px] text-danger-text">{error}</p> : null}
    </div>
  )
}
