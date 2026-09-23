import { useEffect, useRef, useState } from 'react'

/** True only once `active` has held for `delayMs`.
 *
 *  The rule is: match the wait. A request that finishes in 120 ms must not
 *  flash a skeleton, because the flash reads as a glitch rather than as
 *  progress. Anything slower than roughly a third of a second does deserve
 *  one, and this is what decides which is which. */
export function useDelayedFlag(active: boolean, delayMs = 300): boolean {
  const [shown, setShown] = useState(false)
  useEffect(() => {
    if (!active) {
      setShown(false)
      return
    }
    const timer = window.setTimeout(() => setShown(true), delayMs)
    return () => window.clearTimeout(timer)
  }, [active, delayMs])
  return shown
}

/** Poll while `enabled`, pausing whenever the tab is hidden so a console left
 *  open overnight does not keep a worker busy answering /stats. */
export function useInterval(callback: () => void, ms: number, enabled = true) {
  const saved = useRef(callback)
  saved.current = callback
  useEffect(() => {
    if (!enabled) return
    const tick = () => {
      if (!document.hidden) saved.current()
    }
    const id = window.setInterval(tick, ms)
    return () => window.clearInterval(id)
  }, [ms, enabled])
}
