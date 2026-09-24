import { createContext, use, useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { activeModel, type ActiveModel } from './api'

/** Which model the server is serving right now.
 *
 *  Held in one place because it changes under the console's feet: the models
 *  page can switch the running server between architectures, and every page
 *  that shows a duration, a knob or an upload button has to follow. Anything
 *  reading `model` re-renders on a switch.
 */
interface ModelState {
  model: ActiveModel | null
  /** False only before the first answer; a failed refresh keeps the last known
   *  model rather than blanking the page. */
  loaded: boolean
  refresh: () => Promise<void>
}

const Ctx = createContext<ModelState | null>(null)

export function ModelProvider({ children }: { children: ReactNode }) {
  const [model, setModel] = useState<ActiveModel | null>(null)
  const [loaded, setLoaded] = useState(false)

  const refresh = useCallback(async () => {
    try {
      setModel((await activeModel()) ?? null)
    } catch {
      /* offline or unauthorised: the connection badge already says so */
    } finally {
      setLoaded(true)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const value = useMemo(() => ({ model, loaded, refresh }), [model, loaded, refresh])
  return <Ctx value={value}>{children}</Ctx>
}

export function useModel(): ModelState {
  const ctx = use(Ctx)
  if (!ctx) throw new Error('useModel outside ModelProvider')
  return ctx
}
