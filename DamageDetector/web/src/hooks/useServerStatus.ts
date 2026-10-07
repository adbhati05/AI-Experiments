import { useEffect, useState } from 'react'
import { checkHealth } from '../lib/api'

export type ServerStatus = 'checking' | 'ready' | 'down'

export function useServerStatus(): ServerStatus {
  const [status, setStatus] = useState<ServerStatus>('checking')

  useEffect(() => {
    let cancelled = false
    checkHealth().then((ok) => {
      if (!cancelled) setStatus(ok ? 'ready' : 'down')
    })
    return () => {
      cancelled = true
    }
  }, [])

  return status
}
