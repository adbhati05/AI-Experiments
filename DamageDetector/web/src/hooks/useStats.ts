import { useEffect, useState } from 'react'
import { STATS_CHANGED, getStats } from '../lib/api'
import type { Stats } from '../lib/types'

// Fetching the running totals when the component using this hook mounts, and again whenever an answer is saved.
export function useStats(): Stats | null {
  const [stats, setStats] = useState<Stats | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = () => {
      getStats().then((result) => {
        if (!cancelled) setStats(result)
      })
    }

    load()
    window.addEventListener(STATS_CHANGED, load)
    return () => {
      cancelled = true
      window.removeEventListener(STATS_CHANGED, load)
    }
  }, [])

  return stats
}
