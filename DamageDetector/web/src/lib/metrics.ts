import raw from '../data/metrics.json'
import type { Metrics } from './types'

// Importing the metrics file directly so the Training and Performance pages load without calling the API.
export const metrics = raw as unknown as Metrics
