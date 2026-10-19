import type { FeedbackRequest, PredictResponse, Stats } from './types'

const API_URL = import.meta.env.VITE_API_URL

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024 // Matching the 10 MB cap enforced in api/main.py.

// Calling the health endpoint when the page loads, which also wakes the server up if the host put it to sleep.
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`)
    return res.ok
  } catch {
    return false
  }
}

export async function predict(file: File): Promise<PredictResponse> {
  const form = new FormData()
  form.append('file', file) // The name 'file' has to match the parameter name of predict() in api/main.py.

  let res: Response
  try {
    res = await fetch(`${API_URL}/predict`, { method: 'POST', body: form })
  } catch {
    throw new Error('Could not reach the server. Check your connection and try again.')
  }

  const data = await res.json()
  // fetch only throws on network failures, so a 400 or 413 from the API has to be checked for here. The API puts its message in "detail".
  if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Something went wrong while scanning the photo.')
  return data
}

export type FeedbackResult = 'saved' | 'duplicate' | 'failed'

export const STATS_CHANGED = 'stats-changed' // Fired on window after an answer is saved, so the counter in the footer can fetch the new totals.

// Returning a result instead of throwing, since a repeated answer (409) is not really a failure and the form treats it as done.
export async function submitFeedback(feedback: FeedbackRequest): Promise<FeedbackResult> {
  try {
    const res = await fetch(`${API_URL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(feedback),
    })
    if (res.ok) {
      window.dispatchEvent(new Event(STATS_CHANGED))
      return 'saved'
    }
    return res.status === 409 ? 'duplicate' : 'failed'
  } catch {
    return 'failed'
  }
}

// Returning null when the stats cannot be read, so the counter hides itself instead of showing a made up number.
export async function getStats(): Promise<Stats | null> {
  try {
    const res = await fetch(`${API_URL}/stats`)
    return res.ok ? await res.json() : null
  } catch {
    return null
  }
}
