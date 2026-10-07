import type { PredictResponse } from './types'

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
