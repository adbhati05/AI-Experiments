import { useStats } from '../hooks/useStats'

const MIN_REPORTS = 10 // Hiding the counter until there are enough answers for the numbers to mean something.

// The live counter of answers from the questionnaire. It renders nothing if the stats cannot be read or there are too few answers.
// The "|" after it is rendered here rather than in the footer, so the separator only appears when the counter does.
export function ReportedStats() {
  const stats = useStats()
  if (!stats || stats.total < MIN_REPORTS) return null

  return (
    <>
      <span>
        User-reported: {stats.correct} of {stats.total} scans correct, {stats.missed_or_wrong} missed or wrong, {stats.false_positive}{' '}
        {stats.false_positive === 1 ? 'false alarm' : 'false alarms'}
      </span>
      <span aria-hidden="true">|</span>
    </>
  )
}
