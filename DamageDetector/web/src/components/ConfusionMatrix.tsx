import { classColor } from '../lib/classes'
import { metrics } from '../lib/metrics'

const { labels, counts } = metrics.confusion_matrix

// Shorter headers so all seven columns fit on a phone. The full name stays available as the hover text.
const SHORT: Record<string, string> = {
  dent: 'dent',
  scratch: 'scr.',
  crack: 'crack',
  'glass shatter': 'glass',
  'lamp broken': 'lamp',
  'tire flat': 'tire',
  background: 'none',
}

// counts[predicted][actual]. Each column is divided by its total, so a cell reads as the share of that actual class.
const COLUMN_TOTALS = labels.map((_, actual) => counts.reduce((sum, row) => sum + row[actual], 0))

export function ConfusionMatrix() {
  return (
    <div className="card p-4 md:p-5">
      <p className="eyebrow">Confusion matrix, test split at threshold {metrics.confusion_matrix.conf.toFixed(2)}</p>
      <div className="mt-4">
        <div className="grid max-w-[38rem] gap-0.5" style={{ gridTemplateColumns: `4rem repeat(${labels.length}, minmax(0, 1fr))` }}>
          <span className="num flex items-end pb-1 text-[10px] leading-tight text-muted">predicted / actual</span>
          {labels.map((label) => (
            <span key={label} title={label} className="num flex flex-col items-center gap-1 pb-1 text-[10px] text-muted sm:text-[11px]">
              {label !== 'background' && <span className="h-1.5 w-1.5 rounded-[2px]" style={{ backgroundColor: classColor(label) }} />}
              {SHORT[label]}
            </span>
          ))}

          {labels.map((predicted, p) => (
            <div key={predicted} className="contents">
              <span title={predicted} className="num flex items-center gap-1.5 pr-1 text-[10px] text-muted sm:text-[11px]">
                {predicted !== 'background' && <span className="h-1.5 w-1.5 flex-none rounded-[2px]" style={{ backgroundColor: classColor(predicted) }} />}
                {SHORT[predicted]}
              </span>
              {labels.map((actual, a) => {
                const count = counts[p][a]
                const share = COLUMN_TOTALS[a] ? count / COLUMN_TOTALS[a] : 0
                const empty = predicted === 'background' && actual === 'background'
                return (
                  <span
                    key={actual}
                    title={empty ? undefined : `Predicted ${predicted}, actual ${actual}: ${count} of ${COLUMN_TOTALS[a]} (${(share * 100).toFixed(0)}%)`}
                    className={`num flex aspect-square items-center justify-center rounded-[3px] text-xs transition-shadow duration-150 hover:shadow-[0_0_0_2px_var(--color-text)] ${share > 0.5 ? 'text-on-brass' : 'text-text'}`}
                    // One hue from the card color up to brass, so a darker cell always means a smaller share.
                    style={{ backgroundColor: `color-mix(in srgb, var(--color-brass) ${Math.round(share * 100)}%, var(--color-raised))` }}
                  >
                    {empty ? '' : share >= 0.005 ? Math.round(share * 100) : ''}
                  </span>
                )
              })}
            </div>
          ))}
        </div>
      </div>
      <div className="num mt-4 flex items-center gap-2 text-[11px] text-muted">
        <span>0%</span>
        <span className="h-1.5 w-28 rounded-full" style={{ background: 'linear-gradient(to right, var(--color-raised), var(--color-brass))' }} />
        <span>100% of the actual class</span>
      </div>
    </div>
  )
}
