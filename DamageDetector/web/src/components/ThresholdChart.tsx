import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { metrics } from '../lib/metrics'

const SERIES = [
  { key: 'precision', label: 'Precision', color: 'var(--color-text)' },
  { key: 'recall', label: 'Recall', color: 'var(--color-brass)' },
]

const AXIS_TICK = { fill: 'var(--color-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }

// Precision and recall at every confidence threshold. The two lines cross next to the shipping threshold, which is why 0.50 was chosen.
export function ThresholdChart() {
  const { curve, best_f1 } = metrics.thresholds

  return (
    <div className="card p-4 md:p-5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <p className="eyebrow mr-auto">Precision and recall by threshold, validation split</p>
        <ul className="flex gap-4">
          {SERIES.map((s) => (
            <li key={s.key} className="flex items-center gap-1.5 text-xs text-muted">
              <span className="h-0.5 w-4 rounded-full" style={{ backgroundColor: s.color }} />
              {s.label}
            </li>
          ))}
        </ul>
      </div>

      <div className="mt-4 h-64 md:h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={curve} margin={{ top: 16, right: 18, bottom: 0, left: -18 }}>
            <CartesianGrid vertical={false} stroke="var(--color-line)" />
            <XAxis dataKey="conf" type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(v: number) => v.toFixed(2)} tick={AXIS_TICK} axisLine={{ stroke: 'var(--color-line)' }} tickLine={false} />
            <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(v: number) => v.toFixed(2)} tick={AXIS_TICK} axisLine={false} tickLine={false} />
            <ReferenceLine
              x={metrics.model.conf}
              stroke="var(--color-muted)"
              strokeWidth={1}
              label={{ value: `shipped at ${metrics.model.conf.toFixed(2)}`, position: 'top', fill: 'var(--color-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
            />
            <Tooltip
              cursor={{ stroke: 'var(--color-muted)', strokeWidth: 1 }}
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null
                return (
                  <div className="rounded-md border border-line bg-raised px-3 py-2 text-xs shadow-lg">
                    <p className="num mb-1 text-muted">Threshold {Number(label).toFixed(2)}</p>
                    {payload.map((row) => (
                      <p key={String(row.dataKey)} className="flex items-center gap-2">
                        <span className="h-0.5 w-3 rounded-full" style={{ backgroundColor: row.color }} />
                        <span className="num font-medium">{Number(row.value).toFixed(3)}</span>
                        <span className="text-muted">{String(row.dataKey)}</span>
                      </p>
                    ))}
                  </div>
                )
              }}
            />
            {SERIES.map((s) => (
              <Line key={s.key} dataKey={s.key} type="monotone" stroke={s.color} strokeWidth={2} strokeLinecap="round" dot={false} activeDot={{ r: 4, stroke: 'var(--color-card)', strokeWidth: 2 }} animationDuration={900} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="num mt-3 text-xs text-muted">
        Best F1 at {best_f1.conf.toFixed(2)}: precision {best_f1.precision.toFixed(3)}, recall {best_f1.recall.toFixed(3)}
      </p>
    </div>
  )
}
