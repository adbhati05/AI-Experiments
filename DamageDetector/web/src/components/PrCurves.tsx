import { useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { CLASS_NAMES, classColor } from '../lib/classes'
import { metrics } from '../lib/metrics'

// One row per recall value with a precision column for each class, which is the shape Recharts expects.
const DATA = metrics.pr_curves.recall.map((recall, i) => {
  const row: Record<string, number> = { recall }
  for (const name of CLASS_NAMES) row[name] = metrics.pr_curves.precision[name][i]
  return row
})

const AXIS_TICK = { fill: 'var(--color-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }

export function PrCurves() {
  const [active, setActive] = useState<string | null>(null)

  return (
    <div className="card p-4 md:p-5">
      <p className="eyebrow">Precision against recall, test split</p>

      {/* Each legend entry carries the class name and its AP50, and brings its curve forward on hover or focus. */}
      <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5">
        {CLASS_NAMES.map((name) => (
          <li key={name}>
            <button
              type="button"
              onMouseEnter={() => setActive(name)}
              onMouseLeave={() => setActive(null)}
              onFocus={() => setActive(name)}
              onBlur={() => setActive(null)}
              className={`flex items-center gap-1.5 text-xs transition-opacity duration-150 ${active !== null && active !== name ? 'opacity-40' : ''}`}
            >
              <span className="h-0.5 w-4 rounded-full" style={{ backgroundColor: classColor(name) }} />
              {name}
              <span className="num text-muted">{metrics.test.tta.per_class[name].ap50.toFixed(2)}</span>
            </button>
          </li>
        ))}
      </ul>

      <div className="mt-4 h-72 md:h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={DATA} margin={{ top: 8, right: 18, bottom: 16, left: -18 }}>
            <CartesianGrid vertical={false} stroke="var(--color-line)" />
            <XAxis
              dataKey="recall"
              type="number"
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tickFormatter={(v: number) => v.toFixed(2)}
              tick={AXIS_TICK}
              axisLine={{ stroke: 'var(--color-line)' }}
              tickLine={false}
              label={{ value: 'recall', position: 'insideBottom', offset: -10, fill: 'var(--color-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
            />
            <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(v: number) => v.toFixed(2)} tick={AXIS_TICK} axisLine={false} tickLine={false} />
            <Tooltip
              cursor={{ stroke: 'var(--color-muted)', strokeWidth: 1 }}
              content={({ active: shown, payload, label }) => {
                if (!shown || !payload?.length) return null
                const rows = [...payload].sort((a, b) => Number(b.value) - Number(a.value))
                return (
                  <div className="rounded-md border border-line bg-raised px-3 py-2 text-xs shadow-lg">
                    <p className="num mb-1 text-muted">Recall {Number(label).toFixed(2)}</p>
                    {rows.map((row) => (
                      <p key={String(row.dataKey)} className="flex items-center gap-2">
                        <span className="h-0.5 w-3 rounded-full" style={{ backgroundColor: row.color }} />
                        <span className="num font-medium">{Number(row.value).toFixed(2)}</span>
                        <span className="text-muted">{String(row.dataKey)}</span>
                      </p>
                    ))}
                  </div>
                )
              }}
            />
            {CLASS_NAMES.map((name) => (
              <Line
                key={name}
                dataKey={name}
                type="monotone"
                stroke={classColor(name)}
                strokeOpacity={active === null || active === name ? 1 : 0.15}
                strokeWidth={2}
                strokeLinecap="round"
                dot={false}
                activeDot={{ r: 4, stroke: 'var(--color-card)', strokeWidth: 2 }}
                animationDuration={900}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
