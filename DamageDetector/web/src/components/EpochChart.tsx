import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { metrics } from '../lib/metrics'

const SHIPPED = metrics.sessions[metrics.sessions.length - 1].id

// Reshaping the per-session curves into one row per epoch, which is the shape Recharts expects. Session 5 stopped early, so its later epochs are left out.
const DATA = Array.from({ length: 100 }, (_, i) => {
  const row: Record<string, number> = { epoch: i + 1 }
  for (const session of metrics.sessions) {
    const value = session.curve.map50_95[i]
    if (value !== undefined) row[`s${session.id}`] = value
  }
  return row
})

// The shipped session is drawn in brass and the others in cream, so the eye goes to the one that matters.
function strokeFor(id: number) {
  return id === SHIPPED ? 'var(--color-brass)' : 'var(--color-text)'
}

function opacityFor(id: number, active: number | null) {
  if (active === null) return id === SHIPPED ? 1 : 0.32
  return active === id ? 1 : 0.12
}

export function EpochChart({ active, onActive }: { active: number | null; onActive: (id: number | null) => void }) {
  return (
    <div className="card p-4 md:p-5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <p className="eyebrow mr-auto">Validation mAP50-95 by epoch</p>
        {/* The legend doubles as a control: hovering or focusing a session brings its line forward. */}
        <ul className="flex flex-wrap gap-x-3 gap-y-1">
          {metrics.sessions.map((session) => (
            <li key={session.id}>
              <button
                type="button"
                onMouseEnter={() => onActive(session.id)}
                onMouseLeave={() => onActive(null)}
                onFocus={() => onActive(session.id)}
                onBlur={() => onActive(null)}
                className={`flex items-center gap-1.5 text-xs transition-opacity duration-150 ${active !== null && active !== session.id ? 'opacity-40' : ''}`}
              >
                <span className="h-0.5 w-4 rounded-full" style={{ backgroundColor: strokeFor(session.id), opacity: session.id === SHIPPED ? 1 : 0.6 }} />
                <span className="num text-muted">S{session.id}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="mt-4 h-64 md:h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={DATA} margin={{ top: 8, right: 18, bottom: 0, left: -18 }}>
            <CartesianGrid vertical={false} stroke="var(--color-line)" />
            <XAxis
              dataKey="epoch"
              type="number"
              domain={[1, 100]}
              ticks={[1, 25, 50, 75, 100]}
              tick={{ fill: 'var(--color-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}
              axisLine={{ stroke: 'var(--color-line)' }}
              tickLine={false}
            />
            <YAxis
              domain={[0.1, 0.6]}
              ticks={[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
              tickFormatter={(v: number) => v.toFixed(1)}
              tick={{ fill: 'var(--color-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ stroke: 'var(--color-muted)', strokeWidth: 1 }}
              content={({ active: shown, payload, label }) => {
                if (!shown || !payload?.length) return null
                const rows = [...payload].sort((a, b) => Number(b.value) - Number(a.value))
                return (
                  <div className="rounded-md border border-line bg-raised px-3 py-2 text-xs shadow-lg">
                    <p className="num mb-1 text-muted">Epoch {label}</p>
                    {rows.map((row) => (
                      <p key={String(row.dataKey)} className="flex items-center gap-2">
                        <span className="h-0.5 w-3 rounded-full" style={{ backgroundColor: row.color }} />
                        <span className="num font-medium">{Number(row.value).toFixed(3)}</span>
                        <span className="num text-muted">{String(row.dataKey).toUpperCase()}</span>
                      </p>
                    ))}
                  </div>
                )
              }}
            />
            {metrics.sessions.map((session) => (
              <Line
                key={session.id}
                dataKey={`s${session.id}`}
                type="monotone"
                stroke={strokeFor(session.id)}
                strokeOpacity={opacityFor(session.id, active)}
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
