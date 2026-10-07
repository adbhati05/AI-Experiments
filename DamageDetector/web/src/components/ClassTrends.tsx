import { CLASS_NAMES, classColor } from '../lib/classes'
import { metrics } from '../lib/metrics'

// Plot geometry shared by every small panel, so the six classes can be compared against the same scale.
const W = 200
const H = 104
const LEFT = 18
const RIGHT = 182
const TOP = 22
const BOTTOM = 78

const xFor = (i: number) => LEFT + (i * (RIGHT - LEFT)) / (metrics.sessions.length - 1)
const yFor = (v: number) => BOTTOM - v * (BOTTOM - TOP)

// One small chart per class showing its val AP50 across the five sessions. Drawing them separately is clearer than five overlapping lines on one chart.
export function ClassTrends() {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
      {CLASS_NAMES.map((name) => {
        const color = classColor(name)
        const values = metrics.sessions.map((s) => s.val.per_class[name].ap50)
        const change = values[values.length - 1] - values[0]
        const points = values.map((v, i) => `${xFor(i)},${yFor(v)}`).join(' ')
        return (
          <div key={name} className="card p-3">
            <div className="flex items-baseline justify-between gap-2">
              <p className="flex items-center gap-2 text-sm">
                <span className="h-2.5 w-2.5 flex-none rounded-[3px]" style={{ backgroundColor: color }} />
                {name}
              </p>
              <p className="num text-xs text-muted">{change >= 0 ? '+' : '−'}{Math.abs(change).toFixed(2)}</p>
            </div>
            <svg
              viewBox={`0 0 ${W} ${H}`}
              className="mt-1 w-full"
              role="img"
              aria-label={`${name}: AP50 by session, ${values.map((v, i) => `S${i + 1} ${v.toFixed(2)}`).join(', ')}`}
            >
              <line x1={LEFT - 8} x2={RIGHT + 8} y1={yFor(1)} y2={yFor(1)} stroke="var(--color-line)" strokeWidth="1" />
              <line x1={LEFT - 8} x2={RIGHT + 8} y1={yFor(0)} y2={yFor(0)} stroke="var(--color-line)" strokeWidth="1" />
              <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
              {values.map((v, i) => (
                <g key={i}>
                  <circle cx={xFor(i)} cy={yFor(v)} r="4" fill={color} stroke="var(--color-card)" strokeWidth="2" />
                  {/* A larger invisible circle so the point is easy to hover. */}
                  <circle cx={xFor(i)} cy={yFor(v)} r="14" fill="transparent">
                    <title>{`Session ${i + 1}: ${v.toFixed(3)}`}</title>
                  </circle>
                  <text x={xFor(i)} y={H - 6} textAnchor="middle" fontSize="10" fill="var(--color-muted)" fontFamily="var(--font-mono)">
                    S{i + 1}
                  </text>
                </g>
              ))}
              {/* Labeling only the first and last value, and leaving the rest to the hover text. */}
              <text x={xFor(0)} y={yFor(values[0]) - 9} textAnchor="start" fontSize="11" fill="var(--color-text)" fontFamily="var(--font-mono)">
                {values[0].toFixed(2)}
              </text>
              <text x={xFor(values.length - 1)} y={yFor(values[values.length - 1]) - 9} textAnchor="end" fontSize="11" fill="var(--color-text)" fontFamily="var(--font-mono)">
                {values[values.length - 1].toFixed(2)}
              </text>
            </svg>
          </div>
        )
      })}
    </div>
  )
}
