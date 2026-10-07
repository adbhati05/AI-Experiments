import { motion } from 'motion/react'
import { CLASS_NAMES, classColor } from '../lib/classes'
import { metrics } from '../lib/metrics'

const ROWS = CLASS_NAMES.map((name) => ({
  name,
  area: metrics.class_geometry[name].median_box_area_pct,
  fill: metrics.class_geometry[name].box_fill_pct,
  ap50: metrics.test.tta.per_class[name].ap50,
})).sort((a, b) => b.ap50 - a.ap50)

const MAX_AREA = Math.max(...ROWS.map((r) => r.area))

// A value with a thin bar under it. The bar grows in when the table scrolls into view.
function Meter({ value, share, color }: { value: string; share: number; color: string }) {
  return (
    <div>
      <p className="num text-sm">{value}</p>
      <div className="mt-1.5 h-1.5 rounded-r-full bg-raised">
        <motion.div
          className="h-full rounded-r-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          whileInView={{ width: `${Math.max(share * 100, 1.5)}%` }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
    </div>
  )
}

// Three measurements per class, sorted by accuracy. All three columns rank the classes in the same order.
export function GeometryTable() {
  return (
    <div className="card overflow-hidden">
      <div>
        <div className="eyebrow grid grid-cols-[6.75rem_1fr_1fr_1fr] gap-3 border-b border-line px-4 py-2.5 md:grid-cols-[1.15fr_1fr_1fr_1fr] md:gap-4">
          <span>Class</span>
          <span>Box size</span>
          <span>Box fill</span>
          <span>AP50</span>
        </div>
        {ROWS.map((row) => {
          const color = classColor(row.name)
          return (
            <div key={row.name} className="grid grid-cols-[6.75rem_1fr_1fr_1fr] items-center gap-3 border-b border-line px-4 py-3 last:border-b-0 md:grid-cols-[1.15fr_1fr_1fr_1fr] md:gap-4">
              <p className="flex items-center gap-2 text-sm">
                <span className="h-2.5 w-2.5 flex-none rounded-[3px]" style={{ backgroundColor: color }} />
                {row.name}
              </p>
              <Meter value={`${row.area.toFixed(1)}%`} share={row.area / MAX_AREA} color={color} />
              <Meter value={`${row.fill.toFixed(0)}%`} share={row.fill / 100} color={color} />
              <Meter value={row.ap50.toFixed(3)} share={row.ap50} color={color} />
            </div>
          )
        })}
      </div>
    </div>
  )
}
