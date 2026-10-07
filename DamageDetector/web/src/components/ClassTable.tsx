import { motion } from 'motion/react'
import { CLASS_NAMES, classColor } from '../lib/classes'
import { metrics } from '../lib/metrics'

const ROWS = CLASS_NAMES.map((name) => ({
  name,
  ...metrics.test.tta.per_class[name],
  trainBoxes: metrics.dataset.class_instances.train[name],
})).sort((a, b) => b.ap50 - a.ap50)

export function ClassTable() {
  return (
    <div className="card overflow-hidden">
      {/* On phones only the class, AP50 and recall columns are shown, so the table fits without scrolling sideways. */}
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="eyebrow border-b border-line text-left">
            <th className="px-4 py-2.5 font-medium">Class</th>
            <th className="w-[46%] px-4 py-2.5 font-medium md:w-[34%]">AP50</th>
            <th className="hidden px-4 py-2.5 text-right font-medium md:table-cell">AP50-95</th>
            <th className="hidden px-4 py-2.5 text-right font-medium md:table-cell">Precision</th>
            <th className="px-4 py-2.5 text-right font-medium">Recall</th>
            <th className="hidden px-4 py-2.5 text-right font-medium md:table-cell">Training boxes</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map((row) => {
            const color = classColor(row.name)
            return (
              <tr key={row.name} className="border-b border-line transition-colors duration-150 last:border-b-0 hover:bg-raised">
                <td className="px-4 py-3">
                  <span className="flex items-center gap-2 whitespace-nowrap">
                    <span className="h-2.5 w-2.5 flex-none rounded-[3px]" style={{ backgroundColor: color }} />
                    {row.name}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <span className="flex items-center gap-3">
                    <span className="num w-11 flex-none">{row.ap50.toFixed(3)}</span>
                    <span className="h-2 flex-1 rounded-r-full bg-raised">
                      <motion.span
                        className="block h-full rounded-r-full"
                        style={{ backgroundColor: color }}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${row.ap50 * 100}%` }}
                        viewport={{ once: true, margin: '-40px' }}
                        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
                      />
                    </span>
                  </span>
                </td>
                <td className="num hidden px-4 py-3 text-right md:table-cell">{row.ap50_95.toFixed(3)}</td>
                <td className="num hidden px-4 py-3 text-right md:table-cell">{row.precision.toFixed(3)}</td>
                <td className="num px-4 py-3 text-right">{row.recall.toFixed(3)}</td>
                <td className="num hidden px-4 py-3 text-right text-muted md:table-cell">{row.trainBoxes.toLocaleString()}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
