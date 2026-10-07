import { useState } from 'react'
import { motion } from 'motion/react'
import { CountUp } from './CountUp'
import { SegmentedControl } from './SegmentedControl'
import { CLASS_NAMES, classColor } from '../lib/classes'
import { metrics } from '../lib/metrics'

type Conf = '0.25' | '0.50'

const OPTIONS: { value: Conf; label: string }[] = [
  { value: '0.25', label: '0.25' },
  { value: '0.50', label: '0.50' },
]

// The before and after of adding undamaged cars to training, measured on clean cars the model never saw.
export function CleanCars() {
  const [conf, setConf] = useState<Conf>('0.50')
  const before = metrics.clean_cars.before[`conf_${conf}`].all
  const after = metrics.clean_cars.after[`conf_${conf}`].all
  const maxCount = Math.max(...CLASS_NAMES.map((name) => before.by_class[name]), 1)

  const sides = [
    { label: 'Without negatives', data: before, color: 'var(--color-muted)' },
    { label: 'With negatives', data: after, color: 'var(--color-brass)' },
  ]

  return (
    <div className="card p-4 md:p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="eyebrow">Clean cars flagged as damaged</p>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted">Threshold</span>
          <SegmentedControl id="clean-conf" label="Confidence threshold" options={OPTIONS} value={conf} onChange={setConf} />
        </div>
      </div>

      <div className="mt-5 grid gap-6 sm:grid-cols-2">
        {sides.map((side) => (
          <div key={side.label}>
            <p className="text-sm text-muted">{side.label}</p>
            <p className="font-sans text-5xl font-semibold" style={{ color: side.color === 'var(--color-brass)' ? side.color : 'var(--color-text)' }}>
              <CountUp value={side.data.images_flagged_pct} decimals={1} suffix="%" />
            </p>
            <div className="mt-3 h-2 rounded-r-full bg-raised">
              <motion.div
                className="h-full rounded-r-full"
                style={{ backgroundColor: side.color }}
                animate={{ width: `${side.data.images_flagged_pct}%` }}
                transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              />
            </div>
            <p className="num mt-2 text-xs text-muted">{side.data.false_positives} false detections on {side.data.images} photos</p>
          </div>
        ))}
      </div>

      {/* The same comparison split by class. The muted bar is before and the colored bar is after. */}
      <div className="mt-7 border-t border-line pt-4">
        <p className="eyebrow mb-3">False detections by class</p>
        <ul className="flex flex-col gap-3">
          {CLASS_NAMES.map((name) => (
            <li key={name} className="grid grid-cols-[7.5rem_minmax(0,1fr)_4.5rem] items-center gap-3 text-sm">
              <span className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 flex-none rounded-[3px]" style={{ backgroundColor: classColor(name) }} />
                {name}
              </span>
              <span className="flex flex-col gap-0.5" title={`${name}: ${before.by_class[name]} before, ${after.by_class[name]} after`}>
                <motion.span
                  className="block h-1.5 rounded-r-full bg-muted/50"
                  animate={{ width: `${(before.by_class[name] / maxCount) * 100}%` }}
                  transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                  style={{ minWidth: before.by_class[name] ? 2 : 0 }}
                />
                <motion.span
                  className="block h-1.5 rounded-r-full"
                  animate={{ width: `${(after.by_class[name] / maxCount) * 100}%` }}
                  transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                  style={{ backgroundColor: classColor(name), minWidth: after.by_class[name] ? 2 : 0 }}
                />
              </span>
              <span className="num text-right text-xs text-muted">
                {before.by_class[name]} to <span className="text-text">{after.by_class[name]}</span>
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
