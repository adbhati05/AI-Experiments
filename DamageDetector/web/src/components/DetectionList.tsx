import { AnimatePresence, motion } from 'motion/react'
import { classColor } from '../lib/classes'
import type { IndexedDetection } from './PhotoWithBoxes'

export function DetectionList({ items, active, onActive }: {
  items: IndexedDetection[]
  active: number | null
  onActive: (index: number | null) => void
}) {
  const sorted = [...items].sort((a, b) => b.detection.confidence - a.detection.confidence)

  return (
    <ul className="flex flex-col">
      <AnimatePresence initial={false}>
        {sorted.map(({ index, detection }) => (
          <motion.li
            key={index}
            layout
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-b border-line last:border-b-0"
          >
            {/* Hovering or focusing a row highlights its box on the photo. */}
            <button
              type="button"
              onMouseEnter={() => onActive(index)}
              onMouseLeave={() => onActive(null)}
              onFocus={() => onActive(index)}
              onBlur={() => onActive(null)}
              className={`flex w-full items-center gap-3 px-1 py-2.5 text-left text-sm transition-colors duration-150 ${active === index ? 'bg-raised' : ''}`}
            >
              <span className="h-2.5 w-2.5 flex-none rounded-[3px]" style={{ backgroundColor: classColor(detection.class_name) }} />
              <span className="flex-1">{detection.class_name}</span>
              <span className="num text-muted">{detection.confidence.toFixed(2)}</span>
            </button>
          </motion.li>
        ))}
      </AnimatePresence>
    </ul>
  )
}
