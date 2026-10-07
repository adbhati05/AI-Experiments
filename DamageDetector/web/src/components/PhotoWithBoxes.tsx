import { AnimatePresence, motion } from 'motion/react'
import { classColor } from '../lib/classes'
import type { Detection } from '../lib/types'

export interface IndexedDetection {
  index: number
  detection: Detection
}

export function PhotoWithBoxes({ url, width, height, items, active, onActive, scanning = false }: {
  url: string
  width?: number
  height?: number
  items: IndexedDetection[]
  active: number | null
  onActive: (index: number | null) => void
  scanning?: boolean
}) {
  return (
    <div className="flex justify-center">
      <div className="relative inline-block overflow-hidden rounded-lg border border-line bg-card">
        <img
          src={url}
          alt="The uploaded car"
          className={`block max-h-[70vh] max-w-full transition-[filter,opacity] duration-300 ${scanning ? 'opacity-70 saturate-50' : ''}`}
        />

        {scanning && <div className="scan-line" />}

        {/* The box coordinates are in pixels of the original photo, so each one is placed as a percentage of the photo's size. That way the boxes follow the photo at any display size. */}
        {width && height && (
          <AnimatePresence>
            {items.map(({ index, detection }, order) => {
              const { x1, y1, x2, y2 } = detection.box
              const color = classColor(detection.class_name)
              const dimmed = active !== null && active !== index
              const labelInside = y1 / height < 0.08 // Moving the label inside the box when there is no room above it.
              const labelRight = x1 / width > 0.7 // Anchoring the label to the right edge for boxes near the right side of the photo.
              return (
                <motion.div
                  key={index}
                  onMouseEnter={() => onActive(index)}
                  onMouseLeave={() => onActive(null)}
                  initial={{ opacity: 0, scale: 0.96 }}
                  animate={{ opacity: dimmed ? 0.25 : 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.98 }}
                  transition={{ duration: 0.25, delay: active === null ? Math.min(order * 0.05, 0.4) : 0, ease: [0.16, 1, 0.3, 1] }}
                  className="absolute rounded-[3px] border-2"
                  style={{
                    left: `${(x1 / width) * 100}%`,
                    top: `${(y1 / height) * 100}%`,
                    width: `${((x2 - x1) / width) * 100}%`,
                    height: `${((y2 - y1) / height) * 100}%`,
                    borderColor: color,
                    backgroundColor: active === index ? `color-mix(in srgb, ${color} 14%, transparent)` : 'transparent',
                    zIndex: active === index ? 2 : 1,
                  }}
                >
                  {/* Every box carries a text label, since the class colors alone are not enough to tell classes apart for colorblind viewers. */}
                  <span
                    className={`num absolute whitespace-nowrap px-1.5 py-0.5 text-[11px] font-medium leading-none text-bg ${labelInside ? 'top-0' : '-top-0.5 -translate-y-full'} ${labelRight ? '-right-0.5' : '-left-0.5'}`}
                    style={{ backgroundColor: color, borderRadius: labelInside ? '0 0 3px 0' : '3px 3px 0 0' }}
                  >
                    {detection.class_name} {detection.confidence.toFixed(2)}
                  </span>
                </motion.div>
              )
            })}
          </AnimatePresence>
        )}
      </div>
    </div>
  )
}
