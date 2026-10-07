import { motion } from 'motion/react'

interface Option<T extends string> {
  value: T
  label: string
}

// A small toggle between options. The brass pill slides to the selected option through its shared layoutId.
export function SegmentedControl<T extends string>({ id, label, options, value, onChange }: {
  id: string
  label: string
  options: Option<T>[]
  value: T
  onChange: (value: T) => void
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex rounded-md border border-line bg-raised p-0.5">
      {options.map((option) => {
        const selected = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={selected}
            onClick={() => onChange(option.value)}
            className={`num relative rounded px-3 py-1 text-sm transition-colors duration-200 ${selected ? 'text-on-brass' : 'text-muted hover:text-text'}`}
          >
            {selected && (
              <motion.span
                layoutId={`segment-${id}`}
                className="absolute inset-0 rounded bg-brass"
                transition={{ type: 'spring', stiffness: 480, damping: 36 }}
              />
            )}
            <span className="relative">{option.label}</span>
          </button>
        )
      })}
    </div>
  )
}
