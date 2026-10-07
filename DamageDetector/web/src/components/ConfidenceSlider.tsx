import type { CSSProperties } from 'react'

const MIN = 0.1 // The API returns every detection down to 0.10, so that is as low as the slider can usefully go.
const MAX = 1

export function ConfidenceSlider({ value, defaultValue, onChange }: {
  value: number
  defaultValue: number
  onChange: (value: number) => void
}) {
  const fill = ((value - MIN) / (MAX - MIN)) * 100

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label htmlFor="confidence" className="eyebrow">Confidence threshold</label>
        <output htmlFor="confidence" className="num text-lg">{value.toFixed(2)}</output>
      </div>
      <input
        id="confidence"
        type="range"
        min={MIN}
        max={MAX}
        step={0.01}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="slider mt-2"
        style={{ '--fill': `${fill}%` } as CSSProperties}
      />
      <div className="num flex items-center justify-between text-xs text-muted">
        <span>{MIN.toFixed(2)}</span>
        {value !== defaultValue && (
          <button type="button" onClick={() => onChange(defaultValue)} className="text-brass underline-offset-4 hover:underline">
            Reset to {defaultValue.toFixed(2)}
          </button>
        )}
        <span>{MAX.toFixed(2)}</span>
      </div>
    </div>
  )
}
