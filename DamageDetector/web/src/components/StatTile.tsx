import { CountUp } from './CountUp'

export function StatTile({ label, value, decimals, suffix, note }: {
  label: string
  value: number
  decimals: number
  suffix?: string
  note: string
}) {
  return (
    <div className="bg-card px-4 py-4">
      <p className="eyebrow">{label}</p>
      <p className="mt-1 text-3xl font-semibold md:text-4xl">
        <CountUp value={value} decimals={decimals} suffix={suffix} />
      </p>
      <p className="mt-1 text-xs text-muted">{note}</p>
    </div>
  )
}
