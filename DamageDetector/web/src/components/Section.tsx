import type { ReactNode } from 'react'

export function Section({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-4">
      <div>
        <h2 className="font-display text-xl font-medium tracking-wide">{title}</h2>
        {note && <p className="mt-1 max-w-[65ch] text-sm text-muted">{note}</p>}
      </div>
      {children}
    </section>
  )
}
