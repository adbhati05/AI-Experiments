export function PageHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="mb-10 md:mb-12">
      <h1 className="font-display text-3xl font-semibold tracking-wide md:text-4xl">{title}</h1>
      <p className="mt-2 text-muted">{subtitle}</p>
    </div>
  )
}
