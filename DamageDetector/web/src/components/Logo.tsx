// A placeholder mark. Swap the contents of this component for the final logo once it exists.
export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <circle cx="16" cy="16" r="11.5" fill="none" stroke="var(--color-brass)" strokeWidth="2.5" />
      <path d="M11 21 L15 14 L17 18 L21 11" fill="none" stroke="var(--color-text)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
