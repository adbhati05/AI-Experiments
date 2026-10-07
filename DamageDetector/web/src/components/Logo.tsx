// The logo is served from web/public, so the top bar and the browser tab icon both use the same file.
export function Logo({ className }: { className?: string }) {
  return <img src="/DD_Logo.svg" alt="" aria-hidden="true" className={className} />
}
