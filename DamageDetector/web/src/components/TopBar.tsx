import { useEffect, useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'motion/react'
import { Menu, X } from 'lucide-react'
import { Logo } from './Logo'

const LINKS = [
  { to: '/', label: 'Detect' },
  { to: '/training', label: 'Training' },
  { to: '/performance', label: 'Performance' },
]

export function TopBar() {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()

  // Closing the mobile menu with the Escape key while it is open.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/85 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-5 md:h-16">
        <NavLink to="/" onClick={() => setOpen(false)} className="flex items-center gap-2.5">
          <Logo className="h-7 w-7 md:h-8 md:w-8" />
          <span className="font-display text-lg font-semibold tracking-wide md:text-xl">
            Damage<span className="text-brass">Detector</span>
          </span>
        </NavLink>

        {/* Full navigation, shown from the md breakpoint up. The brass underline slides between links through its shared layoutId. */}
        <nav aria-label="Pages" className="hidden items-center gap-1 md:flex">
          {LINKS.map((link) => {
            const active = pathname === link.to
            return (
              <NavLink
                key={link.to}
                to={link.to}
                className={`relative px-3 py-2 text-sm transition-colors duration-200 ${active ? 'text-text' : 'text-muted hover:text-text'}`}
              >
                {link.label}
                {active && (
                  <motion.span
                    layoutId="nav-underline"
                    className="absolute inset-x-3 -bottom-px h-0.5 rounded-full bg-brass"
                    transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                  />
                )}
              </NavLink>
            )
          })}
        </nav>

        {/* On small screens the links collapse behind this button. */}
        <button
          type="button"
          className="-mr-2 flex h-10 w-10 items-center justify-center rounded-md text-text md:hidden"
          aria-label={open ? 'Close menu' : 'Open menu'}
          aria-expanded={open}
          aria-controls="mobile-menu"
          onClick={() => setOpen((v) => !v)}
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={open ? 'close' : 'open'}
              initial={{ rotate: -45, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 45, opacity: 0 }}
              transition={{ duration: 0.14 }}
              className="flex"
            >
              {open ? <X size={22} /> : <Menu size={22} />}
            </motion.span>
          </AnimatePresence>
        </button>
      </div>

      <AnimatePresence>
        {open && (
          <motion.nav
            id="mobile-menu"
            aria-label="Pages"
            className="absolute inset-x-0 top-full overflow-hidden border-y border-line bg-bg md:hidden"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
          >
            <ul className="mx-auto flex max-w-5xl flex-col px-5 py-2">
              {LINKS.map((link, i) => {
                const active = pathname === link.to
                return (
                  <motion.li
                    key={link.to}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.04 * i, duration: 0.2 }}
                  >
                    <NavLink
                      to={link.to}
                      onClick={() => setOpen(false)}
                      className={`flex items-center gap-3 py-3 text-base ${active ? 'text-text' : 'text-muted'}`}
                    >
                      <span className={`h-1.5 w-1.5 rounded-full ${active ? 'bg-brass' : 'bg-line'}`} />
                      {link.label}
                    </NavLink>
                  </motion.li>
                )
              })}
            </ul>
          </motion.nav>
        )}
      </AnimatePresence>
    </header>
  )
}
