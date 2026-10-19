import { Suspense, lazy } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import { MotionConfig, motion } from 'motion/react'
import { ArrowUp } from 'lucide-react'
import { TopBar } from './components/TopBar'
import { ReportedStats } from './components/ReportedStats'
import Detect from './pages/Detect'

// GitHub's mark drawn inline, since lucide no longer ships brand icons. It uses currentColor so the link's hover color applies to it.
function GitHubMark() {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  )
}

// Loading the two chart-heavy pages only when they are visited, so the Detect page does not have to download the charting library.
const Training = lazy(() => import('./pages/Training'))
const Performance = lazy(() => import('./pages/Performance'))

// Scrolling smoothly back to the top, unless the visitor has asked for reduced motion.
function scrollToTop() {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  window.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' })
}

function App() {
  const location = useLocation()

  return (
    // Setting reducedMotion to "user" so every animation respects the visitor's reduced motion setting.
    <MotionConfig reducedMotion="user">
      <div className="flex min-h-dvh flex-col">
        <TopBar />
        {/* Keying the page on its path so each page fades in when it is navigated to. */}
        <motion.main
          key={location.pathname}
          className="mx-auto w-full max-w-5xl flex-1 px-5 py-10 md:py-14"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        >
          <Suspense fallback={null}>
            <Routes location={location}>
              <Route path="/" element={<Detect />} />
              <Route path="/training" element={<Training />} />
              <Route path="/performance" element={<Performance />} />
              <Route path="*" element={<Detect />} />
            </Routes>
          </Suspense>
        </motion.main>
        <footer className="border-t border-line">
          {/* The counter and copyright sit on the left and the back to top button on the right. The counter renders nothing until there are enough answers. */}
          <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-5 py-6 text-xs text-muted">
            <p className="flex flex-wrap items-center gap-x-2 gap-y-1">
              <ReportedStats />
              <span>© {new Date().getFullYear()} Aditya Bhati.</span>
              <a
                href="https://github.com/adbhati05/AI-Experiments/tree/main/DamageDetector"
                target="_blank"
                rel="noreferrer"
                aria-label="Source on GitHub"
                className="transition-colors duration-150 hover:text-brass"
              >
                <GitHubMark />
              </a>
            </p>
            <button
              type="button"
              onClick={scrollToTop}
              aria-label="Back to top"
              className="flex h-7 w-7 flex-none items-center justify-center rounded-md border border-muted transition-colors duration-150 hover:border-brass hover:text-brass"
            >
              <ArrowUp size={14} />
            </button>
          </div>
        </footer>
      </div>
    </MotionConfig>
  )
}

export default App
