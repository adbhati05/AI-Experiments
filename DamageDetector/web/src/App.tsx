import { Suspense, lazy } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import { MotionConfig, motion } from 'motion/react'
import { TopBar } from './components/TopBar'
import Detect from './pages/Detect'

// Loading the two chart-heavy pages only when they are visited, so the Detect page does not have to download the charting library.
const Training = lazy(() => import('./pages/Training'))
const Performance = lazy(() => import('./pages/Performance'))

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
          {/* The data credit sits on the left and the copyright on the right. On small screens the two lines stack. */}
          <div className="mx-auto flex max-w-5xl flex-col gap-2 px-5 py-6 text-xs text-muted sm:flex-row sm:items-center sm:justify-between">
            <p>Trained on CarDD, with undamaged cars from CompCars. Model: YOLOv8 by Ultralytics.</p>
            <p className="sm:text-right">© {new Date().getFullYear()} Aditya Bhati. All rights reserved.</p>
          </div>
        </footer>
      </div>
    </MotionConfig>
  )
}

export default App
