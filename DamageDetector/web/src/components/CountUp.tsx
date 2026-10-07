import { useEffect } from 'react'
import { animate, motion, useMotionValue, useReducedMotion, useTransform } from 'motion/react'

// Animating a number up to its value. A motion value is used instead of React state so the count does not re-render the page on every frame.
export function CountUp({ value, decimals = 0, suffix = '' }: { value: number; decimals?: number; suffix?: string }) {
  const reduced = useReducedMotion()
  const current = useMotionValue(0)
  const text = useTransform(current, (v) => v.toFixed(decimals) + suffix)

  useEffect(() => {
    if (reduced) {
      current.set(value)
      return
    }
    const controls = animate(current, value, { duration: 0.9, ease: [0.16, 1, 0.3, 1] })
    return () => controls.stop()
  }, [current, value, reduced])

  return <motion.span>{text}</motion.span>
}
