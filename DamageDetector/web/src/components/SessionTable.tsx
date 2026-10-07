import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Check, ChevronDown, Minus, X } from 'lucide-react'
import { SESSION_NOTES } from '../data/sessions'
import { metrics } from '../lib/metrics'

// Each verdict gets an icon as well as a label so the outcome never depends on color alone.
function Verdict({ verdict }: { verdict: string }) {
  if (verdict === 'baseline') return <span className="text-xs text-muted">baseline</span>
  const confirmed = verdict === 'confirmed'
  const Icon = confirmed ? Check : verdict === 'not confirmed' ? X : Minus
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs ${confirmed ? 'border-brass text-brass' : 'border-line text-muted'}`}>
      <Icon size={12} strokeWidth={2.5} />
      {verdict}
    </span>
  )
}

export function SessionTable({ active, onActive }: { active: number | null; onActive: (id: number | null) => void }) {
  const [open, setOpen] = useState<number | null>(null)

  return (
    <div className="card overflow-hidden">
      <div className="eyebrow hidden grid-cols-[2.5rem_minmax(0,1fr)_6rem_5rem_4.5rem_10.5rem_1.5rem] items-center gap-3 border-b border-line px-4 py-2.5 md:grid">
        <span>#</span>
        <span>Change</span>
        <span className="text-right">mAP50-95</span>
        <span className="text-right">Epochs</span>
        <span className="text-right">Hours</span>
        <span>Hypothesis</span>
        <span />
      </div>

      {metrics.sessions.map((session) => {
        const expanded = open === session.id
        const notes = SESSION_NOTES[session.id]
        return (
          <div key={session.id} className="border-b border-line last:border-b-0">
            {/* Hovering a row highlights that session's line in the chart below. Clicking it opens the hypothesis and result. */}
            <button
              type="button"
              aria-expanded={expanded}
              onClick={() => setOpen(expanded ? null : session.id)}
              onMouseEnter={() => onActive(session.id)}
              onMouseLeave={() => onActive(null)}
              onFocus={() => onActive(session.id)}
              onBlur={() => onActive(null)}
              className={`grid w-full grid-cols-[2rem_minmax(0,1fr)_auto_1.25rem] items-center gap-3 px-4 py-3 text-left transition-colors duration-150 md:grid-cols-[2.5rem_minmax(0,1fr)_6rem_5rem_4.5rem_10.5rem_1.5rem] ${active === session.id || expanded ? 'bg-raised' : ''}`}
            >
              <span className="num text-muted">{session.id}</span>
              <span className="min-w-0">
                <span className="block truncate">{session.label}</span>
                <span className="block truncate text-xs text-muted">{session.change}</span>
              </span>
              <span className="num text-right">{session.val.map50_95.toFixed(3)}</span>
              <span className="num hidden text-right text-muted md:block">{session.epochs_run}</span>
              <span className="num hidden text-right text-muted md:block">{session.wall_time_hours.toFixed(1)}</span>
              <span className="hidden md:block"><Verdict verdict={session.verdict} /></span>
              <motion.span animate={{ rotate: expanded ? 180 : 0 }} transition={{ duration: 0.2 }} className="flex justify-end text-muted">
                <ChevronDown size={16} />
              </motion.span>
            </button>

            <AnimatePresence initial={false}>
              {expanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                  className="overflow-hidden bg-raised"
                >
                  <dl className="grid gap-x-8 gap-y-3 px-4 pb-4 pt-1 text-sm md:grid-cols-2 md:pl-[4.25rem]">
                    <div>
                      <dt className="eyebrow">Hypothesis</dt>
                      <dd className="mt-1">{notes.hypothesis}</dd>
                    </div>
                    <div>
                      <dt className="eyebrow">Result</dt>
                      <dd className="mt-1">{notes.result}</dd>
                    </div>
                    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-muted md:col-span-2">
                      <span className="md:hidden"><Verdict verdict={session.verdict} /></span>
                      <span className="num">{session.model} at {session.imgsz} px</span>
                      <span className="num">batch {session.batch}</span>
                      <span className="num md:hidden">{session.epochs_run} epochs, {session.wall_time_hours.toFixed(1)} h</span>
                      <span className="num">best epoch {session.best_epoch}</span>
                    </div>
                  </dl>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )
      })}
    </div>
  )
}
