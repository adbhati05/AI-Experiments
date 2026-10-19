import { useState } from 'react'
import type { ReactNode } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { CircleCheck, MessageSquare } from 'lucide-react'
import { submitFeedback } from '../lib/api'
import { CLASS_NAMES, classColor } from '../lib/classes'
import type { ClassName } from '../lib/classes'

type Status = 'closed' | 'answering' | 'sending' | 'saved' | 'failed'

// Each question slides open under the previous one once that one is answered.
const reveal = {
  initial: { opacity: 0, height: 0 },
  animate: { opacity: 1, height: 'auto' },
  exit: { opacity: 0, height: 0 },
  transition: { duration: 0.25, ease: [0.16, 1, 0.3, 1] as const },
}

function Pill({ selected, disabled, onClick, children }: {
  selected: boolean
  disabled?: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      disabled={disabled}
      onClick={onClick}
      className={`flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm transition-colors duration-150 disabled:opacity-60 ${
        selected ? 'border-brass bg-brass text-on-brass' : 'border-line bg-raised text-text hover:border-muted'
      }`}
    >
      {children}
    </button>
  )
}

function YesNo({ question, value, disabled, onAnswer }: {
  question: string
  value: boolean | null
  disabled: boolean
  onAnswer: (answer: boolean) => void
}) {
  return (
    <div role="group" aria-label={question}>
      <p className="text-sm">{question}</p>
      <div className="mt-2 flex gap-2">
        <Pill selected={value === true} disabled={disabled} onClick={() => onAnswer(true)}>Yes</Pill>
        <Pill selected={value === false} disabled={disabled} onClick={() => onAnswer(false)}>No</Pill>
      </div>
    </div>
  )
}

// The optional questionnaire under a scan's results. It is keyed by the prediction token in Detect.tsx, so a new scan always starts it closed.
export function Feedback({ token, threshold, shownClasses }: {
  token: string
  threshold: number
  shownClasses: ClassName[]
}) {
  const [status, setStatus] = useState<Status>('closed')
  const [hasDamage, setHasDamage] = useState<boolean | null>(null)
  const [secondAnswer, setSecondAnswer] = useState<boolean | null>(null)
  const [missed, setMissed] = useState<ClassName[]>([])

  // Only the "damage, but not detected accurately" path asks which classes were on the car.
  const needsClasses = hasDamage === true && secondAnswer === false
  const busy = status === 'sending'

  // Sending the threshold and visible classes as they are at the moment of answering, since the user may have moved the slider since the scan.
  async function send(damage: boolean, second: boolean, classes: ClassName[]) {
    setStatus('sending')
    const result = await submitFeedback({
      prediction_token: token,
      has_damage: damage,
      // Translating the second answer so true always means the model got it right. "Detected any damage?" on a clean car is right when the answer is no.
      model_correct: damage ? second : !second,
      missed_classes: classes,
      threshold,
      shown_classes: shownClasses,
    })
    // A repeated answer means this scan is already counted, so it is shown the same as a saved one.
    setStatus(result === 'failed' ? 'failed' : 'saved')
  }

  // Changing the first answer clears the later ones, since the second question's wording depends on it.
  function answerFirst(answer: boolean) {
    setStatus('answering')
    setHasDamage(answer)
    setSecondAnswer(null)
    setMissed([])
  }

  function answerSecond(answer: boolean) {
    setStatus('answering')
    setSecondAnswer(answer)
    if (hasDamage === true && answer === false) return // Waiting for the class chips before sending.
    send(hasDamage!, answer, [])
  }

  function toggleClass(name: ClassName) {
    setMissed((current) => (current.includes(name) ? current.filter((item) => item !== name) : [...current, name]))
  }

  if (status === 'closed') {
    return (
      <button
        type="button"
        onClick={() => setStatus('answering')}
        className="card flex items-center gap-3 px-4 py-3 text-left text-sm transition-colors duration-150 hover:border-muted"
      >
        <MessageSquare size={18} className="flex-none text-brass" />
        <span className="flex-1">
          How did the model do? <span className="text-muted">Optional, three quick questions at most.</span>
        </span>
      </button>
    )
  }

  if (status === 'saved') {
    return (
      <motion.p initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} className="card flex items-center gap-2 px-4 py-3 text-sm" role="status">
        <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 420, damping: 18, delay: 0.05 }}>
          <CircleCheck size={18} className="text-good" />
        </motion.span>
        Thanks, your answer was saved.
      </motion.p>
    )
  }

  return (
    <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} className="card flex flex-col px-4 py-3">
      <p className="eyebrow">Feedback</p>

      <div className="mt-2">
        <YesNo question="Did your photo show a car with damage?" value={hasDamage} disabled={busy} onAnswer={answerFirst} />
      </div>

      <AnimatePresence initial={false}>
        {hasDamage !== null && (
          // Keying by the first answer so the question animates in again when its wording changes.
          <motion.div key={String(hasDamage)} {...reveal} className="overflow-hidden">
            <div className="pt-4">
              <YesNo
                question={hasDamage ? 'Did the model detect the damage accurately?' : 'Did the model detect any damage?'}
                value={secondAnswer}
                disabled={busy}
                onAnswer={answerSecond}
              />
            </div>
          </motion.div>
        )}

        {needsClasses && (
          <motion.div key="classes" {...reveal} className="overflow-hidden">
            <div className="pt-4">
              <p className="text-sm">What damage was on the car? <span className="text-muted">Pick any that apply.</span></p>
              <div className="mt-2 flex flex-wrap gap-2">
                {CLASS_NAMES.map((name) => (
                  <Pill key={name} selected={missed.includes(name)} disabled={busy} onClick={() => toggleClass(name)}>
                    <span className="h-2.5 w-2.5 rounded-[3px]" style={{ backgroundColor: classColor(name) }} />
                    {name}
                  </Pill>
                ))}
              </div>
              <button
                type="button"
                disabled={busy || missed.length === 0}
                onClick={() => send(true, false, missed)}
                className="mt-3 rounded-md bg-brass px-4 py-2 text-sm font-semibold text-on-brass transition-transform duration-150 hover:brightness-110 active:scale-[0.98] disabled:opacity-50 disabled:hover:brightness-100"
              >
                {busy ? 'Sending' : 'Send'}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {status === 'failed' && (
        <p role="alert" className="mt-3 text-sm text-muted">
          Your answer could not be saved.{' '}
          <button type="button" onClick={() => send(hasDamage!, secondAnswer!, needsClasses ? missed : [])} className="text-brass underline-offset-4 hover:underline">
            Try again
          </button>
        </p>
      )}

      <p className="mt-4 text-xs text-muted">Only your answers and the model's results are saved, never the photo.</p>
    </motion.div>
  )
}
