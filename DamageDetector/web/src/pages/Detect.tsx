import { useEffect, useRef, useState } from 'react'
import { motion } from 'motion/react'
import { RotateCcw, TriangleAlert } from 'lucide-react'
import { Dropzone } from '../components/Dropzone'
import { PhotoWithBoxes } from '../components/PhotoWithBoxes'
import { ConfidenceSlider } from '../components/ConfidenceSlider'
import { DetectionList } from '../components/DetectionList'
import { MAX_UPLOAD_BYTES, predict } from '../lib/api'
import { metrics } from '../lib/metrics'
import { useServerStatus } from '../hooks/useServerStatus'
import type { ServerStatus } from '../hooks/useServerStatus'
import type { PredictResponse } from '../lib/types'

type State =
  | { status: 'idle' }
  | { status: 'loading'; url: string }
  | { status: 'done'; url: string; result: PredictResponse }
  | { status: 'error'; message: string }

const DEFAULT_THRESHOLD = metrics.model.conf // Starting the slider at the shipping threshold, 0.50.

const STATUS_TEXT: Record<ServerStatus, string> = {
  checking: 'Waking the model',
  ready: 'Model ready',
  down: 'Model unreachable',
}

function ServerBadge({ status }: { status: ServerStatus }) {
  const dot = status === 'ready' ? 'bg-good' : status === 'checking' ? 'bg-brass pulse-dot' : 'bg-muted'
  return (
    <p className="num flex items-center justify-center gap-2 text-xs text-muted">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {STATUS_TEXT[status]}
    </p>
  )
}

export default function Detect() {
  const [state, setState] = useState<State>({ status: 'idle' })
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD)
  const [active, setActive] = useState<number | null>(null)
  const [slow, setSlow] = useState(false)
  const previewUrl = useRef<string | null>(null)
  const server = useServerStatus()

  // Showing a longer message if the scan takes more than a few seconds, which happens when the server is waking up.
  useEffect(() => {
    if (state.status !== 'loading') return
    const timer = setTimeout(() => setSlow(true), 3500)
    return () => {
      clearTimeout(timer)
      setSlow(false)
    }
  }, [state.status])

  async function handleFile(file: File) {
    if (file.size > MAX_UPLOAD_BYTES) {
      setState({ status: 'error', message: 'That photo is larger than 10 MB. Please choose a smaller one.' })
      return
    }

    // Releasing the previous preview before creating a new one so repeated uploads do not pile up in memory.
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current)
    const url = URL.createObjectURL(file)
    previewUrl.current = url

    setActive(null)
    setThreshold(DEFAULT_THRESHOLD)
    setState({ status: 'loading', url })

    try {
      const result = await predict(file)
      setState({ status: 'done', url, result })
    } catch (err) {
      setState({ status: 'error', message: err instanceof Error ? err.message : 'Something went wrong.' })
    }
  }

  function reset() {
    setState({ status: 'idle' })
    setActive(null)
  }

  if (state.status === 'idle' || state.status === 'error') {
    return (
      <div className="mx-auto flex max-w-2xl flex-col gap-6">
        <div className="text-center">
          <h1 className="font-display text-3xl font-semibold tracking-wide md:text-5xl">Find the damage</h1>
          <p className="mx-auto mt-3 max-w-md text-muted">
            Dents, scratches, cracks, shattered glass, broken lamps and flat tires, marked on your photo.
          </p>
        </div>

        {state.status === 'error' && (
          <motion.p
            role="alert"
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            className="card flex items-start gap-3 px-4 py-3 text-sm"
          >
            <TriangleAlert size={18} className="mt-0.5 flex-none text-brass" />
            {state.message}
          </motion.p>
        )}

        <Dropzone onFile={handleFile} />
        <ServerBadge status={server} />
      </div>
    )
  }

  if (state.status === 'loading') {
    return (
      <div className="flex flex-col items-center gap-5">
        <PhotoWithBoxes url={state.url} items={[]} active={null} onActive={() => {}} scanning />
        <p className="num text-sm text-muted" role="status">
          {slow ? 'Still scanning. The server may be waking up.' : 'Scanning'}
        </p>
      </div>
    )
  }

  const { result, url } = state
  const all = result.detections.map((detection, index) => ({ index, detection }))
  const visible = all.filter((item) => item.detection.confidence >= threshold)
  const hidden = all.length - visible.length

  return (
    <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1.65fr)_minmax(0,1fr)]">
      <PhotoWithBoxes url={url} width={result.width} height={result.height} items={visible} active={active} onActive={setActive} />

      <motion.aside
        initial={{ opacity: 0, x: 12 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.35, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
        className="flex flex-col gap-6"
      >
        <div>
          <p className="eyebrow">Result</p>
          <p className="mt-1 font-display text-3xl font-semibold tracking-wide">
            {visible.length === 0 ? 'No damage found' : `${visible.length} ${visible.length === 1 ? 'area' : 'areas'} of damage`}
          </p>
          {hidden > 0 && (
            <p className="mt-1 text-sm text-muted">
              {hidden} lower-confidence {hidden === 1 ? 'detection is' : 'detections are'} hidden below the threshold.
            </p>
          )}
        </div>

        <div className="card p-4">
          <ConfidenceSlider value={threshold} defaultValue={DEFAULT_THRESHOLD} onChange={setThreshold} />
        </div>

        {visible.length > 0 && (
          <div className="card px-3 py-1">
            <DetectionList items={visible} active={active} onActive={setActive} />
          </div>
        )}

        <button
          type="button"
          onClick={reset}
          className="flex items-center justify-center gap-2 rounded-md bg-brass px-4 py-2.5 text-sm font-semibold text-on-brass transition-transform duration-150 hover:brightness-110 active:scale-[0.98]"
        >
          <RotateCcw size={16} />
          Scan another photo
        </button>
      </motion.aside>
    </div>
  )
}
