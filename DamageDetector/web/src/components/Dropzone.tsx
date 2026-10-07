import { useState } from 'react'
import type { DragEvent } from 'react'
import { motion } from 'motion/react'
import { ImageUp } from 'lucide-react'

// The four corner brackets that frame the dropzone like a viewfinder. They move inward while a file is dragged over.
function Corners({ active }: { active: boolean }) {
  const offset = active ? 18 : 10
  const corners = [
    { key: 'tl', style: { top: offset, left: offset }, border: 'border-t-2 border-l-2' },
    { key: 'tr', style: { top: offset, right: offset }, border: 'border-t-2 border-r-2' },
    { key: 'bl', style: { bottom: offset, left: offset }, border: 'border-b-2 border-l-2' },
    { key: 'br', style: { bottom: offset, right: offset }, border: 'border-b-2 border-r-2' },
  ]
  return (
    <>
      {corners.map((c) => (
        <span
          key={c.key}
          style={c.style}
          className={`pointer-events-none absolute h-5 w-5 transition-all duration-300 ease-out ${c.border} ${active ? 'border-brass' : 'border-line group-hover:border-brass/70'}`}
        />
      ))}
    </>
  )
}

export function Dropzone({ onFile }: { onFile: (file: File) => void }) {
  const [dragging, setDragging] = useState(false)

  function handleDrop(e: DragEvent<HTMLLabelElement>) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) onFile(file)
  }

  return (
    <motion.label
      htmlFor="photo-input"
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      animate={{ scale: dragging ? 1.01 : 1 }}
      transition={{ type: 'spring', stiffness: 300, damping: 24 }}
      className={`group relative flex cursor-pointer flex-col items-center justify-center gap-4 rounded-xl border px-6 py-16 text-center transition-colors duration-300 focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-brass md:py-24 ${dragging ? 'border-brass bg-raised' : 'border-line bg-card hover:bg-raised/60'}`}
    >
      <Corners active={dragging} />
      <motion.span
        animate={{ y: dragging ? -6 : 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 18 }}
        className={`flex h-14 w-14 items-center justify-center rounded-full border transition-colors duration-300 ${dragging ? 'border-brass text-brass' : 'border-line text-muted group-hover:border-brass/70 group-hover:text-brass'}`}
      >
        <ImageUp size={24} strokeWidth={1.75} />
      </motion.span>
      <span>
        <span className="block font-display text-xl font-medium tracking-wide">
          {dragging ? 'Release to scan' : 'Drop a photo of a car'}
        </span>
        <span className="mt-1 block text-sm text-muted">
          or <span className="text-brass underline-offset-4 group-hover:underline">browse your files</span>
        </span>
      </span>
      <span className="num text-xs text-muted">JPEG, PNG or WebP, up to 10 MB</span>
      {/* Limiting the picker to formats every browser can display, since Chrome and Firefox cannot show HEIC photos. */}
      <input
        id="photo-input"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="sr-only"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          e.target.value = ''
        }}
      />
    </motion.label>
  )
}
