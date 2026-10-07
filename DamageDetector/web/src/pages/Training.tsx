import { useState } from 'react'
import { PageHeader } from '../components/PageHeader'
import { Section } from '../components/Section'
import { SessionTable } from '../components/SessionTable'
import { EpochChart } from '../components/EpochChart'
import { ClassTrends } from '../components/ClassTrends'
import { GeometryTable } from '../components/GeometryTable'
import { CleanCars } from '../components/CleanCars'
import { Accordion } from '../components/Accordion'
import { metrics } from '../lib/metrics'

const { splits, clean_eval_images } = metrics.dataset
const negatives = splits.train.negatives

// Where the data came from and what it was trained on, shown as a compact strip at the top of the page.
const SETUP = [
  { label: 'CarDD', value: `${(splits.train.images - negatives).toLocaleString()} / ${splits.val.images} / ${splits.test.images}`, note: 'train, val and test photos' },
  { label: 'CompCars', value: `${negatives} + ${clean_eval_images}`, note: 'undamaged cars for training and for testing' },
  { label: 'Hardware', value: 'M2 Pro, 16 GB', note: 'one laptop, no cloud GPU' },
]

export default function Training() {
  // Shared between the session table and the chart, so hovering either one highlights the same session in both.
  const [active, setActive] = useState<number | null>(null)

  return (
    <>
      <PageHeader title="Training" subtitle="Five runs, one change at a time, each with a prediction written down first." />

      <div className="flex flex-col gap-14">
        <dl className="grid gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3">
          {SETUP.map((item) => (
            <div key={item.label} className="bg-card px-4 py-3.5">
              <dt className="eyebrow">{item.label}</dt>
              <dd className="num mt-1 text-lg">{item.value}</dd>
              <dd className="text-xs text-muted">{item.note}</dd>
            </div>
          ))}
        </dl>

        <Section title="The five sessions" note="Two predictions failed and one held. Select a row for the details.">
          <SessionTable active={active} onActive={setActive} />
          <EpochChart active={active} onActive={setActive} />
        </Section>

        <Section title="What moved, class by class" note="AP50 on the validation split across the five sessions. Scratch never improved.">
          <ClassTrends />
        </Section>

        <Section title="Why three classes stay hard" note="Larger damage that fills its box scores higher. How often a class appears in training does not predict it.">
          <GeometryTable />
        </Section>

        <Section title="What the benchmark missed" note={`CarDD has no undamaged cars, so it could not show false alarms. These are ${clean_eval_images} clean cars the model never trained on.`}>
          <CleanCars />
        </Section>

        <div className="border-t border-line">
          <Accordion title="The bug that cost a day">
            <p>
              The first runs looked broken: losses rose and accuracy stayed near zero. The same run on the CPU learned normally, which pointed at
              PyTorch's Apple GPU backend and away from the data. Upgrading PyTorch fixed it and made inference 29 times faster.
            </p>
          </Accordion>
          <Accordion title="Not done yet">
            <ul className="flex list-disc flex-col gap-1.5 pl-5">
              <li>Whole-car photos still draw more false alarms than close-ups. Whole-car negatives should close that gap.</li>
              <li>One threshold per class, in place of a single 0.50.</li>
              <li>Segmentation masks. A crack fills only 26% of its box, so a box is a poor fit for it.</li>
            </ul>
          </Accordion>
        </div>
      </div>
    </>
  )
}
